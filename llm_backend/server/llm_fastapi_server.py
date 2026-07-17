from __future__ import annotations

import json
import threading
from queue import Empty
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8012
MAX_BATCH_ITEMS = 32
DEFAULT_BATCH_SIZE = 8
MAX_EMBEDDING_ITEMS = 64
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_MAX_TOKENS = 512
API_KEY: str | None = None
_MODEL_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}


app = FastAPI(title="B4 Raw LLM FastAPI Server", version="1.1.0")


def read_yaml(path: str | Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements.txt") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    return candidate.resolve()


def resolve_from_file(path: str | Path, containing_file: str | Path) -> Path:
    return resolve_path(path, Path(containing_file).resolve().parent)


def _dtype_value(torch_module: Any, configured: str) -> Any:
    if configured == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if configured not in mapping:
        raise ValueError(f"unsupported torch_dtype: {configured}")
    return mapping[configured]


def _model_cache_key(
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[str, ...]:
    try:
        device_map_key = json.dumps(device_map, sort_keys=True, separators=(",", ":"))
    except TypeError:
        device_map_key = repr(device_map)
    try:
        max_memory_key = json.dumps(max_memory, sort_keys=True, separators=(",", ":"))
    except TypeError:
        max_memory_key = repr(max_memory)
    return (
        str(model_path),
        str(tokenizer_path),
        str(local_only),
        str(trust_remote_code),
        str(dtype),
        device_map_key,
        max_memory_key,
    )


def _load_model_bundle(
    auto_model: Any,
    auto_tokenizer: Any,
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[Any, Any]:
    cache_key = _model_cache_key(
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        device_map,
        max_memory,
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    tokenizer = auto_tokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )
    model = auto_model.from_pretrained(
        str(model_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
    )
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _api_error(status_code: int, message: str, error_type: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(HTTPException)
def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "server_error",
                "param": None,
                "code": None,
            }
        },
    )


def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    if request.headers.get("authorization", "") != f"Bearer {API_KEY}":
        raise _api_error(401, "invalid authorization token", "invalid_request_error")


def _model_config_path() -> Path:
    return Path(MODEL_CONFIG_PATH).expanduser().resolve()


def _load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError("model.yaml must contain an object")
    return config


def _validate_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has invalid role: {role}")
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise ValueError(f"message {index} content must be a string or non-empty content block array")
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") not in {"text", "image"}:
                raise ValueError(f"message {index} content block {block_index} must be text or image")
            if block["type"] == "text" and not isinstance(block.get("text"), str):
                raise ValueError(f"message {index} text block {block_index} requires text")
            if block["type"] == "image" and not isinstance(block.get("url"), str):
                raise ValueError(f"message {index} image block {block_index} requires url")
    return messages


def _validate_generation_options(options: Any) -> dict[str, Any]:
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("generation must be an object")
    result = {
        "max_new_tokens": int(options.get("max_new_tokens", 1024)),
        "do_sample": bool(options.get("do_sample", False)),
    }
    if result["max_new_tokens"] <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    for name in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if name in options:
            result[name] = options[name]
    return result


def _options_key(options: dict[str, Any]) -> str:
    try:
        return json.dumps(options, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return repr(sorted(options.items(), key=lambda item: item[0]))


def _merge_generation_options(base_options: Any, override_options: Any) -> dict[str, Any]:
    if base_options is None:
        base_options = {}
    if not isinstance(base_options, dict):
        raise ValueError("generation must be an object")
    if override_options is None:
        return _validate_generation_options(base_options)
    if not isinstance(override_options, dict):
        raise ValueError("item generation must be an object")
    return _validate_generation_options({**base_options, **override_options})


def _validate_batch_size(value: Any) -> int:
    if value is None:
        return DEFAULT_BATCH_SIZE
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size must be a positive integer") from exc
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return min(batch_size, MAX_BATCH_ITEMS)


def _validate_batch_requests(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    items = payload.get("requests")
    if items is None:
        items = payload.get("batch")
    if not isinstance(items, list) or not items:
        raise ValueError("requests must be a non-empty array")
    if len(items) > MAX_BATCH_ITEMS:
        raise ValueError(f"requests length cannot exceed {MAX_BATCH_ITEMS}")

    base_generation = payload.get("generation")
    requests = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"request {index} must be an object")
        request_id = item.get("id", item.get("request_id"))
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError(f"request {index} id must be a string")
        requests.append(
            {
                "index": index,
                "id": request_id,
                "messages": _validate_messages(item.get("messages")),
                "generation": _merge_generation_options(base_generation, item.get("generation")),
            }
        )
    return requests, _validate_batch_size(payload.get("batch_size"))


def _validate_embedding_payload(payload: dict[str, Any]) -> tuple[list[str], int, int]:
    raw_texts = payload.get("texts")
    if raw_texts is None:
        raw_texts = payload.get("input")
    if isinstance(raw_texts, str):
        texts = [raw_texts]
    elif isinstance(raw_texts, list):
        texts = raw_texts
    else:
        raise ValueError("texts or input must be a string or non-empty string array")
    if not texts or len(texts) > MAX_EMBEDDING_ITEMS:
        raise ValueError(f"embedding input length must be between 1 and {MAX_EMBEDDING_ITEMS}")
    clean_texts = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"embedding input {index} must be a non-empty string")
        clean_texts.append(text)
    batch_size = _validate_batch_size(payload.get("batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE)
    try:
        max_tokens = int(payload.get("max_tokens") or DEFAULT_EMBEDDING_MAX_TOKENS)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_tokens must be a positive integer") from exc
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return clean_texts, batch_size, max_tokens


class RawModelServer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._loaded_key: str | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def load(self) -> tuple[Any, Any, Any]:
        path = _model_config_path()
        key = str(path)
        with self._lock:
            if self._loaded_key == key and self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model, self._torch

            try:
                import torch
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError("install torch and transformers before starting the server") from exc

            config = _load_config(path)
            model_config = config.get("model", {})
            if not isinstance(model_config, dict):
                raise ValueError("model config must contain a model object")
            model_setting = model_config.get("model_name_or_path")
            tokenizer_setting = model_config.get("tokenizer_name_or_path", model_setting)
            if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
                raise ValueError("model_name_or_path and tokenizer_name_or_path are required")

            model_path = resolve_from_file(model_setting, path)
            tokenizer_path = resolve_from_file(tokenizer_setting, path)
            if not model_path.exists() or not tokenizer_path.exists():
                raise FileNotFoundError(f"local model path does not exist: {model_path}")

            dtype = _dtype_value(torch, str(model_config.get("torch_dtype", "auto")))
            tokenizer, model = _load_model_bundle(
                AutoModelForMultimodalLM,
                AutoProcessor,
                model_path,
                tokenizer_path,
                bool(model_config.get("local_files_only", True)),
                bool(model_config.get("trust_remote_code", False)),
                dtype,
                model_config.get("device_map", "auto"),
                model_config.get("max_memory"),
            )
            model.eval()
            self._loaded_key = key
            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            return tokenizer, model, torch

    def generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        processor, model, torch = self.load()
        with self._generate_lock:
            inputs = self._apply_chat_template(processor, messages)
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            input_length = int(inputs["input_ids"].shape[-1])
            tokenizer = self._text_tokenizer(processor)
            self._ensure_pad_token(tokenizer, options)
            with torch.no_grad():
                generated = model.generate(**inputs, **options)
            new_tokens = generated[0][input_length:]
            raw_text = self._decode(processor, new_tokens)
            return {
                "raw_text": raw_text,
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": int(new_tokens.shape[-1]),
                    "total_tokens": input_length + int(new_tokens.shape[-1]),
                },
            }

    def generate_batch(self, requests: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
        processor, model, torch = self.load()
        results: list[dict[str, Any] | None] = [None] * len(requests)
        grouped: dict[str, dict[str, Any]] = {}
        for item in requests:
            key = _options_key(item["generation"])
            grouped.setdefault(key, {"generation": item["generation"], "items": []})["items"].append(item)

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        with self._generate_lock:
            for group in grouped.values():
                options = dict(group["generation"])
                items = group["items"]
                for start in range(0, len(items), batch_size):
                    chunk = items[start : start + batch_size]
                    chunk_results = self._generate_batch_same_options(processor, model, torch, chunk, dict(options))
                    for result in chunk_results:
                        usage = result.get("usage", {})
                        total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                        total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
                        total_usage["total_tokens"] += int(usage.get("total_tokens", 0))
                        results[int(result["index"])] = result

        ordered_results = [item for item in results if item is not None]
        return {
            "status": "success",
            "results": ordered_results,
            "usage": total_usage,
            "batch_size": batch_size,
        }

    def embed_texts(self, texts: list[str], batch_size: int, max_tokens: int) -> dict[str, Any]:
        processor, model, torch = self.load()
        tokenizer = self._text_tokenizer(processor)
        embeddings = []
        token_counts = []
        with self._generate_lock:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                )
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                with torch.no_grad():
                    try:
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                    except TypeError:
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    hidden = hidden_states[-1]
                else:
                    hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise RuntimeError("model output does not include hidden states for embeddings")
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones(hidden.shape[:2], device=hidden.device)
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors = pooled.detach().float().cpu().tolist()
                embeddings.extend(vectors)
                token_counts.extend(int(value) for value in attention_mask.sum(dim=1).detach().cpu().tolist())
        dimension = len(embeddings[0]) if embeddings else 0
        return {
            "status": "success",
            "model": str(_model_config_path()),
            "dimension": dimension,
            "embeddings": [
                {"index": index, "embedding": vector, "token_count": token_counts[index]}
                for index, vector in enumerate(embeddings)
            ],
        }

    def _generate_batch_same_options(
        self,
        processor: Any,
        model: Any,
        torch: Any,
        items: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prompts = [self._render_chat_prompt(processor, item["messages"]) for item in items]
        tokenizer = self._text_tokenizer(processor)
        self._ensure_pad_token(tokenizer, options)
        old_padding_side = getattr(tokenizer, "padding_side", None)
        if old_padding_side is not None:
            tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            if old_padding_side is not None:
                tokenizer.padding_side = old_padding_side

        device = next(model.parameters()).device
        inputs = inputs.to(device)
        prompt_width = int(inputs["input_ids"].shape[-1])
        if "attention_mask" in inputs:
            prompt_lengths = [int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()]
        else:
            prompt_lengths = [prompt_width] * len(items)

        with torch.no_grad():
            generated = model.generate(**inputs, **options)

        pad_token_id = options.get("pad_token_id")
        results = []
        for row_index, item in enumerate(items):
            new_tokens = generated[row_index][prompt_width:]
            raw_text = self._decode(processor, new_tokens)
            if pad_token_id is None:
                completion_tokens = int(new_tokens.shape[-1])
            else:
                completion_tokens = int((new_tokens != pad_token_id).sum().item())
            prompt_tokens = prompt_lengths[row_index]
            results.append(
                {
                    "index": item["index"],
                    "id": item["id"],
                    "raw_text": raw_text,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        return results

    def generate_stream(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[str]:
        processor, model, torch = self.load()
        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise RuntimeError("transformers TextIteratorStreamer is required for streaming") from exc

        with self._generate_lock:
            inputs = self._apply_chat_template(processor, messages)
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            tokenizer = self._text_tokenizer(processor)
            self._ensure_pad_token(tokenizer, options)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=1.0,
            )
            generation_options = dict(options)
            generation_options["streamer"] = streamer
            errors: list[BaseException] = []

            def run_generation() -> None:
                try:
                    with torch.no_grad():
                        model.generate(**inputs, **generation_options)
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=run_generation, daemon=True)
            worker.start()
            while worker.is_alive():
                try:
                    chunk = next(streamer)
                except Empty:
                    continue
                except StopIteration:
                    break
                if chunk:
                    yield chunk
            worker.join()
            while True:
                try:
                    chunk = next(streamer)
                except (Empty, StopIteration):
                    break
                if chunk:
                    yield chunk
            if errors:
                yield f"\n[stream_error] {type(errors[0]).__name__}: {errors[0]}"

    @staticmethod
    def _text_tokenizer(processor: Any) -> Any:
        return getattr(processor, "tokenizer", processor)

    @classmethod
    def _decode(cls, processor: Any, tokens: Any) -> str:
        if hasattr(processor, "decode"):
            return processor.decode(tokens, skip_special_tokens=True)
        return cls._text_tokenizer(processor).decode(tokens, skip_special_tokens=True)

    @staticmethod
    def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> Any:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _render_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _ensure_pad_token(tokenizer: Any, options: dict[str, Any]) -> None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            options.setdefault("pad_token_id", tokenizer.pad_token_id)
            return
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("batch generation requires tokenizer.pad_token_id or tokenizer.eos_token_id")
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None and getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = eos_token
        options.setdefault("pad_token_id", tokenizer.eos_token_id)


MODEL_SERVER = RawModelServer()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "b4_raw_llm",
        "endpoints": ["/health", "/generate", "/generate_stream", "/generate_batch", "/embeddings"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "b4_raw_llm",
        "model_config": str(_model_config_path()),
    }


@app.post("/generate")
def generate(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    _check_auth(request)
    try:
        messages = _validate_messages(payload.get("messages"))
        options = _validate_generation_options(payload.get("generation"))
        return MODEL_SERVER.generate(messages, options)
    except ValueError as exc:
        raise _api_error(400, str(exc), "invalid_request_error") from exc
    except Exception as exc:
        raise _api_error(500, str(exc), "server_error") from exc


@app.post("/generate_batch")
def generate_batch(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    _check_auth(request)
    try:
        batch_requests, batch_size = _validate_batch_requests(payload)
        return MODEL_SERVER.generate_batch(batch_requests, batch_size)
    except ValueError as exc:
        raise _api_error(400, str(exc), "invalid_request_error") from exc
    except Exception as exc:
        raise _api_error(500, str(exc), "server_error") from exc


@app.post("/embeddings")
def embeddings(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    _check_auth(request)
    try:
        texts, batch_size, max_tokens = _validate_embedding_payload(payload)
        return MODEL_SERVER.embed_texts(texts, batch_size, max_tokens)
    except ValueError as exc:
        raise _api_error(400, str(exc), "invalid_request_error") from exc
    except Exception as exc:
        raise _api_error(500, str(exc), "server_error") from exc


@app.post("/generate_stream")
def generate_stream(payload: dict[str, Any], request: Request) -> StreamingResponse:
    _check_auth(request)
    try:
        messages = _validate_messages(payload.get("messages"))
        options = _validate_generation_options(payload.get("generation"))
        stream = MODEL_SERVER.generate_stream(messages, options)
        return StreamingResponse(stream, media_type="text/plain; charset=utf-8")
    except ValueError as exc:
        raise _api_error(400, str(exc), "invalid_request_error") from exc
    except Exception as exc:
        raise _api_error(500, str(exc), "server_error") from exc


def main() -> int:
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
