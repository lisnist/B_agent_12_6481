from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8012
MAX_BATCH_ITEMS = 32
DEFAULT_BATCH_SIZE = 8
MAX_EMBEDDING_ITEMS = 64
DEFAULT_EMBEDDING_BATCH_SIZE = 16

DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
API_KEY: str | None = None


app = FastAPI(title="B4 Qwen API LLM FastAPI Server", version="1.0.0")


def _load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_first(*names: str, default: str | None = None) -> str | None:
    _load_dotenv()
    for name in names:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


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
        "max_tokens": int(options.get("max_tokens", options.get("max_new_tokens", 1024))),
    }
    if result["max_tokens"] <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    do_sample = bool(options.get("do_sample", False))
    if do_sample:
        if options.get("temperature") is not None:
            result["temperature"] = options["temperature"]
        if options.get("top_p") is not None:
            result["top_p"] = options["top_p"]
    else:
        result["temperature"] = 0
    return result


def _validate_response_format(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"type"}:
        raise ValueError("response_format must contain only type")
    response_type = value.get("type")
    if response_type not in {"text", "json_object"}:
        raise ValueError("response_format.type must be text or json_object")
    return {"type": response_type}


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
    base_model = payload.get("model")
    requests = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"request {index} must be an object")
        request_id = item.get("id", item.get("request_id"))
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError(f"request {index} id must be a string")
        model = item.get("model", base_model)
        requests.append(
            {
                "index": index,
                "id": request_id,
                "model": model if isinstance(model, str) and model.strip() else None,
                "messages": _validate_messages(item.get("messages")),
                "generation": _merge_generation_options(base_generation, item.get("generation")),
            }
        )
    return requests, _validate_batch_size(payload.get("batch_size"))


def _validate_embedding_payload(payload: dict[str, Any]) -> tuple[list[str], int, str]:
    raw_input = payload.get("texts")
    if raw_input is None:
        raw_input = payload.get("input")
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list):
        texts = [item for item in raw_input if isinstance(item, str)]
        if len(texts) != len(raw_input):
            raise ValueError("embedding input must contain only strings")
    else:
        raise ValueError("embedding input must be a string or string array")
    if not texts or len(texts) > MAX_EMBEDDING_ITEMS:
        raise ValueError(f"embedding input length must be between 1 and {MAX_EMBEDDING_ITEMS}")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        model = _env_first("QWEN_EMBEDDING_MODEL", "DASHSCOPE_EMBEDDING_MODEL", default=DEFAULT_QWEN_EMBEDDING_MODEL)
    return texts, _validate_batch_size(payload.get("batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE), str(model)


def _message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    converted = []
    for block in content:
        if block.get("type") == "text":
            converted.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            converted.append({"type": "image_url", "image_url": {"url": block.get("url", "")}})
    return converted


def _langchain_messages(messages: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    converted = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            role = "user"
        converted.append((role, _message_content(message.get("content", ""))))
    return converted


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value is not None:
                    parts.append(str(value))
        return "".join(parts)
    return str(content)


def _usage_from_response(response: Any) -> dict[str, int]:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage") or metadata.get("usage") or getattr(response, "usage_metadata", None) or {}
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _make_chat_model(model: str | None, options: dict[str, Any], streaming: bool = False) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("install langchain-openai before starting the Qwen API server") from exc
    api_key = _env_first("QWEN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing QWEN_API_KEY or DASHSCOPE_API_KEY in .env or environment")
    base_url = _env_first("QWEN_BASE_URL", "DASHSCOPE_BASE_URL", default=DEFAULT_QWEN_BASE_URL)
    model_name = model or _env_first("QWEN_MODEL", "DASHSCOPE_MODEL", default=DEFAULT_QWEN_MODEL)
    kwargs = dict(options)
    max_tokens = kwargs.pop("max_tokens", None)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        streaming=streaming,
        **kwargs,
    )


class QwenApiServer:
    def generate(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        model: str | None = None,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        llm = _make_chat_model(model, options, streaming=False)
        if response_format is not None:
            llm = llm.bind(response_format=response_format)
        response = llm.invoke(_langchain_messages(messages))
        return {
            "raw_text": _content_to_text(getattr(response, "content", "")),
            "usage": _usage_from_response(response),
        }

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        model: str | None = None,
        response_format: dict[str, str] | None = None,
    ) -> Iterator[str]:
        llm = _make_chat_model(model, options, streaming=True)
        if response_format is not None:
            llm = llm.bind(response_format=response_format)
        for chunk in llm.stream(_langchain_messages(messages)):
            text = _content_to_text(getattr(chunk, "content", ""))
            if text:
                yield text

    def generate_batch(self, requests: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
        results: list[dict[str, Any] | None] = [None] * len(requests)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        workers = min(batch_size, len(requests), MAX_BATCH_ITEMS)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.generate,
                    item["messages"],
                    item["generation"],
                    item.get("model"),
                ): item
                for item in requests
            }
            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                usage = result.get("usage", {})
                for key in total_usage:
                    total_usage[key] += int(usage.get(key, 0) or 0)
                results[item["index"]] = {
                    "index": item["index"],
                    "id": item["id"],
                    "raw_text": result["raw_text"],
                    "usage": usage,
                }
        return {
            "status": "success",
            "results": [item for item in results if item is not None],
            "usage": total_usage,
            "batch_size": batch_size,
        }

    def embed_texts(self, texts: list[str], batch_size: int, model: str) -> dict[str, Any]:
        api_key = _env_first("QWEN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("missing QWEN_API_KEY or DASHSCOPE_API_KEY in .env or environment")
        base_url = _env_first("QWEN_BASE_URL", "DASHSCOPE_BASE_URL", default=DEFAULT_QWEN_BASE_URL)
        if not isinstance(base_url, str) or not base_url.strip():
            raise RuntimeError("missing QWEN_BASE_URL or DASHSCOPE_BASE_URL")

        embeddings: list[dict[str, Any]] = []
        total_usage: dict[str, int] = {}
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            data = json.dumps({"model": model, "input": chunk}, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                base_url.rstrip("/") + "/embeddings",
                data=data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"embedding request failed with HTTP {exc.code}: {detail}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise RuntimeError("embedding response missing data array")
            for offset, item in enumerate(payload["data"]):
                if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                    raise RuntimeError("embedding response item missing embedding")
                raw_index = item.get("index")
                index = int(raw_index) if isinstance(raw_index, int) else offset
                embeddings.append(
                    {
                        "index": start + index,
                        "embedding": item["embedding"],
                    }
                )
            usage = payload.get("usage")
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, int):
                        total_usage[key] = total_usage.get(key, 0) + value
        embeddings.sort(key=lambda item: item["index"])
        return {
            "status": "success",
            "model": model,
            "embeddings": embeddings,
            "usage": total_usage,
            "batch_size": batch_size,
        }


MODEL_SERVER = QwenApiServer()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "b4_qwen_api_llm",
        "endpoints": ["/health", "/generate", "/generate_stream", "/generate_batch", "/embeddings"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "b4_qwen_api_llm",
        "model": _env_first("QWEN_MODEL", "DASHSCOPE_MODEL", default=DEFAULT_QWEN_MODEL),
        "embedding_model": _env_first("QWEN_EMBEDDING_MODEL", "DASHSCOPE_EMBEDDING_MODEL", default=DEFAULT_QWEN_EMBEDDING_MODEL),
        "base_url": _env_first("QWEN_BASE_URL", "DASHSCOPE_BASE_URL", default=DEFAULT_QWEN_BASE_URL),
        "has_api_key": bool(_env_first("QWEN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")),
    }


@app.post("/generate")
def generate(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    _check_auth(request)
    try:
        messages = _validate_messages(payload.get("messages"))
        options = _validate_generation_options(payload.get("generation"))
        response_format = _validate_response_format(payload.get("response_format"))
        model = payload.get("model")
        return MODEL_SERVER.generate(
            messages,
            options,
            model if isinstance(model, str) and model.strip() else None,
            response_format,
        )
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
        texts, batch_size, model = _validate_embedding_payload(payload)
        return MODEL_SERVER.embed_texts(texts, batch_size, model)
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
        response_format = _validate_response_format(payload.get("response_format"))
        model = payload.get("model")
        stream = MODEL_SERVER.generate_stream(
            messages,
            options,
            model if isinstance(model, str) and model.strip() else None,
            response_format,
        )
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
