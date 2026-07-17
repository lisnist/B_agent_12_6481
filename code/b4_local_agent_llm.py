from __future__ import annotations

import argparse
import codecs
import json
import sys
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from common.io_utils import append_jsonl, read_json, read_yaml, write_json
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from common.schemas import make_ai_message, validate_ai_message, validate_messages


_MODEL_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}


def _load_model_config(model_config: str | Path) -> tuple[Path, dict]:
    path = Path(model_config).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError("model.yaml must contain an object")
    return path, config


def _llm_source(config: dict) -> str:
    runtime = config.get("runtime", {})
    source = runtime.get("llm_source", "local") if isinstance(runtime, dict) else "local"
    if source in {"local", "transformers"}:
        return "local"
    if source in {"fastapi", "api"}:
        return "fastapi"
    if source in {"qwen_api", "qwen", "dashscope"}:
        return "qwen_api"
    raise ValueError("runtime.llm_source must be local, fastapi, or qwen_api")


def _generation_options(config: dict) -> dict:
    generation_config = config.get("generation", {})
    if not isinstance(generation_config, dict):
        generation_config = {}
    result = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 1024)),
        "do_sample": bool(generation_config.get("do_sample", False)),
    }
    if result["max_new_tokens"] <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if result["do_sample"]:
        for name in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if name in generation_config and generation_config[name] is not None:
                result[name] = generation_config[name]
    return result


def _max_input_tokens(config: dict) -> int | None:
    context = config.get("context", {})
    if not isinstance(context, dict):
        return None
    value = context.get("max_input_tokens")
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("context.max_input_tokens must be positive")
    return value


def _fastapi_config(config: dict, source: str | None = None) -> dict:
    source = source or _llm_source(config)
    config_key = "qwen_api" if source == "qwen_api" else "fastapi"
    api_config = config.get(config_key, {})
    if not isinstance(api_config, dict):
        raise ValueError(f"{config_key} config must be an object")
    base_url = api_config.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"{config_key}.base_url is required when runtime.llm_source={source}")
    generate_path = api_config.get("generate_path", "/generate")
    if not isinstance(generate_path, str) or not generate_path.startswith("/"):
        raise ValueError(f"{config_key}.generate_path must start with /")
    stream_path = api_config.get("stream_path", "/generate_stream")
    if not isinstance(stream_path, str) or not stream_path.startswith("/"):
        raise ValueError(f"{config_key}.stream_path must start with /")
    timeout = float(api_config.get("timeout_seconds", 600))
    if timeout <= 0:
        raise ValueError(f"{config_key}.timeout_seconds must be positive")
    return {
        "source": source,
        "base_url": base_url.rstrip("/"),
        "generate_path": generate_path,
        "stream_path": stream_path,
        "timeout_seconds": timeout,
        "api_key": api_config.get("api_key"),
        "model": api_config.get("model"),
    }


def _artifact_paths(artifact_dir: str | Path, stem: str | None) -> tuple[Path, Path, Path]:
    directory = Path(artifact_dir)
    prefix = f"{stem}_" if stem else ""
    return (
        directory / f"{prefix}raw_model_output.json",
        directory / f"{prefix}ai_message.json",
        directory / "llm_run_log.jsonl",
    )


def _extract_tool_result(message: dict) -> dict:
    try:
        result = json.loads(message["content"])
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("ToolMessage content is not a SkillResult JSON string") from exc
    if not isinstance(result, dict):
        raise ValueError("ToolMessage content must decode to an object")
    return result


def _three_points(text: str) -> list[str]:
    parts = []
    current = []
    strip_chars = " \t\r\n。！？!?"
    for char in text:
        current.append(char)
        if char in "\r\n。！？!?":
            part = "".join(current).strip(strip_chars)
            if part:
                parts.append(part)
            current = []
    tail = "".join(current).strip(strip_chars)
    if tail:
        parts.append(tail)

    points = []
    for part in parts:
        if part not in points:
            points.append(part)
        if len(points) == 3:
            break
    while len(points) < 3:
        points.append("工具结果未提供更多可提取内容")
    return points


def _mock_generate(messages: list[dict]) -> dict:
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    if not tool_messages:
        return make_ai_message(
            "",
            [
                {
                    "id": "call_001",
                    "name": "file_reader",
                    "args": {"path": "docs/agent_intro.txt", "max_chars": 2000},
                }
            ],
        )
    latest = tool_messages[-1]
    result = _extract_tool_result(latest)
    if latest.get("status") != "success" or result.get("status") != "success":
        error = result.get("error") or {}
        detail = error.get("message", "未知工具错误") if isinstance(error, dict) else str(error)
        return make_ai_message(f"工具调用失败，无法完成请求：{detail}", [])
    output = result.get("output") or {}
    content = output.get("content") if isinstance(output, dict) else None
    if not isinstance(content, str) or not content.strip():
        content = json.dumps(output, ensure_ascii=False)
    points = _three_points(content)
    answer = "三条中文要点如下：\n" + "\n".join(f"{index}. {point}" for index, point in enumerate(points, 1))
    return make_ai_message(answer, [])


def _parse_tool_calls_fragment(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    markers = ['"tool_calls":[', '\\"tool_calls\\":[']
    marker_index = -1
    marker = ""
    for item in markers:
        marker_index = raw_text.find(item)
        if marker_index != -1:
            marker = item
            break
    if marker_index == -1:
        raise original_error
    array_start = marker_index + marker.index("[")
    array_end = raw_text.rfind("]")
    if array_end < array_start:
        raise ValueError("model output contains tool_calls marker but no closing array")
    array_text = raw_text[array_start : array_end + 1]
    try:
        tool_calls = json.loads(array_text)
    except json.JSONDecodeError:
        tool_calls = json.loads(array_text.replace('\\"', '"'))
    if not isinstance(tool_calls, list) or not tool_calls:
        raise original_error
    return {"content": "", "tool_calls": tool_calls}


def _parse_json_with_backtick_tail(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    text = raw_text.strip()
    try:
        candidate, end_index = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        raise original_error
    trailing = text[end_index:].strip()
    if trailing and set(trailing) <= {"`", '"'}:
        return candidate
    raise original_error


def _decode_partial_json_string(fragment: str) -> str:
    text = fragment.rstrip()
    while text.endswith("\\"):
        text = text[:-1]
    while text:
        try:
            return json.loads(f'"{text}"')
        except json.JSONDecodeError:
            text = text[:-1]
    return fragment


def _json_string_value_start(raw_text: str, key: str, search_from: int = 0) -> int:
    key_token = json.dumps(key, ensure_ascii=False)
    while True:
        key_index = raw_text.find(key_token, search_from)
        if key_index == -1:
            return -1
        cursor = key_index + len(key_token)
        while cursor < len(raw_text) and raw_text[cursor].isspace():
            cursor += 1
        if cursor < len(raw_text) and raw_text[cursor] == ":":
            cursor += 1
            while cursor < len(raw_text) and raw_text[cursor].isspace():
                cursor += 1
            if cursor < len(raw_text) and raw_text[cursor] == '"':
                return cursor + 1
        search_from = key_index + 1


def _json_string_value(raw_text: str, key: str, search_from: int = 0) -> str | None:
    value_start = _json_string_value_start(raw_text, key, search_from)
    if value_start == -1:
        return None
    value = _partial_json_string(raw_text, value_start).strip()
    return value or None


def _partial_json_string(raw_text: str, start_index: int) -> str:
    chars = []
    escaped = False
    for char in raw_text[start_index:]:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\":
            chars.append(char)
            escaped = True
            continue
        if char == '"':
            break
        chars.append(char)
    return _decode_partial_json_string("".join(chars))


def _parse_content_fragment(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    value_start = _json_string_value_start(raw_text, "content")
    if value_start == -1:
        raise original_error
    content = _partial_json_string(raw_text, value_start).strip()
    if not content:
        raise original_error
    return {"content": content, "tool_calls": []}


def _parse_plain_text_output(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    text = raw_text.strip()
    if not text or text[0] in "[{" or text.startswith("```"):
        raise original_error
    return {"content": text, "tool_calls": []}


def _parse_malformed_tool_call_fragment(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    marker = json.dumps("tool_calls", ensure_ascii=False)
    marker_index = raw_text.find(marker)
    if marker_index == -1:
        raise original_error
    array_start = raw_text.find("[", marker_index)
    call_start = raw_text.find("{", array_start)
    if array_start == -1 or call_start == -1:
        raise original_error
    name = _json_string_value(raw_text, "name", call_start)
    writer_names = {
        "file_writer",
        "text_file_writer",
        "markdown_file_writer",
        "code_file_writer",
        "docx_writer",
    }
    if name not in writer_names:
        raise original_error
    args_marker = raw_text.find(json.dumps("args", ensure_ascii=False), call_start)
    args_start = raw_text.find("{", args_marker)
    if args_marker == -1 or args_start == -1:
        raise original_error
    filename = _json_string_value(raw_text, "filename", args_start)
    file_type = _json_string_value(raw_text, "file_type", args_start)
    file_content = _json_string_value(raw_text, "content", args_start)
    if not filename or file_content is None:
        raise original_error
    recovered_name = name
    if name == "file_writer":
        recovered_name = {
            "txt": "text_file_writer",
            "markdown": "markdown_file_writer",
            "docx": "docx_writer",
            "code": "code_file_writer",
        }.get(str(file_type or "").strip().lower(), "")
        if not recovered_name:
            raise original_error
    top_content = _json_string_value(raw_text, "content") or ""
    call_id = _json_string_value(raw_text, "id", call_start) or "call_001"
    return {
        "content": top_content,
        "tool_calls": [
            {
                "id": call_id,
                "name": recovered_name,
                "args": {
                    "filename": filename,
                    "content": file_content,
                },
            }
        ],
        "control": {
            "state": "acting",
            "action": "call_tools",
            "reason": f"recover explicit {recovered_name} tool call",
        },
    }


def _streaming_content_prefix(raw_text: str) -> str:
    value_start = _json_string_value_start(raw_text, "content")
    if value_start == -1:
        return ""
    return _partial_json_string(raw_text, value_start)


def _parse_error_fallback_content(raw_text: str) -> str:
    content = _streaming_content_prefix(raw_text).strip()
    if content:
        return content
    stripped = raw_text.strip()
    if stripped:
        return stripped[:1200]
    return "模型返回了空内容。"


def _normalize_agent_step(value: object, action: str | None) -> dict | None:
    if value is None:
        return None
    default_phase = "action" if action == "call_tools" else "final"
    phase_aliases = {
        "planning": "plan",
        "plan": "plan",
        "tool_calling": "action",
        "action": "action",
        "observing": "observation",
        "observation": "observation",
        "answering": "final",
        "finish": "final",
        "final": "final",
    }
    allowed = {"phase", "plan", "observation", "known_facts", "missing_info", "next_step"}
    if isinstance(value, dict):
        step = {key: value[key] for key in allowed if key in value}
        raw_phase = step.get("phase")
    else:
        step = {"next_step": str(value).strip()}
        raw_phase = value
    phase = phase_aliases.get(str(raw_phase or "").strip(), default_phase)
    step["phase"] = phase
    for key in ("plan", "observation", "next_step"):
        current = step.get(key)
        if current is None:
            step[key] = ""
        elif not isinstance(current, str):
            step[key] = json.dumps(current, ensure_ascii=False) if isinstance(current, (dict, list)) else str(current)
    for key in ("known_facts", "missing_info"):
        current = step.get(key)
        if current is None:
            step[key] = []
        elif isinstance(current, str):
            step[key] = [current] if current.strip() else []
        elif isinstance(current, list):
            step[key] = [str(item) for item in current if item is not None]
        else:
            step[key] = [str(current)]
    return step


def _candidate_to_message(candidate: dict, has_tool_messages: bool = False) -> tuple[dict, dict]:
    if not isinstance(candidate, dict):
        raise ValueError("model output JSON must be an object")
    expected_keys = {"content", "tool_calls", "control", "agent_step"}
    unknown_keys = set(candidate) - expected_keys
    if unknown_keys:
        raise ValueError(f"model output JSON contains unknown keys: {', '.join(sorted(unknown_keys))}")
    raw_tool_calls = candidate.get("tool_calls", [])
    normalized_control = None
    raw_control = candidate.get("control")
    if (not raw_tool_calls) and isinstance(raw_control, dict) and isinstance(raw_control.get("tool_calls"), list):
        raw_tool_calls = raw_control.get("tool_calls", [])
    if isinstance(raw_tool_calls, list):
        normalized_calls = []
        for call in raw_tool_calls:
            if not isinstance(call, dict) or "args" in call:
                normalized_calls.append(call)
                continue
            if "arguments" in call:
                normalized_calls.append({**call, "args": call.get("arguments")})
                continue
            if "parameters" in call:
                normalized_calls.append({**call, "args": call.get("parameters")})
                continue
            normalized_calls.append(call)
        raw_tool_calls = normalized_calls
    content = candidate.get("content", "")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)
    if isinstance(raw_control, dict):
        action = raw_control.get("action")
        if action not in {"call_tools", "finish"}:
            action = "call_tools" if raw_tool_calls else "finish"
        state = raw_control.get("state")
        if action == "call_tools" and state not in {"acting", "replanning"}:
            state = "acting" if not has_tool_messages else "replanning"
        if action == "finish" and state not in {"completed", "failed"}:
            state = "completed"
        reason = raw_control.get("reason", "")
        if action == "finish" and state == "failed" and not str(reason or "").strip():
            reason = content.strip() or "模型返回 failed 状态。"
        normalized_control = {
            "state": state,
            "action": action,
            "reason": reason if isinstance(reason, str) else str(reason),
        }
    else:
        action = "call_tools" if raw_tool_calls else "finish"
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": raw_tool_calls,
    }
    if normalized_control is not None:
        message["control"] = normalized_control
    if "agent_step" in candidate:
        agent_step = _normalize_agent_step(candidate["agent_step"], action)
        if agent_step is not None:
            message["agent_step"] = agent_step
    del has_tool_messages
    validate_ai_message(message)
    parsed_candidate = {
        "content": message["content"],
        "tool_calls": message["tool_calls"],
        "control": message["control"],
    }
    if "agent_step" in message:
        parsed_candidate["agent_step"] = message["agent_step"]
    return parsed_candidate, message


def _parse_model_output(raw_text: str, has_tool_messages: bool = False) -> tuple[dict, dict]:
    try:
        candidate = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        try:
            candidate = _parse_json_with_backtick_tail(raw_text, exc)
        except json.JSONDecodeError:
            try:
                candidate = _parse_tool_calls_fragment(raw_text, exc)
            except Exception:
                try:
                    candidate = _parse_malformed_tool_call_fragment(raw_text, exc)
                except Exception:
                    try:
                        candidate = _parse_content_fragment(raw_text, exc)
                    except Exception:
                        candidate = _parse_plain_text_output(raw_text, exc)
    return _candidate_to_message(candidate, has_tool_messages)


def parse_model_output(raw_text: str, has_tool_messages: bool = False) -> dict:
    """Expose B4's existing model-output parser for isolated protocol demos."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string")
    parsed_candidate, ai_message = _parse_model_output(raw_text, has_tool_messages)
    return {
        "parsed_candidate": parsed_candidate,
        "ai_message": ai_message,
    }


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


def _read_model_metadata(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_loader(transformers_module: Any, model_path: Path, model_config: dict) -> tuple[Any, Any, str]:
    requested = str(model_config.get("model_loader", "auto")).lower()
    metadata = _read_model_metadata(model_path)
    architectures = metadata.get("architectures") or []
    model_type = metadata.get("model_type")
    is_qwen35 = model_type == "qwen3_5" or "Qwen3_5ForConditionalGeneration" in architectures

    if requested in {"qwen3_5", "qwen35", "multimodal"} or (requested == "auto" and is_qwen35):
        processor_cls = getattr(transformers_module, "AutoProcessor", None)
        model_cls = getattr(transformers_module, "AutoModelForMultimodalLM", None)
        if processor_cls is not None and model_cls is not None:
            return processor_cls, model_cls, "multimodal"
        direct_cls = getattr(transformers_module, "Qwen3_5ForConditionalGeneration", None)
        if processor_cls is not None and direct_cls is not None:
            return processor_cls, direct_cls, "qwen3_5_direct"
        if requested != "auto":
            raise RuntimeError("transformers does not provide Qwen3.5 multimodal loader classes")

    tokenizer_cls = getattr(transformers_module, "AutoTokenizer")
    causal_cls = getattr(transformers_module, "AutoModelForCausalLM")
    return tokenizer_cls, causal_cls, "causal_lm"


def _from_pretrained_with_dtype(cls: Any, path: Path, kwargs: dict, dtype: Any) -> Any:
    try:
        return cls.from_pretrained(str(path), dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(str(path), torch_dtype=dtype, **kwargs)


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    raise TypeError("chat template output must be a tensor batch or dict")


def _decode_new_tokens(processor: Any, new_tokens: Any) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(new_tokens, skip_special_tokens=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode([new_tokens], skip_special_tokens=True)[0]
    raise TypeError("processor/tokenizer does not provide decode or batch_decode")


def _apply_chat_template(processor: Any, messages: list[dict]) -> Any:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def _model_cache_key(
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
    loader_name: str,
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
        loader_name,
    )


def _load_model_bundle(
    transformers_module: Any,
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
    model_config: dict,
) -> tuple[Any, Any]:
    processor_cls, model_cls, loader_name = _select_loader(transformers_module, model_path, model_config)
    cache_key = _model_cache_key(
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        device_map,
        max_memory,
        loader_name,
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        print("model_cache=hit", file=sys.stderr, flush=True)
        return cached

    print("model_cache=miss", file=sys.stderr, flush=True)
    processor = processor_cls.from_pretrained(
        str(tokenizer_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs = {
        "local_files_only": local_only,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = _from_pretrained_with_dtype(model_cls, model_path, model_kwargs, dtype)
    _MODEL_CACHE[cache_key] = (processor, model)
    return processor, model


def _format_prompt_images(prompt_messages: list[dict]) -> list[dict]:
    formatted = deepcopy(prompt_messages)
    for message in formatted:
        images = message.pop("images", [])
        if not images:
            continue
        text_content = message.get("content", "")
        message["content"] = [
            *({"type": "image", "url": image_url} for image_url in images),
            {"type": "text", "text": text_content},
        ]
    return formatted


def _prompt_messages_for_model(messages: list[dict], tools_schema: list[dict], prompt_ready: bool) -> list[dict]:
    if prompt_ready:
        prompt_messages = deepcopy(messages)
    else:
        from b1_agent_runtime_parts.b1_prompting import build_llm_prompt_messages

        prompt_messages = build_llm_prompt_messages(messages, tools_schema)
    return _format_prompt_images(prompt_messages)


def _prompt_json_generate(
    config_path: Path,
    config: dict,
    messages: list[dict],
    tools_schema: list[dict],
    prompt_ready: bool = False,
) -> str:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("prompt_json mode requires requirements-llm.txt") from exc
    model_config = config.get("model", {})
    model_setting = model_config.get("model_name_or_path")
    tokenizer_setting = model_config.get("tokenizer_name_or_path", model_setting)
    if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
        raise ValueError("model_name_or_path and tokenizer_name_or_path are required")
    model_path = resolve_from_file(model_setting, config_path)
    tokenizer_path = resolve_from_file(tokenizer_setting, config_path)
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path}")
    local_only = bool(model_config.get("local_files_only", True))
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    dtype = _dtype_value(torch, str(model_config.get("torch_dtype", "auto")))
    processor, model = _load_model_bundle(
        transformers,
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        model_config.get("device_map", "auto"),
        model_config.get("max_memory"),
        model_config,
    )
    prompt_messages = _prompt_messages_for_model(messages, tools_schema, prompt_ready)
    inputs = _apply_chat_template(processor, prompt_messages)
    input_length = int(inputs["input_ids"].shape[-1])
    max_input_tokens = _max_input_tokens(config)
    if max_input_tokens is not None and input_length > max_input_tokens:
        raise ValueError(f"prompt has {input_length} tokens, exceeding context.max_input_tokens={max_input_tokens}")
    device = next(model.parameters()).device
    inputs = _move_inputs_to_device(inputs, device)
    options = _generation_options(config)
    eos_token_id = getattr(processor, "eos_token_id", None)
    pad_token_id = getattr(processor, "pad_token_id", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        options.setdefault("pad_token_id", pad_token_id)
    elif eos_token_id is not None:
        options.setdefault("pad_token_id", eos_token_id)
    with torch.no_grad():
        generated = model.generate(**inputs, **options)
    new_tokens = generated[0][input_length:]
    return _decode_new_tokens(processor, new_tokens)


def _fastapi_prompt_json_generate(
    config_path: Path,
    config: dict,
    messages: list[dict],
    tools_schema: list[dict],
    prompt_ready: bool = False,
) -> str:
    del config_path
    api_config = _fastapi_config(config)
    prompt_messages = _prompt_messages_for_model(messages, tools_schema, prompt_ready)
    payload = {
        "messages": prompt_messages,
        "generation": _generation_options(config),
    }
    if api_config["source"] == "qwen_api":
        payload["response_format"] = {"type": "json_object"}
    if isinstance(api_config["model"], str) and api_config["model"].strip():
        payload["model"] = api_config["model"]
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if isinstance(api_config["api_key"], str) and api_config["api_key"]:
        headers["Authorization"] = f"Bearer {api_config['api_key']}"
    request = urllib.request.Request(
        url=api_config["base_url"] + api_config["generate_path"],
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=api_config["timeout_seconds"]) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FastAPI LLM request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"FastAPI LLM request failed: {exc}") from exc
    if not isinstance(response_data, dict) or not isinstance(response_data.get("raw_text"), str):
        raise ValueError("FastAPI LLM response must contain raw_text string")
    return response_data["raw_text"]


def _iter_fastapi_text_response(request: urllib.request.Request, timeout_seconds: float) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            while True:
                chunk = response.read(1)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    yield text
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FastAPI LLM stream request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"FastAPI LLM stream request failed: {exc}") from exc


def _fastapi_prompt_json_stream(
    config_path: Path,
    config: dict,
    messages: list[dict],
    tools_schema: list[dict],
    prompt_ready: bool = False,
) -> Iterator[str]:
    del config_path
    api_config = _fastapi_config(config)
    prompt_messages = _prompt_messages_for_model(messages, tools_schema, prompt_ready)
    payload = {
        "messages": prompt_messages,
        "generation": _generation_options(config),
    }
    if api_config["source"] == "qwen_api":
        payload["response_format"] = {"type": "json_object"}
    if isinstance(api_config["model"], str) and api_config["model"].strip():
        payload["model"] = api_config["model"]
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if isinstance(api_config["api_key"], str) and api_config["api_key"]:
        headers["Authorization"] = f"Bearer {api_config['api_key']}"
    request = urllib.request.Request(
        url=api_config["base_url"] + api_config["stream_path"],
        data=data,
        headers=headers,
        method="POST",
    )
    yield from _iter_fastapi_text_response(request, api_config["timeout_seconds"])


def _write_generation_artifacts(
    mode: str,
    backend: str,
    source: str,
    raw_text: str,
    prompt_messages: list[dict] | None,
    parsed_candidate: dict | None,
    ai_message: dict,
    status: str,
    error: dict | None,
    generated_at: str,
    artifact_dir: str | None,
    artifact_stem: str | None,
) -> None:
    if not artifact_dir:
        return
    raw_record = {
        "mode": mode,
        "backend": backend,
        "llm_source": source,
        "raw_text": raw_text,
        "prompt_messages": prompt_messages,
        "parsed_candidate": parsed_candidate,
        "status": status,
        "error": error,
        "generated_at": generated_at,
    }
    raw_path, message_path, log_path = _artifact_paths(artifact_dir, artifact_stem)
    write_json(raw_record, raw_path)
    write_json(ai_message, message_path)
    append_jsonl(
        {
            "timestamp": generated_at,
            "mode": mode,
            "status": status,
            "raw_output_path": str(raw_path),
            "ai_message_path": str(message_path),
            "error": error,
        },
        log_path,
    )


def generate_ai_message(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    mode: str = "prompt_json",
    artifact_dir: str | None = None,
    artifact_stem: str | None = None,
    prompt_ready: bool = False,
) -> dict:
    config_path, config = _load_model_config(model_config)
    messages = validate_messages(deepcopy(messages))
    if not isinstance(tools_schema, list):
        raise ValueError("tools_schema must be an array")
    generated_at = now_iso()
    source = "mock" if mode == "mock" else _llm_source(config)
    backend = "mock" if mode == "mock" else source
    prompt_messages = None
    if mode == "mock":
        ai_message = _mock_generate(messages)
        parsed_candidate = {
            "content": ai_message["content"],
            "tool_calls": ai_message["tool_calls"],
            "control": ai_message["control"],
        }
        raw_text = json.dumps(parsed_candidate, ensure_ascii=False)
        status = "success"
        error = None
    elif mode == "prompt_json":
        prompt_messages = _prompt_messages_for_model(messages, tools_schema, prompt_ready)
        if source == "local":
            raw_text = _prompt_json_generate(config_path, config, messages, tools_schema, prompt_ready)
        else:
            raw_text = _fastapi_prompt_json_generate(config_path, config, messages, tools_schema, prompt_ready)
        try:
            parsed_candidate, ai_message = _parse_model_output(
                raw_text,
                any(message.get("role") == "tool" for message in messages),
            )
            status = "success"
            error = None
        except Exception as exc:
            parsed_candidate = None
            ai_message = make_ai_message(_parse_error_fallback_content(raw_text), [])
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise ValueError("mode must be mock or prompt_json")
    if artifact_dir:
        _write_generation_artifacts(
            mode,
            backend,
            source,
            raw_text,
            prompt_messages,
            parsed_candidate,
            ai_message,
            status,
            error,
            generated_at,
            artifact_dir,
            artifact_stem,
        )
    return {
        "ai_message": ai_message,
        "status": status,
        "error": error,
        "raw_text": raw_text,
        "prompt_messages": prompt_messages,
    }


def _parse_json_object_output(raw_text: str) -> dict:
    try:
        candidate = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        candidate = _parse_json_with_backtick_tail(raw_text, exc)
    if not isinstance(candidate, dict):
        raise ValueError("model output JSON must be an object")
    return candidate


def generate_json_object(
    model_config: str,
    messages: list[dict],
    mode: str = "prompt_json",
    artifact_dir: str | None = None,
    artifact_stem: str | None = None,
    prompt_ready: bool = True,
) -> dict:
    """Generate a raw JSON object for B1-owned planning/observation stages.

    This is intentionally not an AIMessage parser. B4 still only talks to the
    model and validates transport-level shape; B1 owns the meaning of these
    stage objects.
    """
    if mode != "prompt_json":
        raise ValueError("generate_json_object only supports prompt_json mode")
    config_path, config = _load_model_config(model_config)
    messages = validate_messages(deepcopy(messages))
    generated_at = now_iso()
    source = _llm_source(config)
    backend = source
    prompt_messages = _prompt_messages_for_model(messages, [], prompt_ready)
    if source == "local":
        raw_text = _prompt_json_generate(config_path, config, messages, [], prompt_ready)
    else:
        raw_text = _fastapi_prompt_json_generate(config_path, config, messages, [], prompt_ready)
    parsed_json = None
    status = "success"
    error = None
    try:
        parsed_json = _parse_json_object_output(raw_text)
    except Exception as exc:
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)}
    if artifact_dir:
        raw_path, parsed_path, log_path = _artifact_paths(artifact_dir, artifact_stem)
        raw_record = {
            "mode": mode,
            "backend": backend,
            "llm_source": source,
            "raw_text": raw_text,
            "prompt_messages": prompt_messages,
            "parsed_json": parsed_json,
            "status": status,
            "error": error,
            "generated_at": generated_at,
        }
        write_json(raw_record, raw_path)
        write_json(parsed_json or {}, parsed_path)
        append_jsonl(
            {
                "timestamp": generated_at,
                "mode": mode,
                "status": status,
                "raw_output_path": str(raw_path),
                "parsed_json_path": str(parsed_path),
                "error": error,
            },
            log_path,
        )
    return {
        "json": parsed_json,
        "raw_text": raw_text,
        "status": status,
        "error": error,
        "prompt_messages": prompt_messages,
    }


def stream_ai_message(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    mode: str = "prompt_json",
    artifact_dir: str | None = None,
    artifact_stem: str | None = None,
    prompt_ready: bool = False,
) -> Iterator[dict]:
    config_path, config = _load_model_config(model_config)
    messages = validate_messages(deepcopy(messages))
    if not isinstance(tools_schema, list):
        raise ValueError("tools_schema must be an array")
    generated_at = now_iso()
    source = "mock" if mode == "mock" else _llm_source(config)
    backend = "mock" if mode == "mock" else source
    prompt_messages = None
    parsed_candidate = None
    status = "success"
    error = None
    emitted_chars = 0

    if mode == "mock":
        ai_message = _mock_generate(messages)
        parsed_candidate = {
            "content": ai_message["content"],
            "tool_calls": ai_message["tool_calls"],
            "control": ai_message["control"],
        }
        raw_text = json.dumps(parsed_candidate, ensure_ascii=False)
        if ai_message["content"]:
            yield {"type": "delta", "text": ai_message["content"]}
    elif mode == "prompt_json":
        prompt_messages = _prompt_messages_for_model(messages, tools_schema, prompt_ready)
        if source in {"fastapi", "qwen_api"}:
            raw_parts = []
            for chunk in _fastapi_prompt_json_stream(config_path, config, messages, tools_schema, prompt_ready):
                raw_parts.append(chunk)
                content = _streaming_content_prefix("".join(raw_parts))
                if len(content) > emitted_chars:
                    delta = content[emitted_chars:]
                    emitted_chars = len(content)
                    if delta:
                        yield {"type": "delta", "text": delta}
            raw_text = "".join(raw_parts)
        elif source == "local":
            raw_text = _prompt_json_generate(config_path, config, messages, tools_schema, prompt_ready)
        else:
            raise ValueError("runtime.llm_source must be local, fastapi, or qwen_api")
        try:
            parsed_candidate, ai_message = _parse_model_output(
                raw_text,
                any(message.get("role") == "tool" for message in messages),
            )
            if ai_message["content"] and (source == "local" or emitted_chars == 0):
                yield {"type": "delta", "text": ai_message["content"]}
        except Exception as exc:
            ai_message = make_ai_message(_parse_error_fallback_content(raw_text), [])
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise ValueError("mode must be mock or prompt_json")

    _write_generation_artifacts(
        mode,
        backend,
        source,
        raw_text,
        prompt_messages,
        parsed_candidate,
        ai_message,
        status,
        error,
        generated_at,
        artifact_dir,
        artifact_stem,
    )
    yield {
        "type": "done",
        "result": {
            "ai_message": ai_message,
            "status": status,
            "error": error,
            "raw_text": raw_text,
            "prompt_messages": prompt_messages,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one AIMessage with a local or mock LLM.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--tools_schema", required=True)
    parser.add_argument("--mode", choices=["mock", "prompt_json"], required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outdir = resolve_cli_path(args.outdir)
        generate_ai_message(
            str(resolve_cli_path(args.model_config)),
            read_json(resolve_cli_path(args.messages)),
            read_json(resolve_cli_path(args.tools_schema)),
            args.mode,
            str(outdir),
        )
        print(outdir / "ai_message.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
