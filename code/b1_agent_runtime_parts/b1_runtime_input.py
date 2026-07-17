from __future__ import annotations

from pathlib import Path

from common.io_utils import read_yaml
from common.schemas import normalize_history_messages


def _validate_runtime_input(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("runtime_input.json must contain an object")
    execution_mode = payload.setdefault("execution_mode", "integrated")
    if execution_mode not in {"integrated", "fixture"}:
        raise ValueError("execution_mode must be integrated or fixture")
    if "system_prompt_path" not in payload and "system_prompt" not in payload:
        payload["system_prompt_path"] = "../prompts/agent_system_prompts.json"
    required = ["conversation_id", "user_input", "toolset", "save_memory"]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"runtime input missing: {', '.join(missing)}")
    if not isinstance(payload["conversation_id"], str) or not payload["conversation_id"]:
        raise ValueError("conversation_id must be a non-empty string")
    if not isinstance(payload["user_input"], str) or not payload["user_input"].strip():
        raise ValueError("user_input must be a non-empty string")
    if payload["save_memory"] not in {"none", "conversation", "global"}:
        raise ValueError("save_memory must be none, conversation, or global")
    max_turns = payload.setdefault("max_turns", 10)
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0:
        raise ValueError("max_turns must be a positive integer")
    payload["history_messages"] = normalize_history_messages(payload.get("history_messages", []))
    input_images = payload.setdefault("input_images", [])
    if not isinstance(input_images, list) or not all(
        isinstance(item, str) and item.startswith("data:image/") for item in input_images
    ):
        raise ValueError("input_images must be an array of image data URLs")
    if execution_mode == "fixture":
        fixtures = payload.get("fixtures")
        if not isinstance(fixtures, dict):
            raise ValueError("fixture mode requires a fixtures object")
        required_fixtures = [
            "selected_memory_path",
            "tools_schema_path",
            "ai_messages_path",
            "tool_messages_path",
        ]
        missing_fixtures = [field for field in required_fixtures if not isinstance(fixtures.get(field), str)]
        if missing_fixtures:
            raise ValueError(f"fixtures missing paths: {', '.join(missing_fixtures)}")
        if payload["save_memory"] != "none":
            raise ValueError("fixture mode requires save_memory=none")
    else:
        selected_ids = payload.setdefault("selected_memory_ids", [])
        if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
            raise ValueError("selected_memory_ids must be a list of strings")
        payload.setdefault("use_global_memory", False)
        if not isinstance(payload["use_global_memory"], bool):
            raise ValueError("use_global_memory must be boolean")
    return payload


def _default_llm_mode(model_config: Path) -> str:
    config = read_yaml(model_config)
    return config.get("runtime", {}).get("default_mode", "mock")


def _runtime_base_file(runtime_base: str | Path | None) -> Path:
    if runtime_base is None:
        return (Path(__file__).resolve().parents[1] / "data" / "__runtime_payload__.json").resolve()
    base = Path(runtime_base).expanduser().resolve()
    if base.is_dir():
        return (base / "__runtime_payload__.json").resolve()
    return base

