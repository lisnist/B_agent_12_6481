from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "agent_system_prompts.json"
DEFAULT_CONVERSATION_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "conversation_prompts.json"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"prompt json must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def default_system_prompt(path: str | Path | None = None) -> str:
    prompt_file = Path(path).resolve() if path is not None else DEFAULT_SYSTEM_PROMPTS_PATH
    payload = _read_json(prompt_file)
    default = payload.get("default")
    if isinstance(default, dict) and isinstance(default.get("content"), str):
        return default["content"].strip()
    if isinstance(payload.get("content"), str):
        return payload["content"].strip()
    raise ValueError(f"default system prompt missing content: {prompt_file}")


def load_system_prompt_from_path(path: str | Path) -> str:
    prompt_file = Path(path).resolve()
    if prompt_file.suffix.lower() == ".json":
        return default_system_prompt(prompt_file)
    return prompt_file.read_text(encoding="utf-8").strip()


def _conversation_store(path: str | Path | None = None) -> tuple[Path, dict]:
    store_path = Path(path).resolve() if path is not None else DEFAULT_CONVERSATION_PROMPTS_PATH
    if not store_path.exists():
        return store_path, {"default_prompt_id": "default_local_tool_agent", "conversations": {}}
    payload = _read_json(store_path)
    conversations = payload.setdefault("conversations", {})
    if not isinstance(conversations, dict):
        payload["conversations"] = {}
    payload.setdefault("default_prompt_id", "default_local_tool_agent")
    return store_path, payload


def get_conversation_prompt(
    conversation_id: str,
    store_path: str | Path | None = None,
    default_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string")
    prompt_text = default_system_prompt(default_path)
    path, store = _conversation_store(store_path)
    conversations = store.setdefault("conversations", {})
    entry = conversations.get(conversation_id)
    created = False
    if not isinstance(entry, dict):
        entry = {
            "prompt_id": store.get("default_prompt_id", "default_local_tool_agent"),
            "content": prompt_text,
        }
        conversations[conversation_id] = entry
        _write_json(path, store)
        created = True
    content = entry.get("content")
    if not isinstance(content, str) or not content.strip():
        content = prompt_text
        entry["content"] = content
        _write_json(path, store)
    return {
        "conversation_id": conversation_id,
        "prompt_id": entry.get("prompt_id", store.get("default_prompt_id", "default_local_tool_agent")),
        "content": content,
        "default_content": prompt_text,
        "created": created,
        "locked_default": True,
    }


def update_conversation_prompt(
    conversation_id: str,
    content: str,
    store_path: str | Path | None = None,
    default_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("system prompt content must be a non-empty string")
    path, store = _conversation_store(store_path)
    conversations = store.setdefault("conversations", {})
    entry = conversations.get(conversation_id)
    if not isinstance(entry, dict):
        entry = {
            "prompt_id": store.get("default_prompt_id", "default_local_tool_agent"),
            "content": default_system_prompt(default_path),
        }
        conversations[conversation_id] = entry
    entry["content"] = content.strip()
    _write_json(path, store)
    return get_conversation_prompt(conversation_id, path, default_path)


def delete_conversation_prompt(conversation_id: str, store_path: str | Path | None = None) -> bool:
    path, store = _conversation_store(store_path)
    conversations = store.setdefault("conversations", {})
    if conversation_id not in conversations:
        return False
    conversations.pop(conversation_id, None)
    _write_json(path, store)
    return True
