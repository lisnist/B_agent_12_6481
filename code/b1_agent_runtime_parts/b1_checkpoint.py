from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from common.identifiers import validate_conversation_id


CHECKPOINT_VERSION = 1
CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"


def checkpoint_path(conversation_id: str) -> Path:
    safe_id = validate_conversation_id(conversation_id)
    return CHECKPOINT_ROOT / f"{safe_id}.json"


def save_checkpoint(conversation_id: str, payload: dict[str, Any]) -> Path:
    path = checkpoint_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": CHECKPOINT_VERSION,
        **deepcopy(payload),
        "conversation_id": validate_conversation_id(conversation_id),
    }
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)
    return path


def load_checkpoint(conversation_id: str) -> dict[str, Any]:
    path = checkpoint_path(conversation_id)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be a JSON object")
    if data.get("schema_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    return data


def checkpoint_metadata(conversation_id: str) -> dict[str, Any]:
    path = checkpoint_path(conversation_id)
    return {
        "conversation_id": validate_conversation_id(conversation_id),
        "path": str(path),
        "exists": path.is_file(),
    }
