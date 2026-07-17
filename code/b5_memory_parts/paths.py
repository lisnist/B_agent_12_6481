from __future__ import annotations

from pathlib import Path

from common.identifiers import validate_conversation_id
from common.io_utils import read_json, read_yaml
from common.path_utils import resolve_from_file


def _memory_paths(config_path: str | Path) -> dict[str, Path | int]:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define a memory object")
    memory = config["memory"]
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    root = resolve_from_file(memory["root_dir"], path)
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    return {
        "root": root,
        "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"],
        "conversation_db": root / memory.get("conversation_db_path", "conversation_store.sqlite3"),
        "max_chars": max_chars,
    }

def _conversation_db_path(config_path: str | Path) -> Path:
    return Path(_memory_paths(config_path)["conversation_db"])

def _read_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    return index

def _safe_conversation_id(conversation_id: str) -> str:
    return validate_conversation_id(conversation_id)
