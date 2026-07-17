from __future__ import annotations

from datetime import datetime
from pathlib import Path

from skills import format_workspace_source, resolve_workspace_path
from skills.file_reader import SUPPORTED_SUFFIXES as FILE_READER_SUFFIXES


MAX_ENTRIES_LIMIT = 500
MAX_DEPTH_LIMIT = 8
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx"}


def _modified_time(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return None


def _kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    if path.exists():
        return "other"
    return "missing"


def _entry(path: Path, root: Path, root_alias: str) -> dict:
    source, relative_path = format_workspace_source(path, root, root_alias)
    kind = _kind(path)
    item = {
        "name": path.name or ".",
        "path": source,
        "relative_path": relative_path,
        "root_alias": root_alias,
        "kind": kind,
        "suffix": path.suffix.lower() if kind == "file" else "",
        "modified_time": _modified_time(path) if path.exists() else None,
    }
    if kind == "file":
        item["size_bytes"] = path.stat().st_size
    return {key: value for key, value in item.items() if value is not None}


def _normalize_file_types(file_types: list[str] | None) -> set[str] | None:
    if file_types is None:
        return None
    if not isinstance(file_types, list) or not all(isinstance(item, str) for item in file_types):
        raise ValueError("file_types must be an array of strings")
    return {f".{item.lower().lstrip('.')}" for item in file_types if item.strip()}


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in {"", "."})


def directory_list(
    path: str = ".",
    recursive: bool = False,
    max_depth: int = 1,
    file_types: list[str] | None = None,
    include_hidden: bool = False,
    max_entries: int = 100,
    *,
    data_root: str | None = None,
    allowed_roots: dict[str, str] | None = None,
    default_root: str = "data",
) -> dict:
    if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    if max_entries > MAX_ENTRIES_LIMIT:
        raise ValueError(f"max_entries must not exceed {MAX_ENTRIES_LIMIT}")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    if max_depth > MAX_DEPTH_LIMIT:
        raise ValueError(f"max_depth must not exceed {MAX_DEPTH_LIMIT}")
    extensions = _normalize_file_types(file_types)
    source, root, root_alias = resolve_workspace_path(
        path or ".",
        data_root=data_root,
        allowed_roots=allowed_roots,
        default_root=default_root,
    )
    if not source.exists():
        raise FileNotFoundError(f"path not found: {path}")

    entries: list[dict] = []
    if source.is_file():
        if extensions is None or source.suffix.lower() in extensions:
            entries.append(_entry(source, root, root_alias))
    elif source.is_dir():
        queue = [(child, 0) for child in sorted(source.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))]
        while queue and len(entries) < max_entries:
            current, depth = queue.pop(0)
            relative_from_source = current.relative_to(source)
            if not include_hidden and _is_hidden(relative_from_source):
                continue
            if current.is_dir():
                entries.append(_entry(current, root, root_alias))
                if recursive and depth < max_depth:
                    children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
                    queue.extend((child, depth + 1) for child in children)
            elif extensions is None or current.suffix.lower() in extensions:
                entries.append(_entry(current, root, root_alias))
    else:
        entries.append(_entry(source, root, root_alias))

    source_text, relative_path = format_workspace_source(source, root, root_alias)
    return {
        "path": source_text,
        "relative_path": relative_path,
        "root_alias": root_alias,
        "kind": _kind(source),
        "recursive": bool(recursive),
        "max_depth": max_depth,
        "entry_count": len(entries),
        "truncated": len(entries) >= max_entries,
        "entries": entries,
    }


def file_stat(
    path: str,
    *,
    data_root: str | None = None,
    allowed_roots: dict[str, str] | None = None,
    default_root: str = "data",
) -> dict:
    source, root, root_alias = resolve_workspace_path(
        path,
        data_root=data_root,
        allowed_roots=allowed_roots,
        default_root=default_root,
    )
    source_text, relative_path = format_workspace_source(source, root, root_alias)
    kind = _kind(source)
    suffix = source.suffix.lower() if kind in {"file", "missing"} else ""
    result = {
        "path": source_text,
        "relative_path": relative_path,
        "root_alias": root_alias,
        "exists": source.exists(),
        "kind": kind,
        "suffix": suffix,
        "modified_time": _modified_time(source) if source.exists() else None,
        "readable_by_file_reader": kind == "file" and suffix in FILE_READER_SUFFIXES,
        "analyzable_by_table_analyzer": kind == "file" and suffix in TABLE_SUFFIXES,
    }
    if kind == "file":
        result["size_bytes"] = source.stat().st_size
    if kind == "directory":
        result["child_count"] = sum(1 for _ in source.iterdir())
    if not source.exists():
        parent = source.parent
        parent_source, parent_relative = format_workspace_source(parent, root, root_alias)
        result["parent_exists"] = parent.exists()
        result["parent_path"] = parent_source
        result["parent_relative_path"] = parent_relative
    return {key: value for key, value in result.items() if value is not None}
