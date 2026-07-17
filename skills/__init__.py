from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def _normalize_allowed_roots(
    data_root: str | None = None,
    allowed_roots: dict[str, str] | None = None,
) -> dict[str, Path]:
    roots: dict[str, Path] = {"data": Path(data_root).resolve() if data_root else DEFAULT_DATA_ROOT.resolve()}
    if isinstance(allowed_roots, dict):
        for alias, root in allowed_roots.items():
            if not isinstance(alias, str) or not alias.strip() or not isinstance(root, str):
                continue
            normalized_alias = alias.strip()
            if not normalized_alias.replace("_", "").replace("-", "").isalnum():
                continue
            roots[normalized_alias] = Path(root).resolve()
    return roots


def _match_allowed_root(candidate: Path, roots: dict[str, Path]) -> tuple[str, Path] | None:
    best: tuple[str, Path] | None = None
    for alias, root in roots.items():
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if best is None or len(root.parts) > len(best[1].parts):
            best = (alias, root)
    return best


def resolve_workspace_path(
    path: str,
    data_root: str | None = None,
    allowed_roots: dict[str, str] | None = None,
    default_root: str = "data",
) -> tuple[Path, Path, str]:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    roots = _normalize_allowed_roots(data_root, allowed_roots)
    root_alias = default_root if isinstance(default_root, str) and default_root in roots else "data"
    raw_path = path.strip()
    alias = None
    relative_text = raw_path
    alias_match = raw_path.split(":", 1)
    if (
        len(alias_match) == 2
        and len(alias_match[0]) > 1
        and alias_match[0] in roots
    ):
        alias = alias_match[0]
        relative_text = alias_match[1].lstrip("/\\")
    candidate = Path(raw_path).expanduser()
    if alias:
        root = roots[alias]
        candidate = (root / relative_text).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes allowed root '{alias}': {path}") from exc
        matched = alias, root
    elif candidate.is_absolute():
        candidate = candidate.resolve()
        matched = _match_allowed_root(candidate, roots)
        if matched is None:
            allowed = ", ".join(sorted(roots))
            raise ValueError(f"path is outside allowed roots ({allowed}): {path}")
    else:
        root = roots[root_alias]
        candidate = (root / candidate).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes allowed root '{root_alias}': {path}") from exc
        matched = root_alias, root
    return candidate, matched[1], matched[0]


def resolve_data_path(path: str, data_root: str | None = None) -> tuple[Path, Path]:
    candidate, root, _ = resolve_workspace_path(path, data_root=data_root)
    return candidate, root


def format_workspace_source(path: Path, root: Path, root_alias: str) -> tuple[str, str]:
    relative = path.relative_to(root).as_posix()
    source = relative if root_alias == "data" else f"{root_alias}:{relative}"
    return source, relative
