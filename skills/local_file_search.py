from __future__ import annotations

import re

from skills import format_workspace_source, resolve_workspace_path


SUPPORTED_SEARCH_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".py",
    ".log",
}
MAX_TOP_K = 20
MAX_FILE_CHARS_LIMIT = 100000


def _query_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query.casefold())
    terms: list[str] = []
    seen = set()
    for term in raw_terms:
        candidates = [term]
        if re.fullmatch(r"[\u4e00-\u9fff]+", term) and len(term) > 2:
            candidates.extend(term[index : index + 2] for index in range(len(term) - 1))
        for candidate in candidates:
            if len(candidate) < 2 or candidate in seen:
                continue
            seen.add(candidate)
            terms.append(candidate)
    return terms


def _snippet(text: str, terms: list[str], radius: int = 80) -> tuple[str, int | None]:
    lowered = text.casefold()
    positions = [lowered.find(term.casefold()) for term in terms]
    positions = [position for position in positions if position >= 0]
    first_position = min(positions) if positions else 0
    start = max(0, first_position - radius)
    end = min(len(text), start + radius * 2)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    line_number = text.count("\n", 0, first_position) + 1 if positions else None
    return prefix + text[start:end].replace("\n", " ").strip() + suffix, line_number


def _score_text(path_text: str, text: str, terms: list[str]) -> tuple[int, list[str]]:
    lowered_path = path_text.casefold()
    lowered_text = text.casefold()
    matched_terms = []
    score = 0
    for term in terms:
        path_hits = lowered_path.count(term)
        content_hits = lowered_text.count(term)
        if path_hits or content_hits:
            matched_terms.append(term)
            score += path_hits * 5 + content_hits
    return score, matched_terms


def local_file_search(
    query: str,
    root_dir: str = "docs",
    file_types: list[str] | None = None,
    top_k: int = 5,
    max_file_chars: int = 20000,
    *,
    data_root: str | None = None,
    allowed_roots: dict[str, str] | None = None,
    default_root: str = "data",
) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if top_k > MAX_TOP_K:
        raise ValueError(f"top_k must not exceed {MAX_TOP_K}")
    if not isinstance(max_file_chars, int) or isinstance(max_file_chars, bool) or max_file_chars <= 0:
        raise ValueError("max_file_chars must be a positive integer")
    if max_file_chars > MAX_FILE_CHARS_LIMIT:
        raise ValueError(f"max_file_chars must not exceed {MAX_FILE_CHARS_LIMIT}")
    search_root, root, root_alias = resolve_workspace_path(
        root_dir,
        data_root=data_root,
        allowed_roots=allowed_roots,
        default_root=default_root,
    )
    if not search_root.is_dir():
        raise FileNotFoundError(f"search directory not found: {root_dir}")
    if file_types is not None and (
        not isinstance(file_types, list) or not all(isinstance(item, str) for item in file_types)
    ):
        raise ValueError("file_types must be a list of strings")
    extensions = file_types or ["txt", "md"]
    normalized_extensions = {f".{item.lower().lstrip('.')}" for item in extensions}
    if not normalized_extensions.issubset(SUPPORTED_SEARCH_SUFFIXES):
        supported = ", ".join(sorted(SUPPORTED_SEARCH_SUFFIXES))
        raise ValueError(f"local_file_search only supports text-like files: {supported}")
    terms = _query_terms(query)
    if not terms:
        raise ValueError("query contains no searchable terms")
    results = []
    for path in sorted(search_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in normalized_extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned_text = text[:max_file_chars]
        source_text, relative_path = format_workspace_source(path, root, root_alias)
        score, matched_terms = _score_text(relative_path, scanned_text, terms)
        if score:
            snippet, line_number = _snippet(scanned_text, matched_terms)
            results.append(
                {
                    "path": source_text,
                    "relative_path": relative_path,
                    "root_alias": root_alias,
                    "score": score,
                    "matched_terms": matched_terms,
                    "line_number": line_number,
                    "snippet": snippet,
                    "scanned_chars": min(len(text), max_file_chars),
                    "truncated": len(text) > max_file_chars,
                }
            )
    results.sort(key=lambda item: (-item["score"], item["path"]))
    return {"query_terms": terms, "root_alias": root_alias, "results": results[:top_k]}
