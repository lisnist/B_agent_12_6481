from __future__ import annotations

from urllib.parse import urlparse
from typing import Any


MAX_TOP_K = 10
DEFAULT_REGION = "cn-zh"
DEFAULT_SAFESEARCH = "moderate"
DEFAULT_TIMEOUT = 3
VALID_SEARCH_TYPES = {"auto", "text", "news"}
VALID_TIMELIMITS = {"d", "w", "m", "y"}


def _normalize_text(value: Any, max_chars: int | None = None) -> str:
    text = str(value or "").strip()
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _hostname(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def _normalize_result(raw: dict[str, Any], index: int, category: str) -> dict[str, Any] | None:
    url = _normalize_text(raw.get("href") or raw.get("url"))
    title = _normalize_text(raw.get("title") or raw.get("headline"), 160)
    snippet = _normalize_text(raw.get("body") or raw.get("description") or raw.get("snippet"), 500)
    if not url and not title and not snippet:
        return None
    source_name = _normalize_text(raw.get("source") or _hostname(url), 120)
    published = _normalize_text(raw.get("date") or raw.get("published") or raw.get("published_at"), 80)
    return {
        "rank": index,
        "title": title,
        "url": url,
        "source": url,
        "source_name": source_name,
        "published": published,
        "snippet": snippet,
        "category": category,
    }


def _format_text(results: list[dict[str, Any]]) -> str:
    blocks = []
    for item in results:
        parts = [f"{item['rank']}. {item.get('title') or '(untitled)'}"]
        if item.get("source_name") or item.get("published"):
            meta = " / ".join(part for part in (item.get("source_name"), item.get("published")) if part)
            parts.append(meta)
        if item.get("url"):
            parts.append(item["url"])
        if item.get("snippet"):
            parts.append(item["snippet"])
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _search_attempts(search_type: str) -> list[tuple[str, str]]:
    if search_type == "news":
        return [("news", "duckduckgo"), ("news", "auto")]
    if search_type == "text":
        return [("text", "duckduckgo"), ("text", "auto")]
    return [
        ("text", "duckduckgo"),
        ("news", "duckduckgo"),
        ("text", "auto"),
    ]


def _is_network_block_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    markers = (
        "ConnectError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "WinError 10013",
        "ProxyError",
    )
    return any(marker in text for marker in markers)


def _run_ddgs(
    category: str,
    backend: str,
    query: str,
    top_k: int,
    region: str,
    safesearch: str,
    timelimit: str | None,
    timeout: int,
) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError('Install ddgs first: pip install "ddgs>=9.14,<10"') from exc
    client = DDGS(timeout=timeout)
    method = client.news if category == "news" else client.text
    kwargs: dict[str, Any] = {
        "region": region,
        "safesearch": safesearch,
        "max_results": top_k,
        "backend": backend,
    }
    if timelimit:
        kwargs["timelimit"] = timelimit
    return method(query, **kwargs)


def web_search(
    query: str,
    top_k: int = 5,
    search_type: str = "auto",
    region: str = DEFAULT_REGION,
    timelimit: str | None = None,
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized_query = query.strip()
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        top_k = 5
    requested_top_k = top_k
    top_k = min(top_k, MAX_TOP_K)
    normalized_type = search_type.strip().lower() if isinstance(search_type, str) else "auto"
    if normalized_type not in VALID_SEARCH_TYPES:
        normalized_type = "auto"
    normalized_region = region.strip() if isinstance(region, str) and region.strip() else DEFAULT_REGION
    normalized_timelimit = timelimit.strip().lower() if isinstance(timelimit, str) and timelimit.strip() else None
    if normalized_timelimit not in VALID_TIMELIMITS:
        normalized_timelimit = None

    attempts: list[dict[str, Any]] = []
    last_error = ""
    for index, (category, backend) in enumerate(_search_attempts(normalized_type), 1):
        attempt: dict[str, Any] = {
            "attempt_index": index,
            "category": category,
            "backend": backend,
        }
        try:
            raw_results = _run_ddgs(
                category=category,
                backend=backend,
                query=normalized_query,
                top_k=top_k,
                region=normalized_region,
                safesearch=DEFAULT_SAFESEARCH,
                timelimit=normalized_timelimit,
                timeout=DEFAULT_TIMEOUT,
            )
        except Exception as exc:
            last_error = str(exc)
            attempt["status"] = "error"
            attempt["error"] = {"type": type(exc).__name__, "message": str(exc)}
            attempts.append(attempt)
            if _is_network_block_error(exc):
                break
            continue

        normalized = []
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            item = _normalize_result(raw, len(normalized) + 1, category)
            if item is not None:
                normalized.append(item)
        attempt["status"] = "success"
        attempt["result_count"] = len(normalized)
        attempts.append(attempt)
        if normalized:
            return {
                "query": normalized_query,
                "requested_top_k": requested_top_k,
                "top_k": top_k,
                "search_type": normalized_type,
                "region": normalized_region,
                "timelimit": normalized_timelimit,
                "category": category,
                "backend": backend,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "result_count": len(normalized),
                "results": normalized,
                "text": _format_text(normalized),
            }

    detail = last_error or "no usable search results"
    raise RuntimeError(f"web search failed after {len(attempts)} attempts: {detail}")
