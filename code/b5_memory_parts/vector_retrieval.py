from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common.conversation_store import get_memory_embedding, upsert_memory_embedding
from common.io_utils import read_yaml
from common.path_utils import resolve_from_file

from b5_memory_parts.text_utils import _compact_text, _safe_list


DEFAULT_VECTOR_CONFIG = {
    "enabled": False,
    "provider": "fastapi",
    "embedding_path": "/embeddings",
    "top_k_blocks": 6,
    "top_k_turns": 12,
    "min_score": 0.25,
    "block_weight": 0.85,
    "turn_weight": 0.95,
    "batch_size": 16,
    "max_text_chars": 900,
    "timeout_seconds": 30.0,
}

DEFAULT_RERANK_CONFIG = {
    "enabled": False,
    "max_candidates": 12,
    "max_selected_turns": 5,
}


def _as_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _as_int(value: Any, default: int, minimum: int = 1) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, number)


def _as_float(value: Any, default: float, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, number)


def _as_path(value: Any, default: str) -> str:
    if isinstance(value, str) and value.startswith("/"):
        return value
    return default


def load_retrieval_settings(config_path: str | Path) -> dict:
    config_file = Path(config_path).resolve()
    config = read_yaml(config_file)
    memory = config.get("memory") if isinstance(config, dict) else None
    retrieval = memory.get("retrieval") if isinstance(memory, dict) else None
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    vector_raw = retrieval.get("vector") if isinstance(retrieval.get("vector"), dict) else {}
    rerank_raw = retrieval.get("llm_rerank") if isinstance(retrieval.get("llm_rerank"), dict) else {}
    vector = {
        "enabled": _as_bool(vector_raw.get("enabled"), DEFAULT_VECTOR_CONFIG["enabled"]),
        "provider": vector_raw.get("provider") if isinstance(vector_raw.get("provider"), str) else DEFAULT_VECTOR_CONFIG["provider"],
        "embedding_path": _as_path(vector_raw.get("embedding_path"), DEFAULT_VECTOR_CONFIG["embedding_path"]),
        "base_url": vector_raw.get("base_url") if isinstance(vector_raw.get("base_url"), str) else None,
        "api_key": vector_raw.get("api_key") if isinstance(vector_raw.get("api_key"), str) else None,
        "model": vector_raw.get("model") if isinstance(vector_raw.get("model"), str) else None,
        "timeout_seconds": _as_float(vector_raw.get("timeout_seconds"), DEFAULT_VECTOR_CONFIG["timeout_seconds"], 1.0),
        "top_k_blocks": _as_int(vector_raw.get("top_k_blocks"), DEFAULT_VECTOR_CONFIG["top_k_blocks"]),
        "top_k_turns": _as_int(vector_raw.get("top_k_turns"), DEFAULT_VECTOR_CONFIG["top_k_turns"]),
        "min_score": _as_float(vector_raw.get("min_score"), DEFAULT_VECTOR_CONFIG["min_score"]),
        "block_weight": _as_float(vector_raw.get("block_weight"), DEFAULT_VECTOR_CONFIG["block_weight"]),
        "turn_weight": _as_float(vector_raw.get("turn_weight"), DEFAULT_VECTOR_CONFIG["turn_weight"]),
        "batch_size": _as_int(vector_raw.get("batch_size"), DEFAULT_VECTOR_CONFIG["batch_size"]),
        "max_text_chars": _as_int(vector_raw.get("max_text_chars"), DEFAULT_VECTOR_CONFIG["max_text_chars"], 120),
    }
    rerank = {
        "enabled": _as_bool(rerank_raw.get("enabled"), DEFAULT_RERANK_CONFIG["enabled"]),
        "max_candidates": _as_int(rerank_raw.get("max_candidates"), DEFAULT_RERANK_CONFIG["max_candidates"]),
        "max_selected_turns": _as_int(rerank_raw.get("max_selected_turns"), DEFAULT_RERANK_CONFIG["max_selected_turns"]),
    }
    return {"vector": vector, "llm_rerank": rerank}


def _load_model_fastapi_config(model_config: str | Path | None) -> dict:
    if not model_config:
        return {}
    config_file = Path(model_config).resolve()
    config = read_yaml(config_file)
    if not isinstance(config, dict):
        return {}
    fastapi = config.get("fastapi")
    if not isinstance(fastapi, dict):
        return {}
    model = fastapi.get("model")
    if not isinstance(model, str) or not model.strip():
        model_section = config.get("model")
        model_value = model_section.get("model_name_or_path") if isinstance(model_section, dict) else None
        model = str(resolve_from_file(model_value, config_file)) if isinstance(model_value, str) else "default"
    return {
        "base_url": fastapi.get("base_url") if isinstance(fastapi.get("base_url"), str) else None,
        "api_key": fastapi.get("api_key") if isinstance(fastapi.get("api_key"), str) else None,
        "model": model,
        "timeout_seconds": _as_float(fastapi.get("timeout_seconds"), 60.0, 1.0),
    }


def _endpoint_config(vector_config: dict, model_config: str | Path | None) -> dict:
    if vector_config.get("provider") != "fastapi":
        raise ValueError("only fastapi vector provider is supported")
    model_fastapi = _load_model_fastapi_config(model_config)
    base_url = vector_config.get("base_url") or model_fastapi.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("vector fastapi base_url is unavailable")
    timeout = vector_config.get("timeout_seconds") or model_fastapi.get("timeout_seconds") or 60.0
    return {
        "provider": "fastapi",
        "base_url": base_url.rstrip("/"),
        "embedding_path": vector_config["embedding_path"],
        "api_key": vector_config.get("api_key") or model_fastapi.get("api_key"),
        "model": vector_config.get("model") or model_fastapi.get("model") or "default",
        "timeout_seconds": float(timeout),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _block_text(block: dict, max_chars: int) -> str:
    return _compact_text(
        "\n".join(
            str(item)
            for item in [
                block.get("title"),
                block.get("summary"),
                " ".join(_safe_list(block.get("keywords"))),
            ]
            if item
        ),
        max_chars,
    )


def _turn_text(turn: dict, max_chars: int) -> str:
    return _compact_text(
        "\n".join(
            str(item)
            for item in [
                turn.get("summary"),
                " ".join(_safe_list(turn.get("keywords"))),
                " ".join(_safe_list(turn.get("facts"))),
                " ".join(_safe_list(turn.get("decisions"))),
                " ".join(_safe_list(turn.get("corrections"))),
                " ".join(_safe_list(turn.get("labels"))),
            ]
            if item
        ),
        max_chars,
    )


def _coerce_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("embedding must be a non-empty array")
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding must contain only numbers") from exc
    return result


def _parse_embedding_response(response: dict, expected_count: int) -> list[list[float]]:
    raw_items = response.get("embeddings")
    if raw_items is None:
        raw_items = response.get("data")
    if not isinstance(raw_items, list) or len(raw_items) != expected_count:
        raise ValueError("embedding response count mismatch")
    vectors: list[list[float] | None] = [None] * expected_count
    append_index = 0
    for item in raw_items:
        if isinstance(item, dict):
            vector = item.get("embedding")
            index = item.get("index")
        else:
            vector = item
            index = None
        target_index = index if isinstance(index, int) and 0 <= index < expected_count else append_index
        vectors[target_index] = _coerce_vector(vector)
        append_index += 1
    if any(vector is None for vector in vectors):
        raise ValueError("embedding response contains missing indexes")
    return [vector for vector in vectors if vector is not None]


def _request_embeddings(endpoint: dict, texts: list[str], batch_size: int) -> tuple[list[list[float]], dict]:
    if not texts:
        return [], {"batch_count": 0}
    vectors: list[list[float]] = []
    batches = 0
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        payload = {"texts": chunk, "model": endpoint.get("model")}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if endpoint.get("api_key"):
            headers["Authorization"] = f"Bearer {endpoint['api_key']}"
        request = urllib.request.Request(
            endpoint["base_url"] + endpoint["embedding_path"],
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=endpoint["timeout_seconds"]) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"embedding request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc
        if not isinstance(response_data, dict):
            raise ValueError("embedding response must be an object")
        vectors.extend(_parse_embedding_response(response_data, len(chunk)))
        batches += 1
    return vectors, {"batch_count": batches}


def _cosine_similarity(query_vector: list[float], item_vector: list[float]) -> float:
    if len(query_vector) != len(item_vector) or not query_vector:
        return 0.0
    dot = 0.0
    query_norm_sq = 0.0
    item_norm_sq = 0.0
    for query_value, item_value in zip(query_vector, item_vector):
        dot += query_value * item_value
        query_norm_sq += query_value * query_value
        item_norm_sq += item_value * item_value
    denominator = math.sqrt(query_norm_sq) * math.sqrt(item_norm_sq)
    if denominator <= 0.0:
        return 0.0
    return dot / denominator


def _candidate_key(item_type: str, item: dict) -> str | None:
    key = item.get("id") if item_type == "block" else item.get("turn_id")
    return key if isinstance(key, str) and key else None


def _candidate_text(item_type: str, item: dict, max_chars: int) -> str:
    return _block_text(item, max_chars) if item_type == "block" else _turn_text(item, max_chars)


def _apply_item_score(item: dict, similarity: float, weight: float) -> None:
    similarity = max(0.0, min(1.0, similarity))
    previous = float(item.get("score") or 0.0)
    breakdown = dict(item.get("score_breakdown") or {})
    breakdown["vector_similarity"] = round(similarity, 4)
    breakdown["vector_weight"] = round(weight, 4)
    item["vector_score"] = round(similarity, 4)
    item["score"] = round(previous + similarity * weight, 4)
    item["score_breakdown"] = breakdown


def _load_or_request_item_vectors(
    db_path: str | Path,
    endpoint: dict,
    item_type: str,
    items: list[dict],
    max_chars: int,
    batch_size: int,
) -> tuple[dict[str, list[float]], dict]:
    vectors: dict[str, list[float]] = {}
    missing: list[tuple[str, str, str]] = []
    cache_hits = 0
    provider = endpoint["provider"]
    model = endpoint["model"]
    for item in items:
        item_id = _candidate_key(item_type, item)
        if item_id is None:
            continue
        text = _candidate_text(item_type, item, max_chars)
        if not text:
            continue
        text_hash = _hash_text(text)
        cached = get_memory_embedding(db_path, item_type, item_id, provider, model)
        if cached and cached.get("text_hash") == text_hash and cached.get("vector"):
            vectors[item_id] = _coerce_vector(cached["vector"])
            cache_hits += 1
            continue
        missing.append((item_id, text_hash, text))
    requested = 0
    if missing:
        requested_vectors, _ = _request_embeddings(endpoint, [item[2] for item in missing], batch_size)
        for (item_id, text_hash, _text), vector in zip(missing, requested_vectors):
            upsert_memory_embedding(db_path, item_type, item_id, text_hash, provider, model, vector)
            vectors[item_id] = vector
            requested += 1
    return vectors, {"cache_hits": cache_hits, "requested": requested}


def apply_vector_scores(
    *,
    config_path: str | Path,
    db_path: str | Path,
    model_config: str | Path | None,
    query_text: str,
    blocks: list[dict],
    turns: list[dict],
) -> dict:
    settings = load_retrieval_settings(config_path)
    vector_config = settings["vector"]
    if not vector_config.get("enabled"):
        return {"status": "disabled", "enabled": False}
    if not query_text.strip() or not (blocks or turns):
        return {"status": "skipped", "enabled": True, "reason": "empty query or candidates"}
    try:
        endpoint = _endpoint_config(vector_config, model_config)
        query_vectors, query_stats = _request_embeddings(endpoint, [_compact_text(query_text, vector_config["max_text_chars"])], 1)
        query_vector = query_vectors[0]
        block_vectors, block_stats = _load_or_request_item_vectors(
            db_path,
            endpoint,
            "block",
            blocks,
            vector_config["max_text_chars"],
            vector_config["batch_size"],
        )
        turn_vectors, turn_stats = _load_or_request_item_vectors(
            db_path,
            endpoint,
            "turn",
            turns,
            vector_config["max_text_chars"],
            vector_config["batch_size"],
        )
        for block in blocks:
            item_id = _candidate_key("block", block)
            if item_id in block_vectors:
                _apply_item_score(block, _cosine_similarity(query_vector, block_vectors[item_id]), vector_config["block_weight"])
        for turn in turns:
            item_id = _candidate_key("turn", turn)
            if item_id in turn_vectors:
                _apply_item_score(turn, _cosine_similarity(query_vector, turn_vectors[item_id]), vector_config["turn_weight"])
        return {
            "status": "success",
            "enabled": True,
            "provider": endpoint["provider"],
            "model": endpoint["model"],
            "dimension": len(query_vector),
            "query_batches": query_stats["batch_count"],
            "block_cache_hits": block_stats["cache_hits"],
            "block_requested": block_stats["requested"],
            "turn_cache_hits": turn_stats["cache_hits"],
            "turn_requested": turn_stats["requested"],
            "min_score": vector_config["min_score"],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "enabled": True,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
