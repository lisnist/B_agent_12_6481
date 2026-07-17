from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from b5_memory_parts.text_utils import MAX_RECALLED_BLOCKS, MAX_RECALLED_TURNS, _compact_text, _safe_list

_PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "prompts" / "b5_memory_prompts.json"


def _load_prompt_config() -> dict:
    with _PROMPT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("b5_memory_prompts.json must contain an object")
    return payload


_PROMPTS = _load_prompt_config()


def _prompt(*keys: str) -> str:
    value: object = _PROMPTS
    for key in keys:
        if not isinstance(value, dict):
            raise KeyError(".".join(keys))
        value = value[key]
    if not isinstance(value, str):
        raise KeyError(".".join(keys))
    return value


def _candidate_id(item: dict, key: str) -> str | None:
    value = item.get(key)
    return value if isinstance(value, str) and value else None


def _candidate_score(item: dict) -> float:
    try:
        return float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _block_payload(block: dict) -> dict:
    return {
        "id": block.get("id"),
        "turn_range": [block.get("start_turn_index"), block.get("end_turn_index")],
        "score": _candidate_score(block),
        "summary": _compact_text(str(block.get("summary") or ""), 260),
        "keywords": _safe_list(block.get("keywords"))[:8],
        "score_breakdown": block.get("score_breakdown"),
    }


def _turn_payload(turn: dict) -> dict:
    return {
        "id": turn.get("turn_id"),
        "turn_index": turn.get("turn_index"),
        "block_id": turn.get("block_id"),
        "score": _candidate_score(turn),
        "summary": _compact_text(str(turn.get("summary") or ""), 260),
        "facts": [_compact_text(str(item), 160) for item in _safe_list(turn.get("facts"))[:4] if isinstance(item, str)],
        "decisions": [_compact_text(str(item), 160) for item in _safe_list(turn.get("decisions"))[:3] if isinstance(item, str)],
        "corrections": [_compact_text(str(item), 160) for item in _safe_list(turn.get("corrections"))[:3] if isinstance(item, str)],
        "keywords": _safe_list(turn.get("keywords"))[:8],
        "labels": _safe_list(turn.get("labels"))[:8],
        "has_explicit_fact": bool(turn.get("has_explicit_fact")),
        "has_decision": bool(turn.get("has_decision")),
        "has_user_correction": bool(turn.get("has_user_correction")),
        "context_role": turn.get("context_role"),
        "score_breakdown": turn.get("score_breakdown"),
    }


def _ordered_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _select_by_ids(candidates: list[dict], key: str, selected_ids: list[str], limit: int) -> tuple[list[dict], list[str]]:
    by_id = {_candidate_id(item, key): item for item in candidates if _candidate_id(item, key)}
    selected = []
    invalid = []
    for item_id in selected_ids:
        item = by_id.get(item_id)
        if item is None:
            invalid.append(item_id)
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    if selected:
        return selected, invalid
    return candidates[:limit], invalid


def _rerank_messages(
    current_user_input: str,
    query_text: str,
    tasks: list[dict],
    candidate_blocks: list[dict],
    candidate_turns: list[dict],
    max_selected_turns: int,
) -> list[dict]:
    schema = {
        "selected_block_ids": ["ids copied from candidate_blocks"],
        "selected_turn_ids": ["ids copied from candidate_turns"],
        "reason": "brief selection reason without adding new facts",
    }
    task_payload = [
        {
            "id": task.get("id"),
            "status": task.get("status"),
            "title": task.get("title"),
            "objective": task.get("objective"),
            "phase": task.get("phase"),
        }
        for task in tasks[:4]
        if isinstance(task, dict)
    ]
    payload = {
        "current_user_input": _compact_text(current_user_input, 600),
        "query_context": _compact_text(query_text, 1200),
        "tasks": task_payload,
        "candidate_blocks": [_block_payload(block) for block in candidate_blocks],
        "candidate_turns": [_turn_payload(turn) for turn in candidate_turns],
        "limits": {
            "max_selected_blocks": MAX_RECALLED_BLOCKS,
            "max_selected_turns": max_selected_turns,
        },
    }
    system = _prompt("rerank", "system")
    user = (
        _prompt("rerank", "user_prefix")
        + "\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\n"
        + _prompt("rerank", "payload_label")
        + "\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rerank_memory_candidates(
    *,
    model_config: str | Path | None,
    llm_mode: str | None,
    artifact_dir: str | None,
    current_user_input: str,
    query_text: str,
    tasks: list[dict],
    candidate_blocks: list[dict],
    candidate_turns: list[dict],
    max_candidates: int,
    max_selected_turns: int,
) -> dict:
    if not candidate_blocks and not candidate_turns:
        return {
            "status": "skipped",
            "reason": "no candidates",
            "selected_blocks": [],
            "selected_turns": [],
        }
    if not model_config:
        return {
            "status": "skipped",
            "reason": "model_config unavailable",
            "selected_blocks": candidate_blocks[:MAX_RECALLED_BLOCKS],
            "selected_turns": candidate_turns[:max_selected_turns],
        }
    if llm_mode not in {None, "prompt_json"}:
        return {
            "status": "skipped",
            "reason": f"llm_mode {llm_mode} does not support JSON rerank",
            "selected_blocks": candidate_blocks[:MAX_RECALLED_BLOCKS],
            "selected_turns": candidate_turns[:max_selected_turns],
        }
    block_candidates = candidate_blocks[:max_candidates]
    turn_candidates = candidate_turns[:max_candidates]
    try:
        from b4_local_agent_llm import generate_json_object

        result = generate_json_object(
            str(model_config),
            _rerank_messages(
                current_user_input,
                query_text,
                tasks,
                block_candidates,
                turn_candidates,
                max_selected_turns,
            ),
            mode="prompt_json",
            artifact_dir=artifact_dir,
            artifact_stem="b5_memory_rerank",
            prompt_ready=True,
        )
        if result.get("status") != "success" or not isinstance(result.get("json"), dict):
            return {
                "status": "fallback",
                "reason": "rerank JSON generation failed",
                "error": result.get("error"),
                "selected_blocks": candidate_blocks[:MAX_RECALLED_BLOCKS],
                "selected_turns": candidate_turns[:max_selected_turns],
            }
        payload = result["json"]
        selected_blocks, invalid_block_ids = _select_by_ids(
            block_candidates,
            "id",
            _ordered_ids(payload.get("selected_block_ids")),
            MAX_RECALLED_BLOCKS,
        )
        selected_turns, invalid_turn_ids = _select_by_ids(
            turn_candidates,
            "turn_id",
            _ordered_ids(payload.get("selected_turn_ids")),
            max_selected_turns,
        )
        return {
            "status": "success",
            "reason": payload.get("reason") if isinstance(payload.get("reason"), str) else "",
            "selected_blocks": selected_blocks,
            "selected_turns": selected_turns,
            "invalid_block_ids": invalid_block_ids,
            "invalid_turn_ids": invalid_turn_ids,
            "candidate_block_count": len(block_candidates),
            "candidate_turn_count": len(turn_candidates),
        }
    except Exception as exc:
        return {
            "status": "fallback",
            "reason": "rerank failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "selected_blocks": candidate_blocks[:MAX_RECALLED_BLOCKS],
            "selected_turns": candidate_turns[:max_selected_turns],
        }
