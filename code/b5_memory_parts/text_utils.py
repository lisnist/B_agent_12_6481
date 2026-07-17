from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

RECENT_CONTEXT_TURNS = 4
MAX_RECALLED_BLOCKS = 3
MAX_RECALLED_TURNS = 5
MAX_SOURCE_SNIPPET_CHARS = 360
MAX_TOOL_SNIPPET_CHARS = 280
MAX_WORKSPACE_ITEM_CHARS = 420
FIELD_PREFIXES = ("project:", "file:", "model:", "tool:", "task:")
TASK_LABELS = {"category:task_state", "boundary:task_switch", "boundary:phase_complete", "boundary:task_complete"}
LONG_TERM_LABELS = {"category:preference", "category:decision", "category:correction"}


def _compact_text(value: str, limit: int = 1200) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _unique_strings(values: list[Any], limit: int | None = None) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return result


def _field_values(payload: dict, prefixes: tuple[str, ...] = FIELD_PREFIXES) -> list[str]:
    values = []
    if not isinstance(payload, dict):
        return values
    candidates = [
        *_safe_list(payload.get("keywords")),
        *_safe_list(payload.get("labels")),
    ]
    for refs_key in ("tool_refs", "artifact_refs"):
        for ref in _safe_list(payload.get(refs_key)):
            if isinstance(ref, dict):
                candidates.extend(str(value) for value in ref.values() if isinstance(value, str))
    for item in candidates:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        lowered = stripped.casefold()
        if any(lowered.startswith(prefix) for prefix in prefixes):
            values.append(stripped)
    return _unique_strings(values, 24)


def _format_block_topic(keywords: list[str]) -> str:
    scoped = [item for item in keywords if isinstance(item, str) and any(item.casefold().startswith(prefix) for prefix in FIELD_PREFIXES)]
    selected = scoped[:4] if scoped else keywords[:6]
    return ", ".join(selected) if selected else "一般对话"


def _compact_jsonish(value: Any, limit: int = MAX_WORKSPACE_ITEM_CHARS) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _compact_text(value, limit)
    if isinstance(value, list):
        return [_compact_jsonish(item, limit) for item in value[:8]]
    if isinstance(value, dict):
        return {str(key): _compact_jsonish(item, limit) for key, item in list(value.items())[:16]}
    return _compact_text(str(value), limit)


def _task_query_text(tasks: list[dict]) -> str:
    parts = []
    for task in tasks[:4]:
        parts.extend(
            [
                task.get("title"),
                task.get("objective"),
                task.get("phase"),
                " ".join(_safe_list(task.get("completed_items"))),
                " ".join(_safe_list(task.get("pending_items"))),
                " ".join(_safe_list(task.get("constraints"))),
                " ".join(_safe_list(task.get("active_files"))),
                " ".join(_safe_list(task.get("next_actions"))),
            ]
        )
    return "\n".join(item for item in parts if isinstance(item, str) and item.strip())


def _history_query_text(history_messages: list[dict], message_limit: int = 6) -> str:
    recent = history_messages[-message_limit:]
    return "\n".join(_compact_text(message.get("content", ""), 220) for message in recent)


def _text_similarity(query_text: str, candidate_text: str) -> float:
    query = _compact_text(query_text, 1600).casefold()
    candidate = _compact_text(candidate_text, 1600).casefold()
    if not query or not candidate:
        return 0.0
    return SequenceMatcher(None, query, candidate).ratio()


def _field_overlap_score(query_text: str, candidate_fields: list[str]) -> float:
    query = _compact_text(query_text, 2400).casefold()
    if not query or not candidate_fields:
        return 0.0
    hits = 0
    for field in candidate_fields:
        value = str(field).casefold().strip()
        if not value:
            continue
        bare_value = value.split(":", 1)[1] if ":" in value else value
        if bare_value and bare_value in query:
            hits += 1
    return min(1.0, hits / max(1, len(candidate_fields)))


def _tool_signal_score(query_text: str, tool_refs: list[dict]) -> float:
    tool_names = []
    for ref in _safe_list(tool_refs):
        if isinstance(ref, dict) and isinstance(ref.get("tool_name"), str):
            tool_names.append(f"tool:{ref['tool_name']}")
    return _field_overlap_score(query_text, _unique_strings(tool_names, 12))


def _recency_score(index: Any, newest_turn_index: int) -> float:
    if newest_turn_index <= 0:
        return 0.0
    try:
        numeric_index = float(index or 0)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, numeric_index / newest_turn_index))


def _block_search_text(block: dict) -> str:
    return "\n".join(
        str(item)
        for item in [
            block.get("title"),
            block.get("summary"),
            " ".join(_safe_list(block.get("keywords"))),
        ]
        if item
    )


def _turn_search_text(turn: dict) -> str:
    return "\n".join(
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
    )


def _score_block(block: dict, query_text: str, newest_turn_index: int) -> float:
    return _score_block_detail(block, query_text, newest_turn_index)["score"]


def _score_block_detail(block: dict, query_text: str, newest_turn_index: int) -> dict:
    similarity = _text_similarity(query_text, _block_search_text(block))
    field_overlap = _field_overlap_score(query_text, _field_values(block))
    recency = _recency_score(block.get("end_turn_index"), newest_turn_index)
    score = similarity * 1.7 + field_overlap * 0.65 + recency * 0.2
    return {
        "score": round(score, 4),
        "similarity": round(similarity, 4),
        "field_overlap": round(field_overlap, 4),
        "recency": round(recency, 4),
    }


def _score_turn(turn: dict, query_text: str, newest_turn_index: int) -> float:
    return _score_turn_detail(turn, query_text, newest_turn_index)["score"]


def _score_turn_detail(turn: dict, query_text: str, newest_turn_index: int) -> dict:
    similarity = _text_similarity(query_text, _turn_search_text(turn))
    labels = {item for item in _safe_list(turn.get("labels")) if isinstance(item, str)}
    current_task = float(turn.get("current_task_relevance") or 0.0)
    long_term = float(turn.get("long_term_value") or 0.0)
    has_stable_signal = bool(
        turn.get("has_explicit_fact")
        or turn.get("has_decision")
        or turn.get("has_user_correction")
        or labels.intersection(LONG_TERM_LABELS)
    )
    if "category:noise" in labels:
        current_task = min(current_task, 0.1)
        long_term = min(long_term, 0.1)
    elif "category:casual_chat" in labels and not has_stable_signal:
        current_task = min(current_task, 0.25)
        long_term = min(long_term, 0.25)
    if turn.get("has_explicit_fact"):
        long_term = max(long_term, 0.35)
    field_overlap = _field_overlap_score(query_text, _field_values(turn))
    tool_overlap = _tool_signal_score(query_text, _safe_list(turn.get("tool_refs")))
    signal = 0.0
    if turn.get("has_decision"):
        signal += 0.35
    if turn.get("has_user_correction"):
        signal += 0.3
    if turn.get("has_explicit_fact"):
        signal += 0.2
    recency = _recency_score(turn.get("turn_index"), newest_turn_index)
    noise = float(turn.get("noise_score") or 0.0)
    drop_penalty = 0.5 if turn.get("allow_drop") else 0.0
    score = (
        similarity * 1.6
        + field_overlap * 0.55
        + tool_overlap * 0.25
        + current_task * 0.12
        + long_term * 0.45
        + signal
        + recency * 0.15
        - noise
        - drop_penalty
    )
    return {
        "score": round(score, 4),
        "similarity": round(similarity, 4),
        "field_overlap": round(field_overlap, 4),
        "tool_overlap": round(tool_overlap, 4),
        "current_task": round(current_task, 4),
        "long_term": round(long_term, 4),
        "signal": round(signal, 4),
        "recency": round(recency, 4),
        "noise": round(noise, 4),
        "drop_penalty": round(drop_penalty, 4),
    }


def _turn_context_role(turn: dict) -> str:
    labels = {item for item in _safe_list(turn.get("labels")) if isinstance(item, str)}
    current_task = float(turn.get("current_task_relevance") or 0.0)
    long_term = float(turn.get("long_term_value") or 0.0)
    if "category:casual_chat" in labels and not turn.get("has_explicit_fact"):
        current_task = min(current_task, 0.25)
        long_term = min(long_term, 0.25)
    if labels.intersection(TASK_LABELS):
        return "task_related"
    if labels.intersection(LONG_TERM_LABELS) or long_term >= 0.65 or turn.get("has_decision") or turn.get("has_user_correction"):
        return "durable_memory"
    return "supporting_context"


def _group_turns_by_context_role(turns: list[dict]) -> dict[str, list[dict]]:
    groups = {"task_related": [], "durable_memory": [], "supporting_context": []}
    for turn in turns:
        role = _turn_context_role(turn)
        groups[role].append(turn)
    return groups


def _list_text(title: str, values: Any, limit: int = 4) -> str | None:
    items = _unique_strings(_safe_list(values), limit)
    if not items:
        return None
    return f"{title}: " + "; ".join(items)


def _clip_source_text(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    compact = _compact_text(text, limit)
    return compact


def _source_message_context(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        result.append(
            {
                "message_id": message.get("id"),
                "role": message.get("role"),
                "message_order": message.get("message_order"),
                "created_at": message.get("created_at"),
                "content": _clip_source_text(message.get("content"), MAX_SOURCE_SNIPPET_CHARS),
            }
        )
    return result


def _source_tool_context(tool_steps: list[dict]) -> list[dict]:
    result = []
    for step in tool_steps:
        if not isinstance(step, dict):
            continue
        result.append(
            {
                "tool_step_id": step.get("id"),
                "tool_name": step.get("tool_name"),
                "status": step.get("status"),
                "created_at": step.get("created_at"),
                "input": _compact_jsonish(step.get("input"), MAX_TOOL_SNIPPET_CHARS),
                "output": _compact_jsonish(step.get("output"), MAX_TOOL_SNIPPET_CHARS),
                "error": _compact_jsonish(step.get("error"), MAX_TOOL_SNIPPET_CHARS),
            }
        )
    return result


def _foreground_task(tasks: list[dict]) -> dict | None:
    for task in tasks:
        if isinstance(task, dict) and task.get("status") == "foreground":
            return task
    return None


def _paused_tasks(tasks: list[dict]) -> list[dict]:
    return [task for task in tasks if isinstance(task, dict) and task.get("status") == "paused"]


def _append_budgeted_line(lines: list[str], line: str, budget: dict) -> None:
    if budget["remaining"] <= 0:
        return
    text = str(line).rstrip()
    if not text:
        return
    required = len(text) + 1
    if required > budget["remaining"]:
        if budget["remaining"] <= 8:
            return
        text = text[: budget["remaining"] - 4].rstrip() + "..."
        required = len(text) + 1
        budget["truncated"] = True
    lines.append(text)
    budget["remaining"] -= required


def _build_memory_context_text(
    *,
    tasks: list[dict],
    selected_blocks: list[dict],
    selected_turns: list[dict],
    source_messages: list[dict],
    source_tool_steps: list[dict],
    legacy_docs: list[dict],
    max_chars: int,
) -> tuple[str, bool]:
    has_context = bool(tasks or selected_blocks or selected_turns or source_messages or source_tool_steps or legacy_docs)
    if not has_context:
        return "", False
    lines: list[str] = []
    budget = {"remaining": max(800, int(max_chars)), "truncated": False}
    _append_budgeted_line(lines, "[B5 分层记忆上下文]", budget)
    _append_budgeted_line(
        lines,
        "这些记忆只作为历史上下文；当前用户输入优先。摘要只用于定位来源，精确事实必须来自 source 片段或当前工具结果。",
        budget,
    )
    foreground = [task for task in tasks if task.get("status") == "foreground"]
    paused = [task for task in tasks if task.get("status") == "paused"]
    if foreground or paused:
        _append_budgeted_line(lines, "\n[任务记忆]", budget)
    for task in foreground[:1]:
        _append_budgeted_line(
            lines,
            f"- 前台任务 {task.get('id')}: {task.get('title')} | 目标={task.get('objective') or ''} | 阶段={task.get('phase') or ''}",
            budget,
        )
        for optional in (
            _list_text("已完成", task.get("completed_items")),
            _list_text("待处理", task.get("pending_items")),
            _list_text("约束", task.get("constraints")),
            _list_text("相关文件", task.get("active_files")),
            _list_text("下一步", task.get("next_actions")),
        ):
            if optional:
                _append_budgeted_line(lines, f"  {optional}", budget)
    for task in paused[:3]:
        _append_budgeted_line(
            lines,
            f"- 暂停任务 {task.get('id')}: {task.get('title')} | 阶段={task.get('phase') or ''}",
            budget,
        )

    if selected_blocks:
        _append_budgeted_line(lines, "\n[召回的记忆块]", budget)
    for block in selected_blocks:
        _append_budgeted_line(
            lines,
            f"- 记忆块 {block.get('id')} 轮次 {block.get('start_turn_index')}-{block.get('end_turn_index')}: {block.get('summary')}",
            budget,
        )

    grouped_turns = _group_turns_by_context_role(selected_turns)
    turn_sections = [
        ("task_related", "\n[任务相关召回轮次]"),
        ("durable_memory", "\n[长期偏好、决策和纠正]"),
        ("supporting_context", "\n[辅助上下文召回轮次]"),
    ]
    for group_key, heading in turn_sections:
        turns = grouped_turns[group_key]
        if turns:
            _append_budgeted_line(lines, heading, budget)
        for turn in turns:
            source_ids = _safe_list(turn.get("source_message_ids"))
            tool_ids = _safe_list(turn.get("source_tool_step_ids"))
            _append_budgeted_line(
                lines,
                f"- 轮次 {turn.get('turn_index')} ({turn.get('turn_id')}): {turn.get('summary')} | source_messages={source_ids} | source_tools={tool_ids}",
                budget,
            )
            details = []
            for field, label in (("decisions", "决策"), ("corrections", "纠正"), ("facts", "事实")):
                values = _unique_strings(_safe_list(turn.get(field)), 3)
                if values:
                    details.append(f"{label}: {'; '.join(values)}")
            if details:
                _append_budgeted_line(lines, "  " + " | ".join(details), budget)

    if source_messages:
        _append_budgeted_line(lines, "\n[已加载的原始消息片段]", budget)
    for message in source_messages:
        _append_budgeted_line(
            lines,
            f"- 消息 {message.get('id')} role={message.get('role')} 顺序={message.get('message_order')}: {_clip_source_text(message.get('content'), MAX_SOURCE_SNIPPET_CHARS)}",
            budget,
        )

    if source_tool_steps:
        _append_budgeted_line(lines, "\n[已加载的原始工具片段]", budget)
    for step in source_tool_steps:
        payload = {
            "input": step.get("input"),
            "output": step.get("output"),
            "error": step.get("error"),
        }
        _append_budgeted_line(
            lines,
            f"- 工具步骤 {step.get('id')} name={step.get('tool_name')} status={step.get('status')}: {_clip_source_text(payload, MAX_TOOL_SNIPPET_CHARS)}",
            budget,
        )

    if legacy_docs:
        _append_budgeted_line(lines, "\n[已选旧版记忆文档]", budget)
    for doc in legacy_docs:
        _append_budgeted_line(
            lines,
            f"- {doc.get('memory_id')} {doc.get('title')}: {_clip_source_text(doc.get('content'), MAX_SOURCE_SNIPPET_CHARS)}",
            budget,
        )
    return "\n".join(lines).strip(), bool(budget["truncated"])
