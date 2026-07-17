from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.conversation_store import (
    list_turn_summaries,
    list_unblocked_conversation_turns,
    upsert_conversation_turn,
    upsert_memory_block,
    upsert_task_memory,
    upsert_turn_memory_tags,
    upsert_turn_summary,
)

from b5_memory_parts.conversation_api import list_conversation_tasks, list_message_tool_steps
from b5_memory_parts.paths import _conversation_db_path, _safe_conversation_id
from b5_memory_parts.text_utils import (
    _compact_jsonish,
    _compact_text,
    _field_values,
    _format_block_topic,
    _safe_list,
    _unique_strings,
)

MIN_BLOCK_TURNS = 3
MAX_BLOCK_TURNS = 8
MAX_BLOCK_SUMMARY_CHARS = 1400
BOUNDARY_TOPIC_SHIFT = "boundary:topic_shift"
BOUNDARY_TASK_SWITCH = "boundary:task_switch"
BOUNDARY_TASK_COMPLETE = "boundary:task_complete"
BOUNDARY_PHASE_COMPLETE = "boundary:phase_complete"
NON_TASK_CATEGORIES = {"category:casual_chat", "category:noise", "category:preference"}
TASK_BOUNDARY_LABELS = {BOUNDARY_TASK_SWITCH, BOUNDARY_TASK_COMPLETE, BOUNDARY_PHASE_COMPLETE}
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


def _contains_cjk(text: Any) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def _neutral_locator_summary(raw_user_input: str = "", final_answer: str = "") -> str:
    if _contains_cjk(raw_user_input) or _contains_cjk(final_answer):
        return "本轮记忆反思暂不可用。需要事实、路径、命令、代码、错误和输出时，请以关联的原始消息和工具步骤为准。"
    return (
        "Memory reflection was unavailable for this turn. "
        "Use the linked source messages and tool steps for all facts, paths, commands, code, errors, and outputs."
    )


def _workspace_snapshot(trace: dict) -> dict:
    workspace = trace.get("workspace") if isinstance(trace, dict) and isinstance(trace.get("workspace"), dict) else {}
    if not workspace:
        return {}
    tools = workspace.get("tools") if isinstance(workspace.get("tools"), dict) else {}
    draft = workspace.get("draft") if isinstance(workspace.get("draft"), dict) else {}
    stages = []
    for entry in _safe_list(workspace.get("trace"))[-8:]:
        if not isinstance(entry, dict):
            continue
        stages.append(
            {
                "phase": entry.get("phase"),
                "payload": _compact_jsonish(entry.get("payload")),
            }
        )
    return {
        "task": _compact_jsonish(workspace.get("task") if isinstance(workspace.get("task"), dict) else {}),
        "known_facts": _compact_jsonish(draft.get("known_facts", [])),
        "missing_info": _compact_jsonish(draft.get("missing_info", [])),
        "accepted_evidence": _compact_jsonish(tools.get("accepted_evidence", [])),
        "rejected_evidence": _compact_jsonish(tools.get("rejected_evidence", [])),
        "observations": _compact_jsonish(tools.get("observations", [])),
        "final": _compact_jsonish(workspace.get("final") if isinstance(workspace.get("final"), dict) else {}),
        "stages": stages,
    }


def _safe_summary_items(value: Any) -> list:
    items = []
    if not isinstance(value, list):
        return items
    for item in value:
        if not isinstance(item, str):
            continue
        compact = _compact_text(item, 180)
        if compact:
            items.append(compact)
    return items


def _bounded_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return min(1.0, max(0.0, score))


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _safe_labels(value: Any) -> list[str]:
    return _unique_strings([item for item in _safe_list(value) if isinstance(item, str)], 20)


def _tool_refs_from_steps(tool_steps: list[dict]) -> list[dict]:
    refs = []
    for step in tool_steps:
        if not isinstance(step, dict):
            continue
        refs.append(
            {
                "tool_step_id": step.get("id"),
                "tool_call_id": step.get("tool_call_id"),
                "tool_name": step.get("tool_name"),
                "status": step.get("status"),
            }
        )
    return refs


def _artifact_refs_from_steps(tool_steps: list[dict]) -> list[dict]:
    refs = []
    for step in tool_steps:
        if not isinstance(step, dict):
            continue
        output = step.get("output") if isinstance(step.get("output"), dict) else {}
        if not isinstance(output, dict):
            continue
        if "relative_output_path" in output or "generated_file_path" in output:
            refs.append(
                {
                    "tool_step_id": step.get("id"),
                    "tool_name": step.get("tool_name"),
                    "artifact_type": "generated_file",
                }
            )
    return refs


def _neutral_memory_decision(
    raw_user_input: str,
    final_answer: str,
    trace: dict,
    source_message_ids: list[str],
    source_tool_step_ids: list[str],
    tool_steps: list[dict],
) -> dict:
    tags = {
        "current_task_relevance": 0.5,
        "long_term_value": 0.5,
        "has_explicit_fact": False,
        "has_decision": False,
        "has_user_correction": False,
        "allow_compress": True,
        "allow_drop": False,
        "noise_score": 0.0,
        "labels": ["model_reflection_unavailable"],
    }
    summary = {
        "summary": _neutral_locator_summary(raw_user_input, final_answer),
        "keywords": [],
        "facts": [],
        "decisions": [],
        "corrections": [],
        "tool_refs": _tool_refs_from_steps(tool_steps),
        "artifact_refs": _artifact_refs_from_steps(tool_steps),
        "source_message_ids": source_message_ids,
        "source_tool_step_ids": source_tool_step_ids,
    }
    task_memory = {
        "action": "no_change",
        "status": "foreground",
        "title": "",
        "objective": "",
        "phase": "",
        "completed_items": [],
        "pending_items": [],
        "constraints": [],
        "key_results": [],
        "active_files": [],
        "blocked_items": [],
        "next_actions": [],
        "source_turn_ids": [],
        "confidence": 0.35,
    }
    return {
        "source": "neutral_fallback",
        "turn_tags": tags,
        "turn_summary": summary,
        "task_memory": task_memory,
        "trace_status": trace.get("status") if isinstance(trace, dict) else None,
    }


def _memory_reflection_messages(
    raw_user_input: str,
    final_answer: str,
    trace: dict,
    tool_steps: list[dict],
    existing_tasks: list[dict],
) -> list[dict]:
    compact_trace = {
        "status": trace.get("status"),
        "tool_rounds_used": trace.get("tool_rounds_used"),
        "llm_call_count": trace.get("llm_call_count"),
        "final_state": trace.get("final_state"),
        "finish_reason": trace.get("finish_reason"),
    } if isinstance(trace, dict) else {}
    compact_tools = []
    for step in tool_steps:
        compact_tools.append(
            {
                "tool_step_id": step.get("id"),
                "tool_call_id": step.get("tool_call_id"),
                "tool_name": step.get("tool_name"),
                "status": step.get("status"),
                "has_error": bool(step.get("error")),
                "latency_ms": step.get("latency_ms"),
            }
        )
    observation = {
        "user_input": _compact_text(raw_user_input, 1800),
        "final_answer": _compact_text(final_answer, 1800),
        "trace": compact_trace,
        "workspace": _workspace_snapshot(trace),
        "tool_steps": compact_tools,
        "existing_tasks": [
            {
                "task_id": task.get("id"),
                "status": task.get("status"),
                "title": task.get("title"),
                "objective": task.get("objective"),
                "phase": task.get("phase"),
            }
            for task in existing_tasks[:8]
        ],
    }
    schema_hint = {
        "turn_tags": {
            "current_task_relevance": "0..1",
            "long_term_value": "0..1",
            "has_explicit_fact": "boolean",
            "has_decision": "boolean",
            "has_user_correction": "boolean",
            "allow_compress": "boolean",
            "allow_drop": "boolean; low-priority recall signal only, never permission to delete source turns",
            "noise_score": "0..1",
            "labels": [
                "controlled labels such as category:task_state, category:preference, category:decision, "
                "category:casual_chat, category:noise, boundary:topic_shift, boundary:task_switch, "
                "boundary:phase_complete, boundary:task_complete"
            ],
        },
        "turn_summary": {
            "summary": "short locator summary; never include exact paths, commands, code, parameters, or error text",
            "keywords": [
                "short retrieval keys. Prefer scoped keys when available: project:<name>, file:<name>, "
                "model:<name>, tool:<name>, task:<id-or-title>"
            ],
            "facts": [],
            "decisions": [],
            "corrections": [],
        },
        "task_memory": {
            "action": "no_change | update_foreground | switch_task | resume_task | pause_task | complete_task",
            "target_task_id": "optional existing task id",
            "status": "foreground | paused | completed | abandoned",
            "title": "task title",
            "objective": "task objective",
            "phase": "task phase",
            "completed_items": [],
            "pending_items": [],
            "constraints": [],
            "key_results": [],
            "active_files": [],
            "blocked_items": [],
            "next_actions": [],
            "confidence": "0..1",
        },
    }
    system = _prompt("reflection", "system")
    user = (
        _prompt("reflection", "user_prefix")
        + "\n\n"
        + _prompt("reflection", "classification_rules")
        + "\n\n"
        + _prompt("reflection", "schema_label")
        + "\n"
        + json.dumps(schema_hint, ensure_ascii=False, indent=2)
        + "\n\n"
        + _prompt("reflection", "observation_label")
        + "\n"
        + json.dumps(observation, ensure_ascii=False, indent=2)
        + "\n\n"
        + _prompt("reflection", "return_instruction")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _coerce_memory_decision(
    candidate: dict,
    source_message_ids: list[str],
    source_tool_step_ids: list[str],
    tool_steps: list[dict],
    raw_user_input: str = "",
    final_answer: str = "",
) -> dict:
    if not isinstance(candidate, dict):
        raise ValueError("memory decision must be an object")
    tags = candidate.get("turn_tags")
    summary = candidate.get("turn_summary")
    task = candidate.get("task_memory")
    if not isinstance(tags, dict) or not isinstance(summary, dict) or not isinstance(task, dict):
        raise ValueError("memory decision missing turn_tags, turn_summary, or task_memory")
    tags = dict(tags)
    tags["current_task_relevance"] = _bounded_score(tags.get("current_task_relevance"))
    tags["long_term_value"] = _bounded_score(tags.get("long_term_value"))
    tags["noise_score"] = _bounded_score(tags.get("noise_score"))
    tags["has_explicit_fact"] = _safe_bool(tags.get("has_explicit_fact"))
    tags["has_decision"] = _safe_bool(tags.get("has_decision"))
    tags["has_user_correction"] = _safe_bool(tags.get("has_user_correction"))
    tags["allow_compress"] = _safe_bool(tags.get("allow_compress"), True)
    tags["allow_drop"] = _safe_bool(tags.get("allow_drop"))
    tags["labels"] = _safe_labels(tags.get("labels"))
    labels = set(tags["labels"])
    durable_signal = (
        tags["has_explicit_fact"]
        or tags["has_decision"]
        or tags["has_user_correction"]
        or "category:preference" in labels
        or "category:decision" in labels
        or "category:correction" in labels
    )
    low_value_non_task = bool(labels.intersection({"category:casual_chat", "category:noise"})) and not durable_signal
    if low_value_non_task:
        tags["allow_drop"] = True
        tags["long_term_value"] = min(tags["long_term_value"], 0.3)
        tags["current_task_relevance"] = min(tags["current_task_relevance"], 0.3)
    elif durable_signal or tags["long_term_value"] >= 0.65:
        tags["allow_drop"] = False
    if tags["has_explicit_fact"]:
        tags["long_term_value"] = max(tags["long_term_value"], 0.35)
    if tags["allow_drop"]:
        tags["long_term_value"] = min(tags["long_term_value"], 0.3)
        tags["current_task_relevance"] = min(tags["current_task_relevance"], 0.3)
    summary = dict(summary)
    summary_text = summary.get("summary")
    if isinstance(summary_text, str):
        summary["summary"] = _compact_text(summary_text, 360)
    summary["keywords"] = _safe_summary_items(summary.get("keywords"))
    summary["facts"] = _safe_summary_items(summary.get("facts"))
    summary["decisions"] = _safe_summary_items(summary.get("decisions"))
    summary["corrections"] = _safe_summary_items(summary.get("corrections"))
    summary["tool_refs"] = _tool_refs_from_steps(tool_steps)
    summary["artifact_refs"] = _artifact_refs_from_steps(tool_steps)
    summary["source_message_ids"] = source_message_ids
    summary["source_tool_step_ids"] = source_tool_step_ids
    if not isinstance(summary.get("summary"), str) or not summary["summary"].strip():
        summary["summary"] = _neutral_locator_summary(raw_user_input, final_answer)
    task = dict(task)
    action = task.get("action")
    if action not in {"no_change", "update_foreground", "switch_task", "resume_task", "pause_task", "complete_task"}:
        action = "no_change"
    task["action"] = action
    task_status = task.get("status")
    task["status"] = task_status if task_status in {"foreground", "paused", "completed", "abandoned"} else "foreground"
    task.setdefault("source_turn_ids", [])
    has_task_state = "category:task_state" in labels
    if not has_task_state:
        task["action"] = "no_change"
        tags["labels"] = [label for label in tags["labels"] if label not in TASK_BOUNDARY_LABELS]
        summary["keywords"] = [item for item in summary["keywords"] if not item.lower().startswith("task:")]
    elif labels.intersection(NON_TASK_CATEGORIES - {"category:preference"}):
        task["action"] = "no_change"
    if tags["noise_score"] >= 0.75 and tags["long_term_value"] <= 0.35 and tags["current_task_relevance"] <= 0.35:
        task["action"] = "no_change"
    return {"source": "model", "turn_tags": tags, "turn_summary": summary, "task_memory": task}


def _reflect_memory_with_model(
    model_config: str,
    llm_mode: str | None,
    artifact_dir: str | None,
    raw_user_input: str,
    final_answer: str,
    trace: dict,
    tool_steps: list[dict],
    existing_tasks: list[dict],
    source_message_ids: list[str],
    source_tool_step_ids: list[str],
) -> dict:
    mode = llm_mode or "prompt_json"
    if mode != "prompt_json":
        raise ValueError("memory reflection model is only enabled in prompt_json mode")
    from b4_local_agent_llm import generate_json_object

    result = generate_json_object(
        model_config,
        _memory_reflection_messages(raw_user_input, final_answer, trace, tool_steps, existing_tasks),
        mode,
        artifact_dir,
        "memory_reflection",
        prompt_ready=True,
    )
    if result.get("status") != "success" or not isinstance(result.get("json"), dict):
        raise ValueError(f"memory reflection failed: {result.get('error')}")
    return _coerce_memory_decision(
        result["json"],
        source_message_ids,
        source_tool_step_ids,
        tool_steps,
        raw_user_input,
        final_answer,
    )


def _apply_task_memory_decision(
    config_path: str,
    conversation_id: str,
    turn_id: str,
    task_memory: dict,
) -> dict:
    action = task_memory.get("action")
    if action not in {"update_foreground", "switch_task", "resume_task", "pause_task", "complete_task"}:
        return {"status": "skipped", "reason": "no task memory update requested"}
    task = dict(task_memory)
    task["source_turn_ids"] = list(dict.fromkeys([*(_safe_list(task.get("source_turn_ids"))), turn_id]))
    if action in {"pause_task"}:
        task["status"] = "paused"
    elif action in {"complete_task"}:
        task["status"] = "completed"
    else:
        task["status"] = "foreground"
    title = task.get("title")
    if not isinstance(title, str) or not title.strip():
        task["title"] = "当前任务"
    return upsert_task_memory(
        _conversation_db_path(config_path),
        conversation_id,
        task,
        task_id=task.get("target_task_id") if isinstance(task.get("target_task_id"), str) else None,
    )


def _summary_labels(summary: dict) -> set[str]:
    return {item for item in _safe_list(summary.get("labels")) if isinstance(item, str)}


def _summary_task_keys(summary: dict) -> set[str]:
    return {
        value
        for value in _field_values(summary, prefixes=("task:",))
        if isinstance(value, str) and value.strip()
    }


def _summary_text_size(summaries: list[dict]) -> int:
    total = 0
    for summary in summaries:
        total += len(str(summary.get("summary") or ""))
        total += sum(len(str(item)) for item in _safe_list(summary.get("keywords")))
        total += sum(len(str(item)) for item in _safe_list(summary.get("facts")))
        total += sum(len(str(item)) for item in _safe_list(summary.get("decisions")))
    return total


def _block_boundary_reason(summaries: list[dict]) -> str | None:
    if len(summaries) >= MAX_BLOCK_TURNS:
        return "max_turns"
    if len(summaries) < MIN_BLOCK_TURNS:
        return None
    labels = _summary_labels(summaries[-1])
    if BOUNDARY_TASK_COMPLETE in labels:
        return "task_complete"
    if BOUNDARY_PHASE_COMPLETE in labels:
        return "phase_complete"
    if _summary_text_size(summaries) >= MAX_BLOCK_SUMMARY_CHARS:
        return "context_length"
    return None


def _select_block_summaries(summaries: list[dict]) -> tuple[list[dict], str | None]:
    if not summaries:
        return [], None
    labels = _summary_labels(summaries[-1])
    if labels.intersection({BOUNDARY_TOPIC_SHIFT, BOUNDARY_TASK_SWITCH}) and len(summaries) - 1 >= MIN_BLOCK_TURNS:
        return summaries[:-1], "boundary_before_latest"
    latest_task_keys = _summary_task_keys(summaries[-1])
    previous_task_keys = set().union(*[_summary_task_keys(summary) for summary in summaries[:-1]]) if len(summaries) > 1 else set()
    if latest_task_keys and previous_task_keys and latest_task_keys.isdisjoint(previous_task_keys) and len(summaries) - 1 >= MIN_BLOCK_TURNS:
        return summaries[:-1], "task_key_change"
    reason = _block_boundary_reason(summaries)
    if reason:
        return summaries, reason
    return [], None


def _maybe_create_memory_block(config_path: str, conversation_id: str) -> dict:
    db_path = _conversation_db_path(config_path)
    turns = list_unblocked_conversation_turns(db_path, conversation_id)
    if len(turns) < MIN_BLOCK_TURNS:
        return {"status": "skipped", "reason": "not enough turns"}
    turn_ids = [turn["id"] for turn in turns if isinstance(turn.get("id"), str)]
    summaries = list_turn_summaries(db_path, conversation_id, turn_ids=turn_ids)
    summaries.sort(key=lambda item: int(item.get("turn_index") or 0))
    selected_summaries, boundary_reason = _select_block_summaries(summaries)
    if not selected_summaries or boundary_reason is None:
        return {"status": "skipped", "reason": "waiting for task/topic/length boundary"}
    block_turn_ids = [summary["turn_id"] for summary in selected_summaries if isinstance(summary.get("turn_id"), str)]
    if len(block_turn_ids) < MIN_BLOCK_TURNS:
        return {"status": "skipped", "reason": "selected block is too small"}
    start = int(selected_summaries[0].get("turn_index") or 0)
    end = int(selected_summaries[-1].get("turn_index") or 0)
    keywords = _unique_strings(
        [
            keyword
            for summary in selected_summaries
            for keyword in [
                *_safe_list(summary.get("keywords")),
                *_safe_list(summary.get("labels")),
            ]
        ],
        16,
    )
    representative = []
    for summary in selected_summaries[:4]:
        text = summary.get("summary")
        if isinstance(text, str) and text.strip():
            representative.append(f"轮次 {summary.get('turn_index')}: {_compact_text(text, 120)}")
    topic_text = _format_block_topic(keywords)
    task_keys = _unique_strings([key for summary in selected_summaries for key in _summary_task_keys(summary)], 3)
    block = {
        "title": f"轮次 {start}-{end}: {topic_text}",
        "summary": (
            f"记忆块覆盖轮次 {start}-{end}。边界原因：{boundary_reason}。主题：{topic_text}。"
            + ("代表性轮摘要：" + " | ".join(representative) + "。 " if representative else "")
            + "优先使用关联轮摘要定位信息；需要精确事实时，再加载原始消息或工具步骤。"
        ),
        "status": "active",
        "keywords": keywords,
        "task_id": task_keys[0].split(":", 1)[1].strip() if task_keys and ":" in task_keys[0] else None,
        "source": f"derived_from_turn_summaries:{boundary_reason}",
    }
    return upsert_memory_block(
        db_path,
        conversation_id,
        block,
        block_turn_ids,
    )


def record_completed_turn_memory(
    config_path: str,
    conversation_id: str,
    run_id: str,
    user_message_id: str,
    assistant_message_id: str,
    raw_user_input: str,
    final_answer: str,
    trace: dict,
    model_config: str | None = None,
    llm_mode: str | None = None,
    artifact_dir: str | None = None,
) -> dict:
    """Record layered memory metadata for one completed web Agent turn.

    The original messages and tool_steps remain the source of truth. Summaries
    are locator metadata used for future retrieval and must point back to source
    message/tool-step ids.
    """
    conversation_id = _safe_conversation_id(conversation_id)
    turn = upsert_conversation_turn(
        _conversation_db_path(config_path),
        conversation_id,
        run_id,
        user_message_id,
        assistant_message_id,
        status=trace.get("status", "unknown") if isinstance(trace, dict) else "unknown",
    )
    turn_id = turn["turn_id"]
    tool_steps = list_message_tool_steps(config_path, assistant_message_id)
    source_message_ids = [user_message_id, assistant_message_id]
    source_tool_step_ids = [step["id"] for step in tool_steps if isinstance(step.get("id"), str)]
    existing_tasks = list_conversation_tasks(config_path, conversation_id)
    reflection_error = None
    decision = None
    if model_config:
        try:
            decision = _reflect_memory_with_model(
                model_config,
                llm_mode,
                artifact_dir,
                raw_user_input,
                final_answer,
                trace,
                tool_steps,
                existing_tasks,
                source_message_ids,
                source_tool_step_ids,
            )
        except Exception as exc:
            reflection_error = {"type": type(exc).__name__, "message": str(exc)}
    if decision is None:
        decision = _neutral_memory_decision(
            raw_user_input,
            final_answer,
            trace,
            source_message_ids,
            source_tool_step_ids,
            tool_steps,
        )
    tags_result = upsert_turn_memory_tags(
        _conversation_db_path(config_path),
        turn_id,
        decision["turn_tags"],
        source=decision.get("source", "neutral_fallback"),
    )
    summary_result = upsert_turn_summary(
        _conversation_db_path(config_path),
        turn_id,
        decision["turn_summary"],
        source=decision.get("source", "neutral_fallback"),
    )
    task_result = _apply_task_memory_decision(config_path, conversation_id, turn_id, decision["task_memory"])
    block_result = _maybe_create_memory_block(config_path, conversation_id)
    return {
        "status": "success",
        "turn": turn,
        "tags": tags_result,
        "summary": summary_result,
        "task_memory": task_result,
        "memory_block": block_result,
        "decision_source": decision.get("source", "neutral_fallback"),
        "reflection_error": reflection_error,
    }
