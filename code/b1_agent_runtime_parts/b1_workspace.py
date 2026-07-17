from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from common.schemas import normalize_history_messages


def _memory_overview(selected_memory: dict) -> dict:
    if not isinstance(selected_memory, dict):
        return {"status": "unknown"}
    docs = selected_memory.get("selected_memory_docs")
    return {
        "status": selected_memory.get("status"),
        "selected_memory_count": len(docs) if isinstance(docs, list) else 0,
        "total_chars": selected_memory.get("total_chars"),
        "truncated": selected_memory.get("truncated"),
        "errors": selected_memory.get("errors", []),
        "note": "legacy markdown memory is not injected into the active workspace context",
    }


def _workspace_memory(selected_memory: dict, workspace_memory_context: dict | None = None) -> dict:
    result = {"legacy": _memory_overview(selected_memory)}
    result["layered"] = deepcopy(workspace_memory_context) if isinstance(workspace_memory_context, dict) else {
        "status": "not_requested"
    }
    return result


def _workspace_history_messages(runtime: dict) -> list[dict]:
    recent_history = runtime.get("recent_history_messages")
    if isinstance(recent_history, list):
        return normalize_history_messages(recent_history)
    return normalize_history_messages(runtime.get("history_messages", []))


def _prepare_workspace_runtime_context(
    runtime: dict,
    memory_file: Path,
    selected_memory: dict,
    output_dir: Path,
    model_file: Path | None = None,
    llm_mode: str | None = None,
) -> tuple[dict, dict]:
    from b5_memory import prepare_workspace_memory_context

    updated = deepcopy(runtime)
    memory_package = prepare_workspace_memory_context(
        str(memory_file),
        runtime["conversation_id"],
        runtime["user_input"],
        runtime["history_messages"],
        selected_memory,
        str(output_dir),
        str(model_file) if model_file is not None else None,
        llm_mode,
    )
    recent_history = memory_package.get("recent_history_messages")
    if isinstance(recent_history, list):
        updated["recent_history_messages"] = normalize_history_messages(recent_history)
    workspace_memory = memory_package.get("workspace_memory")
    updated["workspace_memory_context"] = workspace_memory if isinstance(workspace_memory, dict) else {
        "status": "error",
        "error": {"type": "InvalidMemoryPackage", "message": "B5 did not return workspace_memory"},
    }
    updated["workspace_memory_build"] = {
        "status": memory_package.get("status"),
        "history_message_count": memory_package.get("history_message_count"),
        "recent_history_message_count": memory_package.get("recent_history_message_count"),
    }
    return updated, memory_package


def _workspace_from_runtime(runtime: dict, selected_memory: dict) -> dict:
    history_messages = _workspace_history_messages(runtime)
    return {
        "input": {
            "conversation_id": runtime["conversation_id"],
            "user_input": runtime["user_input"],
            "history_messages": history_messages,
            "history_policy": {
                "source": "recent_history_messages" if isinstance(runtime.get("recent_history_messages"), list) else "history_messages",
                "full_history_message_count": len(runtime.get("history_messages", [])),
                "workspace_history_message_count": len(history_messages),
            },
            "input_images_count": len(runtime.get("input_images", [])),
        },
        "memory": _workspace_memory(selected_memory, runtime.get("workspace_memory_context")),
        "task": {
            "user_goal": "",
            "requirements": [],
            "success_criteria": [],
            "required_outputs": [],
            "plan": "",
            "stage": "planning",
            "reason": "",
        },
        "tools": {
            "calls": [],
            "results": [],
            "observations": [],
            "accepted_evidence": [],
            "rejected_evidence": [],
            "no_action_outputs": [],
            "last_tool_intent": "",
        },
        "draft": {
            "known_facts": [],
            "missing_info": [],
        },
        "final": {
            "answer": "",
            "status": "",
        },
        "trace": [],
    }


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"无", "无。", "none", "None", "null"}:
            return []
        return [text]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _merge_unique(target: list[str], values: object) -> None:
    for value in _as_string_list(values):
        if value not in target:
            target.append(value)


def _compact_for_workspace(value: object, limit: int = 900) -> object:
    if isinstance(value, str):
        text = value.strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "..."
    if isinstance(value, list):
        return [_compact_for_workspace(item, max(160, limit // 3)) for item in value[:6]]
    if isinstance(value, dict):
        compact = {}
        for key, item in value.items():
            if key in {"content", "text"}:
                compact[key] = _compact_for_workspace(item, limit)
            elif key in {"generated_file_path", "relative_output_path", "download_url", "filename", "file_type", "status", "error"}:
                compact[key] = _compact_for_workspace(item, limit)
            elif isinstance(item, (str, int, float, bool)) or item is None:
                compact[key] = item
        return compact
    return value


def _tool_message_payload(message: dict) -> dict:
    raw_content = message.get("content")
    try:
        parsed = json.loads(raw_content) if isinstance(raw_content, str) else {}
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _successful_artifacts(workspace: dict) -> list[dict]:
    artifacts: list[dict] = []
    for message in workspace["tools"].get("results", []):
        if not isinstance(message, dict):
            continue
        parsed = _tool_message_payload(message)
        if parsed.get("status") != "success" and message.get("status") != "success":
            continue
        output = parsed.get("output")
        if isinstance(output, dict):
            if any(output.get(key) for key in ("generated_file_path", "relative_output_path", "download_url")):
                artifacts.append(
                    {
                        key: output.get(key)
                        for key in (
                            "filename",
                            "file_type",
                            "suffix",
                            "relative_output_path",
                            "download_url",
                            "num_bytes",
                        )
                        if output.get(key) is not None
                    }
                )
        raw_artifacts = parsed.get("artifacts")
        if isinstance(raw_artifacts, list):
            for artifact in raw_artifacts:
                if isinstance(artifact, dict):
                    artifacts.append(
                        {
                            key: artifact.get(key)
                            for key in (
                                "filename",
                                "file_type",
                                "suffix",
                                "relative_output_path",
                                "download_url",
                                "num_bytes",
                            )
                            if artifact.get(key) is not None
                        }
                    )
    deduped = []
    seen = set()
    for artifact in artifacts:
        key = artifact.get("download_url") or artifact.get("relative_output_path") or artifact.get("filename")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(artifact)
    return deduped


def _required_outputs_pending(workspace: dict) -> list[object]:
    required = workspace["task"].get("required_outputs", [])
    if not isinstance(required, list) or not required:
        return []
    artifacts = _successful_artifacts(workspace)
    if len(artifacts) >= len(required):
        return []
    return required[len(artifacts):]


def _tool_attempts_summary(workspace: dict) -> list[dict]:
    calls = workspace["tools"].get("calls", [])
    results = workspace["tools"].get("results", [])
    summaries = []
    for index, message in enumerate(results):
        if not isinstance(message, dict):
            continue
        call = calls[index] if index < len(calls) and isinstance(calls[index], dict) else {}
        parsed = _tool_message_payload(message)
        output = parsed.get("output")
        error = parsed.get("error")
        summaries.append(
            {
                "index": index + 1,
                "tool_name": message.get("name") or call.get("name") or parsed.get("skill_name"),
                "tool_call_id": message.get("tool_call_id") or call.get("id"),
                "args": call.get("args") or parsed.get("input"),
                "status": message.get("status") or parsed.get("status"),
                "output_summary": _compact_for_workspace(output),
                "error": _compact_for_workspace(error),
            }
        )
    return summaries


def _record_no_tool_action(workspace: dict, ai_message: dict) -> None:
    content = ai_message.get("content", "").strip()
    note = "工具动作阶段没有返回 tool_calls，因此本阶段没有执行任何工具。"
    if content:
        note += " 该阶段输出的自然语言或草稿不能作为工具结果或完成证据。"
    workspace["tools"]["no_action_outputs"].append(
        {
            "content": content,
            "control": ai_message.get("control"),
            "agent_step": ai_message.get("agent_step"),
        }
    )
    _merge_unique(workspace["tools"]["rejected_evidence"], [note])
    _merge_unique(workspace["draft"]["missing_info"], ["缺少本轮工具动作实际执行后的结果。"])
    if _required_outputs_pending(workspace):
        workspace["task"]["stage"] = "tool_calling"
        _merge_unique(workspace["draft"]["missing_info"], ["用户要求生成文件，但本轮没有成功的文件产物。"])
    else:
        workspace["task"]["stage"] = "answering"
    workspace["task"]["reason"] = note


def _record_stage(workspace: dict, phase: str, payload: dict) -> None:
    workspace["trace"].append({"phase": phase, "payload": deepcopy(payload)})


def _agent_step_from_plan(plan: dict) -> dict:
    return {
        "phase": "plan",
        "plan": str(plan.get("plan") or ""),
        "observation": str(plan.get("reason") or ""),
        "known_facts": _as_string_list(plan.get("known_facts")),
        "missing_info": _as_string_list(plan.get("missing_info")),
        "next_step": str(plan.get("next_stage") or ""),
    }


def _agent_step_from_observation(observation: dict) -> dict:
    return {
        "phase": "observation",
        "plan": str(observation.get("reason") or ""),
        "observation": str(observation.get("observation") or ""),
        "known_facts": _as_string_list(observation.get("known_facts")),
        "missing_info": _as_string_list(observation.get("missing_info")),
        "next_step": str(observation.get("next_stage") or ""),
    }

