import json
import re
from pathlib import Path

from backend.api_models import UploadedFileRef


def read_json_file(path_text: str) -> dict | list | None:
    path = Path(path_text)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(path_text: str, payload: dict | list) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_trace(trace_path: str) -> dict:
    trace = read_json_file(trace_path)
    if not isinstance(trace, dict):
        return {}
    return {
        "tool_rounds_used": trace.get("tool_rounds_used"),
        "llm_call_count": trace.get("llm_call_count"),
        "final_state": trace.get("final_state"),
        "finish_reason": trace.get("finish_reason"),
        "memory_save": trace.get("memory_save"),
        "checkpoint": trace.get("checkpoint"),
        "warnings": trace.get("warnings", []),
        "error": trace.get("error"),
    }


def read_trace_full(trace_path: str) -> dict:
    trace = read_json_file(trace_path)
    return trace if isinstance(trace, dict) else {}


def stream_event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def short_title(text: str, limit: int = 18) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return "新对话"
    return compact[:limit] + ("..." if len(compact) > limit else "")


def is_trivial_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).lower()
    return compact in {"你好", "您好", "hi", "hello", "在吗", "你是谁", "谢谢", "好的", "ok"}


def is_trivial_conversation(history: list[dict], current_user_input: str) -> bool:
    user_texts = [message["content"] for message in history if message.get("role") == "user"]
    user_texts.append(current_user_input)
    return bool(user_texts) and all(is_trivial_text(text) for text in user_texts)


def history_title(history: list[dict], current_user_input: str) -> str:
    for message in history:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return short_title(message["content"])
    return short_title(current_user_input)


def extract_tool_steps(trace: dict) -> list[dict]:
    steps = []
    turns = trace.get("turns", [])
    if not isinstance(turns, list):
        return steps
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        ai_message = turn.get("ai_message") if isinstance(turn.get("ai_message"), dict) else {}
        assistant_content = ai_message.get("content") if isinstance(ai_message.get("content"), str) else ""
        agent_step = ai_message.get("agent_step") if isinstance(ai_message.get("agent_step"), dict) else None
        raw_tool_calls = ai_message.get("tool_calls") if isinstance(ai_message, dict) else []
        tool_calls_by_id = {
            call.get("id"): call for call in raw_tool_calls
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        } if isinstance(raw_tool_calls, list) else {}
        turn_tool_messages = turn.get("tool_messages", [])
        if not isinstance(turn_tool_messages, list):
            turn_tool_messages = []
        for tool_message in turn_tool_messages:
            if not isinstance(tool_message, dict):
                continue
            raw_content = tool_message.get("content")
            try:
                parsed = json.loads(raw_content) if isinstance(raw_content, str) else {}
            except json.JSONDecodeError:
                parsed = {}
            tool_call_id = tool_message.get("tool_call_id")
            tool_call = tool_calls_by_id.get(tool_call_id)
            input_data = parsed.get("input")
            if tool_call is not None:
                input_data = {
                    "assistant_content_before_tool": assistant_content,
                    "agent_step_before_tool": agent_step,
                    "tool_call": tool_call,
                    "skill_input": input_data,
                }
            steps.append({
                "tool_call_id": tool_call_id,
                "tool_name": tool_message.get("name") or parsed.get("skill_name") or "unknown",
                "input_data": input_data,
                "output_data": parsed.get("output"),
                "status": tool_message.get("status") or parsed.get("status") or "unknown",
                "error": parsed.get("error"),
                "latency_ms": parsed.get("latency_ms"),
            })
        if (
            not turn_tool_messages
            and isinstance(agent_step, dict)
            and agent_step.get("phase") in {"plan", "observation"}
        ):
            steps.append({
                "tool_call_id": None,
                "tool_name": "agent_observation",
                "input_data": {"agent_step": agent_step},
                "output_data": None,
                "status": "info",
                "error": None,
                "latency_ms": None,
            })
    return steps


def assistant_metadata(result: dict, trace: dict) -> dict:
    return {
        "agent_status": result.get("status"),
        "elapsed_ms": result.get("elapsed_ms"),
        "trace_path": result.get("trace_path"),
        "output_dir": str(Path(result.get("trace_path", "")).parent) if result.get("trace_path") else None,
        "llm_call_count": trace.get("llm_call_count"),
        "tool_rounds_used": trace.get("tool_rounds_used"),
        "final_state": trace.get("final_state"),
        "finish_reason": trace.get("finish_reason"),
        "memory_save": trace.get("memory_save"),
    }


def message_ui_status(message: dict) -> str | None:
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    status = metadata.get("ui_status")
    return status if status in {"pending", "error", "cancelled"} else None


def message_resumable(message: dict) -> bool:
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    return bool(metadata.get("resumable"))


def message_attachments(message: dict) -> list[UploadedFileRef]:
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    raw_attachments = metadata.get("attachments")
    if not isinstance(raw_attachments, list):
        return []
    attachments = []
    for item in raw_attachments:
        try:
            attachments.append(UploadedFileRef.model_validate(item))
        except (TypeError, ValueError):
            continue
    return attachments
