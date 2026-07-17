from __future__ import annotations

from typing import Any


VALID_ROLES = {"system", "user", "assistant", "tool"}
VALID_AGENT_STATES = {"acting", "replanning", "completed", "failed"}
VALID_AGENT_ACTIONS = {"call_tools", "finish"}
VALID_AGENT_STEP_PHASES = {"plan", "action", "observation", "final"}
HISTORY_ROLES = {"user", "assistant"}


def normalize_history_messages(messages: Any) -> list[dict]:
    """Normalize persisted chat history for the Agent runtime.

    Historical tool calls are not reconstructed. Tool and control messages
    belong to the active run; persisted history carries only completed
    user/assistant text with explicit roles.
    """
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise ValueError("history_messages must be an array")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"history message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in HISTORY_ROLES:
            raise ValueError(f"history message {index} has invalid role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"history message {index} content must be a non-empty string")
        normalized.append({"role": role, "content": content})
    return normalized


def make_ai_message(
    content: str = "",
    tool_calls: list[dict] | None = None,
    control: dict | None = None,
    agent_step: dict | None = None,
) -> dict:
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls or [],
    }
    if control is not None:
        message["control"] = control
    if agent_step is not None:
        message["agent_step"] = agent_step
    validate_ai_message(message)
    return message


def make_tool_message(
    tool_call_id: str,
    name: str,
    content: str,
    status: str = "success",
) -> dict:
    if status not in {"success", "error"}:
        raise ValueError(f"invalid tool status: {status}")
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
        "status": status,
    }


def make_skill_result(
    skill_name: str,
    status: str,
    input_data: dict,
    output: dict | None = None,
    error: dict | None = None,
    latency_ms: float | None = None,
) -> dict:
    if status not in {"success", "error"}:
        raise ValueError(f"invalid skill status: {status}")
    return {
        "skill_name": skill_name,
        "status": status,
        "input": input_data,
        "output": output,
        "error": error,
        "latency_ms": latency_ms,
        "summary": _skill_result_summary(status, output, error),
        "sources": _skill_result_sources(output),
        "artifacts": _skill_result_artifacts(output),
    }


def _skill_result_summary(status: str, output: dict | None, error: dict | None) -> dict:
    if status == "error":
        message = ""
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        return {"status": status, "message": message}
    if not isinstance(output, dict):
        return {"status": status, "message": ""}
    counts = {}
    for key in (
        "num_rows",
        "num_columns",
        "num_chars",
        "num_bytes",
        "size_bytes",
        "line_count",
        "slide_count",
        "entry_count",
        "child_count",
    ):
        value = output.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            counts[key] = value
    if isinstance(output.get("results"), list):
        counts["result_count"] = len(output["results"])
    return {
        "status": status,
        "message": _skill_result_message(output),
        "counts": counts,
    }


def _skill_result_message(output: dict) -> str:
    if isinstance(output.get("content"), str):
        text = output["content"].strip().replace("\n", " ")
        return text[:180]
    if isinstance(output.get("formatted_text"), str):
        text = output["formatted_text"].strip().replace("\n", " ")
        return text[:180]
    if isinstance(output.get("text"), str):
        text = output["text"].strip().replace("\n", " ")
        return text[:180]
    if isinstance(output.get("entries"), list):
        return f"{len(output['entries'])} entries"
    if "exists" in output and isinstance(output.get("path"), str):
        status = "exists" if output.get("exists") else "missing"
        return f"{output['path']} {status}"
    if "result" in output:
        return str(output["result"])
    if isinstance(output.get("time"), str) and isinstance(output.get("date"), str):
        return f"{output['date']} {output['time']}"
    return ""


def _skill_result_sources(output: dict | None) -> list[dict]:
    if not isinstance(output, dict):
        return []
    sources = []
    source = output.get("source") or output.get("path")
    if isinstance(source, str):
        item = {
            "source": source,
            "relative_path": output.get("relative_path"),
            "root_alias": output.get("root_alias"),
        }
        sources.append({key: value for key, value in item.items() if value is not None})
    results = output.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            result_source = result.get("source") or result.get("path")
            if not isinstance(result_source, str):
                continue
            item = {
                "source": result_source,
                "relative_path": result.get("relative_path"),
                "root_alias": result.get("root_alias"),
                "score": result.get("score"),
                "line_number": result.get("line_number"),
            }
            sources.append({key: value for key, value in item.items() if value is not None})
    return sources


def _skill_result_artifacts(output: dict | None) -> list[dict]:
    if not isinstance(output, dict):
        return []
    artifacts = []
    generated_path = output.get("generated_file_path")
    if isinstance(generated_path, str):
        item = {
            "path": generated_path,
            "relative_output_path": output.get("relative_output_path"),
            "filename": output.get("filename"),
            "file_type": output.get("file_type"),
            "suffix": output.get("suffix"),
            "num_bytes": output.get("num_bytes"),
        }
        artifacts.append({key: value for key, value in item.items() if value is not None})
    return artifacts


def normalize_tool_call(tool_call: dict[str, Any], index: int = 0) -> dict:
    if not isinstance(tool_call, dict):
        raise ValueError("tool call must be an object")
    if "function" in tool_call:
        function = tool_call.get("function") or {}
        name = function.get("name")
        args = function.get("arguments", {})
    else:
        name = tool_call.get("name")
        args = tool_call.get("args", {})
    if isinstance(args, str):
        import json

        args = json.loads(args)
    if not isinstance(name, str) or not name:
        raise ValueError("tool call name must be a non-empty string")
    if not isinstance(args, dict):
        raise ValueError("tool call args must be an object")
    call_id = tool_call.get("id") or f"call_{index + 1:03d}"
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool call id must be a non-empty string")
    return {"id": call_id, "name": name, "args": args}


def validate_ai_message(message: dict) -> None:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("AIMessage role must be assistant")
    if not isinstance(message.get("content"), str):
        raise ValueError("AIMessage content must be a string")
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        raise ValueError("AIMessage tool_calls must be a list")
    normalized = [normalize_tool_call(call, index) for index, call in enumerate(tool_calls)]
    message["tool_calls"] = normalized
    if not message["content"] and not normalized:
        raise ValueError("AIMessage must contain content or tool_calls")
    default_control = {
        "state": "acting" if normalized else "completed",
        "action": "call_tools" if normalized else "finish",
        "reason": "",
    }
    control = message.setdefault("control", default_control)
    if not isinstance(control, dict):
        raise ValueError("AIMessage control must be an object")
    if set(control) != {"state", "action", "reason"}:
        raise ValueError("AIMessage control must contain exactly state, action, and reason")
    state = control.get("state")
    action = control.get("action")
    reason = control.get("reason")
    if state not in VALID_AGENT_STATES:
        raise ValueError(f"invalid AIMessage control state: {state}")
    if action not in VALID_AGENT_ACTIONS:
        raise ValueError(f"invalid AIMessage control action: {action}")
    if reason is None:
        reason = ""
        control["reason"] = reason
    if not isinstance(reason, str):
        raise ValueError("AIMessage control reason must be a string")
    if action == "call_tools" and not normalized:
        raise ValueError("AIMessage call_tools action requires tool_calls")
    if action == "finish" and normalized:
        raise ValueError("AIMessage finish action requires an empty tool_calls array")
    if action == "call_tools" and state not in {"acting", "replanning"}:
        raise ValueError("AIMessage call_tools action requires acting or replanning state")
    if action == "finish" and state not in {"completed", "failed"}:
        raise ValueError("AIMessage finish action requires completed or failed state")
    if state == "failed" and not reason.strip():
        raise ValueError("AIMessage failed state requires a reason")
    agent_step = message.get("agent_step")
    if agent_step is None:
        return
    if not isinstance(agent_step, dict):
        raise ValueError("AIMessage agent_step must be an object")
    allowed = {"phase", "plan", "observation", "known_facts", "missing_info", "next_step"}
    unknown = set(agent_step) - allowed
    if unknown:
        raise ValueError(f"AIMessage agent_step contains unknown keys: {', '.join(sorted(unknown))}")
    phase = agent_step.get("phase")
    if phase not in VALID_AGENT_STEP_PHASES:
        raise ValueError(f"invalid AIMessage agent_step phase: {phase}")
    for key in ("plan", "observation", "next_step"):
        value = agent_step.get(key, "")
        if value is None:
            agent_step[key] = ""
            continue
        if not isinstance(value, str):
            raise ValueError(f"AIMessage agent_step {key} must be a string")
    for key in ("known_facts", "missing_info"):
        value = agent_step.get(key, [])
        if value is None:
            agent_step[key] = []
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"AIMessage agent_step {key} must be an array of strings")


def validate_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a top-level array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"message {index} has invalid role: {role}")
        if not isinstance(message.get("content", ""), str):
            raise ValueError(f"message {index} content must be a string")
        images = message.get("images", [])
        if not isinstance(images, list) or not all(isinstance(item, str) and item.startswith("data:image/") for item in images):
            raise ValueError(f"message {index} images must be an array of image data URLs")
        if images and role != "user":
            raise ValueError(f"message {index} images are only supported on user messages")
        if role == "assistant":
            message.setdefault("tool_calls", [])
            validate_ai_message(message)
        if role == "tool":
            for field in ("tool_call_id", "name", "status"):
                if field not in message:
                    raise ValueError(f"tool message {index} missing {field}")
    return messages
