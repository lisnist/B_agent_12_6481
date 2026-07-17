from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .b1_workspace import (
    _memory_overview,
    _required_outputs_pending,
    _successful_artifacts,
    _tool_attempts_summary,
    _workspace_history_messages,
    _workspace_memory,
)

_PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "prompts" / "b1_stage_prompts.json"


def _load_prompt_config() -> dict:
    with _PROMPT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("b1_stage_prompts.json must contain an object")
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


def build_llm_prompt_messages(messages: list[dict], tools_schema: list[dict]) -> list[dict]:
    """Build the model-facing prompt for the legacy one-loop Agent path."""
    if not isinstance(tools_schema, list):
        raise ValueError("tools_schema must be an array")
    prompt_messages = deepcopy(messages)
    tool_disclosure = (
        "\n\n本轮可用工具结构：\n"
        + json.dumps(_llm_tool_schemas(tools_schema), ensure_ascii=False)
        + "\n\n"
        + _prompt("legacy", "protocol_instruction")
    )
    if prompt_messages and prompt_messages[0].get("role") == "system":
        prompt_messages[0]["content"] += tool_disclosure
    else:
        prompt_messages.insert(0, {"role": "system", "content": tool_disclosure.strip()})

    if prompt_messages[-1].get("role") == "tool":
        prompt_messages.append(
            {
                "role": "user",
                "content": _prompt("legacy", "tool_result_followup"),
            }
        )
    return prompt_messages


def _json_block(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _short_tool_description(description: object) -> str:
    text = str(description or "").strip()
    for marker in (
        "\n工具执行结果会封装",
        "工具执行结果会封装",
        "\n主要 output 字段",
        "主要 output 字段",
    ):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def _compact_parameter_schema(parameters: object) -> dict:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "required": []}
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    compact_properties = {}
    for name, definition in properties.items():
        if not isinstance(definition, dict):
            continue
        entry = {}
        for key in ("type", "description", "enum"):
            if key in definition:
                entry[key] = definition[key]
        if definition.get("type") == "array" and isinstance(definition.get("items"), dict):
            item_type = definition["items"].get("type")
            entry["items"] = {"type": item_type} if item_type else definition["items"]
        compact_properties[name] = entry
    required = parameters.get("required", [])
    if not isinstance(required, list):
        required = []
    return {
        "type": "object",
        "properties": compact_properties,
        "required": [name for name in required if name in compact_properties],
    }


def _compact_returns_schema(function: dict) -> dict:
    raw_returns = function.get("x-returns")
    properties = raw_returns.get("properties") if isinstance(raw_returns, dict) else None
    if not isinstance(properties, dict):
        return {}
    compact = {}
    for name, definition in properties.items():
        if not isinstance(definition, dict):
            continue
        entry = {}
        for key in ("type", "description"):
            if key in definition:
                entry[key] = definition[key]
        compact[name] = entry
    return compact


def _llm_tool_schemas(tools_schema: list[dict]) -> list[dict]:
    """Return a compact model-facing tool view."""
    compact = []
    for tool in tools_schema:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name") or tool.get("name")
        if not name:
            continue
        entry = {
            "name": name,
            "description": _short_tool_description(function.get("description") or tool.get("description")),
            "parameters": _compact_parameter_schema(function.get("parameters") or tool.get("parameters")),
        }
        returns = _compact_returns_schema(function)
        if returns:
            entry["returns"] = returns
        compact.append(entry)
    return compact


def _tool_briefs(tools_schema: list[dict]) -> list[dict]:
    briefs = []
    for function in _llm_tool_schemas(tools_schema):
        entry = {
            "name": function.get("name"),
            "description": function.get("description", ""),
        }
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            properties = parameters.get("properties")
            if isinstance(properties, dict):
                entry["args"] = list(properties)
        briefs.append(entry)
    return briefs


def _stage_messages(system_prompt: str, stage_name: str, instruction: str, payload: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n"
                + _prompt("stage", "system_suffix")
            ),
        },
        {
            "role": "user",
            "content": (
                f"阶段：{stage_name}\n"
                f"{instruction}\n\n"
                "本轮状态如下：\n"
                f"{_json_block(payload)}\n\n"
                + _prompt("stage", "json_only_suffix")
            ),
        },
    ]


def _workspace_planning_messages(
    system_prompt: str,
    runtime: dict,
    selected_memory: dict,
    tools_schema: list[dict],
) -> list[dict]:
    history_messages = _workspace_history_messages(runtime)
    payload = {
        "user_input": runtime["user_input"],
        "input_images_count": len(runtime.get("input_images", [])),
        "history_messages": history_messages,
        "history_policy": {
            "source": "recent_history_messages" if isinstance(runtime.get("recent_history_messages"), list) else "history_messages",
            "full_history_message_count": len(runtime.get("history_messages", [])),
            "workspace_history_message_count": len(history_messages),
        },
        "legacy_memory": _memory_overview(selected_memory),
        "workspace_memory": _workspace_memory(selected_memory, runtime.get("workspace_memory_context")),
        "available_tools": _tool_briefs(tools_schema),
    }
    return _stage_messages(
        system_prompt,
        "planning",
        _prompt("workspace", "planning_instruction"),
        payload,
    )


def _workspace_tool_messages(
    system_prompt: str,
    workspace: dict,
    tools_schema: list[dict],
) -> list[dict]:
    payload = {
        "user_input": workspace["input"]["user_input"],
        "history_messages": workspace["input"].get("history_messages", []),
        "task": workspace["task"],
        "known_facts": workspace["draft"].get("known_facts", []),
        "missing_info": workspace["draft"].get("missing_info", []),
        "accepted_evidence": workspace["tools"].get("accepted_evidence", []),
        "rejected_evidence": workspace["tools"].get("rejected_evidence", []),
        "previous_tool_attempts": _tool_attempts_summary(workspace),
        "previous_no_action_outputs": workspace["tools"].get("no_action_outputs", []),
        "observations": workspace["tools"].get("observations", []),
        "success_criteria": workspace["task"].get("success_criteria", []),
        "required_outputs": workspace["task"].get("required_outputs", []),
        "produced_artifacts": _successful_artifacts(workspace),
        "pending_required_outputs": _required_outputs_pending(workspace),
        "available_tools_schema": _llm_tool_schemas(tools_schema),
    }
    return [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n"
                + _prompt("workspace", "tool_system_suffix")
            ),
        },
        {
            "role": "user",
            "content": (
                _prompt("workspace", "tool_user_instruction")
                + "\n\n"
                f"本轮状态如下：\n{_json_block(payload)}"
            ),
        },
    ]


def _workspace_observation_messages(
    system_prompt: str,
    workspace: dict,
    tool_messages: list[dict],
) -> list[dict]:
    payload = {
        "user_input": workspace["input"]["user_input"],
        "task": workspace["task"],
        "last_tool_intent": workspace["tools"].get("last_tool_intent", ""),
        "last_tool_messages": tool_messages,
        "previous_tool_attempts": _tool_attempts_summary(workspace),
        "rejected_evidence_before": workspace["tools"].get("rejected_evidence", []),
        "accepted_evidence_before": workspace["tools"].get("accepted_evidence", []),
        "known_facts_before": workspace["draft"].get("known_facts", []),
        "missing_info_before": workspace["draft"].get("missing_info", []),
        "success_criteria": workspace["task"].get("success_criteria", []),
        "required_outputs": workspace["task"].get("required_outputs", []),
        "produced_artifacts": _successful_artifacts(workspace),
        "pending_required_outputs": _required_outputs_pending(workspace),
    }
    return _stage_messages(
        system_prompt,
        "observation",
        _prompt("workspace", "observation_instruction"),
        payload,
    )


def _workspace_answer_messages(system_prompt: str, workspace: dict) -> list[dict]:
    payload = {
        "user_input": workspace["input"]["user_input"],
        "history_messages": workspace["input"].get("history_messages", []),
        "workspace_memory": workspace["memory"],
        "task": workspace["task"],
        "accepted_evidence": workspace["tools"].get("accepted_evidence", []),
        "rejected_evidence": workspace["tools"].get("rejected_evidence", []),
        "known_facts": workspace["draft"].get("known_facts", []),
        "missing_info": workspace["draft"].get("missing_info", []),
        "tool_attempts": _tool_attempts_summary(workspace),
        "observations": workspace["tools"].get("observations", []),
        "success_criteria": workspace["task"].get("success_criteria", []),
        "required_outputs": workspace["task"].get("required_outputs", []),
        "produced_artifacts": _successful_artifacts(workspace),
        "pending_required_outputs": _required_outputs_pending(workspace),
    }
    return [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n"
                + _prompt("workspace", "answer_system_suffix")
            ),
        },
        {
            "role": "user",
            "content": (
                _prompt("workspace", "answer_user_instruction")
                + "\n\n"
                f"本轮状态如下：\n{_json_block(payload)}"
            ),
        },
    ]


def _workspace_stage_failure_answer_messages(
    system_prompt: str,
    workspace: dict,
    failed_stage: str,
    error: dict | None,
    raw_text: str | None = None,
) -> list[dict]:
    payload = {
        "user_input": workspace["input"]["user_input"],
        "history_messages": workspace["input"].get("history_messages", []),
        "workspace_memory": workspace["memory"],
        "task": workspace["task"],
        "accepted_evidence": workspace["tools"].get("accepted_evidence", []),
        "rejected_evidence": workspace["tools"].get("rejected_evidence", []),
        "known_facts": workspace["draft"].get("known_facts", []),
        "missing_info": workspace["draft"].get("missing_info", []),
        "tool_attempts": _tool_attempts_summary(workspace),
        "observations": workspace["tools"].get("observations", []),
        "success_criteria": workspace["task"].get("success_criteria", []),
        "required_outputs": workspace["task"].get("required_outputs", []),
        "produced_artifacts": _successful_artifacts(workspace),
        "pending_required_outputs": _required_outputs_pending(workspace),
        "result_policy": {
            "only_successful_tool_results_are_evidence": True,
            "do_not_claim_completion_without_evidence": True,
        },
    }
    del raw_text
    return [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n"
                + _prompt("workspace", "stage_failure_system_suffix")
            ),
        },
        {
            "role": "user",
            "content": (
                _prompt("workspace", "stage_failure_user_instruction")
                + "\n\n"
                f"本轮状态如下：\n{_json_block(payload)}"
            ),
        },
    ]

