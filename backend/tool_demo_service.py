from __future__ import annotations

import json

from backend.api_models import B3ToolCallsPreviewRequest


def b2_skill_summary(name: str, definition: dict, enabled: bool) -> dict:
    parameters = definition.get("parameters") if isinstance(definition.get("parameters"), dict) else {}
    returns = definition.get("returns") if isinstance(definition.get("returns"), dict) else {}
    required = definition.get("required") if isinstance(definition.get("required"), list) else []
    return {
        "name": name,
        "enabled": enabled,
        "module": definition.get("module"),
        "function": definition.get("function"),
        "description": definition.get("description"),
        "side_effects": bool(definition.get("side_effects")),
        "parameters": parameters,
        "required": required,
        "returns": returns,
        "parameter_count": len(parameters),
        "return_count": len(returns),
    }


def b3_tool_calls_from_request(request: B3ToolCallsPreviewRequest) -> list:
    if request.tool_calls:
        return request.tool_calls
    if isinstance(request.ai_message, dict):
        tool_calls = request.ai_message.get("tool_calls")
        if isinstance(tool_calls, list):
            return tool_calls
    return []


def parse_b3_tool_message(message: dict) -> dict | None:
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
