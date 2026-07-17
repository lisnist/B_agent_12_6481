import re


def validate_conversation_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("conversation_id must be a non-empty string")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return value
