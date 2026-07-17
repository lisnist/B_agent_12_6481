from __future__ import annotations

import sys
from datetime import datetime

from fastapi import HTTPException

from backend.settings import CODE_DIR

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.identifiers import validate_conversation_id


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def safe_conversation_id(value: str | None) -> str:
    if value is None or not value.strip():
        return f"conv_web_{now_stamp()}"
    cleaned = value.strip()
    try:
        return validate_conversation_id(cleaned)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="conversation_id contains unsupported characters") from exc


def safe_run_id(value: str) -> str:
    try:
        return validate_conversation_id(value.strip())
    except (AttributeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="run_id contains unsupported characters") from exc
