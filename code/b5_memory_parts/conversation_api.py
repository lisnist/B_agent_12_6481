from __future__ import annotations

from common.conversation_store import (
    append_message,
    count_memory_retrieval_logs,
    delete_conversation,
    delete_tool_steps,
    get_conversation,
    init_store,
    list_conversation_turns,
    list_conversations,
    list_memory_blocks,
    list_memory_retrieval_logs,
    list_messages,
    list_task_memories,
    list_turn_summaries,
    list_tool_steps,
    record_tool_step,
    update_message,
    upsert_conversation,
)
from common.schemas import normalize_history_messages

from b5_memory_parts.paths import _conversation_db_path, _safe_conversation_id


def init_conversation_db(config_path: str) -> dict:
    return init_store(_conversation_db_path(config_path))

def upsert_conversation_record(
    config_path: str,
    conversation_id: str,
    title: str,
    is_trivial: bool = False,
    trivial_reason: str | None = None,
    summary: str | None = None,
    status: str = "active",
) -> dict:
    _safe_conversation_id(conversation_id)
    return upsert_conversation(
        _conversation_db_path(config_path),
        conversation_id,
        title,
        is_trivial=is_trivial,
        trivial_reason=trivial_reason,
        summary=summary,
        status=status,
    )

def append_conversation_message(
    config_path: str,
    conversation_id: str,
    role: str,
    content: str,
    message_id: str | None = None,
    run_id: str | None = None,
    message_order: int | None = None,
    is_trivial: bool = False,
    token_count: int | None = None,
    metadata: dict | None = None,
) -> dict:
    _safe_conversation_id(conversation_id)
    return append_message(
        _conversation_db_path(config_path),
        conversation_id,
        role,
        content,
        message_id=message_id,
        run_id=run_id,
        message_order=message_order,
        is_trivial=is_trivial,
        token_count=token_count,
        metadata=metadata,
    )

def update_conversation_message(
    config_path: str,
    message_id: str,
    content: str | None = None,
    token_count: int | None = None,
    metadata: dict | None = None,
) -> dict:
    return update_message(
        _conversation_db_path(config_path),
        message_id,
        content=content,
        token_count=token_count,
        metadata=metadata,
    )

def record_conversation_tool_step(
    config_path: str,
    conversation_id: str,
    assistant_message_id: str,
    tool_name: str,
    step_index: int,
    step_id: str | None = None,
    run_id: str | None = None,
    tool_call_id: str | None = None,
    input_data: object = None,
    output_data: object = None,
    status: str = "success",
    error: object = None,
    latency_ms: float | None = None,
) -> dict:
    _safe_conversation_id(conversation_id)
    return record_tool_step(
        _conversation_db_path(config_path),
        conversation_id,
        assistant_message_id,
        tool_name,
        step_id=step_id,
        run_id=run_id,
        step_index=step_index,
        tool_call_id=tool_call_id,
        input_data=input_data,
        output_data=output_data,
        status=status,
        error=error,
        latency_ms=latency_ms,
    )

def list_conversation_records(config_path: str, limit: int = 50) -> list[dict]:
    return list_conversations(_conversation_db_path(config_path), limit)

def get_conversation_memory_snapshot(config_path: str, conversation_id: str, retrieval_limit: int = 20) -> dict:
    """Return a read-only B5 inspection snapshot for one web conversation."""
    _safe_conversation_id(conversation_id)
    db_path = _conversation_db_path(config_path)
    conversation = get_conversation(db_path, conversation_id)
    if conversation is None:
        return {
            "status": "not_found",
            "conversation_id": conversation_id,
            "counts": {
                "messages": 0,
                "tool_steps": 0,
                "turns": 0,
                "turn_summaries": 0,
                "memory_blocks": 0,
                "task_memories": 0,
                "retrieval_logs": 0,
            },
            "messages": [],
            "turns": [],
            "turn_summaries": [],
            "memory_blocks": [],
            "task_memories": [],
            "retrieval_logs": [],
        }

    messages = list_messages(db_path, conversation_id)
    total_tool_steps = 0
    messages_with_steps = []
    for message in messages:
        item = dict(message)
        tool_steps = list_tool_steps(db_path, item["id"]) if item.get("role") == "assistant" else []
        total_tool_steps += len(tool_steps)
        item["tool_steps"] = tool_steps
        item["tool_step_count"] = len(tool_steps)
        messages_with_steps.append(item)

    turns = list_conversation_turns(db_path, conversation_id)
    turn_summaries = list_turn_summaries(db_path, conversation_id)
    memory_blocks = list_memory_blocks(db_path, conversation_id)
    task_memories = list_task_memories(db_path, conversation_id)
    retrieval_logs = list_memory_retrieval_logs(db_path, conversation_id, limit=retrieval_limit)
    retrieval_log_count = count_memory_retrieval_logs(db_path, conversation_id)

    return {
        "status": "success",
        "conversation_id": conversation_id,
        "conversation": conversation,
        "counts": {
            "messages": len(messages_with_steps),
            "tool_steps": total_tool_steps,
            "turns": len(turns),
            "turn_summaries": len(turn_summaries),
            "memory_blocks": len(memory_blocks),
            "task_memories": len(task_memories),
            "retrieval_logs": retrieval_log_count,
        },
        "messages": messages_with_steps,
        "turns": turns,
        "turn_summaries": turn_summaries,
        "memory_blocks": memory_blocks,
        "task_memories": task_memories,
        "retrieval_logs": retrieval_logs,
    }

def delete_conversation_record(config_path: str, conversation_id: str) -> dict:
    _safe_conversation_id(conversation_id)
    return delete_conversation(_conversation_db_path(config_path), conversation_id)

def list_conversation_messages(config_path: str, conversation_id: str) -> list[dict]:
    _safe_conversation_id(conversation_id)
    return list_messages(_conversation_db_path(config_path), conversation_id)

def list_conversation_history(config_path: str, conversation_id: str) -> list[dict]:
    """Expose completed SQLite history using the runtime message protocol."""
    messages = list_conversation_messages(config_path, conversation_id)
    history = []
    for message in messages:
        if message.get("role") not in {"user", "assistant"}:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if metadata.get("ui_status") in {"pending", "error", "cancelled"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            history.append({"role": message["role"], "content": content})
    return normalize_history_messages(history)

def list_message_tool_steps(config_path: str, assistant_message_id: str) -> list[dict]:
    return list_tool_steps(_conversation_db_path(config_path), assistant_message_id)

def clear_message_tool_steps(config_path: str, assistant_message_id: str) -> dict:
    return delete_tool_steps(_conversation_db_path(config_path), assistant_message_id)

def list_conversation_tasks(config_path: str, conversation_id: str, status: str | None = None) -> list[dict]:
    _safe_conversation_id(conversation_id)
    return list_task_memories(_conversation_db_path(config_path), conversation_id, status)
