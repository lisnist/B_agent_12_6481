from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from common.logging_utils import now_iso


SCHEMA_VERSION = 3


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _score(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _flag(value: Any, default: bool = False) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(default)


def init_store(db_path: str | Path) -> dict:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                is_trivial INTEGER NOT NULL DEFAULT 0,
                trivial_reason TEXT,
                summary TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_message_at TEXT
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                content TEXT NOT NULL,
                message_order INTEGER NOT NULL,
                run_id TEXT,
                is_trivial INTEGER NOT NULL DEFAULT 0,
                token_count INTEGER,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE (conversation_id, message_order)
            );

            CREATE TABLE IF NOT EXISTS tool_steps (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL,
                run_id TEXT,
                step_index INTEGER NOT NULL,
                tool_call_id TEXT,
                tool_name TEXT NOT NULL,
                input_json TEXT,
                output_json TEXT,
                status TEXT NOT NULL,
                error_json TEXT,
                latency_ms REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (assistant_message_id) REFERENCES conversation_messages(id) ON DELETE CASCADE,
                UNIQUE (assistant_message_id, step_index)
            );

            CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_order
                ON conversation_messages(conversation_id, message_order);
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_role
                ON conversation_messages(role);
            CREATE INDEX IF NOT EXISTS idx_tool_steps_message_order
                ON tool_steps(assistant_message_id, step_index);
            CREATE INDEX IF NOT EXISTS idx_tool_steps_conversation
                ON tool_steps(conversation_id, step_index);

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                user_message_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (user_message_id) REFERENCES conversation_messages(id) ON DELETE CASCADE,
                FOREIGN KEY (assistant_message_id) REFERENCES conversation_messages(id) ON DELETE CASCADE,
                UNIQUE (conversation_id, run_id),
                UNIQUE (conversation_id, user_message_id),
                UNIQUE (conversation_id, assistant_message_id),
                UNIQUE (conversation_id, turn_index)
            );

            CREATE TABLE IF NOT EXISTS turn_memory_tags (
                turn_id TEXT PRIMARY KEY,
                current_task_relevance REAL NOT NULL,
                long_term_value REAL NOT NULL,
                has_explicit_fact INTEGER NOT NULL,
                has_decision INTEGER NOT NULL,
                has_user_correction INTEGER NOT NULL,
                allow_compress INTEGER NOT NULL,
                allow_drop INTEGER NOT NULL,
                noise_score REAL NOT NULL,
                labels_json TEXT,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (turn_id) REFERENCES conversation_turns(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS turn_summaries (
                turn_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                keywords_json TEXT,
                facts_json TEXT,
                decisions_json TEXT,
                corrections_json TEXT,
                tool_refs_json TEXT,
                artifact_refs_json TEXT,
                source_message_ids_json TEXT NOT NULL,
                source_tool_step_ids_json TEXT,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (turn_id) REFERENCES conversation_turns(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_blocks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                task_id TEXT,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                status TEXT NOT NULL,
                start_turn_index INTEGER NOT NULL,
                end_turn_index INTEGER NOT NULL,
                keywords_json TEXT,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE (conversation_id, start_turn_index, end_turn_index)
            );

            CREATE TABLE IF NOT EXISTS memory_block_turns (
                block_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (block_id, turn_id),
                FOREIGN KEY (block_id) REFERENCES memory_blocks(id) ON DELETE CASCADE,
                FOREIGN KEY (turn_id) REFERENCES conversation_turns(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS task_memories (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                title TEXT NOT NULL,
                objective TEXT,
                phase TEXT,
                completed_items_json TEXT,
                pending_items_json TEXT,
                constraints_json TEXT,
                key_results_json TEXT,
                active_files_json TEXT,
                blocked_items_json TEXT,
                next_actions_json TEXT,
                source_turn_ids_json TEXT,
                confidence REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_retrieval_log (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_context_json TEXT,
                candidate_blocks_json TEXT,
                selected_turns_json TEXT,
                loaded_message_ids_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (item_type, item_id, provider, model)
            );

            CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation_order
                ON conversation_turns(conversation_id, turn_index);
            CREATE INDEX IF NOT EXISTS idx_turn_memory_tags_value
                ON turn_memory_tags(long_term_value, current_task_relevance);
            CREATE INDEX IF NOT EXISTS idx_memory_blocks_conversation_status
                ON memory_blocks(conversation_id, status, start_turn_index);
            CREATE INDEX IF NOT EXISTS idx_memory_block_turns_turn
                ON memory_block_turns(turn_id);
            CREATE INDEX IF NOT EXISTS idx_task_memories_conversation_status
                ON task_memories(conversation_id, status);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_task_memories_one_foreground
                ON task_memories(conversation_id)
                WHERE status = 'foreground';
            CREATE INDEX IF NOT EXISTS idx_memory_embeddings_item
                ON memory_embeddings(item_type, item_id);
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
    return {"status": "success", "db_path": str(Path(db_path).resolve()), "schema_version": SCHEMA_VERSION}


def upsert_conversation(
    db_path: str | Path,
    conversation_id: str,
    title: str,
    *,
    is_trivial: bool = False,
    trivial_reason: str | None = None,
    summary: str | None = None,
    status: str = "active",
) -> dict:
    init_store(db_path)
    now = now_iso()
    with _connect(db_path) as connection:
        existing = connection.execute(
            "SELECT created_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        connection.execute(
            """
            INSERT INTO conversations(
                id, title, is_trivial, trivial_reason, summary, status, created_at, updated_at, last_message_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT last_message_at FROM conversations WHERE id = ?), NULL))
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                is_trivial = excluded.is_trivial,
                trivial_reason = excluded.trivial_reason,
                summary = excluded.summary,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                title,
                int(is_trivial),
                trivial_reason,
                summary,
                status,
                created_at,
                now,
                conversation_id,
            ),
        )
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "created_at": created_at,
        "updated_at": now,
    }


def append_message(
    db_path: str | Path,
    conversation_id: str,
    role: str,
    content: str,
    *,
    message_id: str | None = None,
    run_id: str | None = None,
    message_order: int | None = None,
    is_trivial: bool = False,
    token_count: int | None = None,
    metadata: dict | None = None,
) -> dict:
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("role must be system, user, assistant, or tool")
    if not isinstance(content, str) or not content:
        raise ValueError("content must be a non-empty string")
    init_store(db_path)
    now = now_iso()
    message_id = message_id or _new_id("msg")
    with _connect(db_path) as connection:
        if message_order is None:
            row = connection.execute(
                "SELECT COALESCE(MAX(message_order), 0) + 1 AS next_order FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            message_order = int(row["next_order"])
        connection.execute(
            """
            INSERT INTO conversation_messages(
                id, conversation_id, role, content, message_order, run_id, is_trivial,
                token_count, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                message_order,
                run_id,
                int(is_trivial),
                token_count,
                _json_dumps(metadata),
                now,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ?, last_message_at = ? WHERE id = ?",
            (now, now, conversation_id),
        )
    return {
        "status": "success",
        "message_id": message_id,
        "conversation_id": conversation_id,
        "message_order": message_order,
        "created_at": now,
    }


def update_message(
    db_path: str | Path,
    message_id: str,
    *,
    content: str | None = None,
    token_count: int | None = None,
    metadata: dict | None = None,
) -> dict:
    if content is not None and (not isinstance(content, str) or not content):
        raise ValueError("content must be a non-empty string")
    init_store(db_path)
    now = now_iso()
    assignments = []
    values: list[Any] = []
    if content is not None:
        assignments.append("content = ?")
        values.append(content)
    if token_count is not None:
        assignments.append("token_count = ?")
        values.append(token_count)
    if metadata is not None:
        assignments.append("metadata_json = ?")
        values.append(_json_dumps(metadata))
    if not assignments:
        raise ValueError("update_message requires at least one field to update")
    values.append(message_id)
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT conversation_id FROM conversation_messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            raise ValueError("message_id does not exist")
        connection.execute(
            f"UPDATE conversation_messages SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ?, last_message_at = ? WHERE id = ?",
            (now, now, row["conversation_id"]),
        )
    return {"status": "success", "message_id": message_id, "updated_at": now}


def record_tool_step(
    db_path: str | Path,
    conversation_id: str,
    assistant_message_id: str,
    tool_name: str,
    *,
    step_id: str | None = None,
    run_id: str | None = None,
    step_index: int,
    tool_call_id: str | None = None,
    input_data: Any = None,
    output_data: Any = None,
    status: str,
    error: Any = None,
    latency_ms: float | None = None,
) -> dict:
    if step_index < 1:
        raise ValueError("step_index must be positive")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_name must be a non-empty string")
    if not isinstance(status, str) or not status:
        raise ValueError("status must be a non-empty string")
    init_store(db_path)
    now = now_iso()
    step_id = step_id or _new_id("tool_step")
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO tool_steps(
                id, conversation_id, assistant_message_id, run_id, step_index, tool_call_id,
                tool_name, input_json, output_json, status, error_json, latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id,
                conversation_id,
                assistant_message_id,
                run_id,
                step_index,
                tool_call_id,
                tool_name,
                _json_dumps(input_data),
                _json_dumps(output_data),
                status,
                _json_dumps(error),
                latency_ms,
                now,
            ),
        )
    return {
        "status": "success",
        "tool_step_id": step_id,
        "conversation_id": conversation_id,
        "assistant_message_id": assistant_message_id,
        "step_index": step_index,
        "created_at": now,
    }


def delete_tool_steps(db_path: str | Path, assistant_message_id: str) -> dict:
    init_store(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM tool_steps WHERE assistant_message_id = ?",
            (assistant_message_id,),
        )
    return {
        "status": "success",
        "assistant_message_id": assistant_message_id,
        "deleted_count": cursor.rowcount,
    }


def upsert_conversation_turn(
    db_path: str | Path,
    conversation_id: str,
    run_id: str,
    user_message_id: str,
    assistant_message_id: str,
    *,
    turn_id: str | None = None,
    status: str = "success",
) -> dict:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(status, str) or not status:
        raise ValueError("status must be a non-empty string")
    init_store(db_path)
    now = now_iso()
    with _connect(db_path) as connection:
        existing = connection.execute(
            """
            SELECT id, turn_index, created_at
            FROM conversation_turns
            WHERE conversation_id = ? AND run_id = ?
            """,
            (conversation_id, run_id),
        ).fetchone()
        if existing:
            turn_id = existing["id"]
            turn_index = int(existing["turn_index"])
            created_at = existing["created_at"]
        else:
            turn_id = turn_id or _new_id("turn")
            row = connection.execute(
                "SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_turn FROM conversation_turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            turn_index = int(row["next_turn"])
            created_at = now
        user_row = connection.execute(
            "SELECT created_at FROM conversation_messages WHERE id = ? AND conversation_id = ?",
            (user_message_id, conversation_id),
        ).fetchone()
        assistant_row = connection.execute(
            "SELECT created_at FROM conversation_messages WHERE id = ? AND conversation_id = ?",
            (assistant_message_id, conversation_id),
        ).fetchone()
        if user_row is None or assistant_row is None:
            raise ValueError("turn message ids must exist in the same conversation")
        connection.execute(
            """
            INSERT INTO conversation_turns(
                id, conversation_id, run_id, user_message_id, assistant_message_id,
                turn_index, status, started_at, completed_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id, run_id) DO UPDATE SET
                user_message_id = excluded.user_message_id,
                assistant_message_id = excluded.assistant_message_id,
                status = excluded.status,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at,
                updated_at = excluded.updated_at
            """,
            (
                turn_id,
                conversation_id,
                run_id,
                user_message_id,
                assistant_message_id,
                turn_index,
                status,
                user_row["created_at"],
                assistant_row["created_at"],
                created_at,
                now,
            ),
        )
    return {
        "status": "success",
        "turn_id": turn_id,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "turn_index": turn_index,
        "updated_at": now,
    }


def upsert_turn_memory_tags(
    db_path: str | Path,
    turn_id: str,
    tags: dict,
    *,
    source: str = "unspecified",
) -> dict:
    if not isinstance(tags, dict):
        raise ValueError("tags must be an object")
    init_store(db_path)
    now = now_iso()
    labels = tags.get("labels")
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT created_at FROM turn_memory_tags WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        created_at = row["created_at"] if row else now
        connection.execute(
            """
            INSERT INTO turn_memory_tags(
                turn_id, current_task_relevance, long_term_value, has_explicit_fact,
                has_decision, has_user_correction, allow_compress, allow_drop,
                noise_score, labels_json, source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET
                current_task_relevance = excluded.current_task_relevance,
                long_term_value = excluded.long_term_value,
                has_explicit_fact = excluded.has_explicit_fact,
                has_decision = excluded.has_decision,
                has_user_correction = excluded.has_user_correction,
                allow_compress = excluded.allow_compress,
                allow_drop = excluded.allow_drop,
                noise_score = excluded.noise_score,
                labels_json = excluded.labels_json,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                turn_id,
                _score(tags.get("current_task_relevance")),
                _score(tags.get("long_term_value")),
                _flag(tags.get("has_explicit_fact")),
                _flag(tags.get("has_decision")),
                _flag(tags.get("has_user_correction")),
                _flag(tags.get("allow_compress"), True),
                _flag(tags.get("allow_drop")),
                _score(tags.get("noise_score")),
                _json_dumps(labels if isinstance(labels, list) else []),
                source,
                created_at,
                now,
            ),
        )
    return {"status": "success", "turn_id": turn_id, "updated_at": now}


def upsert_turn_summary(
    db_path: str | Path,
    turn_id: str,
    summary: dict,
    *,
    source: str = "unspecified",
) -> dict:
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    text = summary.get("summary")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("summary.summary must be a non-empty string")
    init_store(db_path)
    now = now_iso()
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT created_at FROM turn_summaries WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        created_at = row["created_at"] if row else now
        connection.execute(
            """
            INSERT INTO turn_summaries(
                turn_id, summary, keywords_json, facts_json, decisions_json,
                corrections_json, tool_refs_json, artifact_refs_json,
                source_message_ids_json, source_tool_step_ids_json,
                source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET
                summary = excluded.summary,
                keywords_json = excluded.keywords_json,
                facts_json = excluded.facts_json,
                decisions_json = excluded.decisions_json,
                corrections_json = excluded.corrections_json,
                tool_refs_json = excluded.tool_refs_json,
                artifact_refs_json = excluded.artifact_refs_json,
                source_message_ids_json = excluded.source_message_ids_json,
                source_tool_step_ids_json = excluded.source_tool_step_ids_json,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                turn_id,
                text.strip(),
                _json_dumps(summary.get("keywords") if isinstance(summary.get("keywords"), list) else []),
                _json_dumps(summary.get("facts") if isinstance(summary.get("facts"), list) else []),
                _json_dumps(summary.get("decisions") if isinstance(summary.get("decisions"), list) else []),
                _json_dumps(summary.get("corrections") if isinstance(summary.get("corrections"), list) else []),
                _json_dumps(summary.get("tool_refs") if isinstance(summary.get("tool_refs"), list) else []),
                _json_dumps(summary.get("artifact_refs") if isinstance(summary.get("artifact_refs"), list) else []),
                _json_dumps(summary.get("source_message_ids") if isinstance(summary.get("source_message_ids"), list) else []),
                _json_dumps(summary.get("source_tool_step_ids") if isinstance(summary.get("source_tool_step_ids"), list) else []),
                source,
                created_at,
                now,
            ),
        )
    return {"status": "success", "turn_id": turn_id, "updated_at": now}


def upsert_task_memory(
    db_path: str | Path,
    conversation_id: str,
    task: dict,
    *,
    task_id: str | None = None,
) -> dict:
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    status = task.get("status", "foreground")
    if status not in {"foreground", "paused", "completed", "abandoned"}:
        raise ValueError("task status must be foreground, paused, completed, or abandoned")
    title = task.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("task.title must be a non-empty string")
    init_store(db_path)
    now = now_iso()
    with _connect(db_path) as connection:
        if not task_id and status == "foreground":
            row = connection.execute(
                """
                SELECT id FROM task_memories
                WHERE conversation_id = ? AND status = 'foreground'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            task_id = row["id"] if row else None
        candidate_task_id = task.get("task_id")
        if task_id is None and isinstance(candidate_task_id, str) and candidate_task_id:
            task_id = candidate_task_id
        task_id = task_id or _new_id("task")
        row = connection.execute(
            "SELECT created_at FROM task_memories WHERE id = ? AND conversation_id = ?",
            (task_id, conversation_id),
        ).fetchone()
        created_at = row["created_at"] if row else now
        if status == "foreground":
            connection.execute(
                """
                UPDATE task_memories
                SET status = 'paused', updated_at = ?
                WHERE conversation_id = ? AND status = 'foreground' AND id <> ?
                """,
                (now, conversation_id, task_id),
            )
        connection.execute(
            """
            INSERT INTO task_memories(
                id, conversation_id, status, title, objective, phase,
                completed_items_json, pending_items_json, constraints_json,
                key_results_json, active_files_json, blocked_items_json,
                next_actions_json, source_turn_ids_json, confidence,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                title = excluded.title,
                objective = excluded.objective,
                phase = excluded.phase,
                completed_items_json = excluded.completed_items_json,
                pending_items_json = excluded.pending_items_json,
                constraints_json = excluded.constraints_json,
                key_results_json = excluded.key_results_json,
                active_files_json = excluded.active_files_json,
                blocked_items_json = excluded.blocked_items_json,
                next_actions_json = excluded.next_actions_json,
                source_turn_ids_json = excluded.source_turn_ids_json,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (
                task_id,
                conversation_id,
                status,
                title.strip(),
                task.get("objective") if isinstance(task.get("objective"), str) else None,
                task.get("phase") if isinstance(task.get("phase"), str) else None,
                _json_dumps(task.get("completed_items") if isinstance(task.get("completed_items"), list) else []),
                _json_dumps(task.get("pending_items") if isinstance(task.get("pending_items"), list) else []),
                _json_dumps(task.get("constraints") if isinstance(task.get("constraints"), list) else []),
                _json_dumps(task.get("key_results") if isinstance(task.get("key_results"), list) else []),
                _json_dumps(task.get("active_files") if isinstance(task.get("active_files"), list) else []),
                _json_dumps(task.get("blocked_items") if isinstance(task.get("blocked_items"), list) else []),
                _json_dumps(task.get("next_actions") if isinstance(task.get("next_actions"), list) else []),
                _json_dumps(task.get("source_turn_ids") if isinstance(task.get("source_turn_ids"), list) else []),
                _score(task.get("confidence"), 0.5),
                created_at,
                now,
            ),
        )
    return {"status": "success", "task_id": task_id, "conversation_id": conversation_id, "updated_at": now}


def record_memory_retrieval(
    db_path: str | Path,
    conversation_id: str,
    query_text: str,
    *,
    query_context: Any = None,
    candidate_blocks: Any = None,
    selected_turns: Any = None,
    loaded_message_ids: Any = None,
) -> dict:
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    init_store(db_path)
    now = now_iso()
    log_id = _new_id("memory_retrieval")
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_retrieval_log(
                id, conversation_id, query_text, query_context_json,
                candidate_blocks_json, selected_turns_json, loaded_message_ids_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                conversation_id,
                query_text.strip(),
                _json_dumps(query_context),
                _json_dumps(candidate_blocks),
                _json_dumps(selected_turns),
                _json_dumps(loaded_message_ids),
                now,
            ),
        )
    return {"status": "success", "retrieval_log_id": log_id, "created_at": now}


def list_memory_retrieval_logs(
    db_path: str | Path,
    conversation_id: str,
    *,
    limit: int = 20,
) -> list[dict]:
    init_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, query_text, query_context_json,
                   candidate_blocks_json, selected_turns_json,
                   loaded_message_ids_json, created_at
            FROM memory_retrieval_log
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, max(1, int(limit))),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["query_context"] = _json_loads(item.get("query_context_json")) or {}
        item["candidate_blocks"] = _json_loads(item.get("candidate_blocks_json")) or []
        item["selected_turns"] = _json_loads(item.get("selected_turns_json")) or []
        item["loaded_message_ids"] = _json_loads(item.get("loaded_message_ids_json")) or []
        result.append(item)
    return result


def count_memory_retrieval_logs(db_path: str | Path, conversation_id: str) -> int:
    init_store(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM memory_retrieval_log WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return int(row["total"]) if row is not None else 0


def get_memory_embedding(
    db_path: str | Path,
    item_type: str,
    item_id: str,
    provider: str,
    model: str,
) -> dict | None:
    if not all(isinstance(value, str) and value for value in (item_type, item_id, provider, model)):
        raise ValueError("embedding lookup keys must be non-empty strings")
    init_store(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT item_type, item_id, text_hash, provider, model, dimension,
                   vector_json, created_at, updated_at
            FROM memory_embeddings
            WHERE item_type = ? AND item_id = ? AND provider = ? AND model = ?
            """,
            (item_type, item_id, provider, model),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    vector = _json_loads(item.get("vector_json"))
    item["vector"] = vector if isinstance(vector, list) else []
    return item


def upsert_memory_embedding(
    db_path: str | Path,
    item_type: str,
    item_id: str,
    text_hash: str,
    provider: str,
    model: str,
    vector: list[float],
) -> dict:
    if not all(isinstance(value, str) and value for value in (item_type, item_id, text_hash, provider, model)):
        raise ValueError("embedding keys must be non-empty strings")
    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding vector must be a non-empty array")
    values = []
    for item in vector:
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vector must contain only numbers") from exc
    init_store(db_path)
    now = now_iso()
    with _connect(db_path) as connection:
        existing = connection.execute(
            """
            SELECT created_at FROM memory_embeddings
            WHERE item_type = ? AND item_id = ? AND provider = ? AND model = ?
            """,
            (item_type, item_id, provider, model),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        connection.execute(
            """
            INSERT INTO memory_embeddings(
                item_type, item_id, text_hash, provider, model, dimension,
                vector_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_type, item_id, provider, model) DO UPDATE SET
                text_hash = excluded.text_hash,
                dimension = excluded.dimension,
                vector_json = excluded.vector_json,
                updated_at = excluded.updated_at
            """,
            (
                item_type,
                item_id,
                text_hash,
                provider,
                model,
                len(values),
                _json_dumps(values),
                created_at,
                now,
            ),
        )
    return {
        "status": "success",
        "item_type": item_type,
        "item_id": item_id,
        "dimension": len(values),
        "updated_at": now,
    }


def upsert_memory_block(
    db_path: str | Path,
    conversation_id: str,
    block: dict,
    turn_ids: list[str],
    *,
    block_id: str | None = None,
) -> dict:
    if not isinstance(block, dict):
        raise ValueError("block must be an object")
    if not isinstance(turn_ids, list) or not all(isinstance(item, str) and item for item in turn_ids):
        raise ValueError("turn_ids must be a non-empty array of strings")
    if not turn_ids:
        raise ValueError("turn_ids must not be empty")
    title = block.get("title")
    summary = block.get("summary")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("block.title must be a non-empty string")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("block.summary must be a non-empty string")
    status = block.get("status", "active")
    if status not in {"active", "sealed"}:
        raise ValueError("block.status must be active or sealed")
    init_store(db_path)
    now = now_iso()
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, turn_index
            FROM conversation_turns
            WHERE conversation_id = ? AND id IN (%s)
            ORDER BY turn_index ASC
            """ % ",".join("?" for _ in turn_ids),
            [conversation_id, *turn_ids],
        ).fetchall()
        if len(rows) != len(set(turn_ids)):
            raise ValueError("all block turn_ids must exist in the same conversation")
        ordered_turn_ids = [row["id"] for row in rows]
        start_turn_index = int(rows[0]["turn_index"])
        end_turn_index = int(rows[-1]["turn_index"])
        if block_id is None:
            row = connection.execute(
                """
                SELECT id FROM memory_blocks
                WHERE conversation_id = ? AND start_turn_index = ? AND end_turn_index = ?
                """,
                (conversation_id, start_turn_index, end_turn_index),
            ).fetchone()
            block_id = row["id"] if row else _new_id("memory_block")
        existing = connection.execute(
            "SELECT created_at FROM memory_blocks WHERE id = ?",
            (block_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        connection.execute(
            """
            INSERT INTO memory_blocks(
                id, conversation_id, task_id, title, summary, status,
                start_turn_index, end_turn_index, keywords_json, source,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id, start_turn_index, end_turn_index) DO UPDATE SET
                task_id = excluded.task_id,
                title = excluded.title,
                summary = excluded.summary,
                status = excluded.status,
                keywords_json = excluded.keywords_json,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                block_id,
                conversation_id,
                block.get("task_id") if isinstance(block.get("task_id"), str) else None,
                title.strip(),
                summary.strip(),
                status,
                start_turn_index,
                end_turn_index,
                _json_dumps(block.get("keywords") if isinstance(block.get("keywords"), list) else []),
                block.get("source") if isinstance(block.get("source"), str) else "unspecified",
                created_at,
                now,
            ),
        )
        connection.execute("DELETE FROM memory_block_turns WHERE block_id = ?", (block_id,))
        for position, turn_id in enumerate(ordered_turn_ids, 1):
            connection.execute(
                """
                INSERT INTO memory_block_turns(block_id, turn_id, position, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (block_id, turn_id, position, now),
            )
    return {
        "status": "success",
        "block_id": block_id,
        "conversation_id": conversation_id,
        "turn_count": len(turn_ids),
        "updated_at": now,
    }


def list_conversation_turns(db_path: str | Path, conversation_id: str, limit: int | None = None) -> list[dict]:
    init_store(db_path)
    sql = """
        SELECT id, conversation_id, run_id, user_message_id, assistant_message_id,
               turn_index, status, started_at, completed_at, created_at, updated_at
        FROM conversation_turns
        WHERE conversation_id = ?
        ORDER BY turn_index ASC
    """
    params: list[Any] = [conversation_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def list_unblocked_conversation_turns(db_path: str | Path, conversation_id: str, limit: int | None = None) -> list[dict]:
    init_store(db_path)
    sql = """
        SELECT t.id, t.conversation_id, t.run_id, t.user_message_id, t.assistant_message_id,
               t.turn_index, t.status, t.started_at, t.completed_at, t.created_at, t.updated_at
        FROM conversation_turns AS t
        LEFT JOIN memory_block_turns AS bt ON bt.turn_id = t.id
        WHERE t.conversation_id = ? AND bt.turn_id IS NULL
        ORDER BY t.turn_index ASC
    """
    params: list[Any] = [conversation_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def list_task_memories(db_path: str | Path, conversation_id: str, status: str | None = None) -> list[dict]:
    init_store(db_path)
    sql = """
        SELECT id, conversation_id, status, title, objective, phase,
               completed_items_json, pending_items_json, constraints_json,
               key_results_json, active_files_json, blocked_items_json,
               next_actions_json, source_turn_ids_json, confidence,
               created_at, updated_at
        FROM task_memories
        WHERE conversation_id = ?
    """
    params: list[Any] = [conversation_id]
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY CASE status WHEN 'foreground' THEN 0 WHEN 'paused' THEN 1 ELSE 2 END, updated_at DESC"
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        for key in (
            "completed_items_json",
            "pending_items_json",
            "constraints_json",
            "key_results_json",
            "active_files_json",
            "blocked_items_json",
            "next_actions_json",
            "source_turn_ids_json",
        ):
            item[key[:-5]] = _json_loads(item.get(key)) or []
        result.append(item)
    return result


def list_memory_blocks(
    db_path: str | Path,
    conversation_id: str,
    *,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    init_store(db_path)
    sql = """
        SELECT id, conversation_id, task_id, title, summary, status,
               start_turn_index, end_turn_index, keywords_json, source,
               created_at, updated_at
        FROM memory_blocks
        WHERE conversation_id = ?
    """
    params: list[Any] = [conversation_id]
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY end_turn_index DESC, updated_at DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["keywords"] = _json_loads(item.get("keywords_json")) or []
        result.append(item)
    return result


def list_turn_summaries(
    db_path: str | Path,
    conversation_id: str,
    *,
    block_ids: list[str] | None = None,
    turn_ids: list[str] | None = None,
    exclude_turn_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    init_store(db_path)
    if block_ids is not None and not block_ids:
        return []
    if turn_ids is not None and not turn_ids:
        return []
    sql = """
        SELECT t.id AS turn_id, t.turn_index, t.run_id,
               t.user_message_id, t.assistant_message_id, t.status AS turn_status,
               s.summary, s.keywords_json, s.facts_json, s.decisions_json,
               s.corrections_json, s.tool_refs_json, s.artifact_refs_json,
               s.source_message_ids_json, s.source_tool_step_ids_json,
               s.source AS summary_source,
               tags.current_task_relevance, tags.long_term_value,
               tags.has_explicit_fact, tags.has_decision, tags.has_user_correction,
               tags.allow_compress, tags.allow_drop, tags.noise_score,
               tags.labels_json, tags.source AS tags_source,
               bt.block_id, bt.position AS block_position
        FROM conversation_turns AS t
        JOIN turn_summaries AS s ON s.turn_id = t.id
        LEFT JOIN turn_memory_tags AS tags ON tags.turn_id = t.id
        LEFT JOIN memory_block_turns AS bt ON bt.turn_id = t.id
        WHERE t.conversation_id = ?
    """
    params: list[Any] = [conversation_id]
    if block_ids is not None:
        sql += " AND bt.block_id IN (%s)" % ",".join("?" for _ in block_ids)
        params.extend(block_ids)
    if turn_ids is not None:
        sql += " AND t.id IN (%s)" % ",".join("?" for _ in turn_ids)
        params.extend(turn_ids)
    if exclude_turn_ids:
        sql += " AND t.id NOT IN (%s)" % ",".join("?" for _ in exclude_turn_ids)
        params.extend(exclude_turn_ids)
    sql += " ORDER BY t.turn_index DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        for key in (
            "keywords_json",
            "facts_json",
            "decisions_json",
            "corrections_json",
            "tool_refs_json",
            "artifact_refs_json",
            "source_message_ids_json",
            "source_tool_step_ids_json",
            "labels_json",
        ):
            item[key[:-5]] = _json_loads(item.get(key)) or []
        for key in (
            "has_explicit_fact",
            "has_decision",
            "has_user_correction",
            "allow_compress",
            "allow_drop",
        ):
            if item.get(key) is not None:
                item[key] = bool(item[key])
        result.append(item)
    return result


def list_messages_by_ids(db_path: str | Path, message_ids: list[str]) -> list[dict]:
    if not message_ids:
        return []
    init_store(db_path)
    placeholders = ",".join("?" for _ in message_ids)
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT id, conversation_id, role, content, message_order, run_id,
                   is_trivial, token_count, metadata_json, created_at
            FROM conversation_messages
            WHERE id IN ({placeholders})
            ORDER BY message_order ASC
            """,
            message_ids,
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = _json_loads(item.get("metadata_json"))
        result.append(item)
    return result


def list_tool_steps_by_ids(db_path: str | Path, tool_step_ids: list[str]) -> list[dict]:
    if not tool_step_ids:
        return []
    init_store(db_path)
    placeholders = ",".join("?" for _ in tool_step_ids)
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT id, conversation_id, assistant_message_id, run_id, step_index,
                   tool_call_id, tool_name, input_json, output_json, status,
                   error_json, latency_ms, created_at
            FROM tool_steps
            WHERE id IN ({placeholders})
            ORDER BY created_at ASC, step_index ASC
            """,
            tool_step_ids,
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["input"] = _json_loads(item.get("input_json"))
        item["output"] = _json_loads(item.get("output_json"))
        item["error"] = _json_loads(item.get("error_json"))
        result.append(item)
    return result


def list_conversations(db_path: str | Path, limit: int = 50) -> list[dict]:
    init_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, title, is_trivial, trivial_reason, summary, status,
                   created_at, updated_at, last_message_at
            FROM conversations
            ORDER BY COALESCE(last_message_at, updated_at, created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(db_path: str | Path, conversation_id: str) -> dict | None:
    init_store(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, title, is_trivial, trivial_reason, summary, status,
                   created_at, updated_at, last_message_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def delete_conversation(db_path: str | Path, conversation_id: str) -> dict:
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("conversation_id must be a non-empty string")
    init_store(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return {"status": "not_found", "conversation_id": conversation_id, "deleted": False}
        connection.execute(
            """
            DELETE FROM memory_embeddings
            WHERE item_type = 'turn'
              AND item_id IN (SELECT id FROM conversation_turns WHERE conversation_id = ?)
            """,
            (conversation_id,),
        )
        connection.execute(
            """
            DELETE FROM memory_embeddings
            WHERE item_type = 'block'
              AND item_id IN (SELECT id FROM memory_blocks WHERE conversation_id = ?)
            """,
            (conversation_id,),
        )
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return {"status": "success", "conversation_id": conversation_id, "deleted": True}


def list_messages(db_path: str | Path, conversation_id: str) -> list[dict]:
    init_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, role, content, message_order, run_id,
                   is_trivial, token_count, metadata_json, created_at
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY message_order ASC
            """,
            (conversation_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = _json_loads(item.get("metadata_json"))
        result.append(item)
    return result


def list_tool_steps(db_path: str | Path, assistant_message_id: str) -> list[dict]:
    init_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, assistant_message_id, run_id, step_index,
                   tool_call_id, tool_name, input_json, output_json, status,
                   error_json, latency_ms, created_at
            FROM tool_steps
            WHERE assistant_message_id = ?
            ORDER BY step_index ASC
            """,
            (assistant_message_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["input"] = _json_loads(item.get("input_json"))
        item["output"] = _json_loads(item.get("output_json"))
        item["error"] = _json_loads(item.get("error_json"))
        result.append(item)
    return result
