from __future__ import annotations

import sys
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Iterator

from fastapi import HTTPException

from backend.api_models import RunRequest, RunResponse, UploadedFileRef
from backend.conversation_utils import (
    assistant_metadata,
    extract_tool_steps,
    history_title,
    is_trivial_conversation,
    is_trivial_text,
    read_trace,
    read_trace_full,
    stream_event,
    write_json_file,
)
from backend.ids import now_stamp, safe_conversation_id
from backend.settings import (
    CODE_DIR,
    DEFAULT_SYSTEM_PROMPTS_PATH,
    MEMORY_CONFIG,
    MODEL_CONFIG,
    OUTPUT_ROOT,
    PROMPT_STORE_PATH,
    RUNTIME_BASE,
    SYSTEM_PROMPT_PATH,
    TOOLS_CONFIG,
)
from backend.uploads import save_run_uploads, uploaded_image_data_urls, user_input_with_uploaded_files
from common.prompt_store import get_conversation_prompt, update_conversation_prompt

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from b1_agent_runtime import resume_stream as resume_agent_runtime_stream  # noqa: E402
from b1_agent_runtime import run as run_agent_runtime  # noqa: E402
from b1_agent_runtime import run_stream as run_agent_runtime_stream  # noqa: E402
from b5_memory import (  # noqa: E402
    append_conversation_message,
    clear_message_tool_steps,
    init_conversation_db,
    list_conversation_history,
    list_conversation_messages,
    list_message_tool_steps,
    record_completed_turn_memory,
    record_conversation_tool_step,
    update_conversation_message,
    upsert_conversation_record,
)


_RUN_CANCEL_EVENTS: dict[str, Event] = {}
_RUN_CANCEL_LOCK = Lock()


def register_cancel_event(conversation_id: str) -> Event:
    cancel_event = Event()
    with _RUN_CANCEL_LOCK:
        _RUN_CANCEL_EVENTS[conversation_id] = cancel_event
    return cancel_event


def request_cancel(conversation_id: str) -> bool:
    with _RUN_CANCEL_LOCK:
        cancel_event = _RUN_CANCEL_EVENTS.get(conversation_id)
    if cancel_event is None:
        return False
    cancel_event.set()
    return True


def clear_cancel_event(conversation_id: str, cancel_event: Event) -> None:
    with _RUN_CANCEL_LOCK:
        if _RUN_CANCEL_EVENTS.get(conversation_id) is cancel_event:
            _RUN_CANCEL_EVENTS.pop(conversation_id, None)


def build_runtime_payload(
    request: RunRequest,
    conversation_id: str,
    user_input: str,
    history_messages: list[dict],
    input_images: list[str],
) -> dict:
    selected_memory_ids = request.selected_memory_ids
    use_global_memory = request.use_global_memory
    prompt_text = request.system_prompt.strip() if isinstance(request.system_prompt, str) and request.system_prompt.strip() else None
    if prompt_text is None:
        prompt_text = get_conversation_prompt(
            conversation_id,
            PROMPT_STORE_PATH,
            DEFAULT_SYSTEM_PROMPTS_PATH,
        )["content"]
    else:
        update_conversation_prompt(
            conversation_id,
            prompt_text,
            PROMPT_STORE_PATH,
            DEFAULT_SYSTEM_PROMPTS_PATH,
        )
    return {
        "conversation_id": conversation_id,
        "user_input": user_input,
        "history_messages": history_messages,
        "input_images": input_images,
        "system_prompt_path": SYSTEM_PROMPT_PATH,
        "system_prompt": prompt_text,
        "selected_memory_ids": selected_memory_ids,
        "use_global_memory": use_global_memory,
        "toolset": request.toolset,
        "save_memory": "none",
    }


def record_tool_steps(
    conversation_id: str,
    assistant_message_id: str,
    run_id: str,
    trace: dict,
) -> None:
    for index, step in enumerate(extract_tool_steps(trace), 1):
        record_conversation_tool_step(
            str(MEMORY_CONFIG),
            conversation_id,
            assistant_message_id,
            step["tool_name"],
            index,
            run_id=run_id,
            tool_call_id=step.get("tool_call_id"),
            input_data=step.get("input_data"),
            output_data=step.get("output_data"),
            status=step.get("status") or "unknown",
            error=step.get("error"),
            latency_ms=step.get("latency_ms"),
        )


def start_run_messages(
    conversation_id: str,
    run_id: str,
    raw_user_input: str,
    history: list[dict],
    uploaded_files: list[UploadedFileRef],
) -> tuple[str, str]:
    user_record = append_conversation_message(
        str(MEMORY_CONFIG),
        conversation_id,
        "user",
        raw_user_input,
        run_id=run_id,
        is_trivial=is_trivial_text(raw_user_input),
        metadata={
            "attachments": [file.model_dump() for file in uploaded_files],
        } if uploaded_files else None,
    )
    assistant_record = append_conversation_message(
        str(MEMORY_CONFIG),
        conversation_id,
        "assistant",
        "...",
        run_id=run_id,
        metadata={"ui_status": "pending", "agent_status": "running"},
    )
    title = history_title(history, raw_user_input)
    trivial = is_trivial_conversation(history, raw_user_input)
    upsert_conversation_record(
        str(MEMORY_CONFIG),
        conversation_id,
        title,
        is_trivial=trivial,
        trivial_reason="only trivial user messages" if trivial else None,
    )
    return user_record["message_id"], assistant_record["message_id"]


def finish_run_message(
    conversation_id: str,
    assistant_message_id: str,
    run_id: str,
    result: dict,
    trace: dict,
) -> None:
    metadata = assistant_metadata(result, trace)
    if result.get("status") == "cancelled":
        metadata["ui_status"] = "cancelled"
        metadata["cancelled"] = True
        metadata["resumable"] = True
    elif result.get("status") != "success":
        metadata["ui_status"] = "error"
    update_conversation_message(
        str(MEMORY_CONFIG),
        assistant_message_id,
        content=result["final_answer"] or "[empty response]",
        metadata=metadata,
    )
    record_tool_steps(conversation_id, assistant_message_id, run_id, trace)


def record_completed_turn(
    conversation_id: str,
    run_id: str,
    user_message_id: str,
    assistant_message_id: str,
    raw_user_input: str,
    result: dict,
    trace: dict,
    llm_mode: str | None,
    output_dir: Path,
) -> dict:
    try:
        return record_completed_turn_memory(
            str(MEMORY_CONFIG),
            conversation_id,
            run_id,
            user_message_id,
            assistant_message_id,
            raw_user_input,
            result.get("final_answer") or "",
            trace,
            str(MODEL_CONFIG) if result.get("status") == "success" else None,
            llm_mode,
            str(output_dir / "memory_reflection"),
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def write_turn_memory_result(trace_path: str, turn_memory: dict) -> None:
    trace = read_trace_full(trace_path)
    memory_save = trace.get("memory_save") if isinstance(trace.get("memory_save"), dict) else {}
    memory_save["turn_memory"] = turn_memory
    trace["memory_save"] = memory_save
    write_json_file(trace_path, trace)


def schedule_completed_turn_memory(
    conversation_id: str,
    run_id: str,
    user_message_id: str,
    assistant_message_id: str,
    raw_user_input: str,
    result: dict,
    trace: dict,
    llm_mode: str | None,
    output_dir: Path,
    trace_path: str,
) -> dict:
    scheduled = {
        "status": "scheduled",
        "mode": "background",
        "reason": "memory reflection and layered memory writes run after the user response",
    }

    def worker() -> None:
        try:
            turn_memory = record_completed_turn(
                conversation_id,
                run_id,
                user_message_id,
                assistant_message_id,
                raw_user_input,
                result,
                trace,
                llm_mode,
                output_dir,
            )
            write_turn_memory_result(trace_path, turn_memory)
        except Exception:
            return

    Thread(target=worker, name=f"memory-{run_id}", daemon=True).start()
    return scheduled


def mark_run_failed(assistant_message_id: str, error: Exception) -> None:
    update_conversation_message(
        str(MEMORY_CONFIG),
        assistant_message_id,
        content=f"请求失败：{type(error).__name__}: {error}",
        metadata={
            "ui_status": "error",
            "agent_status": "backend_error",
            "error": {"type": type(error).__name__, "message": str(error)},
        },
    )


def mark_run_cancelled(assistant_message_id: str, partial_answer: str = "") -> None:
    content = partial_answer.strip() or "已终止回答。"
    update_conversation_message(
        str(MEMORY_CONFIG),
        assistant_message_id,
        content=content,
        metadata={
            "ui_status": "cancelled",
            "agent_status": "cancelled",
            "cancelled": True,
            "resumable": True,
        },
    )


def call_agent(request: RunRequest) -> RunResponse:
    conversation_id = safe_conversation_id(request.conversation_id)
    raw_user_input = request.user_input.strip()
    init_conversation_db(str(MEMORY_CONFIG))
    history = list_conversation_messages(str(MEMORY_CONFIG), conversation_id)
    history_messages = list_conversation_history(str(MEMORY_CONFIG), conversation_id)
    title = history_title(history, raw_user_input)
    trivial = is_trivial_conversation(history, raw_user_input)
    upsert_conversation_record(
        str(MEMORY_CONFIG),
        conversation_id,
        title,
        is_trivial=trivial,
        trivial_reason="only trivial user messages" if trivial else None,
    )
    uploaded_refs = [
        *request.uploaded_files,
        *save_run_uploads(conversation_id, request.uploaded_file_payloads),
    ]
    agent_user_input = user_input_with_uploaded_files(raw_user_input, uploaded_refs)
    input_images = uploaded_image_data_urls(uploaded_refs)
    runtime_payload = build_runtime_payload(
        request, conversation_id, agent_user_input, history_messages, input_images
    )
    run_id = now_stamp()
    user_message_id, assistant_message_id = start_run_messages(
        conversation_id,
        run_id,
        raw_user_input,
        history,
        uploaded_refs,
    )
    output_dir = OUTPUT_ROOT / conversation_id / run_id
    try:
        result = run_agent_runtime(
            runtime_payload,
            str(TOOLS_CONFIG),
            str(MEMORY_CONFIG),
            str(MODEL_CONFIG),
            str(output_dir),
            request.llm_mode,
            RUNTIME_BASE,
        )
    except Exception as exc:
        mark_run_failed(assistant_message_id, exc)
        raise
    full_trace = read_trace_full(result["trace_path"])
    finish_run_message(
        conversation_id,
        assistant_message_id,
        run_id,
        result,
        full_trace,
    )
    full_trace["memory_save"] = {
        "requested": "database",
        "status": "success",
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "storage": "sqlite",
        "turn_memory": {
            "status": "scheduled",
            "mode": "background",
            "reason": "memory reflection and layered memory writes run after the user response",
        },
    }
    write_json_file(result["trace_path"], full_trace)
    schedule_completed_turn_memory(
        conversation_id,
        run_id,
        user_message_id,
        assistant_message_id,
        raw_user_input,
        result,
        full_trace,
        request.llm_mode,
        output_dir,
        result["trace_path"],
    )
    return RunResponse(
        conversation_id=result["conversation_id"],
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        status=result["status"],
        final_answer=result["final_answer"],
        elapsed_ms=result["elapsed_ms"],
        output_dir=str(output_dir),
        trace=read_trace(result["trace_path"]),
    )


def stream_agent(request: RunRequest) -> Iterator[str]:
    conversation_id = safe_conversation_id(request.conversation_id)
    raw_user_input = request.user_input.strip()
    init_conversation_db(str(MEMORY_CONFIG))
    history = list_conversation_messages(str(MEMORY_CONFIG), conversation_id)
    history_messages = list_conversation_history(str(MEMORY_CONFIG), conversation_id)
    title = history_title(history, raw_user_input)
    trivial = is_trivial_conversation(history, raw_user_input)
    upsert_conversation_record(
        str(MEMORY_CONFIG),
        conversation_id,
        title,
        is_trivial=trivial,
        trivial_reason="only trivial user messages" if trivial else None,
    )
    uploaded_refs = [
        *request.uploaded_files,
        *save_run_uploads(conversation_id, request.uploaded_file_payloads),
    ]
    agent_user_input = user_input_with_uploaded_files(raw_user_input, uploaded_refs)
    input_images = uploaded_image_data_urls(uploaded_refs)
    runtime_payload = build_runtime_payload(
        request, conversation_id, agent_user_input, history_messages, input_images
    )
    run_id = now_stamp()
    user_message_id, assistant_message_id = start_run_messages(
        conversation_id,
        run_id,
        raw_user_input,
        history,
        uploaded_refs,
    )
    output_dir = OUTPUT_ROOT / conversation_id / run_id
    cancel_event = register_cancel_event(conversation_id)
    streamed_answer = ""
    candidate_chunks: list[str] = []
    run_finished = False
    try:
        yield stream_event(
            {
                "type": "start",
                "conversation_id": conversation_id,
                "run_id": run_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
            }
        )
        for event in run_agent_runtime_stream(
            runtime_payload,
            str(TOOLS_CONFIG),
            str(MEMORY_CONFIG),
            str(MODEL_CONFIG),
            str(output_dir),
            request.llm_mode,
            RUNTIME_BASE,
            should_cancel=cancel_event.is_set,
        ):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "delta":
                delta = str(event.get("text", ""))
                if not delta:
                    continue
                candidate_chunks.append(delta)
                streamed_answer += delta
                yield stream_event(
                    {
                        "type": "delta",
                        "conversation_id": conversation_id,
                        "assistant_message_id": assistant_message_id,
                        "text": delta,
                    }
                )
            elif event_type == "state":
                candidate_chunks = []
                yield stream_event(
                    {
                        "type": "state",
                        "conversation_id": conversation_id,
                        "assistant_message_id": assistant_message_id,
                        "state": event.get("state"),
                        "action": event.get("action"),
                        "reason": event.get("reason"),
                        "agent_step": event.get("agent_step"),
                        "llm_call_index": event.get("llm_call_index"),
                        "tool_round_index": event.get("tool_round_index"),
                        "detail": event.get("detail"),
                    }
                )
            elif event_type == "tool_start":
                streamed_answer = ""
                candidate_chunks = []
                update_conversation_message(
                    str(MEMORY_CONFIG),
                    assistant_message_id,
                    content="...",
                    metadata={"ui_status": "pending", "agent_status": "running_tool"},
                )
                yield stream_event(
                    {
                        "type": "tool_start",
                        "conversation_id": conversation_id,
                        "assistant_message_id": assistant_message_id,
                        "tool_calls": event.get("tool_calls", []),
                        "assistant_content": event.get("assistant_content", ""),
                        "agent_step": event.get("agent_step"),
                    }
                )
            elif event_type == "tool_done":
                yield stream_event(
                    {
                        "type": "tool_done",
                        "conversation_id": conversation_id,
                        "assistant_message_id": assistant_message_id,
                        "tool_messages": event.get("tool_messages", []),
                    }
                )
            elif event_type == "done":
                result = event.get("result")
                if not isinstance(result, dict):
                    raise ValueError("stream runtime finished without a result object")
                full_trace = read_trace_full(result["trace_path"])
                finish_run_message(
                    conversation_id,
                    assistant_message_id,
                    run_id,
                    result,
                    full_trace,
                )
                if result.get("status") == "cancelled":
                    full_trace["memory_save"] = {
                        "requested": "database",
                        "status": "skipped",
                        "reason": "cancelled",
                        "conversation_id": conversation_id,
                        "user_message_id": user_message_id,
                        "assistant_message_id": assistant_message_id,
                        "storage": "sqlite",
                        "turn_memory": {"status": "skipped", "reason": "cancelled"},
                    }
                    write_json_file(result["trace_path"], full_trace)
                else:
                    full_trace["memory_save"] = {
                        "requested": "database",
                        "status": "success",
                        "conversation_id": conversation_id,
                        "user_message_id": user_message_id,
                        "assistant_message_id": assistant_message_id,
                        "storage": "sqlite",
                        "turn_memory": {
                            "status": "scheduled",
                            "mode": "background",
                            "reason": "memory reflection and layered memory writes run after the user response",
                        },
                    }
                    write_json_file(result["trace_path"], full_trace)
                    schedule_completed_turn_memory(
                        conversation_id,
                        run_id,
                        user_message_id,
                        assistant_message_id,
                        raw_user_input,
                        result,
                        full_trace,
                        request.llm_mode,
                        output_dir,
                        result["trace_path"],
                    )
                tool_steps = list_message_tool_steps(str(MEMORY_CONFIG), assistant_message_id)
                run_finished = True
                yield stream_event(
                    {
                        "type": "done",
                        "conversation_id": result["conversation_id"],
                        "user_message_id": user_message_id,
                        "assistant_message_id": assistant_message_id,
                        "status": result["status"],
                        "final_answer": result["final_answer"],
                        "elapsed_ms": result["elapsed_ms"],
                        "output_dir": str(output_dir),
                        "trace": read_trace(result["trace_path"]),
                        "tool_steps": tool_steps,
                    }
                )
                return
    except GeneratorExit:
        if not run_finished:
            mark_run_cancelled(assistant_message_id, streamed_answer or "".join(candidate_chunks))
        raise
    except Exception as exc:
        if cancel_event.is_set():
            cancelled_answer = streamed_answer or "".join(candidate_chunks)
            mark_run_cancelled(assistant_message_id, cancelled_answer)
            yield stream_event(
                {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                    "status": "cancelled",
                    "final_answer": cancelled_answer.strip() or "已终止回答。",
                    "elapsed_ms": None,
                    "output_dir": str(output_dir),
                    "trace": {
                        "final_state": "failed",
                        "finish_reason": "user cancelled",
                        "memory_save": {"status": "skipped", "reason": "cancelled"},
                    },
                    "tool_steps": list_message_tool_steps(str(MEMORY_CONFIG), assistant_message_id),
                }
            )
            return
        mark_run_failed(assistant_message_id, exc)
        yield stream_event(
            {
                "type": "error",
                "conversation_id": conversation_id,
                "assistant_message_id": assistant_message_id,
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        clear_cancel_event(conversation_id, cancel_event)


def safe_stream_agent(request: RunRequest) -> Iterator[str]:
    try:
        yield from stream_agent(request)
    except Exception as exc:
        yield stream_event(
            {
                "type": "error",
                "conversation_id": request.conversation_id,
                "message": f"{type(exc).__name__}: {exc}",
            }
        )


def resume_message_context(conversation_id: str, assistant_message_id: str) -> dict:
    messages = list_conversation_messages(str(MEMORY_CONFIG), conversation_id)
    target = None
    previous_user = None
    for message in messages:
        if message.get("id") == assistant_message_id:
            target = message
            break
        if message.get("role") == "user":
            previous_user = message
    if target is None or target.get("role") != "assistant":
        raise HTTPException(status_code=404, detail="assistant message not found")
    if previous_user is None:
        raise HTTPException(status_code=400, detail="cannot resume without a previous user message")
    return {
        "run_id": target.get("run_id") or now_stamp(),
        "user_message_id": previous_user["id"],
        "raw_user_input": previous_user["content"],
    }


def stream_resume_agent(conversation_id: str, assistant_message_id: str) -> Iterator[str]:
    safe_conversation = safe_conversation_id(conversation_id)
    init_conversation_db(str(MEMORY_CONFIG))
    context = resume_message_context(safe_conversation, assistant_message_id)
    run_id = context["run_id"]
    user_message_id = context["user_message_id"]
    raw_user_input = context["raw_user_input"]
    update_conversation_message(
        str(MEMORY_CONFIG),
        assistant_message_id,
        content="...",
        metadata={"ui_status": "pending", "agent_status": "resuming", "resumable": True},
    )
    clear_message_tool_steps(str(MEMORY_CONFIG), assistant_message_id)
    cancel_event = register_cancel_event(safe_conversation)
    streamed_answer = ""
    candidate_chunks: list[str] = []
    run_finished = False
    output_dir = OUTPUT_ROOT / safe_conversation / run_id
    try:
        yield stream_event(
            {
                "type": "start",
                "conversation_id": safe_conversation,
                "run_id": run_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "resumed": True,
            }
        )
        for event in resume_agent_runtime_stream(safe_conversation, should_cancel=cancel_event.is_set):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "delta":
                delta = str(event.get("text", ""))
                if not delta:
                    continue
                candidate_chunks.append(delta)
                streamed_answer += delta
                yield stream_event(
                    {
                        "type": "delta",
                        "conversation_id": safe_conversation,
                        "assistant_message_id": assistant_message_id,
                        "text": delta,
                    }
                )
            elif event_type == "state":
                candidate_chunks = []
                yield stream_event(
                    {
                        "type": "state",
                        "conversation_id": safe_conversation,
                        "assistant_message_id": assistant_message_id,
                        "state": event.get("state"),
                        "action": event.get("action"),
                        "reason": event.get("reason"),
                        "agent_step": event.get("agent_step"),
                        "llm_call_index": event.get("llm_call_index"),
                        "tool_round_index": event.get("tool_round_index"),
                        "detail": event.get("detail"),
                    }
                )
            elif event_type == "tool_start":
                streamed_answer = ""
                candidate_chunks = []
                update_conversation_message(
                    str(MEMORY_CONFIG),
                    assistant_message_id,
                    content="...",
                    metadata={"ui_status": "pending", "agent_status": "running_tool", "resumable": True},
                )
                yield stream_event(
                    {
                        "type": "tool_start",
                        "conversation_id": safe_conversation,
                        "assistant_message_id": assistant_message_id,
                        "tool_calls": event.get("tool_calls", []),
                        "assistant_content": event.get("assistant_content", ""),
                        "agent_step": event.get("agent_step"),
                    }
                )
            elif event_type == "tool_done":
                yield stream_event(
                    {
                        "type": "tool_done",
                        "conversation_id": safe_conversation,
                        "assistant_message_id": assistant_message_id,
                        "tool_messages": event.get("tool_messages", []),
                    }
                )
            elif event_type == "done":
                result = event.get("result")
                if not isinstance(result, dict):
                    raise ValueError("resume runtime finished without a result object")
                full_trace = read_trace_full(result["trace_path"])
                clear_message_tool_steps(str(MEMORY_CONFIG), assistant_message_id)
                finish_run_message(
                    safe_conversation,
                    assistant_message_id,
                    run_id,
                    result,
                    full_trace,
                )
                if result.get("status") == "cancelled":
                    full_trace["memory_save"] = {
                        "requested": "database",
                        "status": "skipped",
                        "reason": "cancelled",
                        "conversation_id": safe_conversation,
                        "user_message_id": user_message_id,
                        "assistant_message_id": assistant_message_id,
                        "storage": "sqlite",
                        "turn_memory": {"status": "skipped", "reason": "cancelled"},
                    }
                    write_json_file(result["trace_path"], full_trace)
                else:
                    full_trace["memory_save"] = {
                        "requested": "database",
                        "status": "success",
                        "conversation_id": safe_conversation,
                        "user_message_id": user_message_id,
                        "assistant_message_id": assistant_message_id,
                        "storage": "sqlite",
                        "turn_memory": {
                            "status": "scheduled",
                            "mode": "background",
                            "reason": "memory reflection and layered memory writes run after the user response",
                        },
                    }
                    write_json_file(result["trace_path"], full_trace)
                    schedule_completed_turn_memory(
                        safe_conversation,
                        run_id,
                        user_message_id,
                        assistant_message_id,
                        raw_user_input,
                        result,
                        full_trace,
                        None,
                        output_dir,
                        result["trace_path"],
                    )
                tool_steps = list_message_tool_steps(str(MEMORY_CONFIG), assistant_message_id)
                run_finished = True
                yield stream_event(
                    {
                        "type": "done",
                        "conversation_id": result["conversation_id"],
                        "user_message_id": user_message_id,
                        "assistant_message_id": assistant_message_id,
                        "status": result["status"],
                        "final_answer": result["final_answer"],
                        "elapsed_ms": result["elapsed_ms"],
                        "output_dir": str(output_dir),
                        "trace": read_trace(result["trace_path"]),
                        "tool_steps": tool_steps,
                    }
                )
                return
    except GeneratorExit:
        if not run_finished:
            mark_run_cancelled(assistant_message_id, streamed_answer or "".join(candidate_chunks))
        raise
    except Exception as exc:
        if cancel_event.is_set():
            cancelled_answer = streamed_answer or "".join(candidate_chunks)
            mark_run_cancelled(assistant_message_id, cancelled_answer)
            yield stream_event(
                {
                    "type": "done",
                    "conversation_id": safe_conversation,
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                    "status": "cancelled",
                    "final_answer": cancelled_answer.strip() or "已终止回答。",
                    "elapsed_ms": None,
                    "output_dir": str(output_dir),
                    "trace": {
                        "final_state": "failed",
                        "finish_reason": "user cancelled",
                        "memory_save": {"status": "skipped", "reason": "cancelled"},
                    },
                    "tool_steps": list_message_tool_steps(str(MEMORY_CONFIG), assistant_message_id),
                }
            )
            return
        mark_run_failed(assistant_message_id, exc)
        yield stream_event(
            {
                "type": "error",
                "conversation_id": safe_conversation,
                "assistant_message_id": assistant_message_id,
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        clear_cancel_event(safe_conversation, cancel_event)


def safe_stream_resume_agent(conversation_id: str, assistant_message_id: str) -> Iterator[str]:
    try:
        yield from stream_resume_agent(conversation_id, assistant_message_id)
    except Exception as exc:
        yield stream_event(
            {
                "type": "error",
                "conversation_id": conversation_id,
                "assistant_message_id": assistant_message_id,
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
