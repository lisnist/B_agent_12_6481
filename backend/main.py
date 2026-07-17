from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.api_models import (
    B2SkillRunRequest,
    B3ToolCallsPreviewRequest,
    B4ProtocolTestRequest,
    B5RecallPreviewRequest,
    ConversationDetail,
    ConversationMessage,
    ConversationPromptResponse,
    ConversationSummary,
    DeleteConversationResponse,
    RunRequest,
    RunResponse,
    UpdateConversationPromptRequest,
    UploadRequest,
    UploadResponse,
)
from backend.artifacts import attach_artifact_download_urls, generated_artifact_target
from backend.b4_demo_service import (
    get_b4_call_detail,
    list_b4_calls,
    protocol_test_cases,
    run_b4_protocol_tests,
)
from backend.conversation_utils import (
    message_attachments,
    message_resumable,
    message_ui_status,
)
from backend.ids import now_stamp, safe_conversation_id
from backend.run_service import call_agent, request_cancel, safe_stream_agent, safe_stream_resume_agent
from backend.settings import (
    CODE_DIR,
    DEFAULT_SYSTEM_PROMPTS_PATH,
    HOST,
    MEMORY_CONFIG,
    MODEL_CONFIG,
    OUTPUT_ROOT,
    PORT,
    PROMPT_STORE_PATH,
    TOOLS_CONFIG,
    UPLOAD_ROOT,
)
from backend.tool_demo_service import b2_skill_summary, b3_tool_calls_from_request, parse_b3_tool_message
from backend.uploads import delete_child_directory, save_uploaded_files

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from b1_agent_runtime_parts.b1_checkpoint import checkpoint_metadata, load_checkpoint  # noqa: E402
from b2_run_skill import run_skill as run_b2_skill  # noqa: E402
from b3_tool_layer import execute_tool_calls as execute_b3_tool_calls  # noqa: E402
from b3_tool_layer import get_tools_schema as get_b3_tools_schema  # noqa: E402
from common.io_utils import append_jsonl, write_json  # noqa: E402
from common.logging_utils import now_iso  # noqa: E402
from common.prompt_store import (  # noqa: E402
    default_system_prompt,
    delete_conversation_prompt,
    get_conversation_prompt,
    update_conversation_prompt,
)
from common.tool_config import get_tool_definition, load_tools_config, resolve_toolset  # noqa: E402
from b5_memory import (  # noqa: E402
    delete_conversation_record,
    get_conversation_memory_snapshot,
    init_conversation_db,
    list_conversation_history,
    list_conversation_messages,
    list_conversation_records,
    list_message_tool_steps,
    prepare_workspace_memory_context,
)


app = FastAPI(title="Agent Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "agent_runtime": "b1",
        "model_config": str(MODEL_CONFIG),
        "features": {
            "upload_in_run": True,
            "current_time_tool": True,
        },
    }


@app.post("/api/uploads", response_model=UploadResponse)
async def upload_files(request: UploadRequest) -> UploadResponse:
    normalized = request.model_copy(update={"conversation_id": safe_conversation_id(request.conversation_id)})
    return UploadResponse(files=await asyncio.to_thread(save_uploaded_files, normalized))


@app.get("/api/conversations", response_model=list[ConversationSummary])
def get_conversations(limit: int = 50) -> list[ConversationSummary]:
    init_conversation_db(str(MEMORY_CONFIG))
    records = list_conversation_records(str(MEMORY_CONFIG), max(1, min(limit, 200)))
    return [
        ConversationSummary(
            id=record["id"],
            title=record["title"],
            is_trivial=bool(record["is_trivial"]),
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            last_message_at=record.get("last_message_at"),
        )
        for record in records
    ]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    conversation_id = safe_conversation_id(conversation_id)
    init_conversation_db(str(MEMORY_CONFIG))
    messages = list_conversation_messages(str(MEMORY_CONFIG), conversation_id)
    visible = [
        ConversationMessage(
            id=message["id"],
            role=message["role"],
            content=message["content"],
            message_order=message["message_order"],
            created_at=message["created_at"],
            status=message_ui_status(message),
            resumable=message_resumable(message),
            tool_steps=list_message_tool_steps(str(MEMORY_CONFIG), message["id"]) if message["role"] == "assistant" else [],
            attachments=message_attachments(message) if message["role"] == "user" else [],
        )
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]
    return ConversationDetail(conversation_id=conversation_id, messages=visible)


@app.delete("/api/conversations/{conversation_id}", response_model=DeleteConversationResponse)
def delete_conversation(conversation_id: str) -> DeleteConversationResponse:
    conversation_id = safe_conversation_id(conversation_id)
    init_conversation_db(str(MEMORY_CONFIG))
    record = delete_conversation_record(str(MEMORY_CONFIG), conversation_id)
    if not record.get("deleted"):
        raise HTTPException(status_code=404, detail="conversation not found")
    upload_dir_deleted = delete_child_directory(UPLOAD_ROOT, conversation_id)
    output_dir_deleted = delete_child_directory(OUTPUT_ROOT, conversation_id)
    delete_conversation_prompt(conversation_id, PROMPT_STORE_PATH)
    return DeleteConversationResponse(
        conversation_id=conversation_id,
        deleted=bool(record.get("deleted")),
        upload_dir_deleted=upload_dir_deleted,
        output_dir_deleted=output_dir_deleted,
    )


@app.get("/api/conversations/{conversation_id}/prompt", response_model=ConversationPromptResponse)
def get_conversation_system_prompt(conversation_id: str) -> ConversationPromptResponse:
    prompt = get_conversation_prompt(
        safe_conversation_id(conversation_id),
        PROMPT_STORE_PATH,
        DEFAULT_SYSTEM_PROMPTS_PATH,
    )
    return ConversationPromptResponse(**prompt)


@app.get("/api/prompts/default", response_model=ConversationPromptResponse)
def get_default_system_prompt() -> ConversationPromptResponse:
    default_content = default_system_prompt(DEFAULT_SYSTEM_PROMPTS_PATH)
    return ConversationPromptResponse(
        conversation_id="__default__",
        prompt_id="default_local_tool_agent",
        content=default_content,
        default_content=default_content,
        locked_default=True,
    )


@app.put("/api/conversations/{conversation_id}/prompt", response_model=ConversationPromptResponse)
def update_conversation_system_prompt(
    conversation_id: str,
    request: UpdateConversationPromptRequest,
) -> ConversationPromptResponse:
    try:
        prompt = update_conversation_prompt(
            safe_conversation_id(conversation_id),
            request.content,
            PROMPT_STORE_PATH,
            DEFAULT_SYSTEM_PROMPTS_PATH,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConversationPromptResponse(**prompt)


@app.get("/api/b1/conversations/{conversation_id}/workspace")
def get_b1_workspace_checkpoint(conversation_id: str) -> dict:
    safe_conversation = safe_conversation_id(conversation_id)
    metadata = checkpoint_metadata(safe_conversation)
    if not metadata["exists"]:
        return {
            "status": "missing",
            "module": "B1",
            "conversation_id": safe_conversation,
            "checkpoint": metadata,
            "workspace": None,
        }
    try:
        checkpoint = load_checkpoint(safe_conversation)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    runtime = checkpoint.get("runtime") if isinstance(checkpoint.get("runtime"), dict) else {}
    workspace = checkpoint.get("workspace") if isinstance(checkpoint.get("workspace"), dict) else None
    tools_schema = checkpoint.get("tools_schema")
    selected_memory = checkpoint.get("selected_memory")
    return {
        "status": "success",
        "module": "B1",
        "conversation_id": safe_conversation,
        "checkpoint": {
            **metadata,
            "schema_version": checkpoint.get("schema_version"),
            "status": checkpoint.get("status"),
            "stage": checkpoint.get("stage"),
            "mode": checkpoint.get("mode"),
            "execution_mode": checkpoint.get("execution_mode"),
            "output_dir": checkpoint.get("output_dir"),
        },
        "runtime": runtime,
        "selected_memory": selected_memory if isinstance(selected_memory, dict) else {},
        "tools_schema_count": len(tools_schema) if isinstance(tools_schema, list) else 0,
        "workspace": workspace,
    }


@app.get("/api/b2/skills")
def get_b2_skills(toolset: str | None = None) -> dict:
    try:
        _, config = load_tools_config(str(TOOLS_CONFIG))
        selected, enabled_tools = resolve_toolset(config, toolset)
        enabled = set(enabled_tools)
        tools = [
            b2_skill_summary(name, get_tool_definition(config, name), name in enabled)
            for name in enabled_tools
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    settings = config.get("settings") if isinstance(config.get("settings"), dict) else {}
    workspace_roots = settings.get("workspace_roots") if isinstance(settings.get("workspace_roots"), dict) else {}
    return {
        "status": "success",
        "module": "B2",
        "toolset": selected,
        "tool_count": len(tools),
        "tools": tools,
        "toolsets": config.get("toolsets", {}),
        "settings": {
            "data_root": settings.get("data_root"),
            "default_workspace_root": settings.get("default_workspace_root"),
            "workspace_roots": sorted(workspace_roots.keys()),
        },
    }


@app.post("/api/b2/skills/run")
def run_b2_skill_preview(request: B2SkillRunRequest) -> dict:
    skill_name = request.skill_name.strip()
    if not skill_name:
        raise HTTPException(status_code=400, detail="skill_name is required")
    try:
        _, config = load_tools_config(str(TOOLS_CONFIG))
        selected, enabled_tools = resolve_toolset(config, request.toolset)
        if skill_name not in enabled_tools:
            raise ValueError(f"skill is not available in {selected}: {skill_name}")
        get_tool_definition(config, skill_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = now_stamp()
    output_dir = OUTPUT_ROOT / "b2_demo" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_b2_skill(skill_name, request.input, None, str(output_dir), str(TOOLS_CONFIG))
    attach_artifact_download_urls(result, output_dir)
    write_json(result, output_dir / "b2_skill_result.json")
    append_jsonl(
        {
            "timestamp": now_iso(),
            "module": "B2",
            "toolset": selected,
            "skill_name": skill_name,
            "status": result.get("status"),
            "latency_ms": result.get("latency_ms"),
            "result_path": str(output_dir / "b2_skill_result.json"),
        },
        output_dir / "b2_skill_run_log.jsonl",
    )
    return {
        "status": "success",
        "module": "B2",
        "toolset": selected,
        "skill_name": skill_name,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "result": result,
    }


@app.get("/api/b3/tools-schema")
def get_b3_schema(toolset: str | None = None) -> dict:
    try:
        _, config = load_tools_config(str(TOOLS_CONFIG))
        selected, enabled_tools = resolve_toolset(config, toolset)
        schema = get_b3_tools_schema(str(TOOLS_CONFIG), selected, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "success",
        "module": "B3",
        "toolset": selected,
        "tool_count": len(schema),
        "tools": enabled_tools,
        "tools_schema": schema,
        "toolsets": config.get("toolsets", {}),
    }


@app.post("/api/b3/tool-calls/preview")
def run_b3_tool_calls_preview(request: B3ToolCallsPreviewRequest) -> dict:
    tool_calls = b3_tool_calls_from_request(request)
    if not tool_calls:
        raise HTTPException(status_code=400, detail="tool_calls is required")
    try:
        _, config = load_tools_config(str(TOOLS_CONFIG))
        selected, enabled_tools = resolve_toolset(config, request.toolset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = now_stamp()
    output_dir = OUTPUT_ROOT / "b3_demo" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        schema = get_b3_tools_schema(str(TOOLS_CONFIG), selected, str(output_dir))
        tool_messages = execute_b3_tool_calls(tool_calls, str(TOOLS_CONFIG), selected, str(output_dir))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    results = []
    success_count = 0
    error_count = 0
    for index, message in enumerate(tool_messages):
        parsed = parse_b3_tool_message(message)
        status = str(message.get("status") or (parsed or {}).get("status") or "unknown")
        if status == "success":
            success_count += 1
        elif status == "error":
            error_count += 1
        results.append(
            {
                "index": index,
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name"),
                "status": status,
                "skill_result": parsed,
            }
        )

    return {
        "status": "success",
        "module": "B3",
        "toolset": selected,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "tool_count": len(enabled_tools),
        "tools": enabled_tools,
        "tools_schema": schema,
        "tool_calls": tool_calls,
        "tool_messages": tool_messages,
        "results": results,
        "summary": {
            "tool_call_count": len(tool_calls),
            "tool_message_count": len(tool_messages),
            "success_count": success_count,
            "error_count": error_count,
            "schema_count": len(schema),
            "artifacts": [
                artifact
                for result in results
                for artifact in (
                    (result.get("skill_result") or {}).get("artifacts", [])
                    if isinstance(result.get("skill_result"), dict)
                    and isinstance((result.get("skill_result") or {}).get("artifacts"), list)
                    else []
                )
                if isinstance(artifact, dict)
            ],
        },
    }


@app.get("/api/b4/calls")
def get_b4_calls(conversation_id: str | None = None, limit: int = 60) -> dict:
    try:
        return list_b4_calls(conversation_id, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/b4/calls/detail")
def get_b4_call(call_id: str) -> dict:
    try:
        return get_b4_call_detail(call_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/b4/protocol-tests")
def get_b4_protocol_tests() -> dict:
    return protocol_test_cases()


@app.post("/api/b4/protocol-tests/run")
def run_b4_protocol_test(request: B4ProtocolTestRequest) -> dict:
    try:
        return run_b4_protocol_tests(request.case_id.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/b5/conversations/{conversation_id}/memory")
def get_b5_conversation_memory(conversation_id: str) -> dict:
    conversation_id = safe_conversation_id(conversation_id)
    init_conversation_db(str(MEMORY_CONFIG))
    snapshot = get_conversation_memory_snapshot(str(MEMORY_CONFIG), conversation_id)
    if snapshot.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="conversation not found")
    return snapshot


@app.post("/api/b5/conversations/{conversation_id}/recall-preview")
def run_b5_recall_preview(conversation_id: str, request: B5RecallPreviewRequest) -> dict:
    conversation_id = safe_conversation_id(conversation_id)
    current_user_input = request.current_user_input.strip()
    if not current_user_input:
        raise HTTPException(status_code=400, detail="current_user_input is required")
    init_conversation_db(str(MEMORY_CONFIG))
    snapshot = get_conversation_memory_snapshot(str(MEMORY_CONFIG), conversation_id)
    if snapshot.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="conversation not found")
    try:
        history = list_conversation_history(str(MEMORY_CONFIG), conversation_id)
        result = prepare_workspace_memory_context(
            str(MEMORY_CONFIG),
            conversation_id,
            current_user_input,
            history,
            None,
            None,
            str(MODEL_CONFIG),
            None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    layered = result.get("layered_memory_context") if isinstance(result.get("layered_memory_context"), dict) else {}
    response = dict(result)
    response["current_user_input"] = current_user_input
    response["memory_messages"] = layered.get("memory_messages", [])
    response["recalled_blocks"] = layered.get("recalled_blocks", [])
    response["recalled_turns"] = layered.get("recalled_turns", [])
    response["source_messages"] = layered.get("source_messages", [])
    response["source_tool_steps"] = layered.get("source_tool_steps", [])
    response["vector_retrieval"] = layered.get("vector_retrieval")
    response["llm_rerank"] = layered.get("llm_rerank")
    response["retrieval_log"] = layered.get("retrieval_log")
    return response


@app.get("/api/messages/{message_id}/tool-steps")
def get_message_tool_steps(message_id: str) -> dict:
    init_conversation_db(str(MEMORY_CONFIG))
    return {"message_id": message_id, "tool_steps": list_message_tool_steps(str(MEMORY_CONFIG), message_id)}


@app.get("/api/artifacts/{conversation_id}/{run_id}/{relative_path:path}")
def download_generated_artifact(conversation_id: str, run_id: str, relative_path: str) -> FileResponse:
    target = generated_artifact_target(conversation_id, run_id, relative_path)
    return FileResponse(target, filename=target.name)


@app.post("/api/conversations/{conversation_id}/cancel")
def cancel_conversation_run(conversation_id: str) -> dict:
    safe_conversation = safe_conversation_id(conversation_id)
    return {
        "conversation_id": safe_conversation,
        "cancel_requested": request_cancel(safe_conversation),
    }


@app.post("/api/conversations/{conversation_id}/messages/{assistant_message_id}/resume")
def resume_conversation_run(conversation_id: str, assistant_message_id: str) -> StreamingResponse:
    return StreamingResponse(
        safe_stream_resume_agent(conversation_id, assistant_message_id),
        media_type="application/x-ndjson; charset=utf-8",
    )


@app.post("/api/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")
    try:
        return await asyncio.to_thread(call_agent, request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/run/stream")
async def run_agent_stream(request: RunRequest) -> StreamingResponse:
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")
    return StreamingResponse(
        safe_stream_agent(request),
        media_type="application/x-ndjson; charset=utf-8",
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
