from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

from common.io_utils import append_jsonl, write_json, write_text
from common.logging_utils import now_iso
from common.schemas import make_ai_message

from .b1_checkpoint import checkpoint_metadata, load_checkpoint, save_checkpoint
from .b1_llm_bridge import generate_ai_message, generate_json_object, stream_ai_message
from .b1_prompting import (
    _workspace_answer_messages,
    _workspace_observation_messages,
    _workspace_planning_messages,
    _workspace_stage_failure_answer_messages,
    _workspace_tool_messages,
)
from .b1_workspace import (
    _agent_step_from_observation,
    _agent_step_from_plan,
    _as_string_list,
    _merge_unique,
    _record_no_tool_action,
    _record_stage,
    _required_outputs_pending,
    _workspace_from_runtime,
)


def _write_runtime_outputs(
    runtime: dict,
    execution_mode: str,
    mode: str,
    output_dir: Path,
    started: float,
    selected_memory: dict,
    messages: list[dict],
    all_tool_messages: list[dict],
    final_answer: str,
    status: str,
    tool_rounds: int,
    llm_calls: int,
    turns: list[dict],
    final_control: dict | None,
    warnings: list[str],
    terminal_error: dict | None,
    memory_file: Path | None,
    workspace: dict | None = None,
    streaming: bool = False,
) -> dict:
    write_json(messages, output_dir / "messages.json")
    if execution_mode == "integrated":
        write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_text(final_answer.strip() + "\n", output_dir / "final_answer.md")
    memory_save = {"requested": runtime["save_memory"], "status": "not_requested"}
    if status != "success" and runtime["save_memory"] != "none":
        memory_save = {"requested": runtime["save_memory"], "status": "skipped", "reason": status}
    trace = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": status,
        "toolset": runtime["toolset"],
        "max_turns": runtime["max_turns"],
        "tool_rounds_used": tool_rounds,
        "llm_call_count": llm_calls,
        "final_state": final_control["state"] if final_control else "failed",
        "finish_reason": final_control["reason"] if final_control else "",
        "turns": turns,
        "final_answer_path": "final_answer.md",
        "memory_save": memory_save,
        "warnings": warnings,
        "error": terminal_error,
        "checkpoint": checkpoint_metadata(runtime["conversation_id"]),
    }
    if workspace is not None:
        trace["workspace"] = workspace
    write_json(trace, output_dir / "trace.json")

    saved_memory = None
    if execution_mode == "integrated" and runtime["save_memory"] != "none" and trace["status"] == "success":
        try:
            from b5_memory import save_memory

            saved_memory = save_memory(
                str(memory_file),
                runtime["conversation_id"],
                runtime["save_memory"],
                str(output_dir / "messages.json"),
                str(output_dir / "trace.json"),
                str(output_dir / "final_answer.md"),
                str(output_dir),
            )
            trace["memory_save"] = {"requested": runtime["save_memory"], "status": "success"}
        except Exception as exc:
            trace["memory_save"] = {
                "requested": runtime["save_memory"],
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            trace["warnings"].append("memory save failed")
            if trace["status"] == "success":
                trace["status"] = "partial"
        write_json(trace, output_dir / "trace.json")

    result = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": trace["status"],
        "final_answer": final_answer,
        "messages_path": str(output_dir / "messages.json"),
        "trace_path": str(output_dir / "trace.json"),
        "final_answer_path": str(output_dir / "final_answer.md"),
        "selected_memory": selected_memory,
        "saved_memory": saved_memory,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    if execution_mode == "integrated":
        log_record = {
            "timestamp": now_iso(),
            "conversation_id": runtime["conversation_id"],
            "execution_mode": execution_mode,
            "status": trace["status"],
            "llm_mode": mode,
            "tool_rounds_used": tool_rounds,
            "llm_call_count": llm_calls,
            "elapsed_ms": result["elapsed_ms"],
        }
        if streaming:
            log_record["streaming"] = True
        append_jsonl(log_record, output_dir / "runtime_log.jsonl")
    return result


def _has_successful_tool_result(tool_messages: list[dict] | None) -> bool:
    if not tool_messages:
        return False
    for message in tool_messages:
        if isinstance(message, dict) and message.get("status") == "success":
            return True
    return False


def _apply_observation_next_stage(workspace: dict, observation: dict) -> str:
    next_stage = str(observation.get("next_stage") or "answering")
    if next_stage == "answering":
        pending_outputs = _required_outputs_pending(workspace)
        if pending_outputs:
            next_stage = "tool_calling"
            _merge_unique(workspace["draft"]["missing_info"], ["用户要求生成文件，但还没有成功的文件产物。"])
            _merge_unique(
                workspace["tools"]["rejected_evidence"],
                ["观察阶段试图进入最终回答，但必需文件产物尚未生成。"],
            )
    workspace["task"]["stage"] = next_stage
    workspace["task"]["reason"] = str(observation.get("reason") or "")
    return next_stage


def _workspace_parse_failure(
    runtime: dict,
    execution_mode: str,
    system_prompt: str,
    model_file: Path,
    mode: str,
    output_dir: Path,
    started: float,
    selected_memory: dict,
    messages: list[dict],
    turns: list[dict],
    llm_calls: int,
    warnings: list[str],
    memory_file: Path | None,
    workspace: dict,
    stage: str,
    error: dict | None,
    raw_text: str | None = None,
    all_tool_messages: list[dict] | None = None,
    tool_rounds: int = 0,
    streaming: bool = False,
) -> dict:
    parse_error = {
        "type": "LLMStageParseError",
        "message": f"runtime stage failed to parse: {stage}",
        "stage": stage,
        "cause": error,
    }
    warnings.append(f"workspace stage parse failed: {stage}")
    _record_stage(
        workspace,
        "stage_parse_error",
        {
            "stage": stage,
            "error": error,
            "raw_text": raw_text,
        },
    )
    llm_calls += 1
    answer_messages = _workspace_stage_failure_answer_messages(system_prompt, workspace, stage, error, raw_text)
    if stage == "observation" and _has_successful_tool_result(all_tool_messages or workspace["tools"].get("results", [])):
        answer_messages = _workspace_answer_messages(system_prompt, workspace)
    final_result = generate_ai_message(
        str(model_file),
        answer_messages,
        [],
        mode,
        str(output_dir / "llm_calls"),
        f"workspace_{llm_calls:03d}_{stage}_failure_answering",
        prompt_ready=True,
    )
    if not isinstance(final_result, dict) or not isinstance(final_result.get("ai_message"), dict):
        raise ValueError("B4 result must contain an ai_message object")
    final_ai_message = final_result["ai_message"]
    final_answer = final_ai_message.get("content", "")
    final_control = final_ai_message.get("control")
    status = "success"
    terminal_error = None
    if final_result.get("status") != "success":
        status = "llm_parse_error"
        terminal_error = {
            **parse_error,
            "final_answer_error": final_result.get("error"),
            "llm_call_index": llm_calls,
        }
    elif final_control and final_control.get("state") == "failed":
        status = "agent_failed"
        terminal_error = {
            **parse_error,
            "final_reason": final_control.get("reason", ""),
            "llm_call_index": llm_calls,
        }
    workspace["final"] = {"answer": final_answer, "status": status}
    _record_stage(
        workspace,
        "answering",
        {
            "content": final_answer,
            "control": final_control,
            "agent_step": final_ai_message.get("agent_step"),
        },
    )
    messages.append(final_ai_message)
    turns.append(
        {
            "turn_index": llm_calls,
            "ai_message": final_ai_message,
            "llm_prompt_messages": final_result.get("prompt_messages"),
            "llm_status": final_result.get("status"),
            "llm_error": final_result.get("error"),
            "tool_messages": [],
            "control": final_control,
            "agent_step": final_ai_message.get("agent_step"),
            "latency_ms": None,
        }
    )
    return _write_runtime_outputs(
        runtime,
        execution_mode,
        mode,
        output_dir,
        started,
        selected_memory,
        messages,
        all_tool_messages or [],
        final_answer,
        status,
        tool_rounds,
        llm_calls,
        turns,
        final_control,
        warnings,
        terminal_error,
        memory_file,
        workspace,
        streaming,
    )


def _cancel_requested(should_cancel: Callable[[], bool] | None) -> bool:
    return bool(should_cancel and should_cancel())


def _save_workspace_checkpoint(
    *,
    runtime: dict,
    execution_mode: str,
    mode: str,
    output_dir: Path,
    selected_memory: dict,
    tools_schema: list[dict],
    tools_file: Path,
    memory_file: Path,
    model_file: Path,
    system_prompt: str,
    messages: list[dict],
    all_tool_messages: list[dict],
    tool_rounds: int,
    llm_calls: int,
    turns: list[dict],
    warnings: list[str],
    workspace: dict,
    next_stage: str | None = None,
    status: str = "running",
    partial_answer: str = "",
) -> Path:
    stage = next_stage or workspace.get("task", {}).get("stage") or "planning"
    return save_checkpoint(
        runtime["conversation_id"],
        {
            "status": status,
            "stage": stage,
            "partial_answer": partial_answer,
            "runtime": deepcopy(runtime),
            "execution_mode": execution_mode,
            "mode": mode,
            "output_dir": str(output_dir),
            "selected_memory": deepcopy(selected_memory),
            "tools_schema": deepcopy(tools_schema),
            "tools_file": str(tools_file),
            "memory_file": str(memory_file),
            "model_file": str(model_file),
            "system_prompt": system_prompt,
            "messages": deepcopy(messages),
            "all_tool_messages": deepcopy(all_tool_messages),
            "tool_rounds": tool_rounds,
            "llm_calls": llm_calls,
            "turns": deepcopy(turns),
            "warnings": deepcopy(warnings),
            "workspace": deepcopy(workspace),
        },
    )


def _workspace_cancelled_result(
    runtime: dict,
    execution_mode: str,
    mode: str,
    output_dir: Path,
    started: float,
    selected_memory: dict,
    messages: list[dict],
    all_tool_messages: list[dict],
    partial_answer: str,
    tool_rounds: int,
    llm_calls: int,
    turns: list[dict],
    warnings: list[str],
    memory_file: Path | None,
    workspace: dict,
    system_prompt: str | None = None,
    tools_schema: list[dict] | None = None,
    tools_file: Path | None = None,
    model_file: Path | None = None,
) -> dict:
    final_answer = partial_answer.strip() or "已终止回答。"
    if system_prompt is not None and memory_file is not None and tools_file is not None and model_file is not None:
        _save_workspace_checkpoint(
            runtime=runtime,
            execution_mode=execution_mode,
            mode=mode,
            output_dir=output_dir,
            selected_memory=selected_memory,
            tools_schema=tools_schema or [],
            tools_file=tools_file,
            memory_file=memory_file,
            model_file=model_file,
            system_prompt=system_prompt,
            messages=messages,
            all_tool_messages=all_tool_messages,
            tool_rounds=tool_rounds,
            llm_calls=llm_calls,
            turns=turns,
            warnings=warnings,
            workspace=workspace,
            status="paused",
            partial_answer=partial_answer,
        )
    final_control = {
        "state": "failed",
        "action": "finish",
        "reason": "user cancelled",
    }
    workspace["final"] = {"answer": final_answer, "status": "cancelled"}
    _record_stage(
        workspace,
        "cancelled",
        {
            "content": final_answer,
            "control": final_control,
        },
    )
    return _write_runtime_outputs(
        runtime,
        execution_mode,
        mode,
        output_dir,
        started,
        selected_memory,
        messages,
        all_tool_messages,
        final_answer,
        "cancelled",
        tool_rounds,
        llm_calls,
        turns,
        final_control,
        warnings,
        None,
        memory_file,
        workspace,
        streaming=True,
    )


def _run_workspace(
    runtime: dict,
    execution_mode: str,
    system_prompt: str,
    selected_memory: dict,
    tools_schema: list[dict],
    tools_file: Path,
    memory_file: Path,
    model_file: Path,
    mode: str,
    output_dir: Path,
    started: float,
) -> dict:
    from b3_tool_layer import execute_tool_calls

    workspace = _workspace_from_runtime(runtime, selected_memory)
    current_user_message = {"role": "user", "content": runtime["user_input"]}
    if runtime["input_images"]:
        current_user_message["images"] = runtime["input_images"]
    messages = [
        {"role": "system", "content": system_prompt},
        *workspace["input"].get("history_messages", []),
        current_user_message,
    ]
    turns = []
    all_tool_messages = []
    warnings = []
    if selected_memory.get("status") in {"partial", "error"}:
        warnings.append("memory selection completed with errors")
    layered_status = workspace.get("memory", {}).get("layered", {}).get("status")
    if layered_status == "error":
        warnings.append("layered memory context failed")
    llm_calls = 0
    tool_rounds = 0
    max_turns = runtime["max_turns"]
    max_turns_reached = False
    status = "success"
    terminal_error = None
    final_control = None

    llm_calls += 1
    plan_result = generate_json_object(
        str(model_file),
        _workspace_planning_messages(system_prompt, runtime, selected_memory, tools_schema),
        mode,
        str(output_dir / "llm_calls"),
        f"workspace_{llm_calls:03d}_planning",
        prompt_ready=True,
    )
    if plan_result.get("status") != "success" or not isinstance(plan_result.get("json"), dict):
        return _workspace_parse_failure(
            runtime,
            execution_mode,
            system_prompt,
            model_file,
            mode,
            output_dir,
            started,
            selected_memory,
            messages,
            turns,
            llm_calls,
            warnings,
            memory_file,
            workspace,
            "planning",
            plan_result.get("error"),
            plan_result.get("raw_text"),
        )
    plan = plan_result["json"]
    workspace["task"].update(
        {
            "user_goal": str(plan.get("user_goal") or runtime["user_input"]),
            "requirements": _as_string_list(plan.get("requirements")),
            "success_criteria": _as_string_list(plan.get("success_criteria")),
            "required_outputs": plan.get("required_outputs") if isinstance(plan.get("required_outputs"), list) else [],
            "plan": str(plan.get("plan") or ""),
            "stage": str(plan.get("next_stage") or "answering"),
            "reason": str(plan.get("reason") or ""),
        }
    )
    _merge_unique(workspace["draft"]["known_facts"], plan.get("known_facts"))
    _merge_unique(workspace["draft"]["missing_info"], plan.get("missing_info"))
    _record_stage(workspace, "planning", plan)
    plan_ai_message = make_ai_message(
        str(plan.get("plan") or plan.get("reason") or "已完成任务规划。"),
        [],
        {"state": "completed", "action": "finish", "reason": "workspace planning"},
        _agent_step_from_plan(plan),
    )
    turns.append(
        {
            "turn_index": llm_calls,
            "ai_message": plan_ai_message,
            "llm_prompt_messages": plan_result.get("prompt_messages"),
            "llm_status": plan_result.get("status"),
            "llm_error": plan_result.get("error"),
            "tool_messages": [],
            "control": plan_ai_message.get("control"),
            "agent_step": plan_ai_message.get("agent_step"),
            "latency_ms": None,
        }
    )

    next_stage = workspace["task"]["stage"]
    while next_stage == "tool_calling":
        if tool_rounds >= max_turns:
            max_turns_reached = True
            status = "agent_failed"
            final_control = {
                "state": "failed",
                "action": "finish",
                "reason": f"max_turns reached: {max_turns}",
            }
            terminal_error = {
                "type": "MaxTurnsExceeded",
                "message": final_control["reason"],
                "llm_call_index": llm_calls,
            }
            workspace["task"]["stage"] = "failed"
            workspace["task"]["reason"] = final_control["reason"]
            break
        llm_calls += 1
        turn_start = perf_counter()
        tool_result = generate_ai_message(
            str(model_file),
            _workspace_tool_messages(system_prompt, workspace, tools_schema),
            [],
            mode,
            str(output_dir / "llm_calls"),
            f"workspace_{llm_calls:03d}_tool_calling",
            prompt_ready=True,
        )
        if not isinstance(tool_result, dict) or not isinstance(tool_result.get("ai_message"), dict):
            raise ValueError("B4 result must contain an ai_message object")
        ai_message = tool_result["ai_message"]
        llm_status = tool_result.get("status")
        llm_error = tool_result.get("error")
        turn = {
            "turn_index": llm_calls,
            "ai_message": ai_message,
            "llm_prompt_messages": tool_result.get("prompt_messages"),
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
            "control": ai_message.get("control"),
            "agent_step": ai_message.get("agent_step"),
            "latency_ms": None,
        }
        messages.append(ai_message)
        _save_workspace_checkpoint(
            runtime=runtime,
            execution_mode=execution_mode,
            mode=mode,
            output_dir=output_dir,
            selected_memory=selected_memory,
            tools_schema=tools_schema,
            tools_file=tools_file,
            memory_file=memory_file,
            model_file=model_file,
            system_prompt=system_prompt,
            messages=messages,
            all_tool_messages=all_tool_messages,
            tool_rounds=tool_rounds,
            llm_calls=llm_calls,
            turns=turns,
            warnings=warnings,
            workspace=workspace,
            next_stage="tool_calling",
        )
        if llm_status != "success":
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            return _workspace_parse_failure(
                runtime,
                execution_mode,
                system_prompt,
                model_file,
                mode,
                output_dir,
                started,
                selected_memory,
                messages,
                turns,
                llm_calls,
                warnings,
                memory_file,
                workspace,
                "tool_calling",
                llm_error,
                tool_result.get("raw_text"),
                all_tool_messages,
                tool_rounds,
            )
        tool_calls = ai_message.get("tool_calls", [])
        workspace["tools"]["last_tool_intent"] = ai_message.get("content", "")
        _record_stage(
            workspace,
            "tool_calling",
            {
                "assistant_content": ai_message.get("content", ""),
                "agent_step": ai_message.get("agent_step"),
                "tool_calls": tool_calls,
            },
        )
        _save_workspace_checkpoint(
            runtime=runtime,
            execution_mode=execution_mode,
            mode=mode,
            output_dir=output_dir,
            selected_memory=selected_memory,
            tools_schema=tools_schema,
            tools_file=tools_file,
            memory_file=memory_file,
            model_file=model_file,
            system_prompt=system_prompt,
            messages=messages,
            all_tool_messages=all_tool_messages,
            tool_rounds=tool_rounds,
            llm_calls=llm_calls,
            turns=turns,
            warnings=warnings,
            workspace=workspace,
            next_stage="tool_calling",
        )
        if not tool_calls:
            _record_no_tool_action(workspace, ai_message)
            turns.append(turn)
            next_stage = workspace["task"]["stage"]
            _save_workspace_checkpoint(
                runtime=runtime,
                execution_mode=execution_mode,
                mode=mode,
                output_dir=output_dir,
                selected_memory=selected_memory,
                tools_schema=tools_schema,
                tools_file=tools_file,
                memory_file=memory_file,
                model_file=model_file,
                system_prompt=system_prompt,
                messages=messages,
                all_tool_messages=all_tool_messages,
                tool_rounds=tool_rounds,
                llm_calls=llm_calls,
                turns=turns,
                warnings=warnings,
                workspace=workspace,
                next_stage=next_stage,
            )
            continue
        tool_messages = execute_tool_calls(
            tool_calls,
            str(tools_file),
            runtime["toolset"],
            str(output_dir),
        )
        tool_rounds += 1
        messages.extend(tool_messages)
        all_tool_messages.extend(tool_messages)
        workspace["tools"]["calls"].extend(deepcopy(tool_calls))
        workspace["tools"]["results"].extend(deepcopy(tool_messages))
        turn["tool_messages"] = tool_messages
        turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
        turns.append(turn)

        llm_calls += 1
        observation_result = generate_json_object(
            str(model_file),
            _workspace_observation_messages(system_prompt, workspace, tool_messages),
            mode,
            str(output_dir / "llm_calls"),
            f"workspace_{llm_calls:03d}_observation",
            prompt_ready=True,
        )
        if observation_result.get("status") != "success" or not isinstance(observation_result.get("json"), dict):
            return _workspace_parse_failure(
                runtime,
                execution_mode,
                system_prompt,
                model_file,
                mode,
                output_dir,
                started,
                selected_memory,
                messages,
                turns,
                llm_calls,
                warnings,
                memory_file,
                workspace,
                "observation",
                observation_result.get("error"),
                observation_result.get("raw_text"),
                all_tool_messages,
                tool_rounds,
            )
        observation = observation_result["json"]
        _merge_unique(workspace["tools"]["accepted_evidence"], observation.get("accepted_evidence"))
        _merge_unique(workspace["tools"]["rejected_evidence"], observation.get("rejected_evidence"))
        _merge_unique(workspace["draft"]["known_facts"], observation.get("known_facts"))
        _merge_unique(workspace["draft"]["missing_info"], observation.get("missing_info"))
        workspace["tools"]["observations"].append(str(observation.get("observation") or ""))
        next_stage = _apply_observation_next_stage(workspace, observation)
        _record_stage(workspace, "observation", observation)
        observation_ai_message = make_ai_message(
            str(observation.get("observation") or observation.get("reason") or "已观察工具结果。"),
            [],
            {"state": "completed", "action": "finish", "reason": "workspace observation"},
            _agent_step_from_observation(observation),
        )
        messages.append(observation_ai_message)
        turns.append(
            {
                "turn_index": llm_calls,
                "ai_message": observation_ai_message,
                "llm_prompt_messages": observation_result.get("prompt_messages"),
                "llm_status": observation_result.get("status"),
                "llm_error": observation_result.get("error"),
                "tool_messages": [],
                "control": observation_ai_message.get("control"),
                "agent_step": observation_ai_message.get("agent_step"),
                "latency_ms": None,
            }
        )

    llm_calls += 1
    final_result = generate_ai_message(
        str(model_file),
        _workspace_answer_messages(system_prompt, workspace),
        [],
        mode,
        str(output_dir / "llm_calls"),
        f"workspace_{llm_calls:03d}_answering",
        prompt_ready=True,
    )
    if not isinstance(final_result, dict) or not isinstance(final_result.get("ai_message"), dict):
        raise ValueError("B4 result must contain an ai_message object")
    final_ai_message = final_result["ai_message"]
    final_answer = final_ai_message.get("content", "")
    final_control = final_ai_message.get("control")
    if final_result.get("status") != "success":
        status = "llm_parse_error"
        terminal_error = {
            "type": "LLMParseError",
            "message": "B4 failed to parse final answer output.",
            "llm_call_index": llm_calls,
            "cause": final_result.get("error"),
        }
    elif final_control and final_control.get("state") == "failed":
        status = "agent_failed"
        terminal_error = {
            "type": "AgentDeclaredFailure",
            "message": final_control.get("reason", ""),
            "llm_call_index": llm_calls,
        }
    if max_turns_reached:
        status = "agent_failed"
        final_control = {
            "state": "failed",
            "action": "finish",
            "reason": f"max_turns reached: {max_turns}",
        }
        final_ai_message["control"] = final_control
        terminal_error = {
            "type": "MaxTurnsExceeded",
            "message": final_control["reason"],
            "llm_call_index": llm_calls,
        }
    workspace["final"] = {"answer": final_answer, "status": status}
    _record_stage(
        workspace,
        "answering",
        {
            "content": final_answer,
            "control": final_control,
            "agent_step": final_ai_message.get("agent_step"),
        },
    )
    messages.append(final_ai_message)
    turns.append(
        {
            "turn_index": llm_calls,
            "ai_message": final_ai_message,
            "llm_prompt_messages": final_result.get("prompt_messages"),
            "llm_status": final_result.get("status"),
            "llm_error": final_result.get("error"),
            "tool_messages": [],
            "control": final_ai_message.get("control"),
            "agent_step": final_ai_message.get("agent_step"),
            "latency_ms": None,
        }
    )
    print(f"content: {final_answer}")
    return _write_runtime_outputs(
        runtime,
        execution_mode,
        mode,
        output_dir,
        started,
        selected_memory,
        messages,
        all_tool_messages,
        final_answer,
        status,
        tool_rounds,
        llm_calls,
        turns,
        final_control,
        warnings,
        terminal_error,
        memory_file,
        workspace,
    )


def _run_workspace_stream(
    runtime: dict,
    execution_mode: str,
    system_prompt: str,
    selected_memory: dict,
    tools_schema: list[dict],
    tools_file: Path,
    memory_file: Path,
    model_file: Path,
    mode: str,
    output_dir: Path,
    started: float,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[dict]:
    from b3_tool_layer import execute_tool_calls

    workspace = _workspace_from_runtime(runtime, selected_memory)
    current_user_message = {"role": "user", "content": runtime["user_input"]}
    if runtime["input_images"]:
        current_user_message["images"] = runtime["input_images"]
    messages = [
        {"role": "system", "content": system_prompt},
        *workspace["input"].get("history_messages", []),
        current_user_message,
    ]
    turns = []
    all_tool_messages = []
    warnings = []
    if selected_memory.get("status") in {"partial", "error"}:
        warnings.append("memory selection completed with errors")
    layered_status = workspace.get("memory", {}).get("layered", {}).get("status")
    if layered_status == "error":
        warnings.append("layered memory context failed")
    llm_calls = 0
    tool_rounds = 0
    max_turns = runtime["max_turns"]
    max_turns_reached = False
    status = "success"
    terminal_error = None
    final_control = None
    _save_workspace_checkpoint(
        runtime=runtime,
        execution_mode=execution_mode,
        mode=mode,
        output_dir=output_dir,
        selected_memory=selected_memory,
        tools_schema=tools_schema,
        tools_file=tools_file,
        memory_file=memory_file,
        model_file=model_file,
        system_prompt=system_prompt,
        messages=messages,
        all_tool_messages=all_tool_messages,
        tool_rounds=tool_rounds,
        llm_calls=llm_calls,
        turns=turns,
        warnings=warnings,
        workspace=workspace,
        next_stage="planning",
    )

    def cancelled_done(partial_answer: str = "") -> dict:
        return {
            "type": "done",
            "result": _workspace_cancelled_result(
                runtime,
                execution_mode,
                mode,
                output_dir,
                started,
                selected_memory,
                messages,
                all_tool_messages,
                partial_answer,
                tool_rounds,
                llm_calls,
                turns,
                warnings,
                memory_file,
                workspace,
                system_prompt,
                tools_schema,
                tools_file,
                model_file,
            ),
        }

    if _cancel_requested(should_cancel):
        yield cancelled_done()
        return

    llm_calls += 1
    plan_result = generate_json_object(
        str(model_file),
        _workspace_planning_messages(system_prompt, runtime, selected_memory, tools_schema),
        mode,
        str(output_dir / "llm_calls"),
        f"workspace_{llm_calls:03d}_planning",
        prompt_ready=True,
    )
    if plan_result.get("status") != "success" or not isinstance(plan_result.get("json"), dict):
        result = _workspace_parse_failure(
            runtime,
            execution_mode,
            system_prompt,
            model_file,
            mode,
            output_dir,
            started,
            selected_memory,
            messages,
            turns,
            llm_calls,
            warnings,
            memory_file,
            workspace,
            "planning",
            plan_result.get("error"),
            plan_result.get("raw_text"),
            streaming=True,
        )
        yield {"type": "done", "result": result}
        return
    if _cancel_requested(should_cancel):
        yield cancelled_done()
        return
    plan = plan_result["json"]
    workspace["task"].update(
        {
            "user_goal": str(plan.get("user_goal") or runtime["user_input"]),
            "requirements": _as_string_list(plan.get("requirements")),
            "success_criteria": _as_string_list(plan.get("success_criteria")),
            "required_outputs": plan.get("required_outputs") if isinstance(plan.get("required_outputs"), list) else [],
            "plan": str(plan.get("plan") or ""),
            "stage": str(plan.get("next_stage") or "answering"),
            "reason": str(plan.get("reason") or ""),
        }
    )
    _merge_unique(workspace["draft"]["known_facts"], plan.get("known_facts"))
    _merge_unique(workspace["draft"]["missing_info"], plan.get("missing_info"))
    _record_stage(workspace, "planning", plan)
    plan_ai_message = make_ai_message(
        str(plan.get("plan") or plan.get("reason") or "已完成任务规划。"),
        [],
        {"state": "completed", "action": "finish", "reason": "workspace planning"},
        _agent_step_from_plan(plan),
    )
    turns.append(
        {
            "turn_index": llm_calls,
            "ai_message": plan_ai_message,
            "llm_prompt_messages": plan_result.get("prompt_messages"),
            "llm_status": plan_result.get("status"),
            "llm_error": plan_result.get("error"),
            "tool_messages": [],
            "control": plan_ai_message.get("control"),
            "agent_step": plan_ai_message.get("agent_step"),
            "latency_ms": None,
        }
    )
    _save_workspace_checkpoint(
        runtime=runtime,
        execution_mode=execution_mode,
        mode=mode,
        output_dir=output_dir,
        selected_memory=selected_memory,
        tools_schema=tools_schema,
        tools_file=tools_file,
        memory_file=memory_file,
        model_file=model_file,
        system_prompt=system_prompt,
        messages=messages,
        all_tool_messages=all_tool_messages,
        tool_rounds=tool_rounds,
        llm_calls=llm_calls,
        turns=turns,
        warnings=warnings,
        workspace=workspace,
        next_stage=workspace["task"]["stage"],
    )
    yield {
        "type": "state",
        "state": "planning",
        "action": "workspace_plan",
        "reason": workspace["task"].get("reason", ""),
        "agent_step": plan_ai_message.get("agent_step"),
        "llm_call_index": llm_calls,
    }

    next_stage = workspace["task"]["stage"]
    while next_stage == "tool_calling":
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        if tool_rounds >= max_turns:
            max_turns_reached = True
            status = "agent_failed"
            final_control = {
                "state": "failed",
                "action": "finish",
                "reason": f"max_turns reached: {max_turns}",
            }
            terminal_error = {
                "type": "MaxTurnsExceeded",
                "message": final_control["reason"],
                "llm_call_index": llm_calls,
            }
            workspace["task"]["stage"] = "failed"
            workspace["task"]["reason"] = final_control["reason"]
            yield {"type": "state", **final_control, "llm_call_index": llm_calls}
            break
        llm_calls += 1
        turn_start = perf_counter()
        tool_result = generate_ai_message(
            str(model_file),
            _workspace_tool_messages(system_prompt, workspace, tools_schema),
            [],
            mode,
            str(output_dir / "llm_calls"),
            f"workspace_{llm_calls:03d}_tool_calling",
            prompt_ready=True,
        )
        if not isinstance(tool_result, dict) or not isinstance(tool_result.get("ai_message"), dict):
            raise ValueError("B4 result must contain an ai_message object")
        ai_message = tool_result["ai_message"]
        llm_status = tool_result.get("status")
        llm_error = tool_result.get("error")
        turn = {
            "turn_index": llm_calls,
            "ai_message": ai_message,
            "llm_prompt_messages": tool_result.get("prompt_messages"),
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
            "control": ai_message.get("control"),
            "agent_step": ai_message.get("agent_step"),
            "latency_ms": None,
        }
        messages.append(ai_message)
        if llm_status != "success":
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            result = _workspace_parse_failure(
                runtime,
                execution_mode,
                system_prompt,
                model_file,
                mode,
                output_dir,
                started,
                selected_memory,
                messages,
                turns,
                llm_calls,
                warnings,
                memory_file,
                workspace,
                "tool_calling",
                llm_error,
                tool_result.get("raw_text"),
                all_tool_messages,
                tool_rounds,
                streaming=True,
            )
            yield {"type": "done", "result": result}
            return
        control = ai_message.get("control", {})
        yield {"type": "state", **control, "agent_step": ai_message.get("agent_step"), "llm_call_index": llm_calls}
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        tool_calls = ai_message.get("tool_calls", [])
        workspace["tools"]["last_tool_intent"] = ai_message.get("content", "")
        _record_stage(
            workspace,
            "tool_calling",
            {
                "assistant_content": ai_message.get("content", ""),
                "agent_step": ai_message.get("agent_step"),
                "tool_calls": tool_calls,
            },
        )
        if not tool_calls:
            _record_no_tool_action(workspace, ai_message)
            turns.append(turn)
            next_stage = workspace["task"]["stage"]
            continue
        yield {
            "type": "tool_start",
            "tool_calls": tool_calls,
            "assistant_content": ai_message.get("content", ""),
            "agent_step": ai_message.get("agent_step"),
            "llm_call_index": llm_calls,
        }
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        tool_messages = execute_tool_calls(
            tool_calls,
            str(tools_file),
            runtime["toolset"],
            str(output_dir),
        )
        tool_rounds += 1
        messages.extend(tool_messages)
        all_tool_messages.extend(tool_messages)
        workspace["tools"]["calls"].extend(deepcopy(tool_calls))
        workspace["tools"]["results"].extend(deepcopy(tool_messages))
        turn["tool_messages"] = tool_messages
        turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
        turns.append(turn)
        workspace["task"]["stage"] = "observation"
        _save_workspace_checkpoint(
            runtime=runtime,
            execution_mode=execution_mode,
            mode=mode,
            output_dir=output_dir,
            selected_memory=selected_memory,
            tools_schema=tools_schema,
            tools_file=tools_file,
            memory_file=memory_file,
            model_file=model_file,
            system_prompt=system_prompt,
            messages=messages,
            all_tool_messages=all_tool_messages,
            tool_rounds=tool_rounds,
            llm_calls=llm_calls,
            turns=turns,
            warnings=warnings,
            workspace=workspace,
            next_stage="observation",
        )
        yield {"type": "tool_done", "tool_messages": tool_messages, "llm_call_index": llm_calls}
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return

        llm_calls += 1
        observation_result = generate_json_object(
            str(model_file),
            _workspace_observation_messages(system_prompt, workspace, tool_messages),
            mode,
            str(output_dir / "llm_calls"),
            f"workspace_{llm_calls:03d}_observation",
            prompt_ready=True,
        )
        if observation_result.get("status") != "success" or not isinstance(observation_result.get("json"), dict):
            result = _workspace_parse_failure(
                runtime,
                execution_mode,
                system_prompt,
                model_file,
                mode,
                output_dir,
                started,
                selected_memory,
                messages,
                turns,
                llm_calls,
                warnings,
                memory_file,
                workspace,
                "observation",
                observation_result.get("error"),
                observation_result.get("raw_text"),
                all_tool_messages,
                tool_rounds,
                streaming=True,
            )
            yield {"type": "done", "result": result}
            return
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        observation = observation_result["json"]
        _merge_unique(workspace["tools"]["accepted_evidence"], observation.get("accepted_evidence"))
        _merge_unique(workspace["tools"]["rejected_evidence"], observation.get("rejected_evidence"))
        _merge_unique(workspace["draft"]["known_facts"], observation.get("known_facts"))
        _merge_unique(workspace["draft"]["missing_info"], observation.get("missing_info"))
        workspace["tools"]["observations"].append(str(observation.get("observation") or ""))
        next_stage = _apply_observation_next_stage(workspace, observation)
        _record_stage(workspace, "observation", observation)
        observation_ai_message = make_ai_message(
            str(observation.get("observation") or observation.get("reason") or "已观察工具结果。"),
            [],
            {"state": "completed", "action": "finish", "reason": "workspace observation"},
            _agent_step_from_observation(observation),
        )
        messages.append(observation_ai_message)
        turns.append(
            {
                "turn_index": llm_calls,
                "ai_message": observation_ai_message,
                "llm_prompt_messages": observation_result.get("prompt_messages"),
                "llm_status": observation_result.get("status"),
                "llm_error": observation_result.get("error"),
                "tool_messages": [],
                "control": observation_ai_message.get("control"),
                "agent_step": observation_ai_message.get("agent_step"),
                "latency_ms": None,
            }
        )
        _save_workspace_checkpoint(
            runtime=runtime,
            execution_mode=execution_mode,
            mode=mode,
            output_dir=output_dir,
            selected_memory=selected_memory,
            tools_schema=tools_schema,
            tools_file=tools_file,
            memory_file=memory_file,
            model_file=model_file,
            system_prompt=system_prompt,
            messages=messages,
            all_tool_messages=all_tool_messages,
            tool_rounds=tool_rounds,
            llm_calls=llm_calls,
            turns=turns,
            warnings=warnings,
            workspace=workspace,
            next_stage=next_stage,
        )
        yield {
            "type": "state",
            "state": "observing",
            "action": "workspace_observe",
            "reason": workspace["task"].get("reason", ""),
            "agent_step": observation_ai_message.get("agent_step"),
            "llm_call_index": llm_calls,
        }

    if _cancel_requested(should_cancel):
        yield cancelled_done()
        return

    workspace["task"]["stage"] = "answering"
    _save_workspace_checkpoint(
        runtime=runtime,
        execution_mode=execution_mode,
        mode=mode,
        output_dir=output_dir,
        selected_memory=selected_memory,
        tools_schema=tools_schema,
        tools_file=tools_file,
        memory_file=memory_file,
        model_file=model_file,
        system_prompt=system_prompt,
        messages=messages,
        all_tool_messages=all_tool_messages,
        tool_rounds=tool_rounds,
        llm_calls=llm_calls,
        turns=turns,
        warnings=warnings,
        workspace=workspace,
        next_stage="answering",
    )
    llm_calls += 1
    final_result = None
    final_chunks: list[str] = []
    for event in stream_ai_message(
        str(model_file),
        _workspace_answer_messages(system_prompt, workspace),
        [],
        mode,
        str(output_dir / "llm_calls"),
        f"workspace_{llm_calls:03d}_answering",
        prompt_ready=True,
    ):
        if _cancel_requested(should_cancel):
            yield cancelled_done("".join(final_chunks))
            return
        if not isinstance(event, dict):
            continue
        if event.get("type") == "delta":
            text = str(event.get("text", ""))
            final_chunks.append(text)
            yield {"type": "delta", "text": text, "llm_call_index": llm_calls}
        elif event.get("type") == "done":
            final_result = event.get("result")
    if _cancel_requested(should_cancel):
        yield cancelled_done("".join(final_chunks))
        return
    if not isinstance(final_result, dict) or not isinstance(final_result.get("ai_message"), dict):
        raise ValueError("B4 stream result must contain an ai_message object")
    final_ai_message = final_result["ai_message"]
    final_answer = final_ai_message.get("content", "")
    final_control = final_ai_message.get("control")
    if final_result.get("status") != "success":
        status = "llm_parse_error"
        terminal_error = {
            "type": "LLMParseError",
            "message": "B4 failed to parse final answer output.",
            "llm_call_index": llm_calls,
            "cause": final_result.get("error"),
        }
    elif final_control and final_control.get("state") == "failed":
        status = "agent_failed"
        terminal_error = {
            "type": "AgentDeclaredFailure",
            "message": final_control.get("reason", ""),
            "llm_call_index": llm_calls,
        }
    if max_turns_reached:
        status = "agent_failed"
        final_control = {
            "state": "failed",
            "action": "finish",
            "reason": f"max_turns reached: {max_turns}",
        }
        final_ai_message["control"] = final_control
        terminal_error = {
            "type": "MaxTurnsExceeded",
            "message": final_control["reason"],
            "llm_call_index": llm_calls,
        }
    workspace["final"] = {"answer": final_answer, "status": status}
    _record_stage(
        workspace,
        "answering",
        {
            "content": final_answer,
            "control": final_control,
            "agent_step": final_ai_message.get("agent_step"),
        },
    )
    messages.append(final_ai_message)
    turns.append(
        {
            "turn_index": llm_calls,
            "ai_message": final_ai_message,
            "llm_prompt_messages": final_result.get("prompt_messages"),
            "llm_status": final_result.get("status"),
            "llm_error": final_result.get("error"),
            "tool_messages": [],
            "control": final_ai_message.get("control"),
            "agent_step": final_ai_message.get("agent_step"),
            "latency_ms": None,
        }
    )
    _save_workspace_checkpoint(
        runtime=runtime,
        execution_mode=execution_mode,
        mode=mode,
        output_dir=output_dir,
        selected_memory=selected_memory,
        tools_schema=tools_schema,
        tools_file=tools_file,
        memory_file=memory_file,
        model_file=model_file,
        system_prompt=system_prompt,
        messages=messages,
        all_tool_messages=all_tool_messages,
        tool_rounds=tool_rounds,
        llm_calls=llm_calls,
        turns=turns,
        warnings=warnings,
        workspace=workspace,
        next_stage="done",
        status="done",
    )
    yield {"type": "state", **(final_control or {}), "agent_step": final_ai_message.get("agent_step"), "llm_call_index": llm_calls}
    result = _write_runtime_outputs(
        runtime,
        execution_mode,
        mode,
        output_dir,
        started,
        selected_memory,
        messages,
        all_tool_messages,
        final_answer,
        status,
        tool_rounds,
        llm_calls,
        turns,
        final_control,
        warnings,
        terminal_error,
        memory_file,
        workspace,
        streaming=True,
    )
    yield {"type": "done", "result": result}


def _last_pending_tool_call(messages: list[dict], turns: list[dict]) -> tuple[dict, dict] | None:
    if not messages:
        return None
    ai_message = messages[-1]
    if not isinstance(ai_message, dict) or ai_message.get("role") != "assistant":
        return None
    tool_calls = ai_message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    for turn in reversed(turns):
        if turn.get("ai_message") == ai_message:
            return None
    turn = {
        "turn_index": 0,
        "ai_message": ai_message,
        "llm_prompt_messages": None,
        "llm_status": "success",
        "llm_error": None,
        "tool_messages": [],
        "control": ai_message.get("control"),
        "agent_step": ai_message.get("agent_step"),
        "latency_ms": None,
    }
    return ai_message, turn


def _last_tool_messages(turns: list[dict]) -> list[dict]:
    for turn in reversed(turns):
        messages = turn.get("tool_messages")
        if isinstance(messages, list) and messages:
            return messages
    return []


def resume_workspace_stream(
    conversation_id: str,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[dict]:
    from b3_tool_layer import execute_tool_calls

    checkpoint = load_checkpoint(conversation_id)
    runtime = deepcopy(checkpoint["runtime"])
    execution_mode = str(checkpoint.get("execution_mode") or "integrated")
    mode = str(checkpoint.get("mode") or "prompt_json")
    output_dir = Path(checkpoint["output_dir"]).resolve()
    selected_memory = deepcopy(checkpoint.get("selected_memory") or {})
    tools_schema = deepcopy(checkpoint.get("tools_schema") or [])
    tools_file = Path(checkpoint["tools_file"]).resolve()
    memory_file = Path(checkpoint["memory_file"]).resolve()
    model_file = Path(checkpoint["model_file"]).resolve()
    system_prompt = str(checkpoint.get("system_prompt") or "")
    messages = deepcopy(checkpoint.get("messages") or [])
    all_tool_messages = deepcopy(checkpoint.get("all_tool_messages") or [])
    turns = deepcopy(checkpoint.get("turns") or [])
    warnings = deepcopy(checkpoint.get("warnings") or [])
    workspace = deepcopy(checkpoint.get("workspace") or _workspace_from_runtime(runtime, selected_memory))
    llm_calls = int(checkpoint.get("llm_calls") or 0)
    tool_rounds = int(checkpoint.get("tool_rounds") or 0)
    next_stage = str(checkpoint.get("stage") or workspace.get("task", {}).get("stage") or "planning")
    started = perf_counter()
    max_turns = runtime["max_turns"]
    max_turns_reached = False
    status = "success"
    terminal_error = None
    final_control = None

    if checkpoint.get("status") == "done":
        result = _write_runtime_outputs(
            runtime,
            execution_mode,
            mode,
            output_dir,
            started,
            selected_memory,
            messages,
            all_tool_messages,
            str(workspace.get("final", {}).get("answer") or ""),
            "success",
            tool_rounds,
            llm_calls,
            turns,
            {"state": "completed", "action": "finish", "reason": "checkpoint already completed"},
            warnings,
            None,
            memory_file,
            workspace,
            streaming=True,
        )
        yield {"type": "done", "result": result}
        return

    def save(stage: str, checkpoint_status: str = "running", partial_answer: str = "") -> None:
        _save_workspace_checkpoint(
            runtime=runtime,
            execution_mode=execution_mode,
            mode=mode,
            output_dir=output_dir,
            selected_memory=selected_memory,
            tools_schema=tools_schema,
            tools_file=tools_file,
            memory_file=memory_file,
            model_file=model_file,
            system_prompt=system_prompt,
            messages=messages,
            all_tool_messages=all_tool_messages,
            tool_rounds=tool_rounds,
            llm_calls=llm_calls,
            turns=turns,
            warnings=warnings,
            workspace=workspace,
            next_stage=stage,
            status=checkpoint_status,
            partial_answer=partial_answer,
        )

    def cancelled_done(partial_answer: str = "") -> dict:
        return {
            "type": "done",
            "result": _workspace_cancelled_result(
                runtime,
                execution_mode,
                mode,
                output_dir,
                started,
                selected_memory,
                messages,
                all_tool_messages,
                partial_answer,
                tool_rounds,
                llm_calls,
                turns,
                warnings,
                memory_file,
                workspace,
                system_prompt,
                tools_schema,
                tools_file,
                model_file,
            ),
        }

    if next_stage == "planning":
        yield from _run_workspace_stream(
            runtime,
            execution_mode,
            system_prompt,
            selected_memory,
            tools_schema,
            tools_file,
            memory_file,
            model_file,
            mode,
            output_dir,
            started,
            should_cancel,
        )
        return

    while next_stage == "tool_calling":
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        if tool_rounds >= max_turns:
            max_turns_reached = True
            status = "agent_failed"
            final_control = {
                "state": "failed",
                "action": "finish",
                "reason": f"max_turns reached: {max_turns}",
            }
            terminal_error = {
                "type": "MaxTurnsExceeded",
                "message": final_control["reason"],
                "llm_call_index": llm_calls,
            }
            workspace["task"]["stage"] = "failed"
            workspace["task"]["reason"] = final_control["reason"]
            yield {"type": "state", **final_control, "llm_call_index": llm_calls}
            break

        pending = _last_pending_tool_call(messages, turns)
        if pending is None:
            llm_calls += 1
            turn_start = perf_counter()
            tool_result = generate_ai_message(
                str(model_file),
                _workspace_tool_messages(system_prompt, workspace, tools_schema),
                [],
                mode,
                str(output_dir / "llm_calls"),
                f"workspace_{llm_calls:03d}_tool_calling",
                prompt_ready=True,
            )
            if not isinstance(tool_result, dict) or not isinstance(tool_result.get("ai_message"), dict):
                raise ValueError("B4 result must contain an ai_message object")
            ai_message = tool_result["ai_message"]
            llm_status = tool_result.get("status")
            llm_error = tool_result.get("error")
            turn = {
                "turn_index": llm_calls,
                "ai_message": ai_message,
                "llm_prompt_messages": tool_result.get("prompt_messages"),
                "llm_status": llm_status,
                "llm_error": llm_error,
                "tool_messages": [],
                "control": ai_message.get("control"),
                "agent_step": ai_message.get("agent_step"),
                "latency_ms": None,
            }
            messages.append(ai_message)
            save("tool_calling")
            if llm_status != "success":
                turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
                turns.append(turn)
                result = _workspace_parse_failure(
                    runtime,
                    execution_mode,
                    system_prompt,
                    model_file,
                    mode,
                    output_dir,
                    started,
                    selected_memory,
                    messages,
                    turns,
                    llm_calls,
                    warnings,
                    memory_file,
                    workspace,
                    "tool_calling",
                    llm_error,
                    tool_result.get("raw_text"),
                    all_tool_messages,
                    tool_rounds,
                    streaming=True,
                )
                yield {"type": "done", "result": result}
                return
            control = ai_message.get("control", {})
            yield {"type": "state", **control, "agent_step": ai_message.get("agent_step"), "llm_call_index": llm_calls}
        else:
            ai_message, turn = pending
            turn_start = perf_counter()
            turn["turn_index"] = llm_calls

        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        tool_calls = ai_message.get("tool_calls", [])
        workspace["tools"]["last_tool_intent"] = ai_message.get("content", "")
        _record_stage(
            workspace,
            "tool_calling",
            {
                "assistant_content": ai_message.get("content", ""),
                "agent_step": ai_message.get("agent_step"),
                "tool_calls": tool_calls,
            },
        )
        save("tool_calling")
        if not tool_calls:
            _record_no_tool_action(workspace, ai_message)
            turns.append(turn)
            next_stage = workspace["task"]["stage"]
            save(next_stage)
            continue
        yield {
            "type": "tool_start",
            "tool_calls": tool_calls,
            "assistant_content": ai_message.get("content", ""),
            "agent_step": ai_message.get("agent_step"),
            "llm_call_index": llm_calls,
        }
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        tool_messages = execute_tool_calls(
            tool_calls,
            str(tools_file),
            runtime["toolset"],
            str(output_dir),
        )
        tool_rounds += 1
        messages.extend(tool_messages)
        all_tool_messages.extend(tool_messages)
        workspace["tools"]["calls"].extend(deepcopy(tool_calls))
        workspace["tools"]["results"].extend(deepcopy(tool_messages))
        turn["tool_messages"] = tool_messages
        turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
        turns.append(turn)
        workspace["task"]["stage"] = "observation"
        next_stage = "observation"
        save("observation")
        yield {"type": "tool_done", "tool_messages": tool_messages, "llm_call_index": llm_calls}

    if next_stage == "observation":
        tool_messages = _last_tool_messages(turns)
        if not tool_messages:
            next_stage = "tool_calling"
            workspace["task"]["stage"] = "tool_calling"
            save("tool_calling")
            yield from resume_workspace_stream(conversation_id, should_cancel)
            return
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        llm_calls += 1
        observation_result = generate_json_object(
            str(model_file),
            _workspace_observation_messages(system_prompt, workspace, tool_messages),
            mode,
            str(output_dir / "llm_calls"),
            f"workspace_{llm_calls:03d}_observation",
            prompt_ready=True,
        )
        if observation_result.get("status") != "success" or not isinstance(observation_result.get("json"), dict):
            result = _workspace_parse_failure(
                runtime,
                execution_mode,
                system_prompt,
                model_file,
                mode,
                output_dir,
                started,
                selected_memory,
                messages,
                turns,
                llm_calls,
                warnings,
                memory_file,
                workspace,
                "observation",
                observation_result.get("error"),
                observation_result.get("raw_text"),
                all_tool_messages,
                tool_rounds,
                streaming=True,
            )
            yield {"type": "done", "result": result}
            return
        if _cancel_requested(should_cancel):
            yield cancelled_done()
            return
        observation = observation_result["json"]
        _merge_unique(workspace["tools"]["accepted_evidence"], observation.get("accepted_evidence"))
        _merge_unique(workspace["tools"]["rejected_evidence"], observation.get("rejected_evidence"))
        _merge_unique(workspace["draft"]["known_facts"], observation.get("known_facts"))
        _merge_unique(workspace["draft"]["missing_info"], observation.get("missing_info"))
        workspace["tools"]["observations"].append(str(observation.get("observation") or ""))
        next_stage = _apply_observation_next_stage(workspace, observation)
        _record_stage(workspace, "observation", observation)
        observation_ai_message = make_ai_message(
            str(observation.get("observation") or observation.get("reason") or "已观察工具结果。"),
            [],
            {"state": "completed", "action": "finish", "reason": "workspace observation"},
            _agent_step_from_observation(observation),
        )
        messages.append(observation_ai_message)
        turns.append(
            {
                "turn_index": llm_calls,
                "ai_message": observation_ai_message,
                "llm_prompt_messages": observation_result.get("prompt_messages"),
                "llm_status": observation_result.get("status"),
                "llm_error": observation_result.get("error"),
                "tool_messages": [],
                "control": observation_ai_message.get("control"),
                "agent_step": observation_ai_message.get("agent_step"),
                "latency_ms": None,
            }
        )
        save(next_stage)
        yield {
            "type": "state",
            "state": "observing",
            "action": "workspace_observe",
            "reason": workspace["task"].get("reason", ""),
            "agent_step": observation_ai_message.get("agent_step"),
            "llm_call_index": llm_calls,
        }
        if next_stage == "tool_calling":
            yield from resume_workspace_stream(conversation_id, should_cancel)
            return

    if _cancel_requested(should_cancel):
        yield cancelled_done()
        return
    workspace["task"]["stage"] = "answering"
    save("answering")
    llm_calls += 1
    final_result = None
    final_chunks: list[str] = []
    for event in stream_ai_message(
        str(model_file),
        _workspace_answer_messages(system_prompt, workspace),
        [],
        mode,
        str(output_dir / "llm_calls"),
        f"workspace_{llm_calls:03d}_answering",
        prompt_ready=True,
    ):
        if _cancel_requested(should_cancel):
            yield cancelled_done("".join(final_chunks))
            return
        if not isinstance(event, dict):
            continue
        if event.get("type") == "delta":
            text = str(event.get("text", ""))
            final_chunks.append(text)
            yield {"type": "delta", "text": text, "llm_call_index": llm_calls}
        elif event.get("type") == "done":
            final_result = event.get("result")
    if _cancel_requested(should_cancel):
        yield cancelled_done("".join(final_chunks))
        return
    if not isinstance(final_result, dict) or not isinstance(final_result.get("ai_message"), dict):
        raise ValueError("B4 stream result must contain an ai_message object")
    final_ai_message = final_result["ai_message"]
    final_answer = final_ai_message.get("content", "")
    final_control = final_ai_message.get("control")
    if final_result.get("status") != "success":
        status = "llm_parse_error"
        terminal_error = {
            "type": "LLMParseError",
            "message": "B4 failed to parse final answer output.",
            "llm_call_index": llm_calls,
            "cause": final_result.get("error"),
        }
    elif final_control and final_control.get("state") == "failed":
        status = "agent_failed"
        terminal_error = {
            "type": "AgentDeclaredFailure",
            "message": final_control.get("reason", ""),
            "llm_call_index": llm_calls,
        }
    if max_turns_reached:
        status = "agent_failed"
        final_control = {
            "state": "failed",
            "action": "finish",
            "reason": f"max_turns reached: {max_turns}",
        }
        final_ai_message["control"] = final_control
        terminal_error = {
            "type": "MaxTurnsExceeded",
            "message": final_control["reason"],
            "llm_call_index": llm_calls,
        }
    workspace["final"] = {"answer": final_answer, "status": status}
    _record_stage(
        workspace,
        "answering",
        {
            "content": final_answer,
            "control": final_control,
            "agent_step": final_ai_message.get("agent_step"),
        },
    )
    messages.append(final_ai_message)
    turns.append(
        {
            "turn_index": llm_calls,
            "ai_message": final_ai_message,
            "llm_prompt_messages": final_result.get("prompt_messages"),
            "llm_status": final_result.get("status"),
            "llm_error": final_result.get("error"),
            "tool_messages": [],
            "control": final_ai_message.get("control"),
            "agent_step": final_ai_message.get("agent_step"),
            "latency_ms": None,
        }
    )
    save("done", "done")
    yield {"type": "state", **(final_control or {}), "agent_step": final_ai_message.get("agent_step"), "llm_call_index": llm_calls}
    result = _write_runtime_outputs(
        runtime,
        execution_mode,
        mode,
        output_dir,
        started,
        selected_memory,
        messages,
        all_tool_messages,
        final_answer,
        status,
        tool_rounds,
        llm_calls,
        turns,
        final_control,
        warnings,
        terminal_error,
        memory_file,
        workspace,
        streaming=True,
    )
    yield {"type": "done", "result": result}

