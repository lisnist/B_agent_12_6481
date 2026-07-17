from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

from common.io_utils import append_jsonl, write_json, write_text
from common.logging_utils import now_iso

from .b1_fixture import _fixture_tool_messages
from .b1_llm_bridge import generate_ai_message, stream_ai_message
from .b1_prompting import build_llm_prompt_messages


def _cancel_requested(should_cancel: Callable[[], bool] | None) -> bool:
    return bool(should_cancel and should_cancel())


def _legacy_cancelled_result(
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
) -> dict:
    final_answer = partial_answer.strip() or "已终止回答。"
    final_control = {
        "state": "failed",
        "action": "finish",
        "reason": "user cancelled",
    }
    write_json(messages, output_dir / "messages.json")
    if execution_mode == "integrated":
        write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_text(final_answer.strip() + "\n", output_dir / "final_answer.md")
    trace = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": "cancelled",
        "toolset": runtime["toolset"],
        "max_turns": runtime["max_turns"],
        "tool_rounds_used": tool_rounds,
        "llm_call_count": llm_calls,
        "final_state": final_control["state"],
        "finish_reason": final_control["reason"],
        "turns": turns,
        "final_answer_path": "final_answer.md",
        "memory_save": {"requested": runtime["save_memory"], "status": "skipped", "reason": "cancelled"},
        "warnings": warnings,
        "error": None,
    }
    write_json(trace, output_dir / "trace.json")
    result = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": "cancelled",
        "final_answer": final_answer,
        "messages_path": str(output_dir / "messages.json"),
        "trace_path": str(output_dir / "trace.json"),
        "final_answer_path": str(output_dir / "final_answer.md"),
        "selected_memory": selected_memory,
        "saved_memory": None,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    if execution_mode == "integrated":
        append_jsonl(
            {
                "timestamp": now_iso(),
                "conversation_id": runtime["conversation_id"],
                "execution_mode": execution_mode,
                "status": "cancelled",
                "llm_mode": mode,
                "tool_rounds_used": tool_rounds,
                "llm_call_count": llm_calls,
                "elapsed_ms": result["elapsed_ms"],
                "streaming": True,
            },
            output_dir / "runtime_log.jsonl",
        )
    return result


def run_legacy_loop(
    runtime: dict,
    execution_mode: str,
    system_prompt: str,
    selected_memory: dict,
    tools_schema: list[dict],
    fixture_data: dict | None,
    tools_file: Path | None,
    memory_file: Path | None,
    model_file: Path | None,
    mode: str,
    output_dir: Path,
    started: float,
) -> dict:
    current_user_message = {"role": "user", "content": runtime["user_input"]}
    if runtime["input_images"]:
        current_user_message["images"] = runtime["input_images"]
    messages = [
        {"role": "system", "content": system_prompt},
        *runtime["history_messages"],
        current_user_message,
    ]
    tool_rounds = 0
    llm_calls = 0
    turns = []
    all_tool_messages = []
    final_answer = ""
    status = "success"
    terminal_error = None
    warnings = []
    final_control = None
    if selected_memory.get("status") in {"partial", "error"}:
        warnings.append("memory selection completed with errors")
    max_turns = runtime["max_turns"]

    while True:
        llm_calls += 1
        turn_start = perf_counter()
        if execution_mode == "fixture":
            if llm_calls > len(fixture_data["ai_messages"]):
                raise ValueError("fixture AIMessage sequence ended before a final answer")
            ai_message = deepcopy(fixture_data["ai_messages"][llm_calls - 1])
            llm_status = "success"
            llm_error = None
            llm_prompt_messages = None
        else:
            llm_input_messages = build_llm_prompt_messages(messages, tools_schema)
            llm_result = generate_ai_message(
                str(model_file),
                llm_input_messages,
                [],
                mode,
                str(output_dir / "llm_calls"),
                f"llm_call_{llm_calls:03d}",
                prompt_ready=True,
            )
            if not isinstance(llm_result, dict) or not isinstance(llm_result.get("ai_message"), dict):
                raise ValueError("B4 result must contain an ai_message object")
            ai_message = llm_result["ai_message"]
            llm_status = llm_result.get("status")
            llm_error = llm_result.get("error")
            llm_prompt_messages = llm_result.get("prompt_messages")
        messages.append(ai_message)
        turn = {
            "turn_index": llm_calls,
            "ai_message": ai_message,
            "llm_prompt_messages": llm_prompt_messages,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
            "control": ai_message.get("control"),
            "agent_step": ai_message.get("agent_step"),
            "latency_ms": None,
        }
        if llm_status != "success":
            final_answer = ai_message.get("content", "").strip() or str(llm_error or "")
            status = "llm_parse_error"
            terminal_error = {
                "type": "LLMParseError",
                "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
                "llm_call_index": llm_calls,
                "cause": llm_error,
            }
            final_control = {
                "state": "failed",
                "action": "finish",
                "reason": "LLM output could not be parsed",
            }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break
        control = ai_message["control"]
        tool_calls = ai_message.get("tool_calls", [])
        if control["action"] == "finish":
            final_control = control
            final_answer = ai_message["content"]
            print(f"content: {final_answer}")
            if control["state"] == "failed":
                status = "agent_failed"
                terminal_error = {
                    "type": "AgentDeclaredFailure",
                    "message": control["reason"],
                    "llm_call_index": llm_calls,
                }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break
        if tool_rounds >= max_turns:
            final_answer = ai_message.get("content", "").strip() or f"已达到最大工具轮数 max_turns={max_turns}，停止继续调用工具。"
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
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break
        if execution_mode == "fixture":
            tool_messages = _fixture_tool_messages(
                tool_calls,
                fixture_data["tool_messages"],
            )
        else:
            from b3_tool_layer import execute_tool_calls

            tool_messages = execute_tool_calls(
                tool_calls,
                str(tools_file),
                runtime["toolset"],
                str(output_dir),
            )
        tool_rounds += 1
        messages.extend(tool_messages)
        all_tool_messages.extend(tool_messages)
        turn["tool_messages"] = tool_messages
        turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
        turns.append(turn)

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
    }
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
        append_jsonl(
            {
                "timestamp": now_iso(),
                "conversation_id": runtime["conversation_id"],
                "execution_mode": execution_mode,
                "status": trace["status"],
                "llm_mode": mode,
                "tool_rounds_used": tool_rounds,
                "llm_call_count": llm_calls,
                "elapsed_ms": result["elapsed_ms"],
            },
            output_dir / "runtime_log.jsonl",
        )
    return result


def run_legacy_stream_loop(
    runtime: dict,
    execution_mode: str,
    system_prompt: str,
    selected_memory: dict,
    tools_schema: list[dict],
    fixture_data: dict | None,
    tools_file: Path | None,
    memory_file: Path | None,
    model_file: Path | None,
    mode: str,
    output_dir: Path,
    started: float,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[dict]:
    current_user_message = {"role": "user", "content": runtime["user_input"]}
    if runtime["input_images"]:
        current_user_message["images"] = runtime["input_images"]
    messages = [
        {"role": "system", "content": system_prompt},
        *runtime["history_messages"],
        current_user_message,
    ]
    tool_rounds = 0
    llm_calls = 0
    turns = []
    all_tool_messages = []
    final_answer = ""
    status = "success"
    terminal_error = None
    warnings = []
    final_control = None
    if selected_memory.get("status") in {"partial", "error"}:
        warnings.append("memory selection completed with errors")
    partial_chunks: list[str] = []
    max_turns = runtime["max_turns"]

    while True:
        if _cancel_requested(should_cancel):
            yield {
                "type": "done",
                "result": _legacy_cancelled_result(
                    runtime,
                    execution_mode,
                    mode,
                    output_dir,
                    started,
                    selected_memory,
                    messages,
                    all_tool_messages,
                    "".join(partial_chunks),
                    tool_rounds,
                    llm_calls,
                    turns,
                    warnings,
                ),
            }
            return
        llm_calls += 1
        turn_start = perf_counter()
        if execution_mode == "fixture":
            if llm_calls > len(fixture_data["ai_messages"]):
                raise ValueError("fixture AIMessage sequence ended before a final answer")
            ai_message = deepcopy(fixture_data["ai_messages"][llm_calls - 1])
            if ai_message.get("content"):
                yield {"type": "delta", "text": ai_message["content"], "llm_call_index": llm_calls}
            llm_status = "success"
            llm_error = None
            llm_prompt_messages = None
        else:
            llm_result = None
            llm_input_messages = build_llm_prompt_messages(messages, tools_schema)
            for event in stream_ai_message(
                str(model_file),
                llm_input_messages,
                [],
                mode,
                str(output_dir / "llm_calls"),
                f"llm_call_{llm_calls:03d}",
                prompt_ready=True,
            ):
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "delta":
                    partial_chunks.append(str(event.get("text", "")))
                    yield {
                        "type": "delta",
                        "text": str(event.get("text", "")),
                        "llm_call_index": llm_calls,
                    }
                elif event.get("type") == "done":
                    llm_result = event.get("result")
                if _cancel_requested(should_cancel):
                    yield {
                        "type": "done",
                        "result": _legacy_cancelled_result(
                            runtime,
                            execution_mode,
                            mode,
                            output_dir,
                            started,
                            selected_memory,
                            messages,
                            all_tool_messages,
                            "".join(partial_chunks),
                            tool_rounds,
                            llm_calls,
                            turns,
                            warnings,
                        ),
                    }
                    return
            if not isinstance(llm_result, dict) or not isinstance(llm_result.get("ai_message"), dict):
                raise ValueError("B4 stream result must contain an ai_message object")
            ai_message = llm_result["ai_message"]
            llm_status = llm_result.get("status")
            llm_error = llm_result.get("error")
            llm_prompt_messages = llm_result.get("prompt_messages")
        messages.append(ai_message)
        turn = {
            "turn_index": llm_calls,
            "ai_message": ai_message,
            "llm_prompt_messages": llm_prompt_messages,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
            "control": ai_message.get("control"),
            "agent_step": ai_message.get("agent_step"),
            "latency_ms": None,
        }
        if llm_status != "success":
            final_answer = ai_message.get("content", "").strip() or str(llm_error or "")
            status = "llm_parse_error"
            terminal_error = {
                "type": "LLMParseError",
                "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
                "llm_call_index": llm_calls,
                "cause": llm_error,
            }
            final_control = {
                "state": "failed",
                "action": "finish",
                "reason": "LLM output could not be parsed",
            }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            yield {"type": "state", **final_control, "llm_call_index": llm_calls}
            break
        control = ai_message["control"]
        yield {"type": "state", **control, "agent_step": ai_message.get("agent_step"), "llm_call_index": llm_calls}
        if _cancel_requested(should_cancel):
            yield {
                "type": "done",
                "result": _legacy_cancelled_result(
                    runtime,
                    execution_mode,
                    mode,
                    output_dir,
                    started,
                    selected_memory,
                    messages,
                    all_tool_messages,
                    "".join(partial_chunks),
                    tool_rounds,
                    llm_calls,
                    turns,
                    warnings,
                ),
            }
            return
        tool_calls = ai_message.get("tool_calls", [])
        if control["action"] == "finish":
            final_control = control
            final_answer = ai_message["content"]
            if control["state"] == "failed":
                status = "agent_failed"
                terminal_error = {
                    "type": "AgentDeclaredFailure",
                    "message": control["reason"],
                    "llm_call_index": llm_calls,
                }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break
        if tool_rounds >= max_turns:
            final_answer = ai_message.get("content", "").strip() or f"已达到最大工具轮数 max_turns={max_turns}，停止继续调用工具。"
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
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            yield {"type": "state", **final_control, "llm_call_index": llm_calls}
            break
        yield {
            "type": "tool_start",
            "tool_calls": tool_calls,
            "assistant_content": ai_message.get("content", ""),
            "agent_step": ai_message.get("agent_step"),
            "llm_call_index": llm_calls,
        }
        if _cancel_requested(should_cancel):
            yield {
                "type": "done",
                "result": _legacy_cancelled_result(
                    runtime,
                    execution_mode,
                    mode,
                    output_dir,
                    started,
                    selected_memory,
                    messages,
                    all_tool_messages,
                    "".join(partial_chunks),
                    tool_rounds,
                    llm_calls,
                    turns,
                    warnings,
                ),
            }
            return
        if execution_mode == "fixture":
            tool_messages = _fixture_tool_messages(
                tool_calls,
                fixture_data["tool_messages"],
            )
        else:
            from b3_tool_layer import execute_tool_calls

            tool_messages = execute_tool_calls(
                tool_calls,
                str(tools_file),
                runtime["toolset"],
                str(output_dir),
            )
        tool_rounds += 1
        messages.extend(tool_messages)
        all_tool_messages.extend(tool_messages)
        turn["tool_messages"] = tool_messages
        yield {"type": "tool_done", "tool_messages": tool_messages, "llm_call_index": llm_calls}
        turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
        turns.append(turn)

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
    }
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
        append_jsonl(
            {
                "timestamp": now_iso(),
                "conversation_id": runtime["conversation_id"],
                "execution_mode": execution_mode,
                "status": trace["status"],
                "llm_mode": mode,
                "tool_rounds_used": tool_rounds,
                "llm_call_count": llm_calls,
                "elapsed_ms": result["elapsed_ms"],
                "streaming": True,
            },
            output_dir / "runtime_log.jsonl",
        )
    yield {"type": "done", "result": result}
