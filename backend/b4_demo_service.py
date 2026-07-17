from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from backend.ids import now_stamp, safe_conversation_id
from backend.settings import MODEL_CONFIG, OUTPUT_ROOT, TOOLS_CONFIG


B4_TEST_CASES = [
    {
        "id": "content_response",
        "title": "普通内容回复",
        "kind": "model",
        "level": "基础要求",
        "description": "验证模型原始输出能够被解析为只包含 content 的标准 AIMessage。",
        "expected": "content 非空，tool_calls 为空，control.action=finish",
    },
    {
        "id": "single_tool_call",
        "title": "单工具调用",
        "kind": "model",
        "level": "基础要求",
        "description": "向 B4 传入 calculator schema，验证模型生成一个标准 tool_call。",
        "expected": "生成包含 id、name、args 的 calculator tool_call，并进入 call_tools",
    },
    {
        "id": "multiple_tool_calls",
        "title": "单轮多个工具调用",
        "kind": "model",
        "level": "进阶要求",
        "description": "同时提供 calculator 与 current_time schema，验证单个 AIMessage 携带多个 tool_calls。",
        "expected": "同一 AIMessage 同时包含 calculator 和 current_time",
    },
    {
        "id": "multiple_tool_messages",
        "title": "接收多个 ToolMessage",
        "kind": "model",
        "level": "进阶要求",
        "description": "一次传入两个已完成的 ToolMessage，验证 B4 生成基于全部结果的最终回复。",
        "expected": "content 非空，tool_calls 为空，control.action=finish",
    },
    {
        "id": "error_tool_message",
        "title": "工具错误后收束",
        "kind": "model",
        "level": "基础要求",
        "description": "传入执行失败的 ToolMessage，验证 B4 能基于错误证据生成最终说明。",
        "expected": "不重复调用工具，生成包含错误说明的最终 content",
    },
    {
        "id": "stream_response",
        "title": "流式输出与最终解析",
        "kind": "stream",
        "level": "工程增强",
        "description": "记录流式 delta，并在流结束后验证完整 AIMessage。",
        "expected": "至少产生一个 delta，最终 AIMessage 协议有效",
    },
    {
        "id": "content_with_tool_call",
        "title": "说明与工具调用共存",
        "kind": "parser",
        "level": "工程增强",
        "description": "回放同时包含简短 content 和 tool_calls 的输出，验证当前协议允许二者共存。",
        "expected": "保留 content 与 tool_call，并规范化为 call_tools",
    },
    {
        "id": "normalize_parameters_alias",
        "title": "工具参数别名归一化",
        "kind": "parser",
        "level": "工程增强",
        "description": "回放模型使用 parameters 表示工具参数的输出，验证 B4 将其归一化为标准 args。",
        "expected": "calculator 的 parameters.expression 被保留到 args.expression",
    },
    {
        "id": "recover_trailing_markers",
        "title": "可恢复格式偏差",
        "kind": "parser",
        "level": "工程增强",
        "description": "回放带多余 Markdown 标记的模型输出，验证现有容错解析路径。",
        "expected": "解析成功并生成标准 content AIMessage",
    },
    {
        "id": "reject_empty_message",
        "title": "拒绝无效 AIMessage",
        "kind": "parser",
        "level": "基础要求",
        "description": "输入 content 与 tool_calls 都为空的对象，验证协议校验明确拒绝无效消息。",
        "expected": "解析失败，返回 AIMessage must contain content or tool_calls",
    },
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _model_info() -> dict[str, Any]:
    from common.io_utils import read_yaml

    config = read_yaml(MODEL_CONFIG)
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    source = runtime.get("llm_source", "local") if isinstance(runtime, dict) else "local"
    source_config = config.get("qwen_api" if source == "qwen_api" else "fastapi", {}) if isinstance(config, dict) else {}
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    tool_calling = config.get("tool_calling", {}) if isinstance(config, dict) else {}
    if source == "local":
        model = model_config.get("model_name_or_path") if isinstance(model_config, dict) else None
        endpoint = None
    else:
        model = source_config.get("model") if isinstance(source_config, dict) else None
        endpoint = source_config.get("base_url") if isinstance(source_config, dict) else None
    return {
        "source": source,
        "model": model,
        "endpoint": endpoint,
        "mode": runtime.get("default_mode", "prompt_json") if isinstance(runtime, dict) else "prompt_json",
        "tool_binding": tool_calling.get("mode", "prompt_json") if isinstance(tool_calling, dict) else "prompt_json",
        "config_path": MODEL_CONFIG.resolve().relative_to(MODEL_CONFIG.parents[1].resolve()).as_posix(),
        "available_sources": ["local", "fastapi", "qwen_api"],
    }


def _call_stage(path: Path) -> str:
    stem = path.name.removesuffix("_raw_model_output.json")
    for stage in ("failure_answering", "tool_calling", "planning", "observation", "answering", "memory_reflection"):
        if stage in stem:
            return stage
    return stem


def _call_scope(path: Path) -> str:
    if "memory_reflection" in path.parts or path.name.startswith("b5_memory_"):
        return "memory_support"
    if "llm_calls" in path.parts:
        return "agent_runtime"
    return "standalone"


def _call_summary(path: Path) -> dict[str, Any]:
    record = _read_json(path)
    prompts = record.get("prompt_messages") if isinstance(record, dict) else []
    prompts = prompts if isinstance(prompts, list) else []
    raw_text = record.get("raw_text", "") if isinstance(record, dict) else ""
    relative = path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()
    kind = "json_object" if isinstance(record, dict) and "parsed_json" in record else "ai_message"
    return {
        "id": relative,
        "stage": _call_stage(path),
        "scope": _call_scope(path),
        "kind": kind,
        "status": record.get("status", "unknown") if isinstance(record, dict) else "unknown",
        "source": record.get("llm_source", record.get("backend", "unknown")) if isinstance(record, dict) else "unknown",
        "mode": record.get("mode", "unknown") if isinstance(record, dict) else "unknown",
        "generated_at": record.get("generated_at") if isinstance(record, dict) else None,
        "message_count": len(prompts),
        "roles": [str(item.get("role", "unknown")) for item in prompts if isinstance(item, dict)],
        "raw_chars": len(raw_text) if isinstance(raw_text, str) else 0,
        "run_id": path.parents[1].name if path.parent.name in {"llm_calls", "memory_reflection"} else path.parent.name,
    }


def list_b4_calls(conversation_id: str | None, limit: int = 60) -> dict[str, Any]:
    root = OUTPUT_ROOT
    safe_id = None
    if conversation_id:
        safe_id = safe_conversation_id(conversation_id)
        root = OUTPUT_ROOT / safe_id
    files = list(root.rglob("*_raw_model_output.json")) if root.exists() else []
    files = [path for path in files if "b4_demo" not in path.parts]
    dated_files: list[tuple[float, Path]] = []
    for path in files:
        try:
            dated_files.append((path.stat().st_mtime, path))
        except OSError:
            continue
    dated_files.sort(key=lambda item: item[0], reverse=True)
    calls = []
    for _, path in dated_files[: max(1, min(limit, 200))]:
        try:
            calls.append(_call_summary(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {
        "status": "success",
        "module": "B4",
        "conversation_id": safe_id,
        "model": _model_info(),
        "calls": calls,
    }


def get_b4_call_detail(call_id: str) -> dict[str, Any]:
    target = (OUTPUT_ROOT / call_id).resolve()
    try:
        target.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("call_id escapes output root") from exc
    if not target.is_file() or not target.name.endswith("_raw_model_output.json"):
        raise FileNotFoundError("B4 call artifact not found")
    record = _read_json(target)
    parsed_path = target.with_name(target.name.replace("_raw_model_output.json", "_ai_message.json"))
    standard_output = _read_json(parsed_path) if parsed_path.is_file() else None
    return {
        "status": "success",
        "module": "B4",
        "call": _call_summary(target),
        "record": record,
        "standard_output": standard_output,
    }


def protocol_test_cases() -> dict[str, Any]:
    return {
        "status": "success",
        "module": "B4",
        "model": _model_info(),
        "cases": B4_TEST_CASES,
    }


def _tool_schemas(names: set[str]) -> list[dict]:
    from b3_tool_layer import get_tools_schema

    schemas = get_tools_schema(str(TOOLS_CONFIG), "basic_tools", None)
    return [
        schema
        for schema in schemas
        if isinstance(schema, dict)
        and isinstance(schema.get("function"), dict)
        and schema["function"].get("name") in names
    ]


def _skill_result(
    name: str,
    input_data: dict,
    output: dict | None,
    *,
    status: str = "success",
    error: dict | None = None,
) -> str:
    return json.dumps(
        {
            "skill_name": name,
            "status": status,
            "input": input_data,
            "output": output,
            "error": error,
            "latency_ms": 1.0,
        },
        ensure_ascii=False,
    )


def _model_case(case_id: str) -> tuple[list[dict], list[dict], bool]:
    if case_id == "content_response":
        return (
            [
                {
                    "role": "user",
                    "content": "在本 Agent 系统中，B4 只负责模型通信、tools_schema 注入和 AIMessage 解析，不执行工具。请用一句简短中文准确复述，不要调用工具。",
                }
            ],
            [],
            False,
        )
    if case_id == "single_tool_call":
        return ([{"role": "user", "content": "计算 (18 + 24) * 3，只请求调用 calculator，不要直接计算。"}], _tool_schemas({"calculator"}), False)
    if case_id == "multiple_tool_calls":
        return (
            [{"role": "user", "content": "请在同一个回复中同时请求：计算 7 * 13，并获取 Asia/Shanghai 当前时间。这两个任务互不依赖。"}],
            _tool_schemas({"calculator", "current_time"}),
            False,
        )
    if case_id == "multiple_tool_messages":
        tool_calls = [
            {"id": "call_calc_demo", "name": "calculator", "args": {"expression": "7 * 13"}},
            {"id": "call_time_demo", "name": "current_time", "args": {"timezone": "Asia/Shanghai"}},
        ]
        return (
            [
                {"role": "user", "content": "计算 7 * 13，并告诉我上海当前时间。"},
                {"role": "assistant", "content": "", "tool_calls": tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": "call_calc_demo",
                    "name": "calculator",
                    "status": "success",
                    "content": _skill_result("calculator", {"expression": "7 * 13"}, {"result": 91}),
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_time_demo",
                    "name": "current_time",
                    "status": "success",
                    "content": _skill_result(
                        "current_time",
                        {"timezone": "Asia/Shanghai"},
                        {"timezone": "Asia/Shanghai", "iso": "2026-07-14T12:00:00+08:00"},
                    ),
                },
            ],
            _tool_schemas({"calculator", "current_time"}),
            False,
        )
    if case_id == "error_tool_message":
        tool_call = {"id": "call_error_demo", "name": "calculator", "args": {"expression": "10 / 0"}}
        return (
            [
                {"role": "user", "content": "计算 10 / 0；如果工具报错，请直接说明原因，不要再次调用工具。"},
                {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": "call_error_demo",
                    "name": "calculator",
                    "status": "error",
                    "content": _skill_result(
                        "calculator",
                        {"expression": "10 / 0"},
                        None,
                        status="error",
                        error={"type": "ZeroDivisionError", "message": "division by zero"},
                    ),
                },
            ],
            _tool_schemas({"calculator"}),
            False,
        )
    if case_id == "stream_response":
        return ([{"role": "user", "content": "请用一句简短中文说明流式输出的作用，不要调用工具。"}], [], False)
    raise ValueError(f"unsupported model test case: {case_id}")


def _evaluate(case_id: str, result: dict[str, Any], deltas: list[str]) -> tuple[bool, str]:
    ai_message = result.get("ai_message") if isinstance(result, dict) else None
    if not isinstance(ai_message, dict):
        return False, "未返回标准 AIMessage"
    content = ai_message.get("content")
    tool_calls = ai_message.get("tool_calls")
    control = ai_message.get("control") if isinstance(ai_message.get("control"), dict) else {}
    result_ok = result.get("status") == "success"
    names = {call.get("name") for call in tool_calls if isinstance(call, dict)} if isinstance(tool_calls, list) else set()
    calls_by_name = {
        str(call.get("name")): call
        for call in tool_calls
        if isinstance(call, dict) and isinstance(call.get("name"), str)
    } if isinstance(tool_calls, list) else {}
    if case_id == "content_response":
        semantic_ok = (
            isinstance(content, str)
            and ("模型" in content or "LLM" in content)
            and ("AIMessage" in content or "AI Message" in content or "AI 消息" in content)
        )
        passed = result_ok and semantic_ok and tool_calls == [] and control.get("action") == "finish"
        return passed, "已生成职责准确的最终 content" if passed else "最终 content 的协议或 B4 职责表述不符合预期"
    if case_id in {"multiple_tool_messages", "error_tool_message"}:
        passed = result_ok and isinstance(content, str) and bool(content.strip()) and tool_calls == [] and control.get("action") == "finish"
        return passed, "已生成最终 content" if passed else "未形成最终 content AIMessage"
    if case_id == "single_tool_call":
        calculator = calls_by_name.get("calculator", {})
        args = calculator.get("args") if isinstance(calculator, dict) else None
        has_id = isinstance(calculator.get("id"), str) and bool(calculator["id"].strip()) if isinstance(calculator, dict) else False
        has_expression = isinstance(args, dict) and isinstance(args.get("expression"), str) and bool(args["expression"].strip())
        passed = result_ok and has_id and has_expression and control.get("action") == "call_tools"
        return passed, "已生成结构完整的 calculator tool_call" if passed else "calculator tool_call 的 id、args.expression 或 control.action 不完整"
    if case_id == "multiple_tool_calls":
        calculator = calls_by_name.get("calculator", {})
        calculator_args = calculator.get("args") if isinstance(calculator, dict) else None
        has_expression = isinstance(calculator_args, dict) and isinstance(calculator_args.get("expression"), str) and bool(calculator_args["expression"].strip())
        current_time = calls_by_name.get("current_time", {})
        current_time_args = current_time.get("args") if isinstance(current_time, dict) else None
        has_timezone = isinstance(current_time_args, dict) and isinstance(current_time_args.get("timezone"), str) and bool(current_time_args["timezone"].strip())
        passed = result_ok and {"calculator", "current_time"}.issubset(names) and has_expression and has_timezone and control.get("action") == "call_tools"
        return passed, "同轮生成两个带有效参数的独立 tool_calls" if passed else f"实际工具：{', '.join(sorted(str(name) for name in names)) or '无'}；calculator 参数有效={has_expression}；current_time 参数有效={has_timezone}"
    if case_id == "stream_response":
        passed = result_ok and bool(deltas) and isinstance(content, str) and bool(content.strip()) and control.get("action") == "finish"
        return passed, f"收到 {len(deltas)} 个 delta" if passed else "未同时获得 delta 与最终 AIMessage"
    return False, "未知测试类型"


def _run_parser_case(case_id: str) -> dict[str, Any]:
    from b4_local_agent_llm import parse_model_output

    raw_inputs = {
        "content_with_tool_call": json.dumps(
            {
                "content": "我会先调用计算器核对结果。",
                "tool_calls": [
                    {"id": "call_parser_demo", "name": "calculator", "args": {"expression": "2 + 3"}}
                ],
                "control": {"state": "in_progress", "action": "call_tools", "reason": "需要准确计算"},
            },
            ensure_ascii=False,
        ),
        "normalize_parameters_alias": json.dumps(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_parameters_demo",
                        "name": "calculator",
                        "parameters": {"expression": "8 * 9"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        "recover_trailing_markers": '{"content":"协议修复成功。","tool_calls":[]}```',
        "reject_empty_message": '{"content":"","tool_calls":[]}',
    }
    raw_text = raw_inputs[case_id]
    started = time.perf_counter()
    try:
        parsed = parse_model_output(raw_text)
        error = None
    except Exception as exc:
        parsed = None
        error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    if case_id == "content_with_tool_call":
        ai_message = parsed.get("ai_message") if isinstance(parsed, dict) else None
        tool_calls = ai_message.get("tool_calls") if isinstance(ai_message, dict) else None
        control = ai_message.get("control") if isinstance(ai_message, dict) else None
        passed = (
            isinstance(ai_message, dict)
            and bool(str(ai_message.get("content", "")).strip())
            and isinstance(tool_calls, list)
            and len(tool_calls) == 1
            and isinstance(control, dict)
            and control.get("action") == "call_tools"
        )
        verdict = "content 与 tool_call 均已保留" if passed else "共存协议解析失败"
    elif case_id == "normalize_parameters_alias":
        ai_message = parsed.get("ai_message") if isinstance(parsed, dict) else None
        tool_calls = ai_message.get("tool_calls") if isinstance(ai_message, dict) else None
        first_call = tool_calls[0] if isinstance(tool_calls, list) and tool_calls else None
        args = first_call.get("args") if isinstance(first_call, dict) else None
        passed = isinstance(args, dict) and args.get("expression") == "8 * 9"
        verdict = "parameters 已归一化为标准 args" if passed else "工具参数别名归一化失败"
    elif case_id == "recover_trailing_markers":
        passed = isinstance(parsed, dict) and isinstance(parsed.get("ai_message"), dict)
        verdict = "容错解析成功" if passed else "容错解析失败"
    else:
        passed = bool(error and "must contain content or tool_calls" in error.get("message", ""))
        verdict = "无效 AIMessage 已被拒绝" if passed else "未按预期拒绝无效 AIMessage"
    return {
        "case_id": case_id,
        "test_status": "passed" if passed else "failed",
        "verdict": verdict,
        "elapsed_ms": elapsed_ms,
        "request": {"raw_text": raw_text},
        "raw_text": raw_text,
        "prompt_messages": [],
        "ai_message": parsed.get("ai_message") if isinstance(parsed, dict) else None,
        "parsed_candidate": parsed.get("parsed_candidate") if isinstance(parsed, dict) else None,
        "error": error,
        "stream": {"delta_count": 0, "deltas": []},
    }


def _run_model_case(case_id: str, output_dir: Path) -> dict[str, Any]:
    from b4_local_agent_llm import generate_ai_message, stream_ai_message

    messages, schemas, prompt_ready = _model_case(case_id)
    started = time.perf_counter()
    deltas: list[str] = []
    if case_id == "stream_response":
        final_result = None
        for event in stream_ai_message(
            str(MODEL_CONFIG),
            messages,
            schemas,
            mode="prompt_json",
            artifact_dir=str(output_dir),
            artifact_stem=case_id,
            prompt_ready=prompt_ready,
        ):
            if event.get("type") == "delta" and isinstance(event.get("text"), str):
                deltas.append(event["text"])
            elif event.get("type") == "done" and isinstance(event.get("result"), dict):
                final_result = event["result"]
        result = final_result or {"status": "error", "error": {"message": "stream ended without done"}}
    else:
        result = generate_ai_message(
            str(MODEL_CONFIG),
            messages,
            schemas,
            mode="prompt_json",
            artifact_dir=str(output_dir),
            artifact_stem=case_id,
            prompt_ready=prompt_ready,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    passed, verdict = _evaluate(case_id, result, deltas)
    return {
        "case_id": case_id,
        "test_status": "passed" if passed else "failed",
        "verdict": verdict,
        "elapsed_ms": elapsed_ms,
        "request": {
            "messages": messages,
            "tools_schema": schemas,
            "streaming": case_id == "stream_response",
        },
        "raw_text": result.get("raw_text", ""),
        "prompt_messages": result.get("prompt_messages") or [],
        "ai_message": result.get("ai_message"),
        "parsed_candidate": None,
        "error": result.get("error"),
        "stream": {"delta_count": len(deltas), "deltas": deltas},
    }


def run_b4_protocol_tests(case_id: str) -> dict[str, Any]:
    known = {item["id"] for item in B4_TEST_CASES}
    selected = list(known) if case_id == "all" else [case_id]
    if any(item not in known for item in selected):
        raise ValueError(f"unknown B4 protocol test case: {case_id}")
    ordered = [item["id"] for item in B4_TEST_CASES if item["id"] in selected]
    run_id = now_stamp()
    output_dir = OUTPUT_ROOT / "b4_demo" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in ordered:
        if item in {
            "content_with_tool_call",
            "normalize_parameters_alias",
            "recover_trailing_markers",
            "reject_empty_message",
        }:
            result = _run_parser_case(item)
        else:
            try:
                result = _run_model_case(item, output_dir)
            except Exception as exc:
                result = {
                    "case_id": item,
                    "test_status": "failed",
                    "verdict": "模型调用失败",
                    "elapsed_ms": 0,
                    "request": {},
                    "raw_text": "",
                    "prompt_messages": [],
                    "ai_message": None,
                    "parsed_candidate": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "stream": {"delta_count": 0, "deltas": []},
                }
        results.append(result)
    response = {
        "status": "success",
        "module": "B4",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "model": _model_info(),
        "summary": {
            "total": len(results),
            "passed": sum(item["test_status"] == "passed" for item in results),
            "failed": sum(item["test_status"] == "failed" for item in results),
        },
        "results": results,
    }
    _write_json(output_dir / "b4_protocol_test_result.json", response)
    return response
