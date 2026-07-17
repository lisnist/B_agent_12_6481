from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

from common.io_utils import read_json
from common.path_utils import resolve_cli_path, resolve_from_file
from common.prompt_store import load_system_prompt_from_path

from b1_agent_runtime_parts.b1_fixture import _load_fixture_inputs
from b1_agent_runtime_parts.b1_legacy_loop import run_legacy_loop, run_legacy_stream_loop
from b1_agent_runtime_parts.b1_llm_bridge import (
    generate_ai_message,
    generate_json_object,
    stream_ai_message,
)
from b1_agent_runtime_parts.b1_prompting import build_llm_prompt_messages
from b1_agent_runtime_parts.b1_runtime_input import (
    _default_llm_mode,
    _runtime_base_file,
    _validate_runtime_input,
)
from b1_agent_runtime_parts.b1_workspace import _prepare_workspace_runtime_context
from b1_agent_runtime_parts.b1_workspace_loop import _run_workspace, _run_workspace_stream, resume_workspace_stream


def _runtime_system_prompt(runtime: dict, base_file: Path) -> str:
    direct = runtime.get("system_prompt")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    prompt_path = resolve_from_file(runtime["system_prompt_path"], base_file)
    return load_system_prompt_from_path(prompt_path)


def run(
    runtime_input: dict,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    outdir: str,
    llm_mode: str | None = None,
    runtime_base: str | Path | None = None,
) -> dict:
    """Run the Agent loop from an in-memory runtime payload.

    runtime_base is only used as the reference file for resolving relative paths
    inside the payload, such as system_prompt_path and fixture paths.
    """
    started = perf_counter()
    base_file = _runtime_base_file(runtime_base)
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _validate_runtime_input(deepcopy(runtime_input))
    print(f"user_input: {runtime['user_input']}")
    execution_mode = runtime["execution_mode"]
    system_prompt = _runtime_system_prompt(runtime, base_file)
    fixture_data = None
    tools_file = memory_file = model_file = None
    if execution_mode == "fixture":
        fixture_data = _load_fixture_inputs(base_file, runtime)
        selected_memory = fixture_data["selected_memory"]
        tools_schema = fixture_data["tools_schema"]
        mode = "fixture"
    else:
        if not tools_config or not memory_config or not model_config:
            raise ValueError("integrated mode requires tools_config, memory_config, and model_config")
        from b3_tool_layer import get_tools_schema
        from b5_memory import load_memory

        tools_file = Path(tools_config).resolve()
        memory_file = Path(memory_config).resolve()
        model_file = Path(model_config).resolve()
        selected_memory = load_memory(
            str(memory_file),
            runtime["selected_memory_ids"],
            runtime["use_global_memory"],
            runtime["user_input"],
            str(output_dir),
        )
        tools_schema = get_tools_schema(str(tools_file), runtime["toolset"], str(output_dir))
        mode = llm_mode or _default_llm_mode(model_file)
        if mode == "prompt_json":
            runtime, _ = _prepare_workspace_runtime_context(
                runtime,
                memory_file,
                selected_memory,
                output_dir,
                model_file,
                mode,
            )
    if execution_mode == "integrated" and mode == "prompt_json":
        return _run_workspace(
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
        )
    return run_legacy_loop(
        runtime,
        execution_mode,
        system_prompt,
        selected_memory,
        tools_schema,
        fixture_data,
        tools_file,
        memory_file,
        model_file,
        mode,
        output_dir,
        started,
    )


def run_stream(
    runtime_input: dict,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    outdir: str,
    llm_mode: str | None = None,
    runtime_base: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[dict]:
    """Run the Agent loop and yield UI-safe streaming events.

    This is an additive entry point. The existing run()/run_agent() path remains
    the stable non-streaming module interface.
    """
    started = perf_counter()
    base_file = _runtime_base_file(runtime_base)
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _validate_runtime_input(deepcopy(runtime_input))
    execution_mode = runtime["execution_mode"]
    system_prompt = _runtime_system_prompt(runtime, base_file)
    fixture_data = None
    tools_file = memory_file = model_file = None
    if execution_mode == "fixture":
        fixture_data = _load_fixture_inputs(base_file, runtime)
        selected_memory = fixture_data["selected_memory"]
        tools_schema = fixture_data["tools_schema"]
        mode = "fixture"
    else:
        if not tools_config or not memory_config or not model_config:
            raise ValueError("integrated mode requires tools_config, memory_config, and model_config")
        from b3_tool_layer import get_tools_schema
        from b5_memory import load_memory

        tools_file = Path(tools_config).resolve()
        memory_file = Path(memory_config).resolve()
        model_file = Path(model_config).resolve()
        selected_memory = load_memory(
            str(memory_file),
            runtime["selected_memory_ids"],
            runtime["use_global_memory"],
            runtime["user_input"],
            str(output_dir),
        )
        tools_schema = get_tools_schema(str(tools_file), runtime["toolset"], str(output_dir))
        mode = llm_mode or _default_llm_mode(model_file)
        if mode == "prompt_json":
            runtime, _ = _prepare_workspace_runtime_context(
                runtime,
                memory_file,
                selected_memory,
                output_dir,
                model_file,
                mode,
            )
    if execution_mode == "integrated" and mode == "prompt_json":
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
    yield from run_legacy_stream_loop(
        runtime,
        execution_mode,
        system_prompt,
        selected_memory,
        tools_schema,
        fixture_data,
        tools_file,
        memory_file,
        model_file,
        mode,
        output_dir,
        started,
        should_cancel,
    )


def resume_stream(
    conversation_id: str,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[dict]:
    """Resume a paused workspace run from checkpoints/<conversation_id>.json."""
    yield from resume_workspace_stream(conversation_id, should_cancel)


def run_agent(
    input_path: str,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    outdir: str,
    llm_mode: str | None = None,
) -> dict:
    input_file = Path(input_path).resolve()
    return run(
        read_json(input_file),
        tools_config,
        memory_config,
        model_config,
        outdir,
        llm_mode,
        input_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Agent message and tool loop.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tools_config")
    parser.add_argument("--memory_config")
    parser.add_argument("--model_config")
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json"], default=None)
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_agent(
            str(resolve_cli_path(args.input)),
            str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
            str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
            str(resolve_cli_path(args.model_config)) if args.model_config else None,
            str(resolve_cli_path(args.outdir)),
            args.llm_mode,
        )
        print(result["final_answer_path"])
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
