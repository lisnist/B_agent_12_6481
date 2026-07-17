from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from time import perf_counter

from common.io_utils import append_jsonl, read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import DEFAULT_DATA_ROOT, bootstrap_project_root, resolve_cli_path, resolve_from_file
from common.schemas import make_skill_result
from common.tool_config import DEFAULT_TOOLS_CONFIG, get_tool_definition, load_tool_function, load_tools_config


bootstrap_project_root()


def _workspace_root_settings(config: dict, config_path: Path, resolved_data_root: Path) -> tuple[dict[str, str], str]:
    settings = config.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    configured = settings.get("workspace_roots", {})
    roots: dict[str, str] = {"data": str(resolved_data_root)}
    if isinstance(configured, dict):
        for alias, raw_path in configured.items():
            if not isinstance(alias, str) or not isinstance(raw_path, str):
                continue
            normalized_alias = alias.strip()
            if not normalized_alias.replace("_", "").replace("-", "").isalnum():
                continue
            roots[normalized_alias] = str(resolve_from_file(raw_path, config_path))
    default_root = settings.get("default_workspace_root", "data")
    if not isinstance(default_root, str) or default_root not in roots:
        default_root = "data"
    return roots, default_root


def run_skill(
    skill_name: str,
    input_data: dict,
    data_root: str | None = None,
    output_dir: str | None = None,
    tools_config: str | Path | None = None,
) -> dict:
    if not isinstance(input_data, dict):
        raise ValueError("skill input must be a JSON object")
    config_path, config = load_tools_config(tools_config or DEFAULT_TOOLS_CONFIG)
    definition = get_tool_definition(config, skill_name)
    function = load_tool_function(definition)
    settings = config.get("settings") if isinstance(config.get("settings"), dict) else {}
    data_root_setting = data_root or settings.get("data_root", str(DEFAULT_DATA_ROOT))
    resolved_data_root = resolve_from_file(data_root_setting, config_path) if isinstance(data_root_setting, str) else DEFAULT_DATA_ROOT
    allowed_roots, default_root = _workspace_root_settings(config, config_path, resolved_data_root)
    kwargs = dict(input_data)
    signature = inspect.signature(function)
    if "data_root" in signature.parameters:
        kwargs["data_root"] = str(resolved_data_root)
    if "allowed_roots" in signature.parameters:
        kwargs["allowed_roots"] = allowed_roots
    if "default_root" in signature.parameters:
        kwargs["default_root"] = default_root
    if "output_dir" in signature.parameters:
        kwargs["output_dir"] = output_dir
    start = perf_counter()
    try:
        output = function(**kwargs)
        status = "success"
        error = None
    except Exception as exc:  # Skill exceptions are a structured business result.
        output = None
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)}
    latency_ms = round((perf_counter() - start) * 1000, 3)
    return make_skill_result(skill_name, status, input_data, output, error, latency_ms)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local Agent skill.")
    parser.add_argument("--skill", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--tools_config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = resolve_cli_path(args.input)
        outdir = resolve_cli_path(args.outdir)
        input_data = read_json(input_path)
        data_root = str(resolve_cli_path(args.data_root)) if args.data_root else None
        tools_config = resolve_cli_path(args.tools_config) if args.tools_config else DEFAULT_TOOLS_CONFIG
        outdir.mkdir(parents=True, exist_ok=True)
        result = run_skill(args.skill, input_data, data_root, str(outdir), tools_config)
        result_path = outdir / f"{args.skill}_result.json"
        write_json(result, result_path)
        append_jsonl(
            {
                "timestamp": now_iso(),
                "skill_name": args.skill,
                "status": result["status"],
                "result_path": str(result_path),
                "latency_ms": result["latency_ms"],
            },
            outdir / "skill_run_log.jsonl",
        )
        print(result_path)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
