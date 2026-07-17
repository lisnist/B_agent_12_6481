from __future__ import annotations

import argparse
import sys

from common.io_utils import read_json
from common.path_utils import resolve_cli_path

from b5_memory_parts.conversation_api import (
    append_conversation_message,
    clear_message_tool_steps,
    delete_conversation_record,
    get_conversation_memory_snapshot,
    init_conversation_db,
    list_conversation_history,
    list_conversation_messages,
    list_conversation_records,
    list_conversation_tasks,
    list_message_tool_steps,
    record_conversation_tool_step,
    update_conversation_message,
    upsert_conversation_record,
)
from b5_memory_parts.legacy import load_memory, save_memory
from b5_memory_parts.reflection import record_completed_turn_memory
from b5_memory_parts.retrieval import build_layered_memory_context, prepare_workspace_memory_context

__all__ = [
    "append_conversation_message",
    "clear_message_tool_steps",
    "build_layered_memory_context",
    "build_parser",
    "delete_conversation_record",
    "get_conversation_memory_snapshot",
    "init_conversation_db",
    "list_conversation_history",
    "list_conversation_messages",
    "list_conversation_records",
    "list_conversation_tasks",
    "list_message_tool_steps",
    "load_memory",
    "main",
    "parse_bool",
    "prepare_workspace_memory_context",
    "record_completed_turn_memory",
    "record_conversation_tool_step",
    "save_memory",
    "update_conversation_message",
    "upsert_conversation_record",
]


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select or save local memory documents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--select_memory_ids", nargs="*")
    parser.add_argument("--use_global_memory", type=parse_bool)
    parser.add_argument("--query")
    parser.add_argument("--save_type", choices=["conversation", "global"])
    parser.add_argument("--save_input_path")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        if args.save_type or args.save_input_path:
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            base = input_path.parent
            save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
            )
            print(outdir / "saved_memory.json")
        else:
            if args.select_memory_ids is None and args.use_global_memory is None:
                raise ValueError("select mode requires --select_memory_ids or --use_global_memory")
            load_memory(
                str(config_path),
                args.select_memory_ids or [],
                bool(args.use_global_memory),
                args.query,
                str(outdir),
            )
            print(outdir / "selected_memory.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
