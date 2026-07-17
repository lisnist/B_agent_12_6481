from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from common.io_utils import read_yaml
from common.path_utils import PROJECT_ROOT


DEFAULT_TOOLS_CONFIG = PROJECT_ROOT / "configs" / "tools.yaml"
INJECTED_PARAMETERS = {"data_root", "output_dir", "allowed_roots", "default_root"}


def load_tools_config(tools_config: str | Path | None = None) -> tuple[Path, dict]:
    config_path = Path(tools_config).resolve() if tools_config else DEFAULT_TOOLS_CONFIG.resolve()
    config = read_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("tools.yaml must contain an object")
    if not isinstance(config.get("tools"), dict) or not isinstance(config.get("toolsets"), dict):
        raise ValueError("tools.yaml must define tools and toolsets")
    return config_path, config


def resolve_toolset(config: dict, toolset: str | None) -> tuple[str, list[str]]:
    selected = toolset or config.get("default_toolset")
    if not isinstance(selected, str) or selected not in config["toolsets"]:
        raise ValueError(f"toolset does not exist: {selected}")
    names = config["toolsets"][selected]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"toolset {selected} must be a list of tool names")
    return selected, names


def get_tool_definition(config: dict, tool_name: str) -> dict:
    tools = config.get("tools")
    if not isinstance(tools, dict):
        raise ValueError("tools.yaml must define tools")
    definition = tools.get(tool_name)
    if not isinstance(definition, dict):
        raise ValueError(f"unknown skill: {tool_name}")
    for field in ("module", "function", "description", "returns"):
        if field not in definition:
            raise ValueError(f"tool {tool_name} missing {field}")
    return definition


def load_tool_function(tool: dict) -> Any:
    module_name = tool.get("module")
    function_name = tool.get("function")
    if not isinstance(module_name, str) or not isinstance(function_name, str):
        raise ValueError("tool definition must contain module and function strings")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)
