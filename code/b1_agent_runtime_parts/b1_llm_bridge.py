from __future__ import annotations

from typing import Iterator


def generate_ai_message(*args, **kwargs) -> dict:
    """Lazy B4 proxy retained as the integrated-mode injection point."""
    from b4_local_agent_llm import generate_ai_message as b4_generate_ai_message

    return b4_generate_ai_message(*args, **kwargs)


def stream_ai_message(*args, **kwargs) -> Iterator[dict]:
    """Lazy B4 streaming proxy used only by the opt-in streaming runtime."""
    from b4_local_agent_llm import stream_ai_message as b4_stream_ai_message

    return b4_stream_ai_message(*args, **kwargs)


def generate_json_object(*args, **kwargs) -> dict:
    """Lazy B4 proxy for B1-owned workspace planning stages."""
    from b4_local_agent_llm import generate_json_object as b4_generate_json_object

    return b4_generate_json_object(*args, **kwargs)

