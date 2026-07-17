from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"

HOST = "127.0.0.1"
PORT = 8020
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "backend_runs"
UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
TOOLS_CONFIG = PROJECT_ROOT / "configs" / "tools.yaml"
MEMORY_CONFIG = PROJECT_ROOT / "configs" / "memory.yaml"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
RUNTIME_BASE = PROJECT_ROOT / "data" / "__frontend_runtime__.json"
SYSTEM_PROMPT_PATH = "../prompts/agent_system_prompts.json"
DEFAULT_SYSTEM_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "agent_system_prompts.json"
PROMPT_STORE_PATH = PROJECT_ROOT / "prompts" / "conversation_prompts.json"

MAX_UPLOAD_FILES = 5
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES = {
    ".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
    ".py", ".log", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif",
}
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
