from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import quote

from fastapi import HTTPException

from backend.ids import safe_conversation_id, safe_run_id
from backend.settings import OUTPUT_ROOT


def safe_generated_artifact_path(relative_path: str) -> Path:
    normalized = relative_path.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or path.parts[0] != "generated_files":
        raise HTTPException(status_code=400, detail="artifact path must stay inside generated_files")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise HTTPException(status_code=400, detail="artifact path contains unsupported segments")
    return Path(*path.parts)


def artifact_download_url(output_dir: Path, relative_output_path: object) -> str | None:
    if not isinstance(relative_output_path, str) or not relative_output_path.strip():
        return None
    try:
        artifact_path = safe_generated_artifact_path(relative_output_path)
        output_dir.resolve().relative_to(OUTPUT_ROOT.resolve())
    except (HTTPException, ValueError):
        return None
    conversation_id = output_dir.parent.name
    run_id = output_dir.name
    if not conversation_id or not run_id:
        return None
    encoded_path = "/".join(quote(part, safe="") for part in PurePosixPath(artifact_path.as_posix()).parts)
    return f"/api/artifacts/{quote(conversation_id, safe='')}/{quote(run_id, safe='')}/{encoded_path}"


def attach_artifact_download_urls(result: dict, output_dir: Path) -> None:
    output = result.get("output")
    if isinstance(output, dict):
        download_url = artifact_download_url(output_dir, output.get("relative_output_path"))
        if download_url:
            output.setdefault("download_url", download_url)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        download_url = artifact_download_url(output_dir, artifact.get("relative_output_path"))
        if download_url:
            artifact.setdefault("download_url", download_url)


def generated_artifact_target(conversation_id: str, run_id: str, relative_path: str) -> Path:
    safe_conversation = safe_conversation_id(conversation_id)
    safe_run = safe_run_id(run_id)
    artifact_path = safe_generated_artifact_path(relative_path)
    output_root = OUTPUT_ROOT.resolve()
    run_dir = (OUTPUT_ROOT / safe_conversation / safe_run).resolve()
    try:
        run_dir.relative_to(output_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="artifact run directory is outside output root") from exc
    target = (run_dir / artifact_path).resolve()
    try:
        target.relative_to(run_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="artifact path is outside run directory") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return target
