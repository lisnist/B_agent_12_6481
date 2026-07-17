import base64
import binascii
import re
import shutil
from pathlib import Path

from fastapi import HTTPException

from backend.api_models import UploadFilePayload, UploadRequest, UploadedFileRef
from backend.settings import (
    IMAGE_MIME_TYPES,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    PROJECT_ROOT,
    SUPPORTED_UPLOAD_SUFFIXES,
    UPLOAD_ROOT,
)


def safe_upload_filename(name: str, fallback: str) -> str:
    original = Path(name.strip()).name
    suffix = Path(original).suffix.lower()
    stem = Path(original).stem
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" ._")
    safe_suffix = re.sub(r"[^A-Za-z0-9.]+", "", suffix)
    if safe_suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"unsupported upload file type: {suffix or '(none)'}")
    return f"{safe_stem or fallback}{safe_suffix}"


def _unique_child_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    path = Path(filename)
    for index in range(1, 1000):
        candidate = directory / f"{path.stem}_{index}{path.suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=409, detail="too many uploaded files with the same name")


def _decode_upload_file(payload: UploadFilePayload) -> bytes:
    try:
        data = base64.b64decode(payload.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 content for {payload.name}") from exc
    if payload.size is not None and payload.size != len(data):
        raise HTTPException(status_code=400, detail=f"upload size mismatch for {payload.name}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"uploaded file is too large: {payload.name}")
    return data


def save_uploaded_files(request: UploadRequest) -> list[UploadedFileRef]:
    if not request.files:
        raise HTTPException(status_code=400, detail="files are required")
    if len(request.files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"at most {MAX_UPLOAD_FILES} files can be uploaded at once")
    if request.conversation_id is None:
        raise ValueError("conversation_id is required before saving uploads")
    conversation_id = request.conversation_id
    target_dir = UPLOAD_ROOT / conversation_id
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, file_payload in enumerate(request.files, 1):
        filename = safe_upload_filename(file_payload.name, f"uploaded_{index}")
        data = _decode_upload_file(file_payload)
        target = _unique_child_path(target_dir, filename)
        target.write_bytes(data)
        saved.append(UploadedFileRef(name=file_payload.name, path=f"uploads/{conversation_id}/{target.name}", size=len(data)))
    return saved


def save_run_uploads(conversation_id: str, files: list[UploadFilePayload]) -> list[UploadedFileRef]:
    if not files:
        return []
    return save_uploaded_files(UploadRequest(conversation_id=conversation_id, files=files))


def delete_child_directory(root: Path, child_name: str) -> bool:
    root = root.resolve()
    target = (root / child_name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="resolved delete path escaped allowed root") from exc
    if target == root:
        raise HTTPException(status_code=400, detail="refusing to delete root directory")
    if not target.exists():
        return False
    if not target.is_dir():
        raise HTTPException(status_code=409, detail=f"delete target is not a directory: {target.name}")
    shutil.rmtree(target)
    return True


def user_input_with_uploaded_files(user_input: str, files: list[UploadedFileRef]) -> str:
    readable_files = [file for file in files if Path(file.path).suffix.lower() not in IMAGE_MIME_TYPES]
    if not readable_files:
        return user_input
    lines = ["本次用户上传了以下文件。调用文件工具时，path 必须原样使用这里给出的规范路径："]
    lines.extend(f"- path: {file.path}" for file in readable_files)
    lines.extend([
        "如果当前用户输入中的“这份文档”“这个文件”“附件”“上传文件”等指代不明确，优先使用上述上传文件路径。",
        "",
        "当前用户输入：",
        user_input,
    ])
    return "\n".join(lines)


def uploaded_image_data_urls(files: list[UploadedFileRef]) -> list[str]:
    data_root = (PROJECT_ROOT / "data").resolve()
    images = []
    for file in files:
        suffix = Path(file.path).suffix.lower()
        mime_type = IMAGE_MIME_TYPES.get(suffix)
        if mime_type is None:
            continue
        source = (data_root / file.path).resolve()
        try:
            source.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"uploaded image path escapes data root: {file.path}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"uploaded image not found: {file.path}")
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        images.append(f"data:{mime_type};base64,{encoded}")
    return images
