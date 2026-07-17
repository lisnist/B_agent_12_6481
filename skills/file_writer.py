from __future__ import annotations

import html
import csv
import re
import zipfile
import json
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "file_writer_files"
GENERATED_DIR_NAME = "generated_files"

TEXT_FILE_TYPES = {
    "txt": {".txt"},
    "markdown": {".md"},
}
TABLE_SUFFIXES = {".csv", ".tsv"}
CODE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".sql",
    ".sh",
    ".ps1",
}
SUPPORTED_FILE_TYPES = {"txt", "markdown", "docx", "code", "json", "table"}
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
INVALID_CONTENT_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

DOCUMENT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""


def _safe_base_output_dir(output_dir: str | None) -> Path:
    base = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    target = (base / GENERATED_DIR_NAME).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("generated file directory escapes output_dir") from exc
    target.mkdir(parents=True, exist_ok=True)
    return target


def _validate_filename(filename: str) -> Path:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    raw = filename.strip().replace("\\", "/")
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute() or windows_path.drive or raw.startswith("/"):
        raise ValueError("filename must be a relative path")
    if any(segment == "" for segment in raw.split("/")):
        raise ValueError("filename must not contain empty path segments")
    if raw.endswith("/") or raw in {".", ".."}:
        raise ValueError("filename must include a file name")
    path = Path(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("filename must not contain empty, current, or parent path segments")
    for part in path.parts:
        if INVALID_FILENAME_CHARS.search(part):
            raise ValueError("filename contains characters that are not allowed in generated files")
        if part.rstrip(" .") != part:
            raise ValueError("filename path segments must not end with spaces or dots")
        if part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
            raise ValueError(f"filename uses a reserved Windows name: {part}")
    return path


def _normalize_file_type(file_type: str) -> str:
    if not isinstance(file_type, str):
        raise ValueError("file_type must be a string")
    normalized = file_type.strip().lower()
    if normalized not in SUPPORTED_FILE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_FILE_TYPES))
        raise ValueError(f"file_type must be one of: {supported}")
    return normalized


def _validate_content(content: str) -> str:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    if INVALID_CONTENT_CHARS.search(content):
        raise ValueError("content contains unsupported control characters")
    return content


def _validate_suffix(path: Path, file_type: str) -> str:
    suffix = path.suffix.lower()
    if file_type in TEXT_FILE_TYPES:
        allowed = TEXT_FILE_TYPES[file_type]
        if suffix not in allowed:
            raise ValueError(f"filename suffix for file_type={file_type} must be: {', '.join(sorted(allowed))}")
        return suffix
    if file_type == "docx":
        if suffix != ".docx":
            raise ValueError("filename suffix for file_type=docx must be .docx")
        return suffix
    if file_type == "json":
        if suffix != ".json":
            raise ValueError("filename suffix for file_type=json must be .json")
        return suffix
    if file_type == "table":
        if suffix not in TABLE_SUFFIXES:
            raise ValueError("filename suffix for file_type=table must be .csv or .tsv")
        return suffix
    if file_type == "code":
        if suffix not in CODE_SUFFIXES:
            raise ValueError("filename suffix for file_type=code is not in the supported code suffix whitelist")
        return suffix
    raise ValueError(f"unsupported file_type: {file_type}")


def _unique_output_path(base_dir: Path, relative_path: Path) -> Path:
    candidate = (base_dir / relative_path).resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError("filename escapes generated file directory") from exc
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    parent = candidate.parent
    index = 1
    while True:
        next_candidate = parent / f"{stem}({index}){suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def _write_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _docx_paragraph_xml(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return f'<w:p><w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>'


def _document_xml(content: str) -> str:
    lines = content.splitlines() or [content]
    body = "".join(_docx_paragraph_xml(line) for line in lines)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body>"
        "</w:document>"
    )


def _write_docx_file(path: Path, content: str) -> None:
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
            archive.writestr("_rels/.rels", ROOT_RELS_XML)
            archive.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS_XML)
            archive.writestr("word/document.xml", _document_xml(content))
    except Exception as exc:
        raise RuntimeError(f"failed to generate docx file: {exc}") from exc


def _write_table_file(path: Path, columns: list[str], rows: list) -> None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow([row.get(column, "") for column in columns])
            elif isinstance(row, list):
                writer.writerow(row)
            else:
                raise ValueError("table rows must contain objects or arrays")


def _write_generated_file(filename: str, file_type: str, content: str, output_dir: str | None = None) -> dict:
    normalized_type = _normalize_file_type(file_type)
    validated_content = _validate_content(content)
    relative_path = _validate_filename(filename)
    suffix = _validate_suffix(relative_path, normalized_type)
    base_dir = _safe_base_output_dir(output_dir)
    target = _unique_output_path(base_dir, relative_path)
    if normalized_type == "docx":
        _write_docx_file(target, validated_content)
    else:
        _write_text_file(target, validated_content)
    return _file_result(target, output_dir, normalized_type, suffix, len(validated_content))


def _file_result(target: Path, output_dir: str | None, file_type: str, suffix: str, num_chars: int) -> dict:
    output_base = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    return {
        "generated_file_path": str(target),
        "relative_output_path": target.relative_to(output_base).as_posix(),
        "filename": target.name,
        "file_type": file_type,
        "suffix": suffix,
        "num_chars": num_chars,
        "num_bytes": target.stat().st_size,
        "overwritten": False,
    }


def text_file_writer(filename: str, content: str, output_dir: str | None = None) -> dict:
    return _write_generated_file(filename, "txt", content, output_dir)


def markdown_file_writer(filename: str, content: str, output_dir: str | None = None) -> dict:
    return _write_generated_file(filename, "markdown", content, output_dir)


def code_file_writer(
    filename: str,
    content: str,
    language: str | None = None,
    output_dir: str | None = None,
) -> dict:
    result = _write_generated_file(filename, "code", content, output_dir)
    if isinstance(language, str) and language.strip():
        result["language"] = language.strip()
    return result


def docx_writer(filename: str, content: str, output_dir: str | None = None) -> dict:
    return _write_generated_file(filename, "docx", content, output_dir)


def json_file_writer(filename: str, data: dict, output_dir: str | None = None) -> dict:
    if not isinstance(data, dict):
        raise ValueError("data must be a JSON object")
    content = json_dumps(data)
    return _write_generated_file(filename, "json", content, output_dir)


def table_file_writer(
    filename: str,
    columns: list,
    rows: list,
    output_dir: str | None = None,
) -> dict:
    if not isinstance(columns, list) or not columns or not all(isinstance(column, str) and column for column in columns):
        raise ValueError("columns must be a non-empty array of strings")
    if not isinstance(rows, list):
        raise ValueError("rows must be an array")
    relative_path = _validate_filename(filename)
    suffix = _validate_suffix(relative_path, "table")
    base_dir = _safe_base_output_dir(output_dir)
    target = _unique_output_path(base_dir, relative_path)
    _write_table_file(target, columns, rows)
    num_chars = len(target.read_text(encoding="utf-8"))
    result = _file_result(target, output_dir, "table", suffix, num_chars)
    result["num_rows"] = len(rows)
    result["num_columns"] = len(columns)
    return result


def json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def file_writer(
    filename: str,
    file_type: str,
    content: str,
    output_dir: str | None = None,
) -> dict:
    return _write_generated_file(filename, file_type, content, output_dir)
