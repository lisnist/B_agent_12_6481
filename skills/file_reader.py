from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from skills import format_workspace_source, resolve_workspace_path


SUPPORTED_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".py",
    ".log",
}
SUPPORTED_OFFICE_SUFFIXES = {".docx", ".pptx"}
UNSUPPORTED_LEGACY_OFFICE_SUFFIXES = {".doc", ".ppt"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_OFFICE_SUFFIXES
MAX_CHARS_LIMIT = 50000

_DOCX_TEXT_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_DOCX_TAB_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tab"
_DOCX_BREAK_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"
_DOCX_PARAGRAPH_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
_PPTX_TEXT_TAG = "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
_PPTX_BREAK_TAG = "{http://schemas.openxmlformats.org/drawingml/2006/main}br"
_PPTX_PARAGRAPH_TAG = "{http://schemas.openxmlformats.org/drawingml/2006/main}p"
_SLIDE_PATTERN = re.compile(r"^ppt/slides/slide(\d+)\.xml$")


def _slice_lines(text: str, start_line: int | None, end_line: int | None) -> tuple[str, int, int, int]:
    lines = text.splitlines()
    line_count = len(lines)
    if line_count == 0:
        return "", 0, 0, 0
    start = 1 if start_line is None else start_line
    end = line_count if end_line is None else end_line
    if not isinstance(start, int) or isinstance(start, bool) or start <= 0:
        raise ValueError("start_line must be a positive integer")
    if not isinstance(end, int) or isinstance(end, bool) or end <= 0:
        raise ValueError("end_line must be a positive integer")
    if start > end:
        raise ValueError("start_line must not be greater than end_line")
    selected = lines[start - 1 : end]
    return "\n".join(selected), start, min(end, line_count), line_count


def _paragraph_text(
    paragraph: ET.Element,
    text_tag: str,
    paragraph_tag: str,
    break_tag: str | None = None,
) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node is not paragraph and node.tag == paragraph_tag:
            continue
        if node.tag == text_tag:
            parts.append(node.text or "")
        elif node.tag == _DOCX_TAB_TAG:
            parts.append("\t")
        elif node.tag == _DOCX_BREAK_TAG or (break_tag is not None and node.tag == break_tag):
            parts.append("\n")
    return "".join(parts).strip()


def _extract_docx_text(source: Path) -> tuple[str, dict]:
    try:
        with zipfile.ZipFile(source) as archive:
            document_xml = archive.read("word/document.xml")
    except KeyError as exc:
        raise ValueError("docx file is missing word/document.xml") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError("docx file is not a valid Office Open XML archive") from exc
    root = ET.fromstring(document_xml)
    paragraphs = [
        text
        for text in (
            _paragraph_text(paragraph, _DOCX_TEXT_TAG, _DOCX_PARAGRAPH_TAG)
            for paragraph in root.iter(_DOCX_PARAGRAPH_TAG)
        )
        if text
    ]
    return "\n".join(paragraphs), {"parser": "office-openxml", "document_type": "docx"}


def _slide_sort_key(name: str) -> int:
    match = _SLIDE_PATTERN.match(name)
    return int(match.group(1)) if match else 0


def _extract_pptx_text(source: Path) -> tuple[str, dict]:
    try:
        with zipfile.ZipFile(source) as archive:
            slide_names = sorted(
                (name for name in archive.namelist() if _SLIDE_PATTERN.match(name)),
                key=_slide_sort_key,
            )
            if not slide_names:
                raise ValueError("pptx file contains no slide XML files")
            slides = []
            for index, name in enumerate(slide_names, 1):
                root = ET.fromstring(archive.read(name))
                paragraphs = [
                    text
                    for text in (
                        _paragraph_text(
                            paragraph,
                            _PPTX_TEXT_TAG,
                            _PPTX_PARAGRAPH_TAG,
                            _PPTX_BREAK_TAG,
                        )
                        for paragraph in root.iter(_PPTX_PARAGRAPH_TAG)
                    )
                    if text
                ]
                body = "\n".join(paragraphs).strip()
                slides.append(f"Slide {index}\n{body}" if body else f"Slide {index}")
    except zipfile.BadZipFile as exc:
        raise ValueError("pptx file is not a valid Office Open XML archive") from exc
    return "\n\n".join(slides), {
        "parser": "office-openxml",
        "document_type": "pptx",
        "slide_count": len(slide_names),
    }


def _read_supported_file(source: Path, suffix: str) -> tuple[str, dict]:
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return source.read_text(encoding="utf-8"), {"parser": "utf-8-text", "document_type": "text"}
    if suffix == ".docx":
        return _extract_docx_text(source)
    if suffix == ".pptx":
        return _extract_pptx_text(source)
    raise ValueError(f"unsupported file suffix: {suffix}")


def file_reader(
    path: str,
    max_chars: int = 2000,
    start_line: int | None = None,
    end_line: int | None = None,
    *,
    data_root: str | None = None,
    allowed_roots: dict[str, str] | None = None,
    default_root: str = "data",
) -> dict:
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if max_chars > MAX_CHARS_LIMIT:
        raise ValueError(f"max_chars must not exceed {MAX_CHARS_LIMIT}")
    source, root, root_alias = resolve_workspace_path(
        path,
        data_root=data_root,
        allowed_roots=allowed_roots,
        default_root=default_root,
    )
    suffix = source.suffix.lower()
    if suffix in UNSUPPORTED_LEGACY_OFFICE_SUFFIXES:
        raise ValueError(
            "file_reader does not support legacy binary Office files (.doc/.ppt); "
            "convert them to .docx/.pptx first"
        )
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"file_reader only supports these file types: {supported}")
    if not source.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    original, parser_metadata = _read_supported_file(source, suffix)
    selected_text, actual_start, actual_end, line_count = _slice_lines(original, start_line, end_line)
    content = selected_text[:max_chars]
    source_text, relative_path = format_workspace_source(source, root, root_alias)
    result = {
        "content": content,
        "num_chars": len(content),
        "source": source_text,
        "relative_path": relative_path,
        "root_alias": root_alias,
        "suffix": suffix,
        "line_count": line_count,
        "line_start": actual_start,
        "line_end": actual_end,
        "truncated": len(selected_text) > len(content),
    }
    result.update(parser_metadata)
    return result
