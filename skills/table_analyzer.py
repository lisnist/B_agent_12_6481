from __future__ import annotations

import csv
import statistics
import re
import zipfile
import xml.etree.ElementTree as ET

from skills import format_workspace_source, resolve_workspace_path


SUPPORTED_TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx"}
_CELL_REF_PATTERN = re.compile(r"^([A-Z]+)")
_SPREADSHEET_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _column_index(cell_ref: str) -> int:
    match = _CELL_REF_PATTERN.match(cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    strings = []
    for item in root.findall("main:si", _SPREADSHEET_NS):
        parts = [node.text or "" for node in item.findall(".//main:t", _SPREADSHEET_NS)]
        strings.append("".join(parts))
    return strings


def _first_worksheet_path(archive: zipfile.ZipFile) -> str:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except KeyError as exc:
        raise ValueError("xlsx file is missing workbook metadata") from exc
    sheets = workbook.findall("main:sheets/main:sheet", _SPREADSHEET_NS)
    if not sheets:
        raise ValueError("xlsx workbook contains no worksheets")
    rel_id = sheets[0].attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    if not rel_id:
        raise ValueError("xlsx first worksheet is missing relationship id")
    for relationship in rels.findall("rel:Relationship", _SPREADSHEET_NS):
        if relationship.attrib.get("Id") == rel_id:
            target = relationship.attrib.get("Target", "")
            if not target:
                break
            if target.startswith("/"):
                return target.lstrip("/")
            return f"xl/{target}".replace("\\", "/")
    raise ValueError("xlsx first worksheet relationship was not found")


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        parts = [node.text or "" for node in cell.findall(".//main:t", _SPREADSHEET_NS)]
        return "".join(parts)
    value = cell.find("main:v", _SPREADSHEET_NS)
    if value is None or value.text is None:
        return ""
    raw = value.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def _read_xlsx_table(source) -> tuple[list[str], list[dict], str]:
    try:
        with zipfile.ZipFile(source) as archive:
            shared_strings = _shared_strings(archive)
            worksheet_path = _first_worksheet_path(archive)
            worksheet = ET.fromstring(archive.read(worksheet_path))
    except zipfile.BadZipFile as exc:
        raise ValueError("xlsx file is not a valid Office Open XML archive") from exc
    rows: list[list[str]] = []
    for row in worksheet.findall(".//main:sheetData/main:row", _SPREADSHEET_NS):
        values: list[str] = []
        for cell in row.findall("main:c", _SPREADSHEET_NS):
            index = _column_index(cell.attrib.get("r", ""))
            while len(values) <= index:
                values.append("")
            values[index] = _xlsx_cell_text(cell, shared_strings)
        if any(value != "" for value in values):
            rows.append(values)
    if not rows:
        raise ValueError("xlsx worksheet contains no data")
    columns = [str(value).strip() for value in rows[0]]
    if not any(columns):
        raise ValueError("xlsx first row must contain column names")
    normalized_columns = []
    seen = set()
    for index, column in enumerate(columns):
        name = column or f"column_{index + 1}"
        if name in seen:
            name = f"{name}_{index + 1}"
        seen.add(name)
        normalized_columns.append(name)
    records = []
    for raw_row in rows[1:]:
        padded = raw_row + [""] * max(0, len(normalized_columns) - len(raw_row))
        records.append({column: padded[index] if index < len(padded) else "" for index, column in enumerate(normalized_columns)})
    return normalized_columns, records, worksheet_path


def _read_delimited_table(source) -> tuple[list[str], list[dict], str]:
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("table must contain a header row")
        return list(reader.fieldnames), list(reader), "csv" if delimiter == "," else "tsv"


def table_analyzer(
    path: str,
    max_rows_preview: int = 5,
    describe: bool = True,
    *,
    data_root: str | None = None,
    allowed_roots: dict[str, str] | None = None,
    default_root: str = "data",
) -> dict:
    if not isinstance(max_rows_preview, int) or isinstance(max_rows_preview, bool) or max_rows_preview < 0:
        raise ValueError("max_rows_preview must be a non-negative integer")
    source, root, root_alias = resolve_workspace_path(
        path,
        data_root=data_root,
        allowed_roots=allowed_roots,
        default_root=default_root,
    )
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_TABLE_SUFFIXES:
        raise ValueError("table_analyzer only supports .csv, .tsv and .xlsx files")
    if not source.is_file():
        raise FileNotFoundError(f"table file not found: {path}")
    if suffix == ".xlsx":
        columns, rows, parser = _read_xlsx_table(source)
    else:
        columns, rows, parser = _read_delimited_table(source)
    column_profiles: dict[str, dict] = {}
    for column in columns:
        values = [row.get(column, "") for row in rows]
        stripped_values = [value.strip() for value in values]
        non_empty = [value for value in stripped_values if value != ""]
        column_profiles[column] = {
            "missing_count": len(stripped_values) - len(non_empty),
            "non_empty_count": len(non_empty),
            "unique_count": len(set(non_empty)),
        }
    stats: dict[str, dict] = {}
    if describe:
        for column in columns:
            raw_values = [row.get(column, "").strip() for row in rows if row.get(column, "").strip() != ""]
            if not raw_values:
                continue
            try:
                values = [float(value) for value in raw_values]
            except ValueError:
                continue
            column_stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
            }
            if len(values) >= 2:
                column_stats["stdev"] = statistics.stdev(values)
            stats[column] = column_stats
    source_text, relative_path = format_workspace_source(source, root, root_alias)
    return {
        "path": source_text,
        "relative_path": relative_path,
        "root_alias": root_alias,
        "suffix": suffix,
        "parser": parser,
        "num_rows": len(rows),
        "num_columns": len(columns),
        "columns": columns,
        "preview": rows[:max_rows_preview],
        "column_profiles": column_profiles,
        "describe": stats,
    }
