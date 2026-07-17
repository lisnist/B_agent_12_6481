from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone as datetime_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_OFFSET_PATTERN = re.compile(r"^([+-])(\d{2}):?(\d{2})$")
_WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_FIXED_TIMEZONE_FALLBACKS = {
    "Asia/Shanghai": timedelta(hours=8),
    "Asia/Beijing": timedelta(hours=8),
    "PRC": timedelta(hours=8),
}


def _parse_timezone(value: str) -> tuple[datetime_timezone | ZoneInfo, str]:
    name = value.strip() if isinstance(value, str) else ""
    if not name or name.lower() == "local":
        return datetime.now().astimezone().tzinfo or datetime_timezone.utc, "local"
    if name.upper() in {"UTC", "Z"}:
        return datetime_timezone.utc, "UTC"
    match = _OFFSET_PATTERN.fullmatch(name)
    if match:
        sign, hours_text, minutes_text = match.groups()
        hours = int(hours_text)
        minutes = int(minutes_text)
        if hours > 23 or minutes > 59:
            raise ValueError("timezone offset must be in the range +/-23:59")
        delta = timedelta(hours=hours, minutes=minutes)
        if sign == "-":
            delta = -delta
        normalized = f"{sign}{hours:02d}:{minutes:02d}"
        return datetime_timezone(delta, normalized), normalized
    try:
        return ZoneInfo(name), name
    except ZoneInfoNotFoundError as exc:
        if name in _FIXED_TIMEZONE_FALLBACKS:
            return datetime_timezone(_FIXED_TIMEZONE_FALLBACKS[name], name), name
        raise ValueError(f"unsupported timezone: {value}") from exc


def _offset_text(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None:
        return "+00:00"
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def current_time(timezone: str = "local") -> dict:
    """Return deterministic current time metadata for a requested timezone."""
    tzinfo, timezone_name = _parse_timezone(timezone)
    now = datetime.now(tzinfo).astimezone(tzinfo)
    utc_now = now.astimezone(datetime_timezone.utc)
    return {
        "timezone": timezone_name,
        "iso": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.time().replace(microsecond=0).isoformat(),
        "weekday": _WEEKDAYS[now.weekday()],
        "weekday_index": now.weekday() + 1,
        "utc_offset": _offset_text(now),
        "unix_timestamp": int(now.timestamp()),
        "local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "utc_time": utc_now.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
