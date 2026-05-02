import re
from datetime import datetime

OUTPUT_PREVIEW_LIMIT = 500
TEXT_FIELD_PREVIEW_LIMIT = 2000
LIST_PREVIEW_LIMIT = 25
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
SPINNER_GLYPH_RE = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")


def timestamp_now():
    return datetime.now().isoformat(timespec="seconds")


def clean_text(value):
    cleaned = ANSI_ESCAPE_RE.sub("", value or "").replace("\r", "\n")
    cleaned = SPINNER_GLYPH_RE.sub(" ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def yaml_quote(value):
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return yaml_quote(value)


def yaml_dump_lines(value, indent=0):
    prefix = " " * indent

    if isinstance(value, dict):
        if not value:
            return [prefix + "{}"]

        lines = []
        for key, child in value.items():
            child_prefix = f"{prefix}{key}:"
            if isinstance(child, (dict, list)):
                child_lines = yaml_dump_lines(child, indent + 2)
                if len(child_lines) == 1 and child_lines[0].strip() in {"{}", "[]"}:
                    lines.append(f"{child_prefix} {child_lines[0].strip()}")
                else:
                    lines.append(child_prefix)
                    lines.extend(child_lines)
            else:
                lines.append(f"{child_prefix} {yaml_scalar(child)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [prefix + "[]"]

        lines = []
        for item in value:
            item_prefix = f"{prefix}-"
            if isinstance(item, (dict, list)):
                child_lines = yaml_dump_lines(item, indent + 2)
                if len(child_lines) == 1 and child_lines[0].strip() in {"{}", "[]"}:
                    lines.append(f"{item_prefix} {child_lines[0].strip()}")
                elif isinstance(item, dict):
                    lines.append(f"{item_prefix} {child_lines[0].strip()}")
                    lines.extend(child_lines[1:])
                else:
                    lines.append(item_prefix)
                    lines.extend(child_lines)
            else:
                lines.append(f"{item_prefix} {yaml_scalar(item)}")
        return lines

    return [prefix + yaml_scalar(value)]


def truncate_text(value, limit=OUTPUT_PREVIEW_LIMIT):
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def truncate_with_metadata(value, limit=TEXT_FIELD_PREVIEW_LIMIT):
    if value is None:
        return {"preview": None, "truncated": False, "original_length": 0}

    text = str(value)
    if len(text) <= limit:
        return {
            "preview": text,
            "truncated": False,
            "original_length": len(text),
        }

    return {
        "preview": text[: limit - 3] + "...",
        "truncated": True,
        "original_length": len(text),
    }


def summarize_list(values, preview_limit=LIST_PREVIEW_LIMIT):
    values = list(values)
    preview = values[:preview_limit]
    remainder = len(values) - len(preview)
    return {
        "count": len(values),
        "preview": preview,
        "truncated": remainder > 0,
        "omitted_count": remainder if remainder > 0 else 0,
    }


def format_bytes(num_bytes):
    if num_bytes is None:
        return "unknown"

    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
