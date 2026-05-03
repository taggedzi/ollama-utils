import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

OUTPUT_PREVIEW_LIMIT = 500
TEXT_FIELD_PREVIEW_LIMIT = 2000
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


def detect_ollama_version(timeout=5):
    try:
        result = subprocess.run(
            tool_command("ollama", "--version"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **subprocess_window_kwargs(),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    output = clean_text(result.stdout) or clean_text(result.stderr)
    if result.returncode != 0 or not output:
        return None

    return output


def subprocess_window_kwargs():
    if os.name != "nt":
        return {}

    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    kwargs["startupinfo"] = startupinfo
    return kwargs


def _windows_tool_candidates(tool_name):
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "")

    candidates = {
        "ollama": [
            Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe",
            Path(program_files) / "Ollama" / "ollama.exe",
        ],
        "nvidia-smi": [
            Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        ],
    }
    return candidates.get(tool_name, [])


def resolve_tool_path(tool_name):
    resolved = shutil.which(tool_name)
    if resolved:
        return resolved

    if os.name == "nt":
        for candidate in _windows_tool_candidates(tool_name):
            if candidate.is_file():
                return str(candidate)

    return tool_name


def tool_command(tool_name, *args):
    return [resolve_tool_path(tool_name), *args]
