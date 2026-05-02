import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_TIMEOUT_SECONDS = 300
FAILURE_LOG = Path("ollama_model_failures.yaml")
EMBEDDING_SAMPLE_TEXT = "test"
API_TIMEOUT_SECONDS = 30
STOP_TIMEOUT_SECONDS = 30
OUTPUT_PREVIEW_LIMIT = 500
TEXT_FIELD_PREVIEW_LIMIT = 2000
LIST_PREVIEW_LIMIT = 25
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
SPINNER_GLYPH_RE = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")
NVIDIA_SMI_TIMEOUT_SECONDS = 10
OLLAMA_API_BASE_URL = "http://127.0.0.1:11434/api"


def log(message):
    print(message, flush=True)


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


def parse_parameters_text(parameters_text):
    parsed = {}
    cleaned = clean_text(parameters_text)
    if not cleaned:
        return parsed

    for line in cleaned.splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        key = parts[0]
        value = parts[1] if len(parts) > 1 else ""
        parsed[key] = value

    return parsed


def timestamp_now():
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(args, timeout, input_text=None):
    return subprocess.run(
        args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def run_ollama_api(method, path, payload=None, timeout=API_TIMEOUT_SECONDS):
    url = f"{OLLAMA_API_BASE_URL}{path}"
    body = None
    headers = {}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib_request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        cleaned = clean_text(error_body) or f"HTTP {exc.code}"
        raise RuntimeError(f"Ollama API {method} {path} failed: {cleaned}") from exc
    except urllib_error.URLError as exc:
        reason = clean_text(str(exc.reason)) or str(exc.reason)
        raise RuntimeError(f"Unable to reach Ollama API {method} {path}: {reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to call Ollama API {method} {path}: {exc}") from exc

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        preview = truncate_text(clean_text(raw_body))
        raise RuntimeError(
            f"Ollama API {method} {path} returned invalid JSON: {preview}"
        ) from exc


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "timeout_seconds",
        nargs="?",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-model timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--fit-vram",
        action="store_true",
        default=True,
        help="Only test models whose stored size is less than or equal to the total GPU VRAM measured once at startup.",
    )
    parser.add_argument(
        "--ignore-size",
        dest="fit_vram",
        action="store_false",
        help="Test all models regardless of the total GPU VRAM measured at startup.",
    )

    args = parser.parse_args(argv[1:])

    if args.timeout_seconds <= 0:
        raise ValueError("Timeout must be a positive integer.")

    return args


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


def write_yaml_report(report):
    FAILURE_LOG.write_text("\n".join(yaml_dump_lines(report)) + "\n", encoding="utf-8")


def add_warning(warnings, stage, message):
    warnings.append(
        {
            "timestamp": timestamp_now(),
            "stage": stage,
            "message": message,
        }
    )


def build_failure(model, category, command, returncode, error, output):
    cleaned_error = clean_text(error)
    cleaned_output = clean_text(output)

    if not cleaned_error and returncode not in (None, 0):
        cleaned_error = f"Command exited with return code {returncode}"
    if not cleaned_error and not cleaned_output:
        cleaned_error = "Command failed without stderr or stdout output"

    return {
        "timestamp": timestamp_now(),
        "model": model,
        "category": category,
        "command": " ".join(command),
        "returncode": returncode if returncode is not None else "unknown",
        "error": truncate_text(cleaned_error),
        "output_preview": truncate_text(cleaned_output),
    }


def build_attempt(result, prompt_label, prompt_value):
    return {
        "timestamp": timestamp_now(),
        "prompt_label": prompt_label,
        "prompt_value": prompt_value,
        "command": " ".join(result["command"]),
        "ok": result["ok"],
        "returncode": result["returncode"] if result["returncode"] is not None else "unknown",
        "error": truncate_text(clean_text(result["error"])),
        "output_preview": truncate_text(clean_text(result["output"])),
    }


def classify_failure(error, returncode):
    if "embedding models require input text" in error:
        return "script_input_issue"
    if error.startswith("Timed out after"):
        return "timeout"
    if "llama runner process has terminated" in error:
        return "runner_crash"
    if "not found" in error.lower() and "ollama" in error.lower():
        return "missing_dependency"
    if returncode not in (None, 0):
        return "runtime_error"
    return "unknown_error"


def build_skip(model, model_size_bytes, device_vram_bytes):
    return {
        "timestamp": timestamp_now(),
        "model": model,
        "model_size_bytes": model_size_bytes,
        "model_size_human": format_bytes(model_size_bytes),
        "device_vram_bytes": device_vram_bytes,
        "device_vram_human": format_bytes(device_vram_bytes),
        "reason": (
            f"Model size {format_bytes(model_size_bytes)} exceeds startup device VRAM "
            f"{format_bytes(device_vram_bytes)}."
        ),
    }


def sanitize_model_metadata(metadata):
    if not isinstance(metadata, dict):
        return metadata

    sanitized = dict(metadata)

    parameters_text = sanitized.get("parameters")
    sanitized["parameters_parsed"] = parse_parameters_text(parameters_text)
    sanitized["parameters_preview"] = truncate_with_metadata(parameters_text)

    for field_name in ("license", "modelfile", "template", "system"):
        if field_name in sanitized:
            sanitized[f"{field_name}_available"] = bool(sanitized[field_name])
            sanitized[f"{field_name}_preview"] = truncate_with_metadata(
                sanitized[field_name]
            )
            del sanitized[field_name]

    if "parameters" in sanitized:
        del sanitized["parameters"]
    if "model_info" in sanitized:
        del sanitized["model_info"]

    return sanitized


def build_model_overview(model_summary, model_metadata, fit_vram, device_vram_bytes):
    details = model_summary.get("details")
    if not isinstance(details, dict):
        details = {}

    parsed_parameters = {}
    capabilities = []
    metadata_modified_at = None
    if isinstance(model_metadata, dict):
        parsed_parameters = model_metadata.get("parameters_parsed") or {}
        capabilities = model_metadata.get("capabilities") or []
        metadata_modified_at = model_metadata.get("modified_at")

    size_bytes = model_summary.get("size")
    fits_device_vram = None
    if isinstance(size_bytes, int) and device_vram_bytes is not None:
        fits_device_vram = size_bytes <= device_vram_bytes

    return {
        "name": clean_text(model_summary.get("name")),
        "family": details.get("family"),
        "families": details.get("families"),
        "capabilities": capabilities,
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "format": details.get("format"),
        "parent_model": details.get("parent_model"),
        "size_bytes": size_bytes,
        "size_human": format_bytes(size_bytes) if isinstance(size_bytes, int) else None,
        "modified_at": model_summary.get("modified_at"),
        "show_modified_at": metadata_modified_at,
        "digest": model_summary.get("digest"),
        "context_window": parsed_parameters.get("num_ctx"),
        "parameters": parsed_parameters,
        "fits_device_vram": fits_device_vram,
        "device_vram_human": format_bytes(device_vram_bytes),
        "size_filter_enabled": fit_vram,
    }


def build_model_details(model_metadata):
    if not isinstance(model_metadata, dict):
        return {
            "parameters_preview": truncate_with_metadata(None),
            "license_available": False,
            "license_preview": truncate_with_metadata(None),
            "template_available": False,
            "template_preview": truncate_with_metadata(None),
            "modelfile_available": False,
            "modelfile_preview": truncate_with_metadata(None),
            "system_available": False,
            "system_preview": truncate_with_metadata(None),
        }

    return {
        "parameters_preview": model_metadata.get("parameters_preview"),
        "license_available": model_metadata.get("license_available", False),
        "license_preview": model_metadata.get("license_preview"),
        "template_available": model_metadata.get("template_available", False),
        "template_preview": model_metadata.get("template_preview"),
        "modelfile_available": model_metadata.get("modelfile_available", False),
        "modelfile_preview": model_metadata.get("modelfile_preview"),
        "system_available": model_metadata.get("system_available", False),
        "system_preview": model_metadata.get("system_preview"),
    }


def get_models():
    payload = run_ollama_api("GET", "/tags")
    models = payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama API /tags did not return a 'models' list.")

    normalized_models = []
    seen = set()

    for model in models:
        if not isinstance(model, dict):
            continue

        name = clean_text(model.get("name"))
        if not name or name in seen:
            continue

        seen.add(name)
        normalized_models.append(model)

    return normalized_models


def get_model_metadata(model_name):
    metadata = run_ollama_api(
        "POST",
        "/show",
        payload={"model": model_name},
    )
    return sanitize_model_metadata(metadata)


def get_device_vram_bytes():
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = run_cmd(command, timeout=NVIDIA_SMI_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise RuntimeError("The 'nvidia-smi' command was not found in PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"'nvidia-smi' timed out after {NVIDIA_SMI_TIMEOUT_SECONDS} seconds."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to run 'nvidia-smi': {exc}") from exc

    if result.returncode != 0:
        error = clean_text(result.stderr) or clean_text(result.stdout)
        raise RuntimeError(error or "Failed to run 'nvidia-smi'.")

    total_vram_values = []
    for line in result.stdout.splitlines():
        value = clean_text(line)
        if not value:
            continue
        try:
            total_vram_values.append(int(value) * 1024 * 1024)
        except ValueError as exc:
            raise RuntimeError(f"Unexpected VRAM value from nvidia-smi: {value!r}") from exc

    if not total_vram_values:
        raise RuntimeError("No GPU VRAM values were returned by 'nvidia-smi'.")

    return max(total_vram_values)


def stop_model(model, warnings):
    try:
        result = run_cmd(["ollama", "stop", model], timeout=STOP_TIMEOUT_SECONDS)
    except FileNotFoundError:
        add_warning(warnings, "stop_model", "The 'ollama' command is unavailable during cleanup.")
        return
    except subprocess.TimeoutExpired:
        add_warning(
            warnings,
            "stop_model",
            f"Timed out stopping model {model!r} after {STOP_TIMEOUT_SECONDS} seconds.",
        )
        return
    except OSError as exc:
        add_warning(warnings, "stop_model", f"Unable to stop model {model!r}: {exc}")
        return

    if result.returncode != 0:
        message = clean_text(result.stderr) or clean_text(result.stdout)
        if message and "not running" not in message.lower():
            add_warning(
                warnings,
                "stop_model",
                f"Failed to stop model {model!r}: {message}",
            )


def run_model_command(model, prompt, timeout_seconds):
    command = ["ollama", "run", model, prompt]

    try:
        result = run_cmd(command, timeout=timeout_seconds)
    except FileNotFoundError:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "error": "The 'ollama' command was not found in PATH.",
            "output": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "error": f"Timed out after {timeout_seconds} seconds",
            "output": "",
        }
    except OSError as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "error": f"Unable to execute command: {exc}",
            "output": "",
        }

    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "error": clean_text(result.stderr),
        "output": clean_text(result.stdout),
    }


def test_model(model, timeout_seconds, warnings):
    attempts = []

    if not model:
        failure = build_failure(
            model="<empty>",
            category="invalid_model_name",
            command=["ollama", "run", "<empty>", ""],
            returncode=None,
            error="Encountered an empty model name while iterating the model list.",
            output="",
        )
        return {"status": "failed", "attempts": attempts, "failure": failure}

    first_attempt = run_model_command(model, "", timeout_seconds)
    attempts.append(build_attempt(first_attempt, "empty_prompt", ""))

    if (
        not first_attempt["ok"]
        and "embedding models require input text" in first_attempt["error"]
    ):
        retry_attempt = run_model_command(model, EMBEDDING_SAMPLE_TEXT, timeout_seconds)
        attempts.append(
            build_attempt(retry_attempt, "embedding_sample_text", EMBEDDING_SAMPLE_TEXT)
        )
        if retry_attempt["ok"]:
            add_warning(
                warnings,
                "embedding_retry",
                f"Model {model!r} required sample input and passed on retry.",
            )
            return {
                "status": "passed",
                "attempts": attempts,
                "failure": None,
                "used_embedding_sample_retry": True,
            }
        failure = build_failure(
            model=model,
            category=classify_failure(retry_attempt["error"], retry_attempt["returncode"]),
            command=retry_attempt["command"],
            returncode=retry_attempt["returncode"],
            error=retry_attempt["error"],
            output=retry_attempt["output"],
        )
        return {
            "status": "failed",
            "attempts": attempts,
            "failure": failure,
            "used_embedding_sample_retry": True,
        }

    if first_attempt["ok"]:
        return {
            "status": "passed",
            "attempts": attempts,
            "failure": None,
            "used_embedding_sample_retry": False,
        }

    failure = build_failure(
        model=model,
        category=classify_failure(first_attempt["error"], first_attempt["returncode"]),
        command=first_attempt["command"],
        returncode=first_attempt["returncode"],
        error=first_attempt["error"],
        output=first_attempt["output"],
    )
    return {
        "status": "failed",
        "attempts": attempts,
        "failure": failure,
        "used_embedding_sample_retry": False,
    }


def summarize_exit_code(failures, aborted):
    if aborted:
        return 130
    return 1 if failures else 0


def build_model_record(model_summary, model_metadata, metadata_error, fit_vram, device_vram_bytes):
    name = clean_text(model_summary.get("name"))
    model_size_bytes = model_summary.get("size")
    fits_device_vram = None

    if isinstance(model_size_bytes, int) and device_vram_bytes is not None:
        fits_device_vram = model_size_bytes <= device_vram_bytes

    return {
        "name": name,
        "observed_at": timestamp_now(),
        "overview": build_model_overview(
            model_summary=model_summary,
            model_metadata=model_metadata,
            fit_vram=fit_vram,
            device_vram_bytes=device_vram_bytes,
        ),
        "details": build_model_details(model_metadata),
        "metadata_error": metadata_error,
        "runtime": {
            "size_policy": {
                "filter_enabled": fit_vram,
                "device_vram_bytes": device_vram_bytes,
                "device_vram_human": format_bytes(device_vram_bytes),
                "model_size_bytes": model_size_bytes,
                "model_size_human": format_bytes(model_size_bytes)
                if isinstance(model_size_bytes, int)
                else None,
                "fits_device_vram": fits_device_vram,
            },
            "test": {
                "status": "pending",
                "attempts": [],
                "failure": None,
                "skip": None,
            },
        },
    }


def log_model_record(index, total_models, model_record):
    log(f"[{index}/{total_models}] Model record:")
    for line in yaml_dump_lines(model_record, indent=2):
        log(line)


def main():
    try:
        args = parse_args(sys.argv)
    except ValueError as exc:
        log(f"Argument error: {exc}")
        return 2
    except SystemExit as exc:
        return exc.code

    warnings = []
    failures = []
    skips = []
    model_records = []
    aborted = False
    tested_models = 0
    device_vram_bytes = None

    try:
        models = get_models()
    except RuntimeError as exc:
        log(f"Unable to list models: {exc}")
        failures.append(
            build_failure(
                model="<list_models>",
                category="setup_error",
                command=["GET", "/api/tags"],
                returncode=None,
                error=str(exc),
                output="",
            )
        )
        models = []

    if args.fit_vram:
        try:
            device_vram_bytes = get_device_vram_bytes()
        except RuntimeError as exc:
            add_warning(
                warnings,
                "device_vram",
                f"Unable to determine device VRAM; continuing without size filtering: {exc}",
            )
            device_vram_bytes = None

    log(f"Found {len(models)} models.")
    log(f"Per-model timeout: {args.timeout_seconds} seconds.")
    if args.fit_vram and device_vram_bytes is not None:
        log(f"Startup device VRAM filter: {format_bytes(device_vram_bytes)}.\n")
    elif args.fit_vram:
        log("Startup device VRAM filter unavailable; continuing without size filtering.\n")
    else:
        log("")

    try:
        for index, model_summary in enumerate(models, start=1):
            model = clean_text(model_summary.get("name"))
            log(f"[{index}/{len(models)}] Inspecting {model} ...")

            metadata = None
            metadata_error = None
            try:
                metadata = get_model_metadata(model)
            except RuntimeError as exc:
                metadata_error = str(exc)
                add_warning(
                    warnings,
                    "model_metadata",
                    f"Unable to fetch metadata for model {model!r}: {exc}",
                )

            model_record = build_model_record(
                model_summary=model_summary,
                model_metadata=metadata,
                metadata_error=metadata_error,
                fit_vram=args.fit_vram,
                device_vram_bytes=device_vram_bytes,
            )

            model_size_bytes = model_summary.get("size")
            should_skip = (
                args.fit_vram
                and device_vram_bytes is not None
                and isinstance(model_size_bytes, int)
                and model_size_bytes > device_vram_bytes
            )

            if should_skip:
                skip = build_skip(model, model_size_bytes, device_vram_bytes)
                skips.append(skip)
                model_record["runtime"]["test"] = {
                    "status": "skipped",
                    "attempts": [],
                    "failure": None,
                    "skip": skip,
                }
                model_records.append(model_record)
                log(f"  SKIPPED: {model} [size]")
                log(f"  Reason: {skip['reason']}")
                log_model_record(index, len(models), model_record)
                continue

            test_result = test_model(model, args.timeout_seconds, warnings)
            tested_models += 1
            model_record["runtime"]["test"] = test_result
            model_records.append(model_record)

            if test_result["status"] == "passed":
                log(f"  OK: {model}")
            else:
                failure = test_result["failure"]
                log(f"  FAILED: {model} [{failure['category']}]")
                if failure["error"]:
                    log(f"  Error: {failure['error']}")
                failures.append(failure)

            log_model_record(index, len(models), model_record)
    except KeyboardInterrupt:
        aborted = True
        add_warning(warnings, "interrupt", "Run interrupted by user.")
        log("\nInterrupted. Writing partial YAML report.")

    report = {
        "generated_at": timestamp_now(),
        "timeout_seconds": args.timeout_seconds,
        "fit_vram": args.fit_vram,
        "device_vram_bytes": device_vram_bytes,
        "device_vram_human": format_bytes(device_vram_bytes),
        "total_models": len(models),
        "tested_models": tested_models,
        "skipped_models": len(skips),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "aborted": aborted,
        "warnings": warnings,
        "skips": skips,
        "failures": failures,
        "models": model_records,
    }

    try:
        write_yaml_report(report)
    except OSError as exc:
        log(f"Failed to write YAML report {FAILURE_LOG}: {exc}")
        return 1

    log("")
    log(f"Done. Failed models: {len(failures)}")
    if skips:
        log(f"Skipped models: {len(skips)}")
    if warnings:
        log(f"Warnings: {len(warnings)}")
    log(f"Failure report: {FAILURE_LOG.resolve()}")

    return summarize_exit_code(failures, aborted)


if __name__ == "__main__":
    sys.exit(main())
