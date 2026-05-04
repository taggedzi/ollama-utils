import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

from .common import clean_text
from .common import default_emit
from .common import format_bytes
from .common import is_loopback_host
from .common import normalize_ollama_api_base_url
from .common import read_response_text_limited
from .common import ResponseTooLargeError
from .common import StopRequested
from .common import subprocess_window_kwargs
from .common import timestamp_now
from .common import tool_command
from .common import truncate_text
from .common import truncate_with_metadata
from .common import yaml_dump_lines

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_REPORT_PATH = Path("model_report.yaml")
SMOKE_TEST_PROMPT = "Respond with exactly OK."
EMBEDDING_SAMPLE_TEXT = "test"
API_TIMEOUT_SECONDS = 30
STOP_TIMEOUT_SECONDS = 30
NVIDIA_SMI_TIMEOUT_SECONDS = 10
OLLAMA_API_BASE_URL = "http://127.0.0.1:11434/api"
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?<!\w)[A-Za-z]:[\\/][^\s\"']+")
POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![:\w])/(?:[^/\s\"']+/)*[^/\s\"']+")
URL_RE = re.compile(r"https?://[^\s\"']+")


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


def run_cmd(args, timeout, input_text=None):
    return subprocess.run(
        args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        **subprocess_window_kwargs(),
    )

def run_ollama_api(method, path, api_base_url, payload=None, timeout=API_TIMEOUT_SECONDS):
    url = f"{api_base_url}{path}"
    body = None
    headers = {}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib_request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw_body = read_response_text_limited(response)
    except urllib_error.HTTPError as exc:
        try:
            error_body = read_response_text_limited(exc)
        except ResponseTooLargeError as size_exc:
            raise RuntimeError(
                f"Ollama API {method} {path} returned too much data: {size_exc}"
            ) from exc
        cleaned = clean_text(error_body) or f"HTTP {exc.code}"
        raise RuntimeError(f"Ollama API {method} {path} failed: {cleaned}") from exc
    except ResponseTooLargeError as exc:
        raise RuntimeError(f"Ollama API {method} {path} returned too much data: {exc}") from exc
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


def api_command_label(method, api_base_url, path, model, prompt):
    return [method, f"{api_base_url}{path}", model, prompt]


def is_api_unreachable_error(message):
    cleaned = clean_text(message)
    return cleaned.startswith("Unable to reach Ollama API ") or cleaned.startswith(
        "Unable to call Ollama API "
    )


def check_ollama_server(api_base_url):
    try:
        run_ollama_api("GET", "/tags", api_base_url=api_base_url)
    except RuntimeError as exc:
        return False, str(exc)
    return True, None


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
    parser.add_argument(
        "--vram-bytes",
        type=int,
        default=None,
        help="Override detected device VRAM with an explicit byte value.",
    )
    parser.add_argument(
        "--vram-mib",
        type=int,
        default=None,
        help="Override detected device VRAM with an explicit MiB value.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Write the YAML report to this path. Default: {DEFAULT_REPORT_PATH}.",
    )
    parser.add_argument(
        "--api-base-url",
        default=OLLAMA_API_BASE_URL,
        help=(
            "Base URL for the Ollama HTTP API. Accepts either the server root or "
            f"an explicit /api path. Default: {OLLAMA_API_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow non-loopback Ollama API hosts. Remote hosts must use https.",
    )
    parser.add_argument(
        "--include-metadata-previews",
        action="store_true",
        help=(
            "Include preview text for metadata fields like license, system prompt, "
            "template, modelfile, and parameters in the YAML report."
        ),
    )

    args = parser.parse_args(argv[1:])

    if args.timeout_seconds <= 0:
        raise ValueError("Timeout must be a positive integer.")
    if args.vram_bytes is not None and args.vram_bytes <= 0:
        raise ValueError("VRAM bytes override must be a positive integer.")
    if args.vram_mib is not None and args.vram_mib <= 0:
        raise ValueError("VRAM MiB override must be a positive integer.")
    if args.vram_bytes is not None and args.vram_mib is not None:
        raise ValueError("Use either --vram-bytes or --vram-mib, not both.")
    args.api_base_url = normalize_ollama_api_base_url(
        args.api_base_url, allow_remote=args.allow_remote
    )

    return args


def write_yaml_report(report, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(yaml_dump_lines(report)) + "\n", encoding="utf-8")


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


def _redact_url_match(match):
    url = match.group(0)
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "<redacted-url>"

    if is_loopback_host(parsed.hostname):
        return url

    return "<redacted-remote-url>"


def redact_export_text(value):
    cleaned = clean_text(value)
    if not cleaned:
        return cleaned

    cleaned = URL_RE.sub(_redact_url_match, cleaned)
    cleaned = WINDOWS_ABSOLUTE_PATH_RE.sub("<redacted-path>", cleaned)
    cleaned = POSIX_ABSOLUTE_PATH_RE.sub("<redacted-path>", cleaned)
    return cleaned


def sanitize_export_value(value):
    if isinstance(value, dict):
        return {key: sanitize_export_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_export_value(item) for item in value]
    if isinstance(value, str):
        return redact_export_text(value)
    return value


def redact_report_path_for_export(report_path):
    try:
        return str(report_path.resolve().name)
    except OSError:
        return report_path.name


def sanitize_api_base_url_for_export(api_base_url):
    cleaned = clean_text(api_base_url)
    if not cleaned:
        return cleaned

    parsed = urlsplit(cleaned)
    if is_loopback_host(parsed.hostname):
        return cleaned

    path = parsed.path or ""
    return f"{parsed.scheme}://<redacted-host>{path}"


def sanitize_model_metadata(metadata, include_previews=False):
    if not isinstance(metadata, dict):
        return metadata

    sanitized = dict(metadata)

    parameters_text = sanitized.get("parameters")
    sanitized["parameters_parsed"] = parse_parameters_text(parameters_text)
    sanitized["parameters_available"] = bool(parameters_text)
    sanitized["parameters_preview"] = (
        truncate_with_metadata(parameters_text) if include_previews else None
    )

    for field_name in ("license", "modelfile", "template", "system"):
        if field_name in sanitized:
            sanitized[f"{field_name}_available"] = bool(sanitized[field_name])
            sanitized[f"{field_name}_preview"] = (
                truncate_with_metadata(sanitized[field_name])
                if include_previews else None
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
            "parameters_available": False,
            "parameters_preview": None,
            "license_available": False,
            "license_preview": None,
            "template_available": False,
            "template_preview": None,
            "modelfile_available": False,
            "modelfile_preview": None,
            "system_available": False,
            "system_preview": None,
        }

    return {
        "parameters_preview": model_metadata.get("parameters_preview"),
        "parameters_available": model_metadata.get("parameters_available", False),
        "license_available": model_metadata.get("license_available", False),
        "license_preview": model_metadata.get("license_preview"),
        "template_available": model_metadata.get("template_available", False),
        "template_preview": model_metadata.get("template_preview"),
        "modelfile_available": model_metadata.get("modelfile_available", False),
        "modelfile_preview": model_metadata.get("modelfile_preview"),
        "system_available": model_metadata.get("system_available", False),
        "system_preview": model_metadata.get("system_preview"),
    }


def get_models_from_api(api_base_url):
    payload = run_ollama_api("GET", "/tags", api_base_url=api_base_url)
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


def get_models_from_cli():
    result = run_cmd(tool_command("ollama", "list"), timeout=API_TIMEOUT_SECONDS)

    if result.returncode != 0:
        error = clean_text(result.stderr) or clean_text(result.stdout)
        raise RuntimeError(error or "Failed to run 'ollama list'.")

    lines = result.stdout.splitlines()[1:]
    models = []
    seen = set()

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        name = clean_text(parts[0])
        if not name or name in seen:
            continue

        seen.add(name)
        models.append({"name": name})

    return models


def get_models(api_base_url):
    warnings = []
    api_models = None
    cli_models = None

    try:
        api_models = get_models_from_api(api_base_url)
    except RuntimeError as exc:
        warnings.append(
            f"Unable to list models from Ollama API /tags; falling back to 'ollama list': {exc}"
        )

    try:
        cli_models = get_models_from_cli()
    except RuntimeError as exc:
        if api_models is None:
            raise RuntimeError(
                "Unable to list models from either Ollama API /tags or 'ollama list': "
                f"{exc}"
            ) from exc
        warnings.append(f"Unable to compare against 'ollama list': {exc}")

    if api_models is None and cli_models is not None:
        return cli_models, "cli_fallback", warnings

    if api_models is None:
        raise RuntimeError("Unable to list models.")

    if cli_models is None:
        return api_models, "api", warnings

    api_by_name = {
        clean_text(model.get("name")): model
        for model in api_models
        if isinstance(model, dict) and clean_text(model.get("name"))
    }
    cli_names = [clean_text(model.get("name")) for model in cli_models]

    if len(cli_names) > len(api_models):
        warnings.append(
            "Ollama API /tags returned fewer models than 'ollama list'; using CLI inventory "
            f"for completeness ({len(cli_names)} vs {len(api_models)})."
        )
        merged_models = [api_by_name.get(name, {"name": name}) for name in cli_names]
        return merged_models, "cli_preferred", warnings

    return api_models, "api", warnings


def get_model_metadata(model_name, api_base_url, include_previews=False):
    metadata = run_ollama_api(
        "POST",
        "/show",
        api_base_url=api_base_url,
        payload={"model": model_name},
    )
    return sanitize_model_metadata(metadata, include_previews=include_previews)


def get_device_vram_bytes():
    command = tool_command(
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    )

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


def resolve_device_vram_bytes(args):
    if args.vram_bytes is not None:
        return args.vram_bytes, "manual_bytes"
    if args.vram_mib is not None:
        return args.vram_mib * 1024 * 1024, "manual_mib"
    return get_device_vram_bytes(), "nvidia_smi"


def run_model_command(model, prompt, timeout_seconds, stop_requested=None):
    command = ["ollama", "run", model, prompt]
    actual_command = tool_command("ollama", "run", model, prompt)

    try:
        process = subprocess.Popen(
            actual_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            **subprocess_window_kwargs(),
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "error": "The 'ollama' command was not found in PATH.",
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

    deadline = time.monotonic() + timeout_seconds

    while True:
        if stop_requested and stop_requested():
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            return {
                "ok": False,
                "command": command,
                "returncode": None,
                "error": "Run cancelled by user",
                "output": clean_text(stdout),
            }

        if process.poll() is not None:
            break

        if time.monotonic() >= deadline:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            return {
                "ok": False,
                "command": command,
                "returncode": None,
                "error": f"Timed out after {timeout_seconds} seconds",
                "output": clean_text(stdout),
            }

        time.sleep(0.1)

    stdout, stderr = process.communicate()
    return {
        "ok": process.returncode == 0,
        "command": command,
        "returncode": process.returncode,
        "error": clean_text(stderr),
        "output": clean_text(stdout),
    }


def run_model_api_command(model, prompt, timeout_seconds, api_base_url, path):
    command = api_command_label("POST", api_base_url, path, model, prompt)
    payload = {"model": model, "stream": False}

    if path == "/generate":
        payload["prompt"] = prompt
    elif path == "/embed":
        payload["input"] = prompt
    else:
        raise ValueError(f"Unsupported Ollama API path: {path}")

    try:
        response = run_ollama_api(
            "POST",
            path,
            api_base_url=api_base_url,
            payload=payload,
            timeout=timeout_seconds,
        )
    except RuntimeError as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "error": str(exc),
            "output": "",
        }

    if path == "/generate":
        output = clean_text(response.get("response"))
        return {
            "ok": bool(response.get("done")),
            "command": command,
            "returncode": 0 if response.get("done") else None,
            "error": "" if response.get("done") else "Ollama API generate did not finish.",
            "output": output,
        }

    embeddings = response.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return {
            "ok": True,
            "command": command,
            "returncode": 0,
            "error": "",
            "output": f"Generated {len(embeddings)} embedding(s).",
        }

    return {
        "ok": False,
        "command": command,
        "returncode": None,
        "error": "Ollama API embed did not return embeddings.",
        "output": clean_text(json.dumps(response, ensure_ascii=False)),
    }


def model_supports_embeddings(model_metadata):
    if not isinstance(model_metadata, dict):
        return False

    capabilities = model_metadata.get("capabilities") or []
    return "embedding" in capabilities or "embeddings" in capabilities


def test_model(
    model,
    timeout_seconds,
    warnings,
    api_base_url=None,
    api_server_available=False,
    model_metadata=None,
    stop_requested=None,
):
    attempts = []
    embedding_capability = model_supports_embeddings(model_metadata)

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

    initial_prompt = EMBEDDING_SAMPLE_TEXT if embedding_capability else SMOKE_TEST_PROMPT
    initial_prompt_label = (
        "embedding_sample_text" if embedding_capability else "smoke_test_prompt"
    )

    if api_server_available and api_base_url:
        first_path = "/embed" if embedding_capability else "/generate"
        first_attempt = run_model_api_command(
            model,
            initial_prompt,
            timeout_seconds,
            api_base_url,
            first_path,
        )
    else:
        first_attempt = run_model_command(
            model,
            initial_prompt,
            timeout_seconds,
            stop_requested=stop_requested,
        )
    attempts.append(build_attempt(first_attempt, initial_prompt_label, initial_prompt))

    if first_attempt["error"] == "Run cancelled by user":
        raise StopRequested

    if (
        not first_attempt["ok"]
        and "embedding models require input text" in first_attempt["error"]
    ):
        if api_server_available and api_base_url:
            retry_attempt = run_model_api_command(
                model,
                EMBEDDING_SAMPLE_TEXT,
                timeout_seconds,
                api_base_url,
                "/embed",
            )
        else:
            retry_attempt = run_model_command(
                model,
                EMBEDDING_SAMPLE_TEXT,
                timeout_seconds,
                stop_requested=stop_requested,
            )
        attempts.append(
            build_attempt(retry_attempt, "embedding_sample_text", EMBEDDING_SAMPLE_TEXT)
        )
        if retry_attempt["error"] == "Run cancelled by user":
            raise StopRequested
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


def log_model_record(index, total_models, model_record, emit):
    emit(f"[{index}/{total_models}] Model record:")
    for line in yaml_dump_lines(model_record, indent=2):
        emit(line)


def main(argv=None, emit=None, stop_requested=None, progress=None):
    argv = argv or sys.argv
    emit = emit or default_emit

    try:
        args = parse_args(argv)
    except ValueError as exc:
        emit(f"Argument error: {exc}")
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
    device_vram_source = "unavailable"
    inventory_source = "unknown"
    api_server_available = True
    report_path = args.report_path.resolve()
    api_base_url = args.api_base_url

    api_server_available, server_error = check_ollama_server(api_base_url)
    if not api_server_available:
        guidance = (
            "Ollama API server was not reachable at startup. Start the Ollama app or "
            f"server so {api_base_url} is available."
        )
        emit(guidance)
        add_warning(warnings, "api_server", guidance)
        add_warning(warnings, "api_server", server_error)

    try:
        models, inventory_source, inventory_warnings = get_models(api_base_url)
        for message in inventory_warnings:
            add_warning(warnings, "model_inventory", message)
    except RuntimeError as exc:
        emit(f"Unable to list models: {exc}")
        failures.append(
            build_failure(
                model="<list_models>",
                category="setup_error",
                command=["GET", "/api/tags", "or", "ollama", "list"],
                returncode=None,
                error=str(exc),
                output="",
            )
        )
        models = []

    if args.fit_vram:
        try:
            device_vram_bytes, device_vram_source = resolve_device_vram_bytes(args)
        except RuntimeError as exc:
            add_warning(
                warnings,
                "device_vram",
                f"Unable to determine device VRAM; continuing without size filtering: {exc}",
            )
            device_vram_bytes = None

    emit(f"Found {len(models)} models.")
    emit(f"Per-model timeout: {args.timeout_seconds} seconds.")
    emit(f"Ollama API base URL: {api_base_url}")
    emit(f"Inventory source: {inventory_source}")
    emit(f"Report path: {report_path}")
    if args.fit_vram and device_vram_bytes is not None:
        emit(
            f"Startup device VRAM filter: {format_bytes(device_vram_bytes)}"
            f" [{device_vram_source}].\n"
        )
    elif args.fit_vram:
        emit("Startup device VRAM filter unavailable; continuing without size filtering.\n")
    else:
        emit("")

    try:
        for index, model_summary in enumerate(models, start=1):
            if stop_requested and stop_requested():
                raise KeyboardInterrupt

            if progress:
                progress(index - 1, len(models))

            model = clean_text(model_summary.get("name"))
            emit(f"[{index}/{len(models)}] Inspecting {model} ...")

            metadata = None
            metadata_error = None
            if api_server_available:
                try:
                    metadata = get_model_metadata(
                        model,
                        api_base_url,
                        include_previews=args.include_metadata_previews,
                    )
                except RuntimeError as exc:
                    metadata_error = str(exc)
                    add_warning(
                        warnings,
                        "model_metadata",
                        f"Unable to fetch metadata for model {model!r}: {exc}",
                    )
            else:
                metadata_error = (
                    "Skipped metadata fetch because the Ollama API server was unavailable "
                    "at startup."
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
                emit(f"  SKIPPED: {model} [size]")
                emit(f"  Reason: {skip['reason']}")
                log_model_record(index, len(models), model_record, emit)
                continue

            test_result = test_model(
                model,
                args.timeout_seconds,
                warnings,
                api_base_url=api_base_url,
                api_server_available=api_server_available,
                model_metadata=metadata,
                stop_requested=stop_requested,
            )
            tested_models += 1

            model_record["runtime"]["test"] = test_result
            model_records.append(model_record)

            if test_result["status"] == "passed":
                emit(f"  OK: {model}")
            else:
                failure = test_result["failure"]
                emit(f"  FAILED: {model} [{failure['category']}]")
                if failure["error"]:
                    emit(f"  Error: {failure['error']}")
                failures.append(failure)

            log_model_record(index, len(models), model_record, emit)
    except (KeyboardInterrupt, StopRequested):
        aborted = True
        add_warning(warnings, "interrupt", "Run interrupted by user.")
        emit("\nInterrupted. Writing partial YAML report.")

    report = {
        "generated_at": timestamp_now(),
        "report_path": redact_report_path_for_export(report_path),
        "api_base_url": sanitize_api_base_url_for_export(api_base_url),
        "api_server_available": api_server_available,
        "inventory_source": inventory_source,
        "timeout_seconds": args.timeout_seconds,
        "fit_vram": args.fit_vram,
        "metadata_previews_included": args.include_metadata_previews,
        "device_vram_bytes": device_vram_bytes,
        "device_vram_human": format_bytes(device_vram_bytes),
        "device_vram_source": device_vram_source,
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
        "retained_data": {
            "report_path": "filename only",
            "api_base_url": "loopback preserved; remote host redacted",
            "metadata_previews": (
                "included by opt-in" if args.include_metadata_previews else "not included"
            ),
            "warnings_failures_attempts": "absolute paths and remote URLs redacted",
        },
    }

    report = sanitize_export_value(report)

    try:
        write_yaml_report(report, report_path)
    except OSError as exc:
        emit(f"Failed to write YAML report {report_path}: {exc}")
        return 1

    emit("")
    emit(f"Done. Failed models: {len(failures)}")
    if skips:
        emit(f"Skipped models: {len(skips)}")
    if warnings:
        emit(f"Warnings: {len(warnings)}")
    emit(f"Model report: {report_path}")

    return summarize_exit_code(failures, aborted)


if __name__ == "__main__":
    sys.exit(main())
