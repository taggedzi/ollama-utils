import subprocess
import sys
import time
from datetime import datetime

from .common import clean_text
from .common import default_emit
from .common import StopRequested
from .common import subprocess_window_kwargs
from .common import tool_command


def log(message, emit):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emit(f"[{timestamp}] {message}")


def get_models():
    result = subprocess.run(
        tool_command("ollama", "list"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        **subprocess_window_kwargs(),
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to run ollama list")

    lines = result.stdout.splitlines()[1:]
    models = []

    for line in lines:
        parts = line.split()
        if parts:
            models.append(parts[0])

    return models


def pull_model(model, index, total, emit, stop_requested=None):
    log(f"[{index}/{total}] Updating {model}", emit)

    process = subprocess.Popen(
        tool_command("ollama", "pull", model),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **subprocess_window_kwargs(),
    )

    if process.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")

    while True:
        if stop_requested and stop_requested():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise StopRequested

        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue

        cleaned_line = clean_text(line)
        if cleaned_line:
            for output_line in cleaned_line.splitlines():
                emit(f"    {output_line}")

    return process.wait()


def main(argv=None, emit=None, stop_requested=None):
    emit = emit or default_emit

    try:
        models = get_models()
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        log(f"Unable to list models: {exc}", emit)
        return 1

    if not models:
        log("No models found.", emit)
        return 0

    failures = []
    log(f"Found {len(models)} models to update.", emit)

    try:
        for index, model in enumerate(models, start=1):
            if stop_requested and stop_requested():
                raise StopRequested

            try:
                returncode = pull_model(
                    model,
                    index,
                    len(models),
                    emit,
                    stop_requested=stop_requested,
                )
            except OSError as exc:
                log(f"ERROR: Failed to start update for {model}: {exc}", emit)
                failures.append(model)
                continue

            if returncode == 0:
                log(f"Completed {model}", emit)
            else:
                log(f"FAILED {model} with exit code {returncode}", emit)
                failures.append(model)
    except (KeyboardInterrupt, StopRequested):
        log("Interrupted. Update stopped before processing all models.", emit)
        return 130

    log(f"Done. Updated: {len(models) - len(failures)}/{len(models)}", emit)

    if failures:
        log(f"Failures: {', '.join(failures)}", emit)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
