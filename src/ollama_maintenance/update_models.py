import subprocess
import sys
from datetime import datetime


def default_emit(message):
    print(message, flush=True)


def log(message, emit):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emit(f"[{timestamp}] {message}")


def get_models():
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
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


def pull_model(model, index, total, emit):
    log(f"[{index}/{total}] Updating {model}", emit)

    process = subprocess.Popen(
        ["ollama", "pull", model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None

    for line in process.stdout:
        line = line.rstrip()
        if line:
            emit(f"    {line}")

    return process.wait()


def main(argv=None, emit=None):
    del argv
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

    for index, model in enumerate(models, start=1):
        try:
            returncode = pull_model(model, index, len(models), emit)
        except OSError as exc:
            log(f"ERROR: Failed to start update for {model}: {exc}", emit)
            failures.append(model)
            continue

        if returncode == 0:
            log(f"Completed {model}", emit)
        else:
            log(f"FAILED {model} with exit code {returncode}", emit)
            failures.append(model)

    log(f"Done. Updated: {len(models) - len(failures)}/{len(models)}", emit)

    if failures:
        log(f"Failures: {', '.join(failures)}", emit)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
