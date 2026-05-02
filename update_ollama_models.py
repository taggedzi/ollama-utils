import subprocess
import sys
from datetime import datetime


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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

    lines = result.stdout.splitlines()[1:]  # skip header
    models = []

    for line in lines:
        parts = line.split()
        if parts:
            models.append(parts[0])

    return models


def pull_model(model, index, total):
    log(f"[{index}/{total}] Updating {model}")

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
            print(f"    {line}", flush=True)

    return process.wait()


def main():
    try:
        models = get_models()
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        log(f"Unable to list models: {exc}")
        return 1

    if not models:
        log("No models found.")
        return 0

    failures = []
    log(f"Found {len(models)} models to update.")

    for index, model in enumerate(models, start=1):
        try:
            returncode = pull_model(model, index, len(models))
        except OSError as exc:
            log(f"ERROR: Failed to start update for {model}: {exc}")
            failures.append(model)
            continue

        if returncode == 0:
            log(f"Completed {model}")
        else:
            log(f"FAILED {model} with exit code {returncode}")
            failures.append(model)

    log(f"Done. Updated: {len(models) - len(failures)}/{len(models)}")

    if failures:
        log(f"Failures: {', '.join(failures)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
