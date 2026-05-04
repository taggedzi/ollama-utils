import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_VERSION_PATTERN = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']\s*$')


def _candidate_pyproject_paths():
    candidates = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "pyproject.toml")

    candidates.append(Path(__file__).resolve().parents[2] / "pyproject.toml")
    return candidates


def _read_version_from_pyproject():
    for path in _candidate_pyproject_paths():
        if not path.is_file():
            continue

        in_project_table = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("["):
                in_project_table = line == "[project]"
                continue

            if not in_project_table:
                continue

            match = _VERSION_PATTERN.match(line)
            if match:
                return match.group(1)
    return None


__version__ = _read_version_from_pyproject() or "dev"
if __version__ == "dev":
    try:
        __version__ = version("tz-ollama-utils")
    except PackageNotFoundError:
        pass

__all__ = ["__version__"]
