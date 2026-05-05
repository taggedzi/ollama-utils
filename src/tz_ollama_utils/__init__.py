import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


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

        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue

        project = data.get("project")
        if not isinstance(project, dict):
            continue

        version_value = project.get("version")
        if isinstance(version_value, str) and version_value.strip():
            return version_value.strip()
    return None


__version__ = _read_version_from_pyproject() or "dev"
if __version__ == "dev":
    try:
        __version__ = version("tz-ollama-utils")
    except PackageNotFoundError:
        pass

__all__ = ["__version__"]
