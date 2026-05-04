import json
from datetime import datetime
from pathlib import Path
from urllib import request as urllib_request
from urllib import error as urllib_error

from .common import format_bytes


class ModelSearchCache:
    DEFAULT_CACHE_PATH = Path.home() / ".tz_ollama_utils" / "model_cache.json"

    def __init__(self):
        self._models: dict = {}
        self._api_base_url: str = ""

    def load(self, cache_path: Path | None = None) -> bool:
        path = cache_path if cache_path is not None else self.DEFAULT_CACHE_PATH
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._api_base_url = data.get("api_base_url", "")
            self._models = data.get("models", {})
            return True
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

    def save(self, cache_path: Path | None = None) -> None:
        path = cache_path if cache_path is not None else self.DEFAULT_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"api_base_url": self._api_base_url, "models": self._models}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_models(self) -> list:
        return [self._normalize(name, entry) for name, entry in self._models.items()]

    def invalidate(self, cache_path: Path | None = None) -> None:
        self._models.clear()
        self._api_base_url = ""
        path = cache_path if cache_path is not None else self.DEFAULT_CACHE_PATH
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def remove_model(self, name: str, cache_path: Path | None = None) -> None:
        self._models.pop(name, None)
        self.save(cache_path)

    def update_model_show(self, name: str, show_data: dict, cache_path: Path | None = None) -> None:
        if name in self._models:
            self._models[name]["show"] = show_data
            self._models[name]["cached_at"] = datetime.now().isoformat(timespec="seconds")
        self.save(cache_path)

    def refresh(self, api_base_url: str, on_progress=None, cache_path: Path | None = None) -> None:
        self._api_base_url = api_base_url
        tags_data = self._fetch_tags(api_base_url)
        current_names = {m["name"] for m in tags_data}

        for name in list(self._models.keys()):
            if name not in current_names:
                del self._models[name]

        needs_show = []
        for model_info in tags_data:
            name = model_info["name"]
            cached = self._models.get(name, {})
            if cached.get("digest") != model_info.get("digest") or "show" not in cached:
                needs_show.append(model_info)
            else:
                self._models[name]["tags"] = model_info

        total = len(needs_show)
        for i, model_info in enumerate(needs_show):
            name = model_info["name"]
            show_data = self._fetch_show(api_base_url, name)
            self._models[name] = {
                "digest": model_info.get("digest", ""),
                "cached_at": datetime.now().isoformat(timespec="seconds"),
                "tags": model_info,
                "show": show_data,
            }
            if on_progress:
                on_progress(name, i + 1, total)

        self.save(cache_path)

    def _fetch_tags(self, api_base_url: str) -> list:
        url = f"{api_base_url.rstrip('/')}/tags"
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read()).get("models", [])
        except (urllib_error.URLError, urllib_error.HTTPError, OSError, json.JSONDecodeError):
            return []

    def _fetch_show(self, api_base_url: str, name: str) -> dict:
        url = f"{api_base_url.rstrip('/')}/show"
        body = json.dumps({"name": name}).encode("utf-8")
        req = urllib_request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except (urllib_error.URLError, urllib_error.HTTPError, OSError, json.JSONDecodeError):
            return {}

    def _normalize(self, name: str, entry: dict) -> dict:
        return {"name": name}  # stub — expanded in Task 3


def filter_models(models: list, filters: dict) -> list:
    return list(models)  # stub — expanded in Task 4
