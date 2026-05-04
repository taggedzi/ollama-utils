import json
import os
from datetime import datetime
from pathlib import Path
from urllib import request as urllib_request
from urllib import error as urllib_error

from .common import format_bytes
from .common import read_response_text_limited
from .common import ResponseTooLargeError


class ModelSearchCache:
    CACHE_SCHEMA_VERSION = 2
    DEFAULT_CACHE_PATH = Path.home() / ".tz_ollama_utils" / "model_cache.json"
    SENSITIVE_SHOW_FIELDS = ("license", "modelfile", "template", "system", "parameters")
    PERSISTED_TAGS_FIELDS = ("name", "size", "modified_at", "digest", "details")
    PERSISTED_DETAILS_FIELDS = (
        "family",
        "families",
        "parameter_size",
        "quantization_level",
        "format",
        "parent_model",
        "capabilities",
    )

    def __init__(self, persist_sensitive_text: bool | None = None):
        self._models: dict = {}
        self._api_base_url: str = ""
        if persist_sensitive_text is None:
            persist_sensitive_text = (
                os.environ.get("TZ_OLLAMA_UTILS_PERSIST_MODEL_TEXT", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
        self._persist_sensitive_text = persist_sensitive_text

    def load(self, cache_path: Path | None = None) -> bool:
        path = cache_path if cache_path is not None else self.DEFAULT_CACHE_PATH
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._api_base_url = data.get("api_base_url", "")
            raw_models = data.get("models", {})
            if not isinstance(raw_models, dict):
                raw_models = {}
            self._models = {
                name: self._load_cached_entry(entry)
                for name, entry in raw_models.items()
                if isinstance(entry, dict)
            }
            return True
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

    def save(self, cache_path: Path | None = None) -> None:
        path = cache_path if cache_path is not None else self.DEFAULT_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        data = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "api_base_url": self._api_base_url,
            "models": {
                name: self._persist_model_entry(entry)
                for name, entry in self._models.items()
            },
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

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
        if tags_data is None:
            return

        current_names = {m["name"] for m in tags_data}

        for name in list(self._models.keys()):
            if name not in current_names:
                del self._models[name]

        needs_show = []
        for model_info in tags_data:
            name = model_info["name"]
            cached = self._models.get(name, {})
            if cached.get("digest") != model_info.get("digest") or cached.get("show") is None:
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

    def _fetch_tags(self, api_base_url: str) -> list | None:
        url = f"{api_base_url.rstrip('/')}/tags"
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=10) as resp:
                return json.loads(read_response_text_limited(resp)).get("models", [])
        except (
            urllib_error.URLError,
            urllib_error.HTTPError,
            OSError,
            json.JSONDecodeError,
            ResponseTooLargeError,
        ):
            return None

    def _fetch_show(self, api_base_url: str, name: str) -> dict | None:
        url = f"{api_base_url.rstrip('/')}/show"
        body = json.dumps({"model": name}).encode("utf-8")
        req = urllib_request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                return json.loads(read_response_text_limited(resp))
        except (
            urllib_error.URLError,
            urllib_error.HTTPError,
            OSError,
            json.JSONDecodeError,
            ResponseTooLargeError,
        ):
            return None

    def _load_cached_entry(self, entry: dict) -> dict:
        return {
            "digest": entry.get("digest", ""),
            "cached_at": entry.get("cached_at", ""),
            "tags": self._persist_tags(entry.get("tags") or {}),
            "show": self._persist_show(entry.get("show")),
        }

    def _persist_model_entry(self, entry: dict) -> dict:
        return {
            "digest": entry.get("digest", ""),
            "cached_at": entry.get("cached_at", ""),
            "tags": self._persist_tags(entry.get("tags") or {}),
            "show": self._persist_show(entry.get("show")),
        }

    def _persist_tags(self, tags: dict) -> dict:
        if not isinstance(tags, dict):
            return {}
        return {
            key: tags.get(key)
            for key in self.PERSISTED_TAGS_FIELDS
            if key in tags
        }

    def _persist_show(self, show: dict | None) -> dict | None:
        if show is None:
            return None
        if not isinstance(show, dict):
            return {}

        details = show.get("details")
        if not isinstance(details, dict):
            details = {}

        persisted = {
            "details": {
                key: details.get(key)
                for key in self.PERSISTED_DETAILS_FIELDS
                if key in details
            },
            "capabilities": show.get("capabilities") or details.get("capabilities") or [],
            "modified_at": show.get("modified_at"),
            "context_window": self._extract_context_window(show),
            "license_short": self._extract_license_short(show),
            "has_system_prompt": self._extract_has_system_prompt(show),
        }

        if self._persist_sensitive_text:
            for field_name in self.SENSITIVE_SHOW_FIELDS:
                value = show.get(field_name)
                if value:
                    persisted[field_name] = value

        return persisted

    def _normalize(self, name: str, entry: dict) -> dict:
        tags = entry.get("tags") or {}
        show = entry.get("show") or {}
        details = show.get("details") or tags.get("details") or {}
        context_window = self._extract_context_window(show)
        license_full = show.get("license") or None
        license_short = self._extract_license_short(show)
        system_prompt = show.get("system_prompt")
        if system_prompt is None:
            system_prompt = show.get("system") or None
        has_system_prompt = self._extract_has_system_prompt(show)

        size_bytes = tags.get("size")

        return {
            "name": name,
            "family": details.get("family"),
            "families": details.get("families") or [],
            "capabilities": show.get("capabilities") or details.get("capabilities") or [],
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "format": details.get("format"),
            "size_bytes": size_bytes,
            "size_human": format_bytes(size_bytes),
            "modified_at": tags.get("modified_at"),
            "digest": tags.get("digest") or entry.get("digest", ""),
            "context_window": context_window,
            "license_full": license_full,
            "license_short": license_short,
            "system_prompt": system_prompt,
            "has_system_prompt": has_system_prompt,
            "template": show.get("template") or None,
            "parent_model": details.get("parent_model") or None,
            "parameters": show.get("parameters") or None,
            "model_info": show.get("model_info") or {},
        }

    def _extract_context_window(self, show: dict) -> int | None:
        context_window = show.get("context_window")
        if context_window is not None:
            try:
                return int(context_window)
            except (ValueError, TypeError):
                pass

        modelinfo = show.get("model_info") or {}
        ctx_raw = next((v for k, v in modelinfo.items() if k.endswith(".context_length")), None)
        if ctx_raw is not None:
            try:
                return int(ctx_raw)
            except (ValueError, TypeError):
                pass

        return None

    def _extract_license_short(self, show: dict) -> str | None:
        license_short = show.get("license_short")
        if license_short:
            return license_short

        license_full = show.get("license") or None
        if license_full:
            first_line = license_full.split("\n")[0].strip()
            return first_line[:80] if first_line else None

        return None

    def _extract_has_system_prompt(self, show: dict) -> bool:
        if "has_system_prompt" in show:
            return bool(show.get("has_system_prompt"))
        return bool(show.get("system_prompt") or show.get("system"))


def filter_models(models: list, filters: dict) -> list:
    return [m for m in models if _model_matches(m, filters)]


def _model_matches(model: dict, filters: dict) -> bool:
    name_query = (filters.get("name") or "").strip().lower()
    if name_query and name_query not in (model.get("name") or "").lower():
        return False

    family = filters.get("family") or ""
    if family and family != model.get("family"):
        return False

    capabilities = filters.get("capabilities") or set()
    if capabilities:
        if not capabilities.issubset(set(model.get("capabilities") or [])):
            return False

    param_size = filters.get("parameter_size") or ""
    if param_size and param_size != model.get("parameter_size"):
        return False

    quantization = filters.get("quantization_level") or ""
    if quantization and quantization != model.get("quantization_level"):
        return False

    max_size = filters.get("max_size_bytes")
    if max_size and (model.get("size_bytes") or 0) > max_size:
        return False

    min_ctx = filters.get("min_context")
    if min_ctx and (model.get("context_window") or 0) < min_ctx:
        return False

    lic = filters.get("license_short") or ""
    if lic and lic != model.get("license_short"):
        return False

    if filters.get("has_system_prompt") is True and not (
        model.get("has_system_prompt") or model.get("system_prompt")
    ):
        return False

    parent_query = (filters.get("parent_model") or "").strip().lower()
    if parent_query and parent_query not in (model.get("parent_model") or "").lower():
        return False

    return True
