import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from tz_ollama_utils.search_models import ModelSearchCache, filter_models


def test_load_returns_false_when_file_missing(tmp_path):
    cache = ModelSearchCache()
    result = cache.load(tmp_path / "missing.json")
    assert result is False
    assert cache.get_models() == []


def test_load_returns_false_on_corrupt_json(tmp_path):
    p = tmp_path / "cache.json"
    p.write_text("not json")
    cache = ModelSearchCache()
    assert cache.load(p) is False


def test_load_populates_models(tmp_path):
    data = {
        "api_base_url": "http://localhost:11434/api",
        "models": {
            "llama3.2:3b": {
                "digest": "sha256:abc",
                "cached_at": "2026-01-01T00:00:00",
                "tags": {"name": "llama3.2:3b", "size": 2000000000},
                "show": {},
            }
        },
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(data))
    cache = ModelSearchCache()
    assert cache.load(p) is True
    models = cache.get_models()
    assert len(models) == 1
    assert models[0]["name"] == "llama3.2:3b"


def test_save_writes_json(tmp_path):
    cache = ModelSearchCache()
    cache._api_base_url = "http://localhost:11434/api"
    cache._models = {
        "test:latest": {
            "digest": "sha256:123",
            "cached_at": "2026-01-01T00:00:00",
            "tags": {},
            "show": {},
        }
    }
    p = tmp_path / "cache.json"
    cache.save(p)
    data = json.loads(p.read_text())
    assert data["api_base_url"] == "http://localhost:11434/api"
    assert "test:latest" in data["models"]


def test_save_creates_parent_dirs(tmp_path):
    cache = ModelSearchCache()
    p = tmp_path / "nested" / "dir" / "cache.json"
    cache.save(p)
    assert p.exists()


def test_invalidate_clears_state_and_deletes_file(tmp_path):
    p = tmp_path / "cache.json"
    p.write_text(json.dumps({"api_base_url": "", "models": {"x": {"digest": "", "cached_at": "", "tags": {}, "show": {}}}}))
    cache = ModelSearchCache()
    cache.load(p)
    cache.invalidate(p)
    assert cache.get_models() == []
    assert not p.exists()


def test_invalidate_is_safe_when_file_missing(tmp_path):
    cache = ModelSearchCache()
    cache.invalidate(tmp_path / "nonexistent.json")  # must not raise


def test_remove_model_removes_entry_and_saves(tmp_path):
    cache = ModelSearchCache()
    cache._models = {
        "a:latest": {"digest": "x", "cached_at": "", "tags": {}, "show": {}},
        "b:latest": {"digest": "y", "cached_at": "", "tags": {}, "show": {}},
    }
    p = tmp_path / "cache.json"
    cache.remove_model("a:latest", cache_path=p)
    assert len(cache.get_models()) == 1
    assert cache.get_models()[0]["name"] == "b:latest"
    data = json.loads(p.read_text())
    assert "a:latest" not in data["models"]


def test_update_model_show_persists(tmp_path):
    cache = ModelSearchCache()
    cache._models = {
        "m:latest": {"digest": "d", "cached_at": "", "tags": {}, "show": {}},
    }
    p = tmp_path / "cache.json"
    cache.update_model_show("m:latest", {"license": "MIT"}, cache_path=p)
    data = json.loads(p.read_text())
    assert data["models"]["m:latest"]["show"]["license"] == "MIT"


def _mock_response(data: dict):
    body = json.dumps(data).encode()
    return _mock_bytes_response(body)


def _mock_bytes_response(body: bytes):
    mock = MagicMock()
    mock.headers = {}
    offset = {"value": 0}

    def _read(amount=-1):
        if amount < 0:
            amount = len(body) - offset["value"]
        chunk = body[offset["value"]:offset["value"] + amount]
        offset["value"] += len(chunk)
        return chunk

    mock.read.side_effect = _read
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_refresh_fetches_tags_and_show(tmp_path):
    tags_resp = _mock_response({"models": [{"name": "llama3.2:3b", "digest": "sha256:new", "size": 2000000000}]})
    show_resp = _mock_response({"details": {"family": "llama"}, "license": "MIT"})
    responses = iter([tags_resp, show_resp])

    cache = ModelSearchCache()
    progress = []
    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", side_effect=lambda r, timeout=None: next(responses)):
        cache.refresh("http://localhost:11434/api", on_progress=lambda n, d, t: progress.append((n, d, t)), cache_path=tmp_path / "c.json")

    models = cache.get_models()
    assert len(models) == 1
    assert models[0]["name"] == "llama3.2:3b"
    assert progress == [("llama3.2:3b", 1, 1)]


def test_refresh_skips_show_when_digest_unchanged(tmp_path):
    cache = ModelSearchCache()
    cache._models = {
        "llama3.2:3b": {
            "digest": "sha256:same",
            "cached_at": "2026-01-01",
            "tags": {"name": "llama3.2:3b", "size": 1},
            "show": {"details": {"family": "llama"}},
        }
    }
    tags_resp = _mock_response({"models": [{"name": "llama3.2:3b", "digest": "sha256:same", "size": 1}]})

    show_calls = []
    def fake_urlopen(req, timeout=None):
        if hasattr(req, "data") and req.data:  # POST = /show call
            show_calls.append(req.full_url)
        return tags_resp

    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", side_effect=fake_urlopen):
        cache.refresh("http://localhost:11434/api", cache_path=tmp_path / "c.json")

    assert show_calls == []


def test_refresh_removes_models_absent_from_api(tmp_path):
    cache = ModelSearchCache()
    cache._models = {"old:latest": {"digest": "x", "cached_at": "", "tags": {}, "show": {}}}
    tags_resp = _mock_response({"models": []})

    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", return_value=tags_resp):
        cache.refresh("http://localhost:11434/api", cache_path=tmp_path / "c.json")

    assert cache.get_models() == []


def test_refresh_tolerates_show_fetch_error(tmp_path):
    tags_resp = _mock_response({"models": [{"name": "bad:latest", "digest": "sha256:xyz", "size": 100}]})
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        if call_count[0] == 0:
            call_count[0] += 1
            return tags_resp
        raise OSError("connection refused")

    cache = ModelSearchCache()
    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", side_effect=fake_urlopen):
        cache.refresh("http://localhost:11434/api", cache_path=tmp_path / "c.json")

    models = cache.get_models()
    assert len(models) == 1
    assert models[0]["name"] == "bad:latest"
    # show should be stored as None (sentinel for failed fetch, not empty dict)
    assert cache._models["bad:latest"]["show"] is None

    # On the next refresh with a working show endpoint, the model must be re-fetched
    # because cached.get("show") is None triggers the staleness gate.
    tags_resp2 = _mock_response({"models": [{"name": "bad:latest", "digest": "sha256:xyz", "size": 100}]})
    show_resp2 = _mock_response({"details": {"family": "llama"}})
    responses2 = iter([tags_resp2, show_resp2])
    show_calls = [0]

    def fake_urlopen2(req, timeout=None):
        resp = next(responses2)
        if hasattr(req, "data") and req.data:
            show_calls[0] += 1
        return resp

    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", side_effect=fake_urlopen2):
        cache.refresh("http://localhost:11434/api", cache_path=tmp_path / "c.json")

    assert show_calls[0] == 1, "show should be re-fetched when previously stored as None"
    assert cache._models["bad:latest"]["show"] == {"details": {"family": "llama"}}


def test_refresh_aborts_on_tags_fetch_error(tmp_path):
    cache = ModelSearchCache()
    cache._models = {"existing:latest": {"digest": "sha256:abc", "cached_at": "2026-01-01", "tags": {}, "show": {"details": {}}}}

    save_calls = [0]
    original_save = cache.save

    def counting_save(*args, **kwargs):
        save_calls[0] += 1
        return original_save(*args, **kwargs)

    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", side_effect=OSError("network error")):
        with patch.object(cache, "save", side_effect=counting_save):
            cache.refresh("http://localhost:11434/api", cache_path=tmp_path / "c.json")

    # Cache must not be wiped and save must not be called
    assert len(cache.get_models()) == 1
    assert cache.get_models()[0]["name"] == "existing:latest"
    assert save_calls[0] == 0, "save should not be called when tags fetch fails"


def test_refresh_aborts_on_oversized_tags_response(tmp_path):
    response = _mock_bytes_response(b"x" * (4 * 1024 * 1024 + 1))

    cache = ModelSearchCache()
    cache._models = {
        "existing:latest": {
            "digest": "sha256:abc",
            "cached_at": "2026-01-01",
            "tags": {},
            "show": {"details": {}},
        }
    }

    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", return_value=response):
        cache.refresh("http://localhost:11434/api", cache_path=tmp_path / "c.json")

    assert cache.get_models()[0]["name"] == "existing:latest"


def _cache_with(tags, show):
    cache = ModelSearchCache()
    cache._models = {
        "test:latest": {"digest": tags.get("digest", ""), "cached_at": "2026-01-01", "tags": tags, "show": show}
    }
    return cache.get_models()[0]


def test_normalize_name():
    m = _cache_with({"name": "test:latest"}, {})
    assert m["name"] == "test:latest"


def test_normalize_family_and_params():
    m = _cache_with({}, {"details": {"family": "llama", "parameter_size": "3.6B", "quantization_level": "Q4_K_M", "format": "gguf"}})
    assert m["family"] == "llama"
    assert m["parameter_size"] == "3.6B"
    assert m["quantization_level"] == "Q4_K_M"
    assert m["format"] == "gguf"


def test_normalize_size_bytes_and_human():
    m = _cache_with({"size": 2000000000}, {})
    assert m["size_bytes"] == 2000000000
    assert m["size_human"] == "1.86 GiB"


def test_normalize_capabilities():
    m = _cache_with({}, {"capabilities": ["completion", "tools"]})
    assert m["capabilities"] == ["completion", "tools"]


def test_normalize_capabilities_empty_when_missing():
    m = _cache_with({}, {})
    assert m["capabilities"] == []


def test_normalize_capabilities_falls_back_to_details():
    m = _cache_with({}, {"details": {"capabilities": ["embedding"]}})
    assert m["capabilities"] == ["embedding"]


def test_normalize_context_window_from_modelinfo():
    m = _cache_with({}, {"model_info": {"llama.context_length": 131072}})
    assert m["context_window"] == 131072


def test_normalize_context_window_none_when_missing():
    m = _cache_with({}, {})
    assert m["context_window"] is None


def test_normalize_license_short_is_first_line():
    m = _cache_with({}, {"license": "MIT License\n\nCopyright (c) 2024..."})
    assert m["license_short"] == "MIT License"
    assert m["license_full"] == "MIT License\n\nCopyright (c) 2024..."


def test_normalize_license_none_when_empty():
    m = _cache_with({}, {})
    assert m["license_short"] is None
    assert m["license_full"] is None


def test_normalize_system_prompt():
    m = _cache_with({}, {"system": "You are helpful."})
    assert m["system_prompt"] == "You are helpful."


def test_normalize_system_prompt_none_when_missing():
    m = _cache_with({}, {})
    assert m["system_prompt"] is None


def test_normalize_parent_model():
    m = _cache_with({}, {"details": {"parent_model": "llama3.2"}})
    assert m["parent_model"] == "llama3.2"


def test_normalize_digest_from_tags():
    m = _cache_with({"digest": "sha256:abc"}, {})
    assert m["digest"] == "sha256:abc"


def test_normalize_handles_fully_empty_show():
    m = _cache_with({"name": "x:latest", "size": 0}, {})
    assert m["name"] == "test:latest"
    assert m["family"] is None
    assert m["capabilities"] == []
    assert m["size_bytes"] == 0


def _m(**kwargs):
    base = {
        "name": "test:latest", "family": "llama", "capabilities": ["completion"],
        "parameter_size": "7B", "quantization_level": "Q4_K_M",
        "size_bytes": 4_000_000_000, "context_window": 8192,
        "license_short": "MIT", "system_prompt": None, "parent_model": None,
    }
    base.update(kwargs)
    return base


def test_filter_empty_filters_returns_all():
    models = [_m(name="a:latest"), _m(name="b:latest")]
    assert filter_models(models, {}) == models


def test_filter_by_name_substring():
    models = [_m(name="llama3.2:3b"), _m(name="mistral:7b")]
    result = filter_models(models, {"name": "llama"})
    assert [r["name"] for r in result] == ["llama3.2:3b"]


def test_filter_by_name_case_insensitive():
    assert len(filter_models([_m(name="Llama3:latest")], {"name": "llama"})) == 1


def test_filter_by_family():
    models = [_m(family="llama"), _m(family="gemma")]
    assert filter_models(models, {"family": "llama"})[0]["family"] == "llama"
    assert len(filter_models(models, {"family": "llama"})) == 1


def test_filter_capabilities_requires_all():
    models = [_m(capabilities=["completion", "tools"]), _m(capabilities=["completion"])]
    result = filter_models(models, {"capabilities": {"completion", "tools"}})
    assert len(result) == 1
    assert "tools" in result[0]["capabilities"]


def test_filter_by_parameter_size():
    models = [_m(parameter_size="7B"), _m(parameter_size="3B")]
    assert filter_models(models, {"parameter_size": "7B"})[0]["parameter_size"] == "7B"
    assert len(filter_models(models, {"parameter_size": "7B"})) == 1


def test_filter_by_quantization():
    models = [_m(quantization_level="Q4_K_M"), _m(quantization_level="Q8_0")]
    assert len(filter_models(models, {"quantization_level": "Q8_0"})) == 1


def test_filter_by_max_size_bytes():
    models = [_m(size_bytes=2_000_000_000), _m(size_bytes=8_000_000_000)]
    result = filter_models(models, {"max_size_bytes": 4_000_000_000})
    assert len(result) == 1
    assert result[0]["size_bytes"] == 2_000_000_000


def test_filter_by_min_context():
    models = [_m(context_window=4096), _m(context_window=32768)]
    result = filter_models(models, {"min_context": 8192})
    assert len(result) == 1
    assert result[0]["context_window"] == 32768


def test_filter_by_license():
    models = [_m(license_short="MIT"), _m(license_short="Apache-2.0")]
    assert len(filter_models(models, {"license_short": "MIT"})) == 1


def test_filter_has_system_prompt():
    models = [_m(system_prompt="You are helpful."), _m(system_prompt=None)]
    result = filter_models(models, {"has_system_prompt": True})
    assert len(result) == 1
    assert result[0]["system_prompt"] is not None


def test_filter_by_parent_model_substring():
    models = [_m(parent_model="llama3.2"), _m(parent_model="gemma2")]
    assert len(filter_models(models, {"parent_model": "llama"})) == 1


def test_filter_and_combination():
    models = [
        _m(family="llama", capabilities=["completion", "tools"]),
        _m(family="llama", capabilities=["completion"]),
        _m(family="gemma", capabilities=["completion", "tools"]),
    ]
    result = filter_models(models, {"family": "llama", "capabilities": {"tools"}})
    assert len(result) == 1
    assert result[0]["family"] == "llama" and "tools" in result[0]["capabilities"]


def test_fetch_show_uses_model_field_in_body(tmp_path):
    captured_bodies = []

    def fake_urlopen(req, timeout=None):
        if req.data:
            captured_bodies.append(json.loads(req.data))
        return _mock_response({"details": {"family": "llama"}})

    cache = ModelSearchCache()
    with patch("tz_ollama_utils.search_models.urllib_request.urlopen", side_effect=fake_urlopen):
        cache._fetch_show("http://localhost:11434/api", "llama3.2:3b")

    assert len(captured_bodies) == 1
    assert captured_bodies[0] == {"model": "llama3.2:3b"}
    assert "name" not in captured_bodies[0]
