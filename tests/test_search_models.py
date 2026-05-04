import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from tz_ollama_utils.search_models import ModelSearchCache


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
    mock = MagicMock()
    mock.read.return_value = body
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
