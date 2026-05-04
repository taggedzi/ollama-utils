import json
import pytest
from pathlib import Path
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
