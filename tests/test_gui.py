from types import SimpleNamespace
from unittest.mock import patch

from tz_ollama_utils.gui import (
    DESTRUCTIVE_ACTIONS_ENV_VAR,
    OllamaUtilsApp,
    destructive_actions_enabled_from_env,
    parse_args,
)


def test_destructive_actions_enabled_by_default():
    assert destructive_actions_enabled_from_env({}) is True


def test_destructive_actions_disabled_by_env_truthy_values():
    assert destructive_actions_enabled_from_env({DESTRUCTIVE_ACTIONS_ENV_VAR: "1"}) is False
    assert destructive_actions_enabled_from_env({DESTRUCTIVE_ACTIONS_ENV_VAR: "true"}) is False
    assert destructive_actions_enabled_from_env({DESTRUCTIVE_ACTIONS_ENV_VAR: "On"}) is False


def test_parse_args_defaults_to_destructive_actions_enabled(monkeypatch):
    monkeypatch.delenv(DESTRUCTIVE_ACTIONS_ENV_VAR, raising=False)
    args = parse_args(["tz-ollama-utils-gui"])
    assert args.disable_destructive_actions is False
    assert args.destructive_actions_enabled is True


def test_parse_args_disables_destructive_actions_with_flag(monkeypatch):
    monkeypatch.delenv(DESTRUCTIVE_ACTIONS_ENV_VAR, raising=False)
    args = parse_args(["tz-ollama-utils-gui", "--disable-destructive-actions"])
    assert args.disable_destructive_actions is True
    assert args.destructive_actions_enabled is False


def test_parse_args_disables_destructive_actions_with_env(monkeypatch):
    monkeypatch.setenv(DESTRUCTIVE_ACTIONS_ENV_VAR, "yes")
    args = parse_args(["tz-ollama-utils-gui"])
    assert args.destructive_actions_enabled is False


def test_search_delete_logs_and_blocks_when_destructive_actions_disabled():
    app = object.__new__(OllamaUtilsApp)
    logged = []
    app._destructive_actions_enabled = False
    app.root = object()
    app.append_log = logged.append
    app._search_model_by_name = lambda name: {"name": name, "size_human": "1.00 GiB"}

    mock_messagebox = SimpleNamespace()
    with patch("tz_ollama_utils.gui._mb", mock_messagebox):
        with patch.object(mock_messagebox, "showwarning", create=True) as showwarning:
            app._search_delete_model("llama3.2:3b")

    assert any("Blocked delete" in line for line in logged)
    showwarning.assert_called_once()


def test_search_delete_logs_cancelled_delete():
    app = object.__new__(OllamaUtilsApp)
    logged = []
    app._destructive_actions_enabled = True
    app.append_log = logged.append
    app._search_model_by_name = lambda name: {"name": name, "size_human": "1.00 GiB"}
    app._confirm_destructive_model_delete = lambda name, size_text: False

    app._search_delete_model("llama3.2:3b")

    assert any("Delete cancelled" in line for line in logged)
