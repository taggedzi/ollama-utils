from types import SimpleNamespace
from unittest.mock import patch

import pytest

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


def test_selected_report_path_rejects_existing_file(tmp_path):
    report_path = tmp_path / "report.yaml"
    report_path.write_text("existing\n", encoding="utf-8")

    app = object.__new__(OllamaUtilsApp)
    app.report_path_var = SimpleNamespace(get=lambda: str(report_path))
    app.report_force_var = SimpleNamespace(get=lambda: False)

    selected, error = app._selected_report_path()

    assert selected is None
    assert "--force" in error


def test_selected_report_path_allows_existing_file_with_overwrite_opt_in(tmp_path):
    report_path = tmp_path / "report.yaml"
    report_path.write_text("existing\n", encoding="utf-8")

    app = object.__new__(OllamaUtilsApp)
    app.report_path_var = SimpleNamespace(get=lambda: str(report_path))
    app.report_force_var = SimpleNamespace(get=lambda: True)

    selected, error = app._selected_report_path()

    assert error is None
    assert selected == str(report_path)


def test_selected_report_path_rejects_non_yaml_suffix(tmp_path):
    app = object.__new__(OllamaUtilsApp)
    app.report_path_var = SimpleNamespace(get=lambda: str(tmp_path / "report.yml"))
    app.report_force_var = SimpleNamespace(get=lambda: False)

    selected, error = app._selected_report_path()

    assert selected is None
    assert ".yaml" in error


def test_selected_report_path_rejects_symlink_target(tmp_path):
    target = tmp_path / "report.yaml"
    target.write_text("existing\n", encoding="utf-8")
    symlink_path = tmp_path / "linked.yaml"
    try:
        symlink_path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")

    app = object.__new__(OllamaUtilsApp)
    app.report_path_var = SimpleNamespace(get=lambda: str(symlink_path))
    app.report_force_var = SimpleNamespace(get=lambda: True)

    selected, error = app._selected_report_path()

    assert selected is None
    assert "symlink" in error
