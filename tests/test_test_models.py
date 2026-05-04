import pytest

from tz_ollama_utils.test_models import (
    classify_failure,
    model_supports_embeddings,
    normalize_ollama_api_base_url,
    parse_parameters_text,
    parse_args,
    redact_export_text,
    sanitize_api_base_url_for_export,
    sanitize_model_metadata,
    validate_report_path,
    write_yaml_report,
)


# --- normalize_ollama_api_base_url ---

def test_normalize_empty_raises():
    with pytest.raises(ValueError):
        normalize_ollama_api_base_url("")


def test_normalize_whitespace_only_raises():
    with pytest.raises(ValueError):
        normalize_ollama_api_base_url("   ")


def test_normalize_none_raises():
    with pytest.raises(ValueError):
        normalize_ollama_api_base_url(None)


def test_normalize_base_url_appends_api():
    assert normalize_ollama_api_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/api"


def test_normalize_already_has_api_suffix():
    assert normalize_ollama_api_base_url("http://127.0.0.1:11434/api") == "http://127.0.0.1:11434/api"


def test_normalize_trailing_slash_stripped():
    assert normalize_ollama_api_base_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434/api"


def test_normalize_strips_whitespace():
    assert normalize_ollama_api_base_url("  http://127.0.0.1:11434  ") == "http://127.0.0.1:11434/api"


def test_normalize_rejects_remote_without_opt_in():
    with pytest.raises(ValueError, match="--allow-remote"):
        normalize_ollama_api_base_url("https://example.com:11434")


def test_parse_args_accepts_secure_remote_with_allow_remote():
    args = parse_args(["tz-ollama-utils-test", "--allow-remote", "--api-base-url", "https://example.com:11434"])
    assert args.allow_remote is True
    assert args.api_base_url == "https://example.com:11434/api"


def test_parse_args_metadata_previews_default_false():
    args = parse_args(["tz-ollama-utils-test"])
    assert args.include_metadata_previews is False


def test_parse_args_metadata_previews_opt_in():
    args = parse_args(["tz-ollama-utils-test", "--include-metadata-previews"])
    assert args.include_metadata_previews is True


def test_parse_args_force_default_false():
    args = parse_args(["tz-ollama-utils-test"])
    assert args.force is False


def test_parse_args_force_opt_in():
    args = parse_args(["tz-ollama-utils-test", "--force"])
    assert args.force is True


# --- parse_parameters_text ---

def test_parse_parameters_empty_string():
    assert parse_parameters_text("") == {}


def test_parse_parameters_none():
    assert parse_parameters_text(None) == {}


def test_parse_parameters_single_line():
    assert parse_parameters_text("temperature 0.7") == {"temperature": "0.7"}


def test_parse_parameters_multiple_lines():
    result = parse_parameters_text("num_ctx 4096\ntemperature 0.7")
    assert result == {"num_ctx": "4096", "temperature": "0.7"}


def test_parse_parameters_value_with_spaces():
    result = parse_parameters_text('stop "</s>"')
    assert result == {"stop": '"</s>"'}


def test_sanitize_model_metadata_excludes_previews_by_default():
    result = sanitize_model_metadata(
        {
            "license": "MIT",
            "system": "You are helpful.",
            "template": "{{ .Prompt }}",
            "modelfile": "FROM llama3.2",
            "parameters": "num_ctx 4096",
        }
    )
    assert result["license_available"] is True
    assert result["license_preview"] is None
    assert result["system_available"] is True
    assert result["system_preview"] is None
    assert result["parameters_available"] is True
    assert result["parameters_preview"] is None
    assert result["parameters_parsed"] == {"num_ctx": "4096"}


def test_sanitize_model_metadata_includes_previews_when_opted_in():
    result = sanitize_model_metadata(
        {"system": "You are helpful.", "parameters": "num_ctx 4096"},
        include_previews=True,
    )
    assert result["system_preview"]["preview"] == "You are helpful."
    assert result["parameters_preview"]["preview"] == "num_ctx 4096"


def test_redact_export_text_redacts_remote_urls_and_absolute_paths():
    value = (
        "See https://example.com:11434/api/show and "
        "/home/alice/private/model_report.yaml and C:\\Users\\alice\\secret.txt"
    )
    result = redact_export_text(value)
    assert "<redacted-remote-url>" in result
    assert "/home/alice/private/model_report.yaml" not in result
    assert "C:\\Users\\alice\\secret.txt" not in result
    assert result.count("<redacted-path>") == 2


def test_sanitize_api_base_url_preserves_loopback_and_redacts_remote_host():
    assert sanitize_api_base_url_for_export("http://127.0.0.1:11434/api") == "http://127.0.0.1:11434/api"
    assert sanitize_api_base_url_for_export("https://example.com:11434/api") == "https://<redacted-host>/api"


def test_validate_report_path_rejects_non_yaml_suffix(tmp_path):
    with pytest.raises(ValueError, match=r"\.yaml"):
        validate_report_path(tmp_path / "report.txt")


def test_validate_report_path_rejects_existing_file_without_force(tmp_path):
    report_path = tmp_path / "report.yaml"
    report_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--force"):
        validate_report_path(report_path)


def test_validate_report_path_accepts_existing_file_with_force(tmp_path):
    report_path = tmp_path / "report.yaml"
    report_path.write_text("existing\n", encoding="utf-8")

    assert validate_report_path(report_path, force=True) == report_path


def test_validate_report_path_rejects_symlink_target(tmp_path):
    target = tmp_path / "report.yaml"
    target.write_text("existing\n", encoding="utf-8")
    symlink_path = tmp_path / "linked.yaml"
    try:
        symlink_path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        validate_report_path(symlink_path, force=True)


def test_validate_report_path_rejects_symlink_parent(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "linked-dir"
    try:
        link_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symlinked directories"):
        validate_report_path(link_dir / "report.yaml", force=True)


def test_write_yaml_report_force_overwrites_existing_file(tmp_path):
    report_path = tmp_path / "report.yaml"
    report_path.write_text("old\n", encoding="utf-8")

    write_yaml_report({"status": "new"}, report_path, force=True)

    assert 'status: "new"' in report_path.read_text(encoding="utf-8")


# --- classify_failure ---

def test_classify_embedding_input_issue():
    assert classify_failure("embedding models require input text", 1) == "script_input_issue"


def test_classify_timeout():
    assert classify_failure("Timed out after 300 seconds", None) == "timeout"


def test_classify_runner_crash():
    assert classify_failure("llama runner process has terminated", 1) == "runner_crash"


def test_classify_missing_dependency():
    assert classify_failure("ollama not found in PATH", 1) == "missing_dependency"


def test_classify_runtime_error():
    assert classify_failure("some generic error", 1) == "runtime_error"


def test_classify_unknown_error_zero_returncode():
    assert classify_failure("some generic error", 0) == "unknown_error"


def test_classify_unknown_error_none_returncode():
    assert classify_failure("some generic error", None) == "unknown_error"


# --- model_supports_embeddings ---

def test_model_supports_embeddings_true_for_embedding_capability():
    assert model_supports_embeddings({"capabilities": ["completion", "embedding"]}) is True


def test_model_supports_embeddings_false_for_non_embedding_capabilities():
    assert model_supports_embeddings({"capabilities": ["completion", "tools"]}) is False


def test_model_supports_embeddings_false_for_missing_metadata():
    assert model_supports_embeddings(None) is False
