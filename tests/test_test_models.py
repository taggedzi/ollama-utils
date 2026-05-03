import pytest

from ollama_utils.test_models import (
    classify_failure,
    normalize_ollama_api_base_url,
    parse_parameters_text,
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
