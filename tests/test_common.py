import pytest

from tz_ollama_utils.common import (
    clean_text,
    format_bytes,
    truncate_text,
    truncate_with_metadata,
    yaml_dump_lines,
    yaml_quote,
    yaml_scalar,
)


# --- format_bytes ---

def test_format_bytes_none():
    assert format_bytes(None) == "unknown"


def test_format_bytes_zero():
    assert format_bytes(0) == "0 B"


def test_format_bytes_bytes():
    assert format_bytes(512) == "512 B"


def test_format_bytes_kib():
    assert format_bytes(1024) == "1.00 KiB"


def test_format_bytes_fractional_kib():
    assert format_bytes(1536) == "1.50 KiB"


def test_format_bytes_mib():
    assert format_bytes(1024 * 1024) == "1.00 MiB"


def test_format_bytes_gib():
    assert format_bytes(1024 ** 3) == "1.00 GiB"


# --- clean_text ---

def test_clean_text_none():
    assert clean_text(None) == ""


def test_clean_text_empty():
    assert clean_text("") == ""


def test_clean_text_strips_ansi():
    assert clean_text("\x1b[31mfoo\x1b[0m") == "foo"


def test_clean_text_strips_spinner():
    assert clean_text("⠋loading") == "loading"


def test_clean_text_strips_surrounding_whitespace_per_line():
    assert clean_text("  hello  \n  world  ") == "hello\nworld"


def test_clean_text_removes_blank_lines():
    assert clean_text("hello\n\n\nworld") == "hello\nworld"


def test_clean_text_cr_becomes_lf():
    assert clean_text("hello\r\nworld") == "hello\nworld"


# --- truncate_text ---

def test_truncate_text_short():
    assert truncate_text("hello", 10) == "hello"


def test_truncate_text_exact_limit():
    assert truncate_text("hello", 5) == "hello"


def test_truncate_text_over_limit():
    result = truncate_text("hello world", 8)
    assert result == "hello..."
    assert len(result) == 8


# --- truncate_with_metadata ---

def test_truncate_with_metadata_none():
    result = truncate_with_metadata(None)
    assert result == {"preview": None, "truncated": False, "original_length": 0}


def test_truncate_with_metadata_short():
    result = truncate_with_metadata("hi")
    assert result == {"preview": "hi", "truncated": False, "original_length": 2}


def test_truncate_with_metadata_over_limit():
    long = "a" * 2001
    result = truncate_with_metadata(long)
    assert result["truncated"] is True
    assert result["original_length"] == 2001
    assert result["preview"].endswith("...")
    assert len(result["preview"]) == 2000


# --- yaml_quote ---

def test_yaml_quote_plain():
    assert yaml_quote("hello") == '"hello"'


def test_yaml_quote_embedded_double_quote():
    assert yaml_quote('say "hi"') == '"say \\"hi\\""'


def test_yaml_quote_newline():
    assert yaml_quote("line1\nline2") == '"line1\\nline2"'


def test_yaml_quote_backslash():
    assert yaml_quote("back\\slash") == '"back\\\\slash"'


# --- yaml_scalar ---

def test_yaml_scalar_none():
    assert yaml_scalar(None) == "null"


def test_yaml_scalar_true():
    assert yaml_scalar(True) == "true"


def test_yaml_scalar_false():
    assert yaml_scalar(False) == "false"


def test_yaml_scalar_int():
    assert yaml_scalar(42) == "42"


def test_yaml_scalar_float():
    assert yaml_scalar(3.14) == "3.14"


def test_yaml_scalar_string():
    assert yaml_scalar("hello") == '"hello"'


# --- yaml_dump_lines ---

def test_yaml_dump_lines_empty_dict():
    assert yaml_dump_lines({}) == ["{}"]


def test_yaml_dump_lines_empty_list():
    assert yaml_dump_lines([]) == ["[]"]


def test_yaml_dump_lines_flat_dict():
    assert yaml_dump_lines({"a": 1, "b": 2}) == ['a: 1', 'b: 2']


def test_yaml_dump_lines_nested_dict():
    assert yaml_dump_lines({"outer": {"inner": 3}}) == ['outer:', '  inner: 3']


def test_yaml_dump_lines_list_of_scalars():
    assert yaml_dump_lines([1, 2]) == ['- 1', '- 2']


def test_yaml_dump_lines_list_of_dicts():
    result = yaml_dump_lines([{"k": "v"}])
    assert result == ['- k: "v"']


def test_yaml_dump_lines_scalar():
    assert yaml_dump_lines("hello") == ['"hello"']


def test_yaml_dump_lines_indent():
    assert yaml_dump_lines({"a": 1}, indent=2) == ['  a: 1']


# --- StopRequested ---

def test_stop_requested_is_exception():
    from tz_ollama_utils.common import StopRequested
    assert issubclass(StopRequested, Exception)


def test_stop_requested_can_be_raised_and_caught():
    from tz_ollama_utils.common import StopRequested
    with pytest.raises(StopRequested):
        raise StopRequested
