"""Output parsing: plain JSON, fences, whitespace, garbage."""

from docutune.evaluation.parser import parse_model_output


def test_plain_json():
    result = parse_model_output('{"name": "A"}')
    assert result.json_valid
    assert result.parsed == {"name": "A"}
    assert result.is_object


def test_surrounding_whitespace():
    result = parse_model_output('   \n{"name": "A"}  \n ')
    assert result.json_valid


def test_markdown_code_fence():
    raw = '```json\n{"name": "A", "skills": []}\n```'
    result = parse_model_output(raw)
    assert result.json_valid
    assert result.matched_via == "code_fence"


def test_json_with_leading_text():
    raw = 'Here is the extraction:\n{"name": "A"}'
    result = parse_model_output(raw)
    assert result.json_valid
    assert result.matched_via == "brace_slice"


def test_invalid_json():
    result = parse_model_output("not json at all")
    assert not result.json_valid
    assert result.parsed is None
    assert result.error


def test_empty_output():
    result = parse_model_output("")
    assert not result.json_valid
    assert "empty" in (result.error or "")


def test_non_object_json_counts_as_not_object():
    result = parse_model_output('["a", "b"]')
    assert result.json_valid  # valid JSON...
    assert not result.is_object  # ...but not a resume object


def test_truncated_json_fails():
    result = parse_model_output('{"name": "A", "skills": [')
    assert not result.json_valid


def test_none_safe():
    result = parse_model_output(None)
    assert not result.json_valid
