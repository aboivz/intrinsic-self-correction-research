import pytest

from self_correction.evaluation import parse_yes_no


@pytest.mark.parametrize(("raw", "status", "answer"), [("Yes", "valid_yes", "yes"), (" no.", "valid_no", "no"), ("YES\n", "valid_yes", "yes"), ("Yesterday...", "invalid", None), ("I cannot determine", "invalid", None), ("Yes, but actually no", "ambiguous", None), ("", "invalid", None)])
def test_parser_cases(raw, status, answer):
    parsed = parse_yes_no(raw)
    assert (parsed.status, parsed.answer) == (status, answer)
