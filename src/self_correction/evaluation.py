from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

YES_NO = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
ANY_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    status: str


def parse_yes_no(raw_output: str | None, error: str | None = None) -> ParsedAnswer:
    if error is not None or raw_output is None:
        return ParsedAnswer(None, "error")
    text = unicodedata.normalize("NFKC", raw_output).strip()
    match = YES_NO.match(text)
    if not match:
        return ParsedAnswer(None, "invalid")
    answer = match.group(1).lower()
    answers = {word.lower() for word in ANY_YES_NO.findall(text)}
    if len(answers) > 1:
        return ParsedAnswer(None, "ambiguous")
    return ParsedAnswer(answer, f"valid_{answer}")


def is_correct(answer: str | None, label: bool) -> bool:
    return answer == ("yes" if label else "no")
