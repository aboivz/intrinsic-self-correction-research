from __future__ import annotations

import time
from pathlib import Path

from .artifacts import append_event, completed_request_ids, read_events, sha256
from .backend import GenerationBackend
from .evaluation import is_correct, parse_yes_no
from .models import DecodingConfig, JournalEvent, Manifest, Message


def request_id(run_id: str, sample_id: str, condition: str, replicate: int, round_number: int) -> str:
    return sha256([run_id, sample_id, condition, replicate, round_number])[:32]


def run_experiment(
    manifest: Manifest,
    backend: GenerationBackend,
    journal_path: Path,
    run_id: str,
    condition: str,
    rounds: int,
    prompts: dict,
    decoding: DecodingConfig,
    prompt_hash: str,
    config_hash: str,
) -> int:
    """Run every item each round. Labels enter only after backend generation."""
    existing = read_events(journal_path)
    by_request = {event.request_id: event for event in existing}
    calls = 0
    for item in manifest.items:
        initial = Message(role="user", content=prompts["initial"]["user"].format(question=item.question))
        messages = [initial]
        previous_output = ""
        for round_number in range(rounds + 1):
            identifier = request_id(run_id, item.sample_id, condition, 0, round_number)
            event = by_request.get(identifier)
            if event is None:
                attempts = 0
                while True:
                    result = backend.generate(messages, decoding, identifier)
                    calls += 1
                    if not result.transient or attempts >= 2:
                        break
                    attempts += 1
                    time.sleep(0.1 * (2**attempts))
                parsed = parse_yes_no(result.raw_output, result.error)
                event = JournalEvent(
                    run_id=run_id, request_id=identifier, sample_id=item.sample_id, condition=condition,
                    round=round_number, messages=messages, raw_output=result.raw_output,
                    parsed_answer=parsed.answer, parse_status=parsed.status,
                    is_correct=is_correct(parsed.answer, item.label), label=item.label, usage=result.usage,
                    api_metadata=result.metadata, retry_attempts=attempts, model=backend.identity,
                    prompt_hash=prompt_hash, config_hash=config_hash, error=result.error,
                )
                append_event(journal_path, event)
                by_request[identifier] = event
            previous_output = event.raw_output or ""
            if round_number < rounds:
                messages = [*messages, Message(role="assistant", content=previous_output), Message(role="user", content=prompts[condition]["user"])]
    return calls


def terminal_ids(journal_path: Path) -> set[str]:
    return completed_request_ids(read_events(journal_path))
