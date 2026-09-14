from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import yaml

from .models import JournalEvent


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def append_event(path: Path, event: JournalEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(event.model_dump_json() + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_events(path: Path) -> list[JournalEvent]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [JournalEvent.model_validate_json(line) for line in file if line.strip()]


def completed_request_ids(events: Iterable[JournalEvent]) -> set[str]:
    return {event.request_id for event in events}
