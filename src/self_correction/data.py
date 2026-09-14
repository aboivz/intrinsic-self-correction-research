from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from .artifacts import sha256
from .models import Manifest, ManifestItem

SOURCE_COMMIT = "69aa1012e2dfc4c57905bedf057c4e2fbc036101"


def normalized_question(question: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question).strip())


def _item(index: int, source: dict) -> ManifestItem:
    question = normalized_question(source["question"])
    question_hash = hashlib.sha256(question.encode()).hexdigest()
    return ManifestItem(source_index=index, sample_id=f"boolq-val-{index:06d}-{question_hash[:8]}", question=question, label=bool(source["answer"]), source_commit=SOURCE_COMMIT, question_sha256=question_hash)


def select_stratified(source: list[dict], count: int, seed: int) -> list[ManifestItem]:
    if count > len(source):
        raise ValueError(f"requested {count} items from {len(source)} records")
    groups = {False: [], True: []}
    for index, row in enumerate(source):
        groups[bool(row["answer"])].append(_item(index, row))
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    false_count = count // 2
    true_count = count - false_count
    if len(groups[False]) < false_count or len(groups[True]) < true_count:
        raise ValueError("cannot stratify requested count")
    selected = groups[False][:false_count] + groups[True][:true_count]
    rng.shuffle(selected)
    return selected


def select_item_subset(items: list[ManifestItem], count: int, seed: int) -> list[ManifestItem]:
    if count > len(items):
        raise ValueError(f"requested {count} items from {len(items)} records")
    groups = {False: [], True: []}
    for item in items:
        groups[item.label].append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = groups[False][: count // 2] + groups[True][: count - count // 2]
    if len(selected) != count:
        raise ValueError("cannot stratify requested count")
    rng.shuffle(selected)
    return selected


def make_manifest(source: list[dict], items: list[ManifestItem], seed: int) -> Manifest:
    body = {"dataset": "BoolQ validation from USC pinned commit", "source_count": len(source), "selection": "label-stratified random sample without replacement", "seed": seed, "items": [item.model_dump() for item in items]}
    return Manifest(**body, sha256=sha256(body))


def write_manifests(source_path: Path, destination: Path, main_n: int, smoke_n: int, wavering_n: int, seed: int) -> None:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    main_items = select_stratified(source, main_n, seed)
    smoke_items = select_item_subset(main_items, smoke_n, seed + 1)
    wavering_items = select_item_subset(main_items, wavering_n, seed + 2)
    destination.mkdir(parents=True, exist_ok=True)
    for name, items in ((f"boolq_main_{main_n}.json", main_items), (f"boolq_smoke_{smoke_n}.json", smoke_items), (f"boolq_wavering_{wavering_n}.json", wavering_items)):
        (destination / name).write_text(make_manifest(source, items, seed).model_dump_json(indent=2) + "\n", encoding="utf-8")
