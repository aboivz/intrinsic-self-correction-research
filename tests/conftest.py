from __future__ import annotations

from self_correction.artifacts import sha256
from self_correction.models import Manifest, ManifestItem


def manifest(count: int = 10) -> Manifest:
    items = [ManifestItem(source_index=index, sample_id=f"boolq-val-{index:06d}-deadbeef", question=f"Question {index}?", label=bool(index % 2), source_commit="pinned", question_sha256="0" * 64) for index in range(count)]
    body = {"dataset": "test", "source_count": count, "selection": "test", "seed": 20260912, "items": [item.model_dump() for item in items]}
    return Manifest(**body, sha256=sha256(body))
