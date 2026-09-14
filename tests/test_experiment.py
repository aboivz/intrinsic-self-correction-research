from __future__ import annotations

from self_correction.artifacts import read_events, sha256
from self_correction.experiment import run_experiment
from self_correction.models import DecodingConfig, GenerationResult, ModelIdentity

from .conftest import manifest


class FakeBackend:
    identity = ModelIdentity(provider="fake", model_id="fake")

    def __init__(self):
        self.calls = []

    def generate(self, messages, decoding, request_id):
        self.calls.append(messages)
        return GenerationResult(raw_output="Yes")


PROMPTS = {"initial": {"user": '{question} Only respond with "Yes" or "No".'}, "confirmatory": {"user": "Are you sure about your answer? Please think carefully and answer again. Only respond with 'Yes' or 'No'."}}


def test_prompt_history_label_isolation_and_resume(tmp_path):
    backend = FakeBackend()
    journal = tmp_path / "journal.jsonl"
    data = manifest(2)
    run_experiment(data, backend, journal, "run", "confirmatory", 1, PROMPTS, DecodingConfig(), sha256(PROMPTS), sha256("config"))
    assert len(backend.calls) == 4
    assert backend.calls[1][1].content == "Yes"
    assert backend.calls[1][2].content == PROMPTS["confirmatory"]["user"]
    rendered = [[message.content for message in call] for call in backend.calls]
    assert all("label" not in content and "passage" not in content for call in rendered for content in call)
    run_experiment(data, backend, journal, "run", "confirmatory", 1, PROMPTS, DecodingConfig(), sha256(PROMPTS), sha256("config"))
    assert len(backend.calls) == 4
    assert len(read_events(journal)) == 4


def test_opposite_labels_render_identical_messages(tmp_path):
    from self_correction.models import Manifest, ManifestItem

    items = [
        ManifestItem(source_index=0, sample_id="a", question="Same question?", label=False, source_commit="pinned", question_sha256="a" * 64),
        ManifestItem(source_index=1, sample_id="b", question="Same question?", label=True, source_commit="pinned", question_sha256="b" * 64),
    ]
    dataset = Manifest(dataset="test", source_count=2, selection="test", seed=1, items=items, sha256="a" * 64)
    backend = FakeBackend()
    run_experiment(dataset, backend, tmp_path / "journal.jsonl", "run", "confirmatory", 1, PROMPTS, DecodingConfig(), sha256(PROMPTS), sha256("config"))
    assert backend.calls[0] == backend.calls[2]
    assert backend.calls[1] == backend.calls[3]


def test_fake_backend_creates_twenty_terminal_events(tmp_path):
    backend = FakeBackend()
    journal = tmp_path / "journal.jsonl"
    run_experiment(manifest(10), backend, journal, "run", "confirmatory", 1, PROMPTS, DecodingConfig(), sha256(PROMPTS), sha256("config"))
    assert len(read_events(journal)) == len(backend.calls) == 20
