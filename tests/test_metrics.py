from __future__ import annotations

from self_correction.artifacts import sha256
from self_correction.metrics import aggregate
from self_correction.models import JournalEvent, Message, ModelIdentity


def event(sample: int, round_number: int, correct: bool) -> JournalEvent:
    return JournalEvent(run_id="r", request_id=f"{sample}-{round_number}", sample_id=str(sample), condition="confirmatory", round=round_number, messages=[Message(role="user", content="q")], raw_output="Yes", parsed_answer="yes", parse_status="valid_yes", is_correct=correct, label=True, model=ModelIdentity(provider="fake", model_id="fake"), prompt_hash=sha256("p"), config_hash=sha256("c"))


def test_hand_transition_fixture_and_delta_identity():
    transitions = [(True, True)] * 3 + [(True, False)] * 2 + [(False, True)] + [(False, False)] * 4
    metrics = aggregate([part for index, pair in enumerate(transitions) for part in (event(index, 0, pair[0]), event(index, 1, pair[1]))])
    assert [metrics[key] for key in ("CC", "CW", "WC", "WW")] == [3, 2, 1, 4]
    assert metrics["delta_A"] == (metrics["WC"] - metrics["CW"]) / 10
    assert metrics["conditional_recovery"] == 1 / 5
    assert metrics["conditional_regression"] == 2 / 5


def test_mcnemar_zero_discordant_and_bootstrap_reproducible():
    events = [event(0, 0, True), event(0, 1, True)]
    assert aggregate(events)["mcnemar_exact_two_sided_p"] == 1.0
    many = events + [event(index, 0, False) for index in range(1, 6)] + [event(index, 1, True) for index in range(1, 6)]
    assert aggregate(many)["bootstrap_delta_A_95_ci"] == aggregate(many)["bootstrap_delta_A_95_ci"]
    assert aggregate(many) == aggregate(list(reversed(many)))
