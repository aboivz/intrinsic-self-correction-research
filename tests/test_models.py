import pytest

from self_correction.models import JournalEvent, Message, ModelIdentity


def test_pydantic_json_round_trip_and_schema_rejection():
    event = JournalEvent(run_id="r", request_id="id", sample_id="s", condition="confirmatory", round=0, messages=[Message(role="user", content="q")], parse_status="invalid", label=True, model=ModelIdentity(provider="fake", model_id="fake"), prompt_hash="p", config_hash="c")
    assert JournalEvent.model_validate_json(event.model_dump_json()) == event
    bad = event.model_dump() | {"schema_version": "2.0.0"}
    with pytest.raises(Exception):
        JournalEvent.model_validate(bad)
