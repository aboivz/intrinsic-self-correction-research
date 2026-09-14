from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(FrozenModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DecodingConfig(FrozenModel):
    temperature: float = 0.0
    seed: int = 20260912
    num_predict: int = Field(default=4, ge=1)
    repeat_penalty: float = 1.0


class ModelIdentity(FrozenModel):
    provider: str
    model_id: str
    digest: str | None = None
    template_hash: str | None = None


class Usage(FrozenModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    eval_duration_ns: int | None = None


class GenerationResult(FrozenModel):
    raw_output: str | None = None
    usage: Usage = Field(default_factory=Usage)
    metadata: dict[str, object] = Field(default_factory=dict)
    error: str | None = None
    transient: bool = False


class ManifestItem(FrozenModel):
    source_index: int = Field(ge=0)
    sample_id: str
    question: str
    label: bool
    source_commit: str
    question_sha256: str


class Manifest(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    dataset: str
    source_count: int
    selection: str
    seed: int
    items: list[ManifestItem]
    sha256: str


class JournalEvent(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str
    request_id: str
    sample_id: str
    condition: str
    replicate: int = 0
    round: int = Field(ge=0)
    messages: list[Message]
    raw_output: str | None = None
    parsed_answer: Literal["yes", "no"] | None = None
    parse_status: Literal["valid_yes", "valid_no", "ambiguous", "invalid", "error"]
    is_correct: bool = False
    label: bool
    usage: Usage = Field(default_factory=Usage)
    api_metadata: dict[str, object] = Field(default_factory=dict)
    retry_attempts: int = 0
    model: ModelIdentity
    prompt_hash: str
    config_hash: str
    error: str | None = None
