# Audited BoolQ Smoke Replication — Skeleton Plan

## Summary

Build an isolated `self-correction-replication/` project and run an audited 10-item BoolQ smoke experiment entirely through WSL2:

- Model: `qwen2.5:3b-instruct-q8_0` via Ollama, pinned by model digest.
- Condition: initial answer plus one confirmatory revision for every item.
- Evaluation: `free_generation`; retain invalid outputs and failures.
- Classification: **adaptation**, because model family, GGUF Q8 quantization, runtime, and chat template differ from the ACL experiment.
- Deliverable: manifest, resolved configuration, append-only raw journal, metrics, diagnostics, and English Markdown smoke report.
- Defer the 300-item runs, plots, `legacy_forced_logits`, API backend, tuned lens, PACT, MLflow/W&B, and Hydra.

The scientific guardrails follow the ACL paper :codex-file-citation{path="D:\2026\Personal Research\papers\Understanding the Dark Side of LLMs Intrinsic Self-Correction.pdf" purpose="source"} and the oracle-filtering/compute-comparison cautions in the ICLR paper :codex-file-citation{path="D:\2026\Personal Research\papers\2310.01798v2.pdf" purpose="source"}.

## Architecture and Interfaces

Use a small `src/` package with five responsibility-based modules:

```text
self-correction-replication/
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── prompts.yaml
│   └── smoke.yaml
├── src/self_correction/
│   ├── cli.py
│   ├── models.py
│   ├── backend.py
│   ├── experiment.py
│   ├── artifacts.py
│   └── evaluation.py
├── tests/
├── data/manifests/
├── runs/
├── reports/
└── vendor/USC/
```

- `models.py`: immutable Pydantic contracts for messages, manifests, decoding config, generation results, journal events, usage, errors, and model identity.
- `backend.py`: a narrow `GenerationBackend` Protocol plus `OllamaBackend.generate()`. Do not add candidate scoring until legacy comparison begins.
- `experiment.py`: the only conversation state machine. It always appends the exact preceding raw assistant output and revises every item without consulting correctness.
- `artifacts.py`: YAML loading, stable IDs/hashes, JSONL append with flush/fsync, run metadata, and resume indexing.
- `evaluation.py`: strict Yes/No parsing, transition/statistical calculations, validation, aggregation, and Markdown report generation.
- `cli.py`: orchestration only; no scientific calculations.

Use `argparse`, `urllib`, `pathlib`, `logging`, and `json` from stdlib; Pydantic and PyYAML at I/O boundaries; NumPy/SciPy for bootstrap and exact McNemar; pytest for checks. Use `uv.lock` and locked execution for reproducibility, consistent with [uv’s lock/sync model](https://docs.astral.sh/uv/concepts/projects/sync/).

Public CLI:

```bash
uv run python -m self_correction.cli doctor --config configs/smoke.yaml
uv run python -m self_correction.cli prepare-data \
  --source vendor/USC/dataset/boolq_json/boolq_validation.json \
  --main-n 300 --smoke-n 10 --wavering-n 50 --seed 20260912
uv run python -m self_correction.cli run \
  --config configs/smoke.yaml \
  --manifest data/manifests/boolq_smoke_10.json \
  --condition confirmatory --rounds 1
uv run python -m self_correction.cli validate-run runs/<run_id>
uv run python -m self_correction.cli aggregate runs/<run_id>
uv run python -m self_correction.cli report runs/<run_id>
```

## Environment, Data, and Execution

- Initialize a separate Git repository under `/mnt/d/2026/Personal Research/self-correction-replication`.
- Add the [official USC repository](https://github.com/qingjiesjtu/USC) as a read-only submodule pinned to `69aa1012e2dfc4c57905bedf057c4e2fbc036101`.
- Keep source and auditable artifacts on `/mnt/d`; keep `.venv`, Ollama models, and caches under `/home/antd` on WSL ext4.
- Install Ollama inside Ubuntu 24.04 and pull the explicit Q8 tag. Before downloading, review the Qwen Research License: the Ollama library notes that the 3B model is not Apache-2.0 licensed. The Q8 artifact is approximately 3.3 GB and explicitly identifies its quantization and digest. [Ollama Qwen2.5 model entry](https://ollama.com/library/qwen2.5%3A3b-instruct-q8_0)
- `doctor` must verify Ollama reachability, version, model tag/digest, template/Modelfile hash, writable paths, WSL/CPU/RAM/swap, GPU visibility/VRAM, and configuration validity. Save the Modelfile because it defines the actual model-specific template. [Ollama Modelfile reference](https://docs.ollama.com/modelfile)
- Use `/api/chat` with `stream=false`, `think=false`, output logprobs enabled, and resolved options: `temperature=0`, `seed=20260912`, `num_predict=4`, `repeat_penalty=1.0`. Save exact request JSON and response metadata, including prompt/output token counts and durations exposed by the API. [Ollama chat API](https://docs.ollama.com/api/chat)
- Record adaptation reasons: `different_model_family`, `gguf_q8_quantization`, `ollama_runtime`, `ollama_chat_template`, and `subset_10`.
- Current WSL allocation—about 7.8 GB RAM, 2 GB swap, and visible 11 GB GTX 1080 Ti VRAM—is sufficient for this Q8 smoke plan; do not change `.wslconfig` unless `doctor` or the real smoke demonstrates memory pressure.

Data preparation:

- Produce deterministic 300-, 50-, and 10-item manifests before inference; the 50 and 10 are label-stratified subsets of the 300.
- Each item contains source index, stable `sample_id`, question, label, source commit, and hashes; exclude the passage.
- Define `sample_id` as source index plus the first eight hex characters of SHA-256 over normalized question text.
- Never replace failed or invalid items.

Raw storage:

- Use one immutable JSONL event per round, not one nested line per sample.
- Deterministic `request_id` covers run, sample, condition, replicate, and round.
- Each event stores exact messages, raw output, parse status, correctness, API metadata, retry attempts, model identity, prompt/config hashes, and future-compatible trajectory fields.
- Resume scans terminal request IDs and makes zero duplicate calls. Retry only transient transport/server failures within the same request; never retry an invalid model answer.

## Evaluation and Tests

Parser states: `valid_yes`, `valid_no`, `ambiguous`, `invalid`, and `error`. Preserve raw output unchanged; require Yes/No at the beginning, reject words such as `Yesterday`, and mark contradictory outputs such as “Yes, but actually no” ambiguous.

Primary metrics count invalid/error as not correct while reporting their statuses separately. Also report valid-only accuracy with its denominator. Produce:

- attempted/completed/error and parsing counts;
- \(A_0\), \(A_1\), and \(\Delta A=A_1-A_0\);
- CC, CW, WC, WW counts and the all-item 3×3 status transition table;
- conditional recovery \(u\), regression \(v\), and answer-change rate;
- exact two-sided McNemar test;
- paired 10,000-resample bootstrap CI with seed `20260912`;
- token counts and latency;
- an English `reports/smoke_summary.md` clearly labeled as an adaptation.

Required automated checks:

- Prompt snapshots and exact conversation-history construction.
- Label-isolation: opposite dummy labels produce byte-identical backend messages; no request contains `label` or `passage`.
- All specified valid, invalid, and ambiguous parser cases.
- Hand fixture: `CC=3, CW=2, WC=1, WW=4`, including the \(\Delta A\) identity and conditional denominators.
- Known McNemar cases and zero-discordant handling.
- Bootstrap reproducibility.
- Pydantic/JSON round-trip and schema-version rejection.
- JSONL aggregation invariant to event order.
- Interrupted resume produces no duplicate backend requests.
- Fake backend integration produces 20 terminal round events for 10 two-round conversations.

Smoke acceptance criteria:

- `doctor` passes and captures the Ollama model digest/template.
- Fake-backend tests pass before model inference.
- Exactly 10 unique samples and 20 terminal round events are present.
- Every sample receives revision regardless of initial correctness.
- Invalid/error items remain in the denominator.
- Re-running in resume mode makes zero model calls.
- Exact messages and five representative raw conversations are manually inspected.
- Metrics satisfy \(\Delta A=(WC-CW)/N\).
- The complete audit bundle can be regenerated from manifest, resolved config, and raw journal alone.

## Assumptions and Next Gate

- Research artifacts and code comments are English; collaboration and handoff remain Vietnamese.
- Internet access and permission to install Ollama, download the model, and initialize the USC submodule will be available during implementation.
- No numerical result from the 10-item smoke will be treated as evidence for or against the paper’s hypothesis; its purpose is to validate experimental integrity.
- The next gate is the 300-item confirmatory run. Only after that passes should the project add question repetition, 50×10 wavering, figures, or a Transformers backend for paper-closer candidate scoring.
