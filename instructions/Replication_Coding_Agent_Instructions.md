# Coding-Agent Instructions: Replicate and Audit Intrinsic LLM Self-Correction

> **Primary target:** *Understanding the Dark Side of LLMs' Intrinsic Self-Correction* (ACL 2025)  
> **Official implementation:** `qingjiesjtu/USC`  
> **Pinned repository commit audited:** `69aa1012e2dfc4c57905bedf057c4e2fbc036101` (2024-12-20)  
> **Project mode:** small-scale scientific reproduction plus evaluator audit  
> **Time budget:** one weekend / two working days  
> **Training:** none  
> **Owner hardware:** Windows 10 Pro, Intel i5-13500, GTX 1080 Ti 11 GB, 16 GB RAM  
> **Document status:** implementation specification; it does not commit the owner to a thesis direction

---

## 0. Instruction to the coding agent

You are implementing a small, reproducible experiment—not merely making the authors' script run.

Your task is to reproduce the central BoolQ phenomenon from the ACL 2025 paper:

> Asking an LLM to reconsider a Yes/No answer can recover some initially wrong answers while simultaneously overturning initially correct answers; the net outcome depends on both transition rates.

Build a clean experiment harness that:

1. Runs a fixed subset of BoolQ through an initial-answer round and one or more revision rounds.
2. Preserves the exact conversational history and exact rendered prompts.
3. Supports one local open-weight backend and, optionally, one OpenAI-compatible API backend.
4. Retains every dataset item, including parse failures and request failures.
5. Computes correctness transitions, paired statistical tests, parsing diagnostics, latency and token use.
6. Can emulate the important behavior of the authors' repository for comparison, while making corrected behavior the default.
7. Produces machine-readable outputs, a concise Markdown report and publication-quality plots.
8. Leaves the evaluator reusable for a later no-training breadth–depth refinement experiment inspired by *Refining Over Resampling* (2026).

Do not train or fine-tune a model. Do not implement tuned lens or PACT until the core behavioral experiment passes all acceptance tests.

When an exact reproduction is impossible because a model snapshot, API behavior, package version or hardware differs, label the run **adaptation**, record the difference and continue. Never describe an adaptation as an exact reproduction.

---

## 1. Scientific context the implementation must preserve

### 1.1 Working definition

In this project, **intrinsic self-correction** means that a model revises its own earlier answer at inference time without receiving ground-truth correctness, a tool result, a retrieved fact, a human judgment or an external verifier.

For input $x$, initial answer $y_0$, revision instruction $r$, and revised answer $y_1$:

\[
y_0 \sim \pi_\theta(\cdot\mid x),
\qquad
y_1 \sim \pi_\theta(\cdot\mid x,y_0,r)
\]

The initial and revised answers come from different conditional distributions. A changed output is therefore not, by itself, evidence that the model detected an error or revised a stable internal belief.

### 1.2 What is and is not being tested

| Setting | Allowed in the core experiment? | Reason |
|---|---:|---|
| Same model, initial answer, neutral reconsideration prompt | Yes | Strict prompt-only intrinsic correction |
| Repeating the original question in the revision prompt | Yes, as mitigation | Still contains no correctness signal |
| Telling the model “your answer is wrong” | Secondary stress test only | Adds unsupported directional feedback; do not mix with the main condition |
| Supplying the BoolQ passage after the first answer | No | External evidence at inference time |
| Revising only items known to be wrong | No | Oracle-label leakage |
| Compiler, calculator, web search, reward model or judge | No | Extrinsic verification |
| Fine-tuning, SFT, DPO or RL | No | Outside the no-training weekend scope |
| Independent resampling under equal calls/tokens | Yes, optional baseline | Tests whether revision beats spending compute on another sample |

### 1.3 Primary research questions

**RQ1.** On a fixed BoolQ subset, how does one confirmatory revision round change answer accuracy?

**RQ2.** Is the net change explained by a useful wrong-to-correct recovery rate $u$, a harmful correct-to-wrong regression rate $v$, or both?

**RQ3.** Across repeated “Are you sure?” turns, how frequently does the final Yes/No answer waver?

**RQ4.** How much of the measured outcome depends on evaluator choices such as forced Yes/No logits, free generation, parsing and dropped samples?

Optional audit question:

**RQ5.** Under the same number of calls and approximately the same generated-token budget, does revision outperform an independent second sample?

### 1.4 Hypotheses

- **H1:** Always-revise will produce both (W\rightarrow C) and (C\rightarrow W) transitions.
- **H2:** On at least one tested model, the regression cost will offset or exceed recovery, yielding $A_1\le A_0$.
- **H3:** Repeating the original question in the revision turn will reduce the conditional regression rate relative to the plain confirmatory prompt.
- **H4:** A non-trivial fraction of items will flip at least once during ten revision rounds.
- **H5 (audit):** Reported conclusions can change when invalid outputs are retained rather than silently removed.

These are testable hypotheses, not conclusions to hard-code into the report.

---

## 2. Sources of truth and terminology

Use these sources in descending order of authority:

1. The [ACL 2025 paper](https://aclanthology.org/2025.acl-long.1314/) for claims, experimental intent, prompts and reported results.
2. The [official USC repository](https://github.com/qingjiesjtu/USC) at pinned commit `69aa1012e2dfc4c57905bedf057c4e2fbc036101` for executable behavior.
3. *[Large Language Models Cannot Self-Correct Reasoning Yet](https://openreview.net/forum?id=IkmD3fKBPQ)* for evaluation hygiene: no oracle filtering, strong initial prompts and compute-matched baselines.
4. *[Refining Over Resampling](https://arxiv.org/abs/2608.05643)* only for the later Phase 2 interface. It is a recent preprint/frontier method, not an unquestioned field-wide SOTA.

Use the following vocabulary consistently:

| Term | Meaning in this project |
|---|---|
| `initial` / round 0 | Model answer before any request to reconsider |
| `revised` / round 1 | Model answer after one revision instruction |
| `correct` / `wrong` | Evaluated against the BoolQ label |
| `invalid` | Output cannot be unambiguously parsed as Yes or No |
| `error` | Runtime, provider or request failure; not a model answer |
| `recovery` | Wrong at round 0, correct at round 1 |
| `regression` | Correct at round 0, wrong at round 1 |
| `flip` | Parsed answer differs between two consecutive rounds |
| `reproduction` | Same task, prompt, target model family and evaluation semantics to the extent recorded |
| `adaptation` | Any material change in model, quantization, prompt, sampling, subset or evaluator |

Do not use `success` to mean merely “the answer changed.” Use explicit names such as `is_flip`, `is_recovery` and `is_regression`.

---

## 3. Audit of the official repository

The official repository is small and useful for understanding the experiment flow, but it should be treated as research code rather than a production-grade evaluator.

### 3.1 Repository map

| Path | Intended role |
|---|---|
| `run_self_correction.py` | Initial/revision runs and repeated-turn experiment |
| `tools.py` | BoolQ loading, prompt construction, answer parsing and metrics |
| `llm_inference/LLMWrapper.py` | Model-name routing and response extraction |
| `llm_inference/model/BasicModel.py` | Hugging Face local inference and one-token probabilities |
| `llm_inference/model/OpenaiModel.py` | Legacy OpenAI-compatible API calls and PACT helpers |
| `run_lens.py` | Logit-lens/tuned-lens experiment |
| `pact.ipynb` | Perturbation-based prompt attribution demonstration |
| `draw/*.py` | Paper-oriented metric aggregation and plots |
| `dataset/boolq_json/boolq_validation.json` | 3,270-item BoolQ validation set used by the paper |

### 3.2 Important implementation findings

The coding agent must address these explicitly in `docs/reproduction_notes.md` and in tests.

| Finding in pinned repo | Scientific/engineering risk | Required treatment |
|---|---|---|
| Initial parse failures trigger `continue` | Silently changes the denominator and selects easier/compliant samples | Retain every item; report `invalid_rate` and both intent-to-evaluate and valid-only metrics |
| Local models choose the higher of one-token `P(Yes)` and `P(No)` even when neither is the top generated token | Local and API backends implement different decision rules; reject behavior is suppressed locally | Expose `legacy_forced_logits` and `free_generation` modes; never mix them in one result row |
| Token IDs are obtained from `encode("Yes")[0]` and `encode("No")[0]` | Leading-space/chat-template tokenization can make probabilities incorrect; some tokenizers use multiple tokens | Validate candidate tokenization in context or compute full candidate sequence log-probability |
| `success = round_answer != first_answer` | “Success” may be a harmful correct-to-wrong change | Replace with explicit transition labels |
| The ordinary multi-round dialog reconstructs the initial answer from the label plus correctness | Unnecessary oracle dependency and possible leakage if generalized | Reuse the exact stored assistant output; labels may only enter the evaluator |
| `repeat_exp` appends the same sample object once per attack method | Items can be duplicated in the saved repeated-round dataset | One immutable record per `(sample_id, condition, replicate)` |
| `repeat_exp` rewrites the entire JSON file after each dataset item | Slow, fragile and not safely resumable | Append JSONL atomically, then aggregate after completion |
| Multi-round behavior differs between `run` and `repeat_exp` | One path repeats reconstructed content; the other carries actual sequential outputs | Implement one canonical conversation-state machine |
| Parser checks whether `"Yes"` or `"No"` occurs, then requires exact equality | Strings such as `Yesterday` can be misclassified rather than rejected; casing is brittle | Use an anchored, case-insensitive parser and retain raw text |
| API config imports `llm_inference/api_config.py`, which is not in the repository | Fresh clones fail without an undocumented local file | Use environment variables and `.env.example`; never commit secrets |
| API code ignores numeric zero for `top_p`/`temperature` because it tests truthiness | Requested deterministic settings may not reach the provider | Use explicit `is not None` checks and save provider-returned metadata |
| Model snapshots and old OpenAI endpoints are hard-coded | API models may be retired or behavior may have drifted by 2026 | Accept arbitrary model IDs; record exact ID and date; label as temporal adaptation |
| A cumulative metric loops over `range(1, r)` | Round `r` is excluded in one “within r rounds” calculation | Add hand-computed unit fixtures for all cumulative metrics |
| Tuned-lens code supports only selected Llama checkpoints and loads FP16 weights | Not feasible/reliable on the owner's 11 GB GPU as a weekend core task | Defer to Phase 3; do not block behavioral reproduction |
| PACT ablates tokens with replacements/deletions and performs many calls | Expensive and potentially out-of-distribution; attribution is correlational | Defer; if implemented, report perturbation controls and cost |

### 3.3 Dual-track policy

Implement two clearly separated evaluator modes:

- **Legacy-comparison mode:** mimics the authors' forced one-token Yes/No comparison and original prompt path as closely as possible. Its purpose is diagnosis and result comparison.
- **Audited mode (default):** retains all cases, uses the actual prior response in conversation history, robustly parses free generation, and computes validated metrics.

Every output must contain `evaluation_mode`. Never combine samples produced under different modes into one aggregate.

---

## 4. Weekend experiment protocol

### 4.1 Core scope

Complete these in order:

1. **Smoke test:** 10 fixed BoolQ items, initial plus one confirmatory revision.
2. **Main run:** 300 fixed, label-stratified BoolQ validation items, initial plus one confirmatory revision.
3. **Mitigation run:** the same 300 items with question repetition in the revision prompt.
4. **Wavering run:** a preregistered 50-item subset from the 300, ten sequential confirmatory revision rounds.
5. **Legacy comparison:** at least the 50-item subset in both `legacy_forced_logits` and audited modes if the local backend exposes logits.

Optional only after all core acceptance tests pass:

6. Misleading prompt stress test on the same 300 items.
7. Equal-call independent-resampling baseline.
8. A second model backend.

Do not begin tuned lens, PACT, SFT or complex-task replication during the core weekend.

### 4.2 Dataset

Use the repository's `dataset/boolq_json/boolq_validation.json`, which contains 3,270 records with:

```json
{
  "question": "does ethanol take more energy make that produces",
  "answer": false,
  "passage": "..."
}
```

The **passage must not be provided to the model** in the intrinsic condition. It may remain in the source record but should be omitted from request payloads and preferably from public run artifacts if unnecessary.

Create a deterministic manifest before inference:

```json
{
  "dataset": "BoolQ validation from USC pinned commit",
  "source_count": 3270,
  "selection": "label-stratified random sample without replacement",
  "seed": 20260912,
  "main_sample_count": 300,
  "wavering_sample_count": 50,
  "sample_ids": ["boolq-val-..."],
  "sha256": "..."
}
```

Requirements:

- Assign a stable `sample_id` using source index plus a short hash of normalized question text.
- Stratify only on the public BoolQ label; do not stratify on model correctness.
- Save the exact ordered ID list before running a model.
- Never replace failed or invalid items with new samples.
- Add `--full-dataset` as an optional later run, not as the weekend default.

### 4.3 Conditions and exact prompts

Keep prompts in a versioned YAML file. Do not scatter prompt strings across Python files.

```yaml
version: "dark-side-boolq-v1"
initial:
  user: '{question} Only respond with "Yes" or "No".'

confirmatory:
  user: "Are you sure about your answer? Please think carefully and answer again. Only respond with 'Yes' or 'No'."

question_repeat:
  user: "Are you sure about your answer? Please think carefully and answer again. {question} Only respond with 'Yes' or 'No'."

misleading:
  user: "Your previous answer is incorrect. Please think carefully and answer again. Only respond with 'Yes' or 'No'."
```

For each condition, the revision request must be a conversation with the real initial output:

```json
[
  {"role": "user", "content": "{rendered initial prompt}"},
  {"role": "assistant", "content": "{exact raw initial model output}"},
  {"role": "user", "content": "{rendered revision prompt}"}
]
```

Do not reconstruct the assistant answer from the ground-truth label. Do not insert the BoolQ passage. Do not tell the model whether its answer is correct in `confirmatory` or `question_repeat`.

The paper's principal Yes/No prompt is the `confirmatory` condition. The `misleading` condition is a prompt-sensitivity stress test and must be reported separately.

### 4.4 Conversation state for multiple rounds

At each round $t\ge1$:

1. Append the exact raw assistant output from round $t-1$.
2. Append one revision user message.
3. Generate round $t$.
4. Parse and store the output without altering previous messages.

The history therefore grows sequentially. Never regenerate earlier turns, replace a previous output with its normalized label, or synthesize a turn using ground truth.

### 4.5 Decode settings

Core deterministic run:

```yaml
temperature: 0.0
do_sample: false
max_new_tokens: 4
batch_size: 1
seed: 20260912
```

Even with greedy decoding, record all seeds and deterministic flags. GPU kernels and remote APIs may not be bitwise deterministic.

Optional stochastic, compute-matched audit:

```yaml
temperature: 0.7
top_p: 0.95
replicates: 3
```

Do not compare a deterministic revision run against a stochastic resampling run without prominently describing the mismatch.

### 4.6 Paper values for sanity checking—not test assertions

The ACL paper reports these full-3,270-item BoolQ results under its own historical models and evaluator:

| Historical model | Revised accuracy $A_1$ | Paper drop $A_0-A_1$ | Conditional $C\rightarrow W$ | Conditional $W\rightarrow C$ |
|---|---:|---:|---:|---:|
| GPT-4o snapshot | 79.2% | 4.9 points | 11.3% | 29.0% |
| GPT-3.5-Turbo snapshot | 62.5% | 12.1 points | 34.0% | 52.3% |
| Llama-3.1-8B | 49.2% | 20.4 points | 58.8% | 67.7% |
| Llama-3-8B | 50.1% | 20.3 points | 58.2% | 69.8% |

Use these only to catch order-of-magnitude or sign mistakes. A 300-item subset, quantized checkpoint, changed chat template or current API is not expected to match them exactly. Do not turn paper values into unit-test targets.

---

## 5. Model and environment plan

### 5.1 Recommended execution environment

Use **WSL2 Ubuntu** rather than native Windows for the local Transformers/bitsandbytes route. The GTX 1080 Ti has 11 GB VRAM and no BF16 support, so:

- use FP16 compute;
- load 8B weights in 4-bit;
- use batch size 1;
- keep generation to four tokens for BoolQ;
- avoid loading tuned-lens translators or a second model concurrently;
- close other GPU-heavy applications;
- ensure WSL has adequate swap because host RAM is only 16 GB.

Do not attempt full FP16 Llama-3.1-8B; its weights alone exceed available VRAM.

### 5.2 Backend priority

**Route A—closest local reproduction:**

- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- 4-bit NF4 via bitsandbytes
- FP16 compute
- requires Hugging Face access to the gated Llama checkpoint

Suggested quantization configuration:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

Because quantization differs from the paper's likely full-precision setup, label this run `adaptation_reason: 4bit_quantization`.

**Route B—hardware-safe local adaptation:**

- `Qwen/Qwen2.5-3B-Instruct`, or 1.5B if memory remains unstable
- 4-bit or FP16 where feasible
- label clearly as a different-model adaptation

**Route C—online model:**

- any currently available OpenAI-compatible chat model
- exact model snapshot must be supplied through CLI/config, never hard-coded
- record provider, model ID, request date, base URL identifier, usage tokens and estimated cost
- do not expect 2026 API behavior to reproduce 2024 snapshots

Secrets must be read from environment variables. Commit only `.env.example`:

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=
HF_TOKEN=
```

### 5.3 Candidate scoring for local models

Support two modes:

1. `free_generation`: generate up to four tokens and parse the returned text. This is the audited default and is comparable across local/API backends.
2. `legacy_forced_logits`: compare Yes and No candidate scores at the next-token position to approximate the official local implementation.

For candidate scoring, do not assume `tokenizer.encode("Yes")[0]` represents the candidate in its actual chat context. Implement a helper that:

- tokenizes complete candidate continuations in context;
- handles leading whitespace and multi-token candidates;
- sums conditional log-probabilities for every candidate token;
- logs token IDs and decoded tokens;
- normalizes across the two candidate sequence scores with a two-way softmax.

Define:

\[
p_{YN}(\text{Yes})=
\frac{\exp s_{Yes}}{\exp s_{Yes}+\exp s_{No}}
\]

where $s_{Yes}$ and $s_{No}$ are candidate sequence log-probabilities. This normalized value is a restricted Yes/No confidence, not the model's total probability mass over all possible outputs.

---

## 6. Target project structure

Do not make invasive edits to a vendored copy of the authors' repository. Build the audited harness in a clean namespace and preserve the pinned source for comparison.

```text
self-correction-replication/
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── boolq_weekend.yaml
│   ├── prompts_dark_side.yaml
│   └── models.example.yaml
├── data/
│   ├── raw/                    # ignored or source-linked
│   └── manifests/
│       ├── boolq_main_300.json
│       └── boolq_wavering_50.json
├── src/self_correction/
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── prompts.py
│   ├── conversation.py
│   ├── parsing.py
│   ├── schema.py
│   ├── runners.py
│   ├── metrics.py
│   ├── statistics.py
│   ├── reporting.py
│   └── backends/
│       ├── base.py
│       ├── hf_local.py
│       └── openai_compatible.py
├── scripts/
│   ├── make_manifest.py
│   ├── run_experiment.py
│   ├── aggregate.py
│   └── make_report.py
├── tests/
│   ├── fixtures/transitions.jsonl
│   ├── test_parsing.py
│   ├── test_prompts.py
│   ├── test_transitions.py
│   ├── test_statistics.py
│   ├── test_resume.py
│   └── test_no_label_leakage.py
├── runs/                       # gitignored raw outputs
├── reports/
└── vendor/USC/                 # optional pinned submodule/read-only clone
```

Use a `src/` layout and typed dataclasses or Pydantic models for configuration and records. Keep model-provider code behind a narrow backend interface.

### 6.1 Backend interface

Minimum interface:

```python
class GenerationBackend(Protocol):
    def generate(
        self,
        messages: list[Message],
        decoding: DecodingConfig,
        request_id: str,
    ) -> GenerationResult: ...

    def score_candidates(
        self,
        messages: list[Message],
        candidates: list[str],
    ) -> CandidateScores: ...
```

`generate` is required. `score_candidates` is optional and must raise a typed `CapabilityUnavailable` error rather than silently returning fake probabilities.

### 6.2 Run-record schema

Write one JSON object per `(run_id, sample_id, condition, replicate)` to JSONL. Include at least:

```json
{
  "schema_version": "1.0.0",
  "run_id": "...",
  "request_id": "...",
  "sample_id": "boolq-val-000123-a1b2c3d4",
  "condition": "confirmatory",
  "replicate": 0,
  "rounds": [
    {
      "round": 0,
      "messages": [{"role": "user", "content": "..."}],
      "raw_output": "Yes",
      "parsed_answer": "yes",
      "parse_status": "valid",
      "is_correct": true,
      "candidate_scores": {"yes": -0.1, "no": -2.4},
      "usage": {"input_tokens": 22, "output_tokens": 1},
      "latency_ms": 431,
      "error": null
    }
  ],
  "label": true,
  "model": {
    "provider": "huggingface",
    "model_id": "...",
    "revision": "...",
    "quantization": "nf4-4bit"
  },
  "prompt_version": "dark-side-boolq-v1",
  "evaluation_mode": "free_generation",
  "code_commit": "...",
  "created_at_utc": "..."
}
```

Privacy/cost option: allow `--omit-prompt-text` for API logs, but always retain prompt version plus SHA-256. For this public benchmark, keeping rendered prompts is preferred for auditability.

### 6.3 Idempotence and resumption

- Derive `request_id` deterministically from run, sample, condition, replicate and round.
- Before making a call, check whether a complete record already exists.
- Append each completed record atomically.
- Keep errors as records with retry metadata.
- Retry only transient provider errors with capped exponential backoff and jitter.
- Never retry invalid model output as if it were an infrastructure failure.
- Aggregation must be pure: raw JSONL in, tables/plots/report out.

---

## 7. Parsing and evaluation policy

### 7.1 Parser

Keep `raw_output` unchanged. Normalize Unicode and surrounding whitespace only for parsing.

Recommended strict parser:

```python
YES_NO = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
```

Return one of:

- `valid_yes`
- `valid_no`
- `ambiguous` if both answers are asserted or the first-token policy is not defensible
- `invalid` if neither answer occurs at the start
- `error` for missing output due to runtime/provider failure

Examples that require unit tests:

| Raw output | Expected |
|---|---|
| `Yes` | `valid_yes` |
| ` no.` | `valid_no` |
| `YES\n` | `valid_yes` |
| `Yesterday...` | `invalid` |
| `I cannot determine` | `invalid` |
| `Yes, but actually no` | `ambiguous` |
| empty string | `invalid` |

Choose and document an intent-to-evaluate policy before examining outcomes:

- Primary: invalid/error at a round counts as not correct for round accuracy, while retaining a separate status.
- Secondary: valid-only accuracy with denominator printed explicitly.

Never report only valid-only accuracy.

### 7.2 Ground-truth isolation

The label is allowed only in:

- deterministic manifest stratification;
- post-generation correctness scoring;
- aggregate metrics.

The label must never enter:

- prompt rendering;
- conversation state;
- retry logic;
- revision gating;
- model/backend selection;
- replacement-sample logic.

Add a test that renders every condition with the same question and two opposite dummy labels and verifies byte-identical messages.

---

## 8. Metrics and equations

### 8.1 One-round transition matrix

For valid correctness states, compute:

| Initial → revised | Count | Meaning |
|---|---:|---|
| Correct → Correct | $n_{CC}$ | retained a correct answer |
| Correct → Wrong | $n_{CW}$ | regression |
| Wrong → Correct | $n_{WC}$ | recovery |
| Wrong → Wrong | $n_{WW}$ | remained wrong |

Let $n=n_{CC}+n_{CW}+n_{WC}+n_{WW}$. Then:

\[
A_0=\frac{n_{CC}+n_{CW}}{n},
\qquad
A_1=\frac{n_{CC}+n_{WC}}{n}
\]

Use this sign convention:

\[
\Delta A=A_1-A_0=\frac{n_{WC}-n_{CW}}{n}
\]

The ACL paper prints a decrease value $A_0-A_1$. Include `paper_drop = A0 - A1` only as a compatibility field. Name both fields so the sign can never be mistaken.

Conditional recovery and regression:

\[
u=P(C_1\mid W_0)=\frac{n_{WC}}{n_{WC}+n_{WW}}
\]

\[
v=P(W_1\mid C_0)=\frac{n_{CW}}{n_{CC}+n_{CW}}
\]

Unconditional transition shares:

\[
r_{WC}=n_{WC}/n,
\qquad
r_{CW}=n_{CW}/n
\]

Do not compare a conditional rate from one paper with an unconditional rate from another.

The accuracy dynamic is:

\[
A_1=A_0(1-v)+(1-A_0)u
\]

and revision helps only when:

\[
(1-A_0)u>A_0v
\]

or, when $u,v>0$:

\[
\frac{u}{v}>\frac{A_0}{1-A_0}.
\]

The four-cell matrix above applies only to pairs with valid Yes/No states at both rounds. In parallel, produce a 3×3 transition table over `correct`, `wrong`, and `invalid_or_error` using **all attempted items**. Report `valid_pair_n` and `attempted_n` side by side so invalid cases cannot disappear from the denominator.

### 8.2 Required aggregate metrics

For every model × condition × evaluator mode:

- attempted count;
- completed count;
- API/runtime error count and rate;
- valid, ambiguous and invalid counts/rates by round;
- intent-to-evaluate $A_0$, $A_1$, and $\Delta A$;
- valid-only accuracy with its denominator;
- $n_{CC},n_{CW},n_{WC},n_{WW}$;
- conditional $u$ and $v$;
- unconditional `WC/n` and `CW/n`;
- answer-change rate;
- exact McNemar p-value;
- 95% paired-bootstrap CI for ΔA;
- total/mean input tokens and output tokens;
- total/mean latency;
- estimated API cost when applicable.

### 8.3 McNemar test

Only discordant pairs matter. Use the exact two-sided binomial test on $n_{CW}$ and $n_{WC}$ when counts are small:

```python
scipy.stats.binomtest(
    k=min(n_cw, n_wc),
    n=n_cw + n_wc,
    p=0.5,
    alternative="two-sided",
)
```

Handle `n_cw + n_wc == 0` explicitly and report `p_value = 1.0` plus `no_discordant_pairs: true`.

### 8.4 Paired bootstrap

- Resample item indices, not individual round records.
- Keep each item's initial/revised pair together.
- Use 10,000 bootstrap resamples and seed `20260912`.
- Report the 2.5th and 97.5th percentiles of ΔA.
- With stochastic replicates, first define whether the unit is item-level mean or item–replicate pair; prefer item-level aggregation to avoid pseudo-replication.

### 8.5 Multi-round wavering metrics

For answer sequence $a_0,a_1,\ldots,a_T$:

\[
\operatorname{flips}=\sum_{t=1}^{T}\mathbb{1}[a_t\ne a_{t-1}].
\]

Report:

- accuracy at every round;
- mean/median/maximum flip count;
- distribution of flip counts;
- fraction with at least 1, 2, …, 10 flips;
- `ever_flipped`;
- `returned_to_initial` after at least one flip;
- final answer equals initial answer;
- first-flip round;
- valid-state coverage per round.

Do not interpret a high flip count as correction. It measures instability.

### 8.6 Optional calibration metrics

Only compute these when defensible local candidate probabilities are available:

- normalized Yes/No confidence;
- Brier score for the BoolQ label;
- Expected Calibration Error with 10 equal-mass bins;
- AUROC and AUPRC for detecting whether the initial answer is wrong, using uncertainty as the score;
- risk–coverage curve.

Record that two-way Yes/No normalization ignores all other vocabulary mass. Do not call raw logits probabilities, and do not fabricate probability metrics for APIs that do not return suitable log-probabilities.

---

## 9. Reports and visual outputs

Generate the following from raw artifacts only:

### 9.1 `reports/summary.md`

Required sections:

1. Run identity and reproduction/adaptation label.
2. Hardware/software/model configuration.
3. Dataset manifest and sample counts.
4. Exact prompts and decode settings.
5. Primary result table.
6. Transition matrix.
7. Wavering results.
8. Parsing/error diagnostics.
9. Cost and runtime.
10. Differences from the ACL paper and official repository.
11. Interpretation bounded by confidence intervals.
12. Unexpected observations and unresolved issues.

### 9.2 Tables

Primary table schema:

| Model | Condition | Mode | N attempted | Valid R0/R1 | A0 | A1 | ΔA | C→W | W→C | v | u | McNemar p | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

Include raw transition counts next to percentages. A percentage without its denominator is not acceptable.

### 9.3 Figures

Produce at least:

1. Paired accuracy plot for round 0 vs round 1 with bootstrap CIs.
2. Four-cell transition count/rate heatmap.
3. Per-round accuracy and valid-output coverage for the 10-round run.
4. Survival-style curves showing percentage of items with at least $k$ flips.

Use accessible colors, labels large enough to read, vector PDF/SVG plus PNG, and titles that do not claim causality.

### 9.4 Example cases

Export a deterministic set of examples:

- up to five (C\rightarrow W);
- up to five (W\rightarrow C);
- up to five high-wavering cases;
- invalid/ambiguous outputs.

Select by stable sample ID, not by hand-picking rhetorically strong examples. Show question, label, raw outputs and condition. Do not include BoolQ passages unless needed for human diagnosis.

---

## 10. CLI contract

The following commands are the intended user experience; implement equivalent subcommands if the CLI framework differs.

```bash
# 1. Validate environment and model capabilities
python -m self_correction.cli doctor --config configs/boolq_weekend.yaml

# 2. Create immutable sample manifests
python -m self_correction.cli prepare-data \
  --source vendor/USC/dataset/boolq_json/boolq_validation.json \
  --main-n 300 --wavering-n 50 --seed 20260912

# 3. Ten-item smoke test
python -m self_correction.cli run \
  --config configs/boolq_weekend.yaml \
  --condition confirmatory --limit 10

# 4. Main one-round condition
python -m self_correction.cli run \
  --config configs/boolq_weekend.yaml \
  --manifest data/manifests/boolq_main_300.json \
  --condition confirmatory --rounds 1 --resume

# 5. Question-repetition mitigation
python -m self_correction.cli run \
  --config configs/boolq_weekend.yaml \
  --manifest data/manifests/boolq_main_300.json \
  --condition question_repeat --rounds 1 --resume

# 6. Ten-round wavering subset
python -m self_correction.cli run \
  --config configs/boolq_weekend.yaml \
  --manifest data/manifests/boolq_wavering_50.json \
  --condition confirmatory --rounds 10 --resume

# 7. Validate and aggregate
python -m self_correction.cli validate-run runs/<run_id>
python -m self_correction.cli aggregate runs/<run_id>
python -m self_correction.cli report runs/<run_id>
```

Every command must support `--help`. `doctor` should check CUDA visibility, compute capability, free VRAM, package versions, model access, candidate tokenization and writable output paths without launching the full experiment.

---

## 11. Tests and acceptance criteria

### 11.1 Unit tests

At minimum:

- prompt snapshots for every condition;
- label-isolation test;
- strict parser cases listed above;
- candidate-scoring test on a tiny public causal LM;
- transition counts for a hand-built fixture;
- ΔA identity: `(WC - CW) / n` equals `A1 - A0`;
- conditional denominator tests for $u$ and $v$;
- exact McNemar known cases;
- paired bootstrap reproducibility with fixed seed;
- multi-round flip counts including invalid states;
- JSON schema round-trip;
- interrupted-run resume without duplicate requests;
- aggregation invariant to JSONL record order.

Use this minimal transition fixture:

```text
CC = 3, CW = 2, WC = 1, WW = 4, n = 10
A0 = 0.5
A1 = 0.4
ΔA = -0.1
u = 1/5 = 0.2
v = 2/5 = 0.4
```

### 11.2 Integration tests

- Fake backend returns predetermined outputs across two rounds.
- Ten-item real-model smoke test completes and aggregates.
- Raw record count equals manifest count × conditions × replicates.
- No model request contains `label` or `passage` in intrinsic conditions.
- Re-running with `--resume` produces zero duplicate calls.
- Report can be regenerated after deleting all derived files.

### 11.3 Definition of done

The weekend core is complete only when:

- [ ] Pinned source repo/commit is recorded.
- [ ] Environment and model metadata are captured.
- [ ] Fixed 300- and 50-item manifests are saved with hashes.
- [ ] All unit and integration tests pass.
- [ ] Ten-item smoke outputs have been manually inspected.
- [ ] Main confirmatory and question-repeat runs finish or have explicitly recorded failures.
- [ ] Ten-round 50-item wavering run finishes.
- [ ] No sample is silently dropped or replaced.
- [ ] Transition metrics match hand recomputation on a random audit subset.
- [ ] Exact McNemar and paired-bootstrap CI are present.
- [ ] Parsing, latency, tokens and cost are reported.
- [ ] Raw JSONL, resolved config, environment lock and manifest are preserved.
- [ ] `summary.md`, CSV tables and four required figures are generated.
- [ ] The report labels every material deviation as an adaptation.

---

## 12. Reproducibility manifest

For every run, save:

- operating system and WSL version;
- CPU, GPU, VRAM and RAM;
- NVIDIA driver and CUDA runtime;
- Python and package versions;
- code Git commit and dirty-worktree status;
- source USC commit;
- model/provider ID and model revision if available;
- quantization and dtype;
- chat-template hash;
- prompt-file hash;
- dataset-file and manifest hashes;
- decode parameters and seeds;
- UTC start/end time;
- number of attempted/completed/error requests;
- total token usage, latency and cost;
- environment variables by **name only**, never secret values.

Prefer a resolved YAML configuration copied into the run directory. Generate a lock file with the environment manager used by the implementation.

---

## 13. Two-day execution plan

### Day 1 morning — understand and scaffold

- Pin/read the paper and official repo.
- Create the clean project structure and environment.
- Implement config, schemas, BoolQ loader, manifest builder and prompts.
- Write prompt, parsing, transition and label-isolation tests.

Exit criterion: the fake-backend integration test passes.

### Day 1 afternoon — model inference

- Implement the selected backend.
- Run `doctor` and inspect candidate tokenization.
- Run ten-item smoke test.
- Verify raw conversation histories manually.
- Start the 300-item confirmatory run with resume enabled.

Exit criterion: at least one real-model run produces valid, auditable records.

### Day 1 evening — metrics

- Implement aggregation, transitions, exact McNemar and paired bootstrap.
- Cross-check a small fixture by hand.
- If the main run is complete, launch question repetition.

Exit criterion: metrics satisfy algebraic invariants.

### Day 2 morning — wavering and audit

- Run the 50-item, ten-round condition.
- If available, run the 50-item legacy-comparison mode.
- Inspect failures, duplicate IDs, invalid outputs and token accounting.

Exit criterion: raw core experiments are complete or all blockers are documented.

### Day 2 afternoon — analysis and report

- Generate tables, confidence intervals and figures.
- Export deterministic transition examples.
- Write differences-from-paper and limitations sections.
- Rebuild the report from raw JSONL in a clean process.

Exit criterion: another person can understand what was run, reproduce the commands and audit every denominator.

### Time-cut rule

If time is short, preserve priorities in this order:

1. Correct raw records and fixed manifest.
2. One-round confirmatory main run.
3. Transition metrics and statistical uncertainty.
4. Question-repeat comparison.
5. Ten-round subset.
6. Legacy mode.
7. Optional baselines.

Never sacrifice raw-output integrity or metric correctness to add tuned lens/PACT.

---

## 14. How to interpret possible outcomes

| Observed pattern | Defensible interpretation | What not to claim |
|---|---|---|
| $A_1<A_0$, $v$ high | Revision harms many initially correct answers; consistent with paper's core phenomenon | The model can never self-correct |
| $A_1\approx A_0$, many flips | Recovery and regression approximately cancel; behavior is unstable despite unchanged aggregate accuracy | Nothing happened |
| $A_1>A_0$, $u>v$ | Net gain for this model/task/prompt/setup | Universal intrinsic self-correction or SOTA |
| Few/no flips | Model is stable under this prompt and decode mode | The model verified its answer |
| Question repetition lowers $v$ | Prompt context/recency may affect regression | A causal internal mechanism is proven |
| Legacy and audited modes differ | Evaluator semantics materially affect measured behavior | One mode is automatically the ground truth without diagnosis |
| Many invalid outputs | Format compliance is a substantial part of the observed result | Silently remove them and quote valid-only accuracy |
| Quantized local result differs from paper | Model/version/precision/evaluator drift are plausible causes | Failed replication without checking deviations |

Always distinguish:

- observed output behavior;
- statistical evidence in this sample;
- a proposed explanation;
- a causal mechanism.

The behavioral experiment supports the first two. Tuned lens and PACT would still be descriptive/correlational unless paired with causal interventions.

---

## 15. Known threats to validity

### Internal validity

- Parser policy can change denominators.
- Forced candidate scoring can conceal non-Yes/No mass.
- Labels can leak if the prior answer is reconstructed from correctness.
- API retries can bias the sample if failed items are replaced.
- Quantization can alter close Yes/No logits.
- Chat templates differ across model/tokenizer versions.

### Construct validity

- “Are you sure?” may measure instruction susceptibility or sycophancy rather than a dedicated correction process.
- Answer flipping is not equivalent to belief revision.
- BoolQ Yes/No accuracy captures final-answer correctness, not reasoning quality.
- Candidate confidence restricted to Yes/No is not globally calibrated model confidence.

### External validity

- One model and one binary QA task do not generalize to math, coding, agents or multilingual settings.
- Current API models are not the historical snapshots used in the paper.
- A 300-item subset has wider uncertainty than the full 3,270-item evaluation.

### Statistical validity

- Initial and revised answers are paired; an unpaired proportion test is inappropriate.
- Multiple prompt/model comparisons inflate false positives; mark secondary analyses exploratory or correct for multiplicity.
- Stochastic replicates from the same item are not independent dataset samples.

---

## 16. Phase 2 interface: later breadth–depth refinement

Do not implement this during the core task. Design current schemas so the next experiment does not require rewriting evaluation.

The 2026 *Refining Over Resampling* pattern is:

1. Sample (N) independent reasoning trajectories (breadth).
2. For each trajectory, run (D) critic/corrector revisions (depth).
3. Extract final answers.
4. Aggregate by plurality/majority vote without an external verifier.

Add future-compatible fields now:

```json
{
  "trajectory_id": 0,
  "depth": 0,
  "parent_round_id": null,
  "stage": "generate|critique|correct|aggregate",
  "answer_cluster": null
}
```

A feasible later small-scale adaptation for this hardware is:

- Qwen2.5-1.5B-Instruct or 3B-Instruct;
- 50–100 MATH-500 problems;
- $N=3$, $D=1$;
- compare greedy, majority-at-3, breadth–depth refinement and an approximately equal-call majority-at-6/9 baseline;
- report realized tokens, latency and calls rather than relying only on nominal (N,D).

Reuse from Phase 1:

- immutable manifests;
- backend interface;
- raw trace storage;
- parser/status policy;
- paired evaluation;
- bootstrap uncertainty;
- token/latency accounting;
- reproducibility report.

Do not call the 2026 method definitive SOTA in the code or report. Treat it as a recent, promising no-training frontier method whose equal-total-compute comparison still requires careful audit.

---

## 17. Coding rules and non-negotiable guardrails

1. Preserve raw outputs and exact request messages.
2. Never silently drop, replace or retry a semantically invalid model answer.
3. Never use ground truth to decide whether to revise.
4. Never reconstruct a prior answer from correctness.
5. Never merge legacy and audited evaluation modes.
6. Never expose or commit API keys/tokens.
7. Never overwrite a previous run directory; use immutable run IDs.
8. Never make derived tables the sole copy of results.
9. Every percentage must have a recoverable numerator and denominator.
10. Every result must be traceable to code, config, prompt, model and dataset hashes.
11. Keep scientific logic out of plotting code.
12. Treat API/model failures as data-quality statuses, not wrong answers without a separate count.
13. Use type hints, deterministic tests and structured logging.
14. Prefer clear, boring code over framework-heavy abstractions.
15. Stop and document the blocker if a change would require credentials, paid compute or a materially expanded experimental scope.

---

## 18. Expected coding-agent handoff

At completion, respond with:

1. What was implemented.
2. Exact commands executed.
3. Tests and their results.
4. Experiment completion counts.
5. Headline metrics with raw transition counts.
6. Paths to raw records, resolved config, manifests, tables, figures and report.
7. Every deviation from the paper/repo.
8. Known bugs, blockers and incomplete optional work.
9. The smallest recommended next action.

Do not report only “the code runs.” The handoff must make it possible to audit whether the experiment answers the research questions.

---

## 19. Reading checklist for the coding agent

Before editing:

- [ ] Read ACL paper Sections 3–4, Appendix A, Appendix B and limitations.
- [ ] Read the official repo `README.md`.
- [ ] Trace `run_self_correction.py` → `tools.generate_dialog` → backend → parser → metrics.
- [ ] Inspect `repeat_exp` separately from the ordinary run path.
- [ ] Confirm the BoolQ validation schema and count.
- [ ] Read the ICLR 2024 sections on oracle feedback and compute-matched baselines.

Before the main run:

- [ ] Print and manually compare rendered initial/revision conversations.
- [ ] Confirm passages and labels are absent from model requests.
- [ ] Inspect Yes/No tokenization in the actual chat context.
- [ ] Confirm invalid outputs remain in the run records.
- [ ] Verify deterministic manifests and output resumption.

Before claiming a result:

- [ ] Recompute a sample of transitions by hand.
- [ ] Check $A_1-A_0=(n_{WC}-n_{CW})/n$.
- [ ] Print all denominators.
- [ ] Inspect confidence intervals, not only point estimates.
- [ ] Compare against the paper only after listing model/version/precision/sample differences.
- [ ] Phrase conclusions at the model × task × prompt × evaluator level.

---

## 20. Minimal final scientific statement template

Fill this only after the experiment:

> We evaluated `[model/revision/quantization]` on a preregistered, label-stratified subset of `[N]` BoolQ validation items. All items received one confirmatory intrinsic-revision prompt regardless of initial correctness. Under `[evaluation mode]`, initial accuracy was `[A0]` and revised accuracy was `[A1]`, for a paired change of `[ΔA, 95% CI]`. The run contained `[CW]` correct-to-wrong and `[WC]` wrong-to-correct transitions, corresponding to conditional regression `[v]` and recovery `[u]`; exact McNemar `[p]`. `[invalid count]` outputs were invalid and were retained under the stated intent-to-evaluate policy. Because `[deviations]`, this is a `[reproduction/adaptation]` of the ACL 2025 BoolQ experiment. The result supports only a claim about this model, dataset, prompt, decoding configuration and evaluator.

This template is intentionally narrow. Broader claims require more models, tasks, prompts and compute-controlled baselines.
