# Repository Guidelines

## Research Collaboration

This research workspace supports publication-quality work in AI and data
mining. Do not assume a fixed thesis topic: treat ideas as revisable until
novelty, feasibility, and empirical value are established.

Communicate primarily in Vietnamese while retaining standard English technical
terms. Lead with the recommended action. Distinguish verified facts,
interpretations, hypotheses, and speculation; never invent citations,
results, capabilities, or dataset properties. Prefer primary sources,
peer-reviewed papers, official repositories, and current documentation. Check
the closest prior work before calling an idea novel.

## Project Structure & Artifacts

The repository currently has no implementation or toolchain. Add a minimal,
discoverable structure when needed:

- `src/` for reusable code; `tests/` for matching automated checks.
- `data/` for small, non-sensitive reproducible samples only.
- `configs/` for versioned experiment settings and pinned identifiers.
- `docs/` for literature notes, decisions, methods, and limitations.
- `outputs/` only when deliberately versioned; keep large artifacts outside
  Git.

Use descriptive lowercase names, such as `literature-review.md` or
`evaluate_retrieval.py`. Do not commit credentials, private data, caches, or
local environments.

## Research Design & Experiments

For each serious idea, record the problem, precise question, falsifiable
hypotheses, related-work gap, data-quality risks, compute-matched baselines,
metrics and statistical tests, error analysis, ablations, robustness checks,
resource needs, and the smallest experiment that could disprove the claim.
Challenge data leakage, unfair comparisons, and unsupported conclusions.

Favor reproducible experiments on local hardware or modest cloud resources.
Pin dependencies, model IDs, prompts, seeds, and configurations. Preserve raw
outputs, trajectories, structured logs, and resumable pipelines. Test parsers
and evaluators before large runs. Report negative results, failures,
uncertainty, and limitations; never alter or discard existing work without
explicit confirmation.

## Build, Test, and Review

No build or test commands are configured yet. When adding code, choose the
smallest appropriate toolchain, document its root-level commands, and add a
focused test (for example, `test_normalizes_missing_values`). Run relevant
checks before review.

Use short, imperative Conventional Commit-style messages, e.g., `docs: add
dataset risk notes`. Pull requests should state purpose, validation, linked
issue or research decision, limitations, and screenshots for user-visible
changes.

## Progress Updates

Make work supervisor-ready: close with key findings, evidence for or against
the hypothesis, blockers and risks, decisions and open questions, three
highest-priority next actions, and one concrete deliverable for the next
supervisor meeting. Optimize for scientific validity, reproducibility,
learning value, and publishable insight—not impressive demos alone.
