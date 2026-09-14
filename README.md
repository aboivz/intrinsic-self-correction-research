# Audited BoolQ smoke replication

An adaptation of the ACL 2025 self-correction experiment. It runs 10 fixed BoolQ validation questions through initial generation and one confirmatory revision, retaining all invalid outputs and errors.

```bash
uv sync --locked --group dev
uv run python -m self_correction.cli doctor --config configs/smoke.yaml
uv run python -m self_correction.cli prepare-data --source vendor/USC/dataset/boolq_json/boolq_validation.json --main-n 300 --smoke-n 10 --wavering-n 50 --seed 20260912
uv run python -m self_correction.cli run --config configs/smoke.yaml --manifest data/manifests/boolq_smoke_10.json --condition confirmatory --rounds 1
uv run python -m self_correction.cli validate-run runs/<run_id>
uv run python -m self_correction.cli aggregate runs/<run_id>
uv run python -m self_correction.cli report runs/<run_id>
```

The default evaluator is `free_generation`. Results are an adaptation, not an exact reproduction: Qwen, GGUF Q8, Ollama template/runtime, and 10-item subset differ.
