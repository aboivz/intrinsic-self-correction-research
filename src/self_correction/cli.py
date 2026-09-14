from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .artifacts import load_yaml, read_events, sha256
from .backend import OllamaBackend
from .data import write_manifests
from .experiment import run_experiment
from .metrics import aggregate
from .models import DecodingConfig, Manifest, ModelIdentity


def _json_request(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None, headers={"Content-Type": "application/json"} if payload else {}, method="POST" if payload else "GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _config(path: Path) -> dict:
    config = load_yaml(path)
    if config.get("schema_version") != "1.0.0" or config.get("evaluation_mode") != "free_generation":
        raise ValueError("config requires schema_version 1.0.0 and free_generation")
    return config


def doctor(config_path: Path) -> int:
    config = _config(config_path)
    model_id = config["model"]["model_id"]
    report: dict[str, object] = {
        "config_valid": True, "python": sys.version, "platform": platform.platform(),
        "wsl": "microsoft" in Path("/proc/version").read_text(errors="ignore").lower() if Path("/proc/version").exists() else False,
        "ollama_binary": shutil.which("ollama") or ("/home/antd/.local/bin/ollama" if Path("/home/antd/.local/bin/ollama").exists() else None), "model_id": model_id, "cpu_count": os.cpu_count(),
        "writable_runs": os.access("runs", os.W_OK), "writable_reports": os.access("reports", os.W_OK),
    }
    try:
        report["ollama_version"] = _json_request("http://127.0.0.1:11434/api/version")
        tags = _json_request("http://127.0.0.1:11434/api/tags").get("models", [])
        tag = next((item for item in tags if item.get("name") == model_id), None)
        report["model_tag_present"] = tag is not None
        report["model_digest"] = tag.get("digest") if tag else None
        show = _json_request("http://127.0.0.1:11434/api/show", {"model": model_id})
        modelfile = show.get("modelfile", "")
        report["modelfile"] = modelfile
        report["template_hash"] = sha256(modelfile)
        Path("runs/doctor_modelfile.txt").write_text(modelfile, encoding="utf-8")
        report["ollama_reachable"] = True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as exc:
        report["ollama_reachable"] = False
        report["ollama_error"] = str(exc)
    if Path("/proc/meminfo").exists():
        report["meminfo"] = Path("/proc/meminfo").read_text()
    nvidia_smi = shutil.which("nvidia-smi")
    report["nvidia_smi"] = nvidia_smi
    if nvidia_smi:
        report["gpu"] = subprocess.run([nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    Path("runs/doctor.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("ollama_reachable") and report.get("model_tag_present") else 1


def run(config_path: Path, manifest_path: Path, condition: str, rounds: int, run_id: str | None) -> int:
    config = _config(config_path)
    prompts = load_yaml(Path(config["prompt_file"]))
    if condition not in prompts:
        raise ValueError(f"unknown condition: {condition}")
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    decoding = DecodingConfig.model_validate(config["decoding"])
    config_hash, prompt_hash = sha256(config), sha256(prompts)
    run_id = run_id or f"smoke-{condition}-{sha256([manifest.sha256, config_hash])[:12]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model = config["model"]
    try:
        tag = next(item for item in _json_request("http://127.0.0.1:11434/api/tags").get("models", []) if item.get("name") == model["model_id"])
        modelfile = _json_request("http://127.0.0.1:11434/api/show", {"model": model["model_id"]}).get("modelfile", "")
        identity = ModelIdentity(provider=model["provider"], model_id=model["model_id"], digest=tag.get("digest"), template_hash=sha256(modelfile))
    except (StopIteration, urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        identity = ModelIdentity(provider=model["provider"], model_id=model["model_id"], digest=model.get("digest"))
    resolved = {"run_id": run_id, "config": config, "manifest_sha256": manifest.sha256, "prompt_hash": prompt_hash, "config_hash": config_hash, "adaptation": True}
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    calls = run_experiment(manifest, OllamaBackend(model["model_id"], identity), run_dir / "journal.jsonl", run_id, condition, rounds, prompts, decoding, prompt_hash, config_hash)
    print(json.dumps({"run_id": run_id, "backend_calls": calls, "journal": str(run_dir / "journal.jsonl")}))
    return 0


def validate(run_dir: Path) -> int:
    events = read_events(run_dir / "journal.jsonl")
    request_ids = [event.request_id for event in events]
    duplicates = len(request_ids) - len(set(request_ids))
    grouped: dict[str, set[int]] = {}
    for event in events:
        grouped.setdefault(event.sample_id, set()).add(event.round)
    incomplete = sorted(sample_id for sample_id, rounds in grouped.items() if rounds != {0, 1})
    result = {"events": len(events), "unique_samples": len(grouped), "duplicate_request_ids": duplicates, "incomplete_samples": incomplete, "valid": not duplicates and not incomplete}
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


def make_report(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else aggregate(read_events(run_dir / "journal.jsonl"))
    report = Path("reports") / "smoke_summary.md"
    valid_initial = "n/a" if metrics["valid_only_A0"] is None else f"{metrics['valid_only_A0']:.1%} (n={metrics['valid_only_A0_denominator']})"
    text = f"""# BoolQ smoke summary — adaptation

This 10-item smoke run validates experimental integrity; it is not evidence for or against intrinsic self-correction.

## Results

- Completed paired items: {metrics['completed_pairs']}
- Initial accuracy (all items): {metrics['A0']:.1%}
- Revised accuracy (all items): {metrics['A1']:.1%}
- Delta accuracy (`A1 - A0`): {metrics['delta_A']:.1%}
- Transitions: CC={metrics['CC']}, CW={metrics['CW']}, WC={metrics['WC']}, WW={metrics['WW']}
- Exact two-sided McNemar p-value: {metrics['mcnemar_exact_two_sided_p']:.4g}
- Paired bootstrap 95% CI for delta: [{metrics['bootstrap_delta_A_95_ci'][0]:.1%}, {metrics['bootstrap_delta_A_95_ci'][1]:.1%}]
- Valid-only initial accuracy: {valid_initial}

## Limits

Adaptation reasons: different model family, GGUF Q8 quantization, Ollama runtime/chat template, and 10-item subset. Invalid and error outputs remain in all-item denominators. The raw journal stores exact messages, output, request/response metadata, and retries.
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("doctor"); command.add_argument("--config", type=Path, required=True)
    command = sub.add_parser("prepare-data"); command.add_argument("--source", type=Path, required=True); command.add_argument("--main-n", type=int, required=True); command.add_argument("--smoke-n", type=int, required=True); command.add_argument("--wavering-n", type=int, required=True); command.add_argument("--seed", type=int, required=True)
    command = sub.add_parser("run"); command.add_argument("--config", type=Path, required=True); command.add_argument("--manifest", type=Path, required=True); command.add_argument("--condition", required=True); command.add_argument("--rounds", type=int, required=True); command.add_argument("--run-id")
    command = sub.add_parser("validate-run"); command.add_argument("run_dir", type=Path)
    command = sub.add_parser("aggregate"); command.add_argument("run_dir", type=Path)
    command = sub.add_parser("report"); command.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "doctor": return doctor(args.config)
    if args.command == "prepare-data":
        write_manifests(args.source, Path("data/manifests"), args.main_n, args.smoke_n, args.wavering_n, args.seed); return 0
    if args.command == "run": return run(args.config, args.manifest, args.condition, args.rounds, args.run_id)
    if args.command == "validate-run": return validate(args.run_dir)
    if args.command == "aggregate":
        metrics = aggregate(read_events(args.run_dir / "journal.jsonl")); (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8"); print(json.dumps(metrics, indent=2)); return 0
    report = make_report(args.run_dir); print(report); return 0


if __name__ == "__main__":
    raise SystemExit(main())
