from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from .models import JournalEvent


def _pairs(events: list[JournalEvent]) -> list[tuple[JournalEvent, JournalEvent]]:
    grouped: dict[str, dict[int, JournalEvent]] = defaultdict(dict)
    for event in events:
        grouped[event.sample_id][event.round] = event
    return [(rounds[0], rounds[1]) for rounds in grouped.values() if 0 in rounds and 1 in rounds]


def aggregate(events: list[JournalEvent], bootstrap_seed: int = 20260912, resamples: int = 10_000) -> dict:
    pairs = _pairs(events)
    transitions = Counter()
    status_transitions = Counter()
    for first, revised in pairs:
        transitions[("C" if first.is_correct else "W") + ("C" if revised.is_correct else "W")] += 1
        status_transitions[(_status_class(first.parse_status), _status_class(revised.parse_status))] += 1
    cc, cw, wc, ww = (transitions[key] for key in ("CC", "CW", "WC", "WW"))
    n = len(pairs)
    a0 = (cc + cw) / n if n else 0.0
    a1 = (cc + wc) / n if n else 0.0
    delta = (wc - cw) / n if n else 0.0
    discordant = cw + wc
    p_value = 1.0 if not discordant else binomtest(min(cw, wc), discordant, 0.5).pvalue
    rng = np.random.default_rng(bootstrap_seed)
    changes = np.array([int(second.is_correct) - int(first.is_correct) for first, second in pairs])
    if n:
        values = changes[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
        ci = [float(x) for x in np.quantile(values, [0.025, 0.975])]
    else:
        ci = [0.0, 0.0]
    parse_counts = Counter(event.parse_status for event in events)
    valid_first = [pair for pair in pairs if pair[0].parsed_answer is not None]
    valid_revised = [pair for pair in pairs if pair[1].parsed_answer is not None]
    return {
        "attempted_round_events": len(events), "completed_pairs": n,
        "errors": parse_counts["error"], "parse_counts": dict(sorted(parse_counts.items())),
        "A0": a0, "A1": a1, "delta_A": delta, "paper_drop": -delta,
        "valid_only_A0": sum(first.is_correct for first, _ in valid_first) / len(valid_first) if valid_first else None,
        "valid_only_A0_denominator": len(valid_first),
        "valid_only_A1": sum(second.is_correct for _, second in valid_revised) / len(valid_revised) if valid_revised else None,
        "valid_only_A1_denominator": len(valid_revised),
        "CC": cc, "CW": cw, "WC": wc, "WW": ww,
        "conditional_recovery": wc / (wc + ww) if wc + ww else None,
        "conditional_regression": cw / (cc + cw) if cc + cw else None,
        "answer_change_rate": sum(first.parsed_answer != second.parsed_answer for first, second in pairs) / n if n else 0.0,
        "mcnemar_exact_two_sided_p": p_value,
        "bootstrap_delta_A_95_ci": ci, "bootstrap_seed": bootstrap_seed, "bootstrap_resamples": resamples,
        "status_transitions": {f"{left}|{right}": value for (left, right), value in sorted(status_transitions.items())},
        "latency_ns": sum(event.usage.total_duration_ns or 0 for event in events),
        "input_tokens": sum(event.usage.input_tokens or 0 for event in events),
        "output_tokens": sum(event.usage.output_tokens or 0 for event in events),
    }


def _status_class(status: str) -> str:
    if status == "valid_yes":
        return "yes"
    if status == "valid_no":
        return "no"
    return "other"
