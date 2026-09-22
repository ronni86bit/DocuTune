"""Deterministic bootstrap confidence intervals.

Bootstrap resampling (percentile method, seeded RNG) over the held-out test
examples. This quantifies sampling uncertainty ON THIS BENCHMARK only - it
does not establish statistical significance for any universal claim.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any


def bootstrap_stat_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float],
                      n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0, "point": 0.0}
    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(stat(sample))
    stats.sort()
    lo_idx = max(0, int((alpha / 2) * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return {
        "low": round(stats[lo_idx], 4),
        "high": round(stats[hi_idx], 4),
        "point": round(stat(values), 4),
    }


def bootstrap_ratio_ci(numerators: Sequence[float], denominators: Sequence[float],
                       n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    """CI for sum(num)/sum(den) under resampling (ratio-of-sums)."""

    def ratio(sample_num: Sequence[float], sample_den: Sequence[float]) -> float:
        den = sum(sample_den)
        return (sum(sample_num) / den) if den else 0.0

    if not numerators:
        return {"low": 0.0, "high": 0.0, "point": 0.0}
    rng = random.Random(seed)
    n = len(numerators)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(ratio([numerators[i] for i in idx], [denominators[i] for i in idx]))
    stats.sort()
    lo_idx = max(0, int((alpha / 2) * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return {
        "low": round(stats[lo_idx], 4),
        "high": round(stats[hi_idx], 4),
        "point": round(ratio(list(numerators), list(denominators)), 4),
    }


def summarize_bootstrap(evals: list[dict[str, Any]], n_boot: int = 1000,
                        seed: int = 42) -> dict[str, dict[str, float]]:
    """Bootstrap CIs for the headline metrics from per-example evaluations.

    Expected keys per eval: schema_valid, exact_match, tp/fp/fn,
    unsupported_units, predicted_units.
    """
    schema = [1.0 if ev.get("schema_valid") else 0.0 for ev in evals]
    exact = [1.0 if ev.get("exact_match") else 0.0 for ev in evals]
    unsup_num = [float(ev.get("unsupported_units", 0)) for ev in evals]
    unsup_den = [float(ev.get("predicted_units", 0)) for ev in evals]

    def f1_of(sample: list[dict[str, Any]]) -> float:
        tp = sum(e["tp"] for e in sample)
        fp = sum(e["fp"] for e in sample)
        fn = sum(e["fn"] for e in sample)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    field_f1_ci = bootstrap_stat_ci(evals, f1_of, n_boot=n_boot, seed=seed)
    return {
        "schema_validity": bootstrap_stat_ci(schema, lambda s: sum(s) / len(s) if s else 0.0,
                                             n_boot=n_boot, seed=seed + 1),
        "exact_match": bootstrap_stat_ci(exact, lambda s: sum(s) / len(s) if s else 0.0,
                                         n_boot=n_boot, seed=seed + 2),
        "field_f1": field_f1_ci,
        "unsupported_value_rate": bootstrap_ratio_ci(unsup_num, unsup_den,
                                                     n_boot=n_boot, seed=seed + 3),
    }
