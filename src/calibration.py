"""Empirical confidence bins from human-labeled outcomes, never LLM confidence."""

import math


def signal_score(signals):
    # Ranking signal only; the returned value is not a probability until fitted.
    relevance = max(0., min(1., signals.get("max_dense_similarity") or 0.))
    coverage = max(0., min(1., signals.get("evidence_coverage") or 0.))
    reranked = [v for v in signals.get("reranker_scores", []) if v is not None and math.isfinite(v)]
    rerank = 1 / (1 + math.exp(-max(-20., min(20., max(reranked))))) if reranked else relevance
    return .35 * relevance + .25 * rerank + .2 * coverage + .2 * bool(signals.get("verification_passed"))


def fit_calibration(rows, dataset_version, minimum_per_bin=20):
    if not rows or any(r.get("reviewed") is not True or not isinstance(r.get("correct"), bool) for r in rows):
        raise ValueError("Calibration requires human-reviewed boolean correctness labels")
    bins = []
    for index in range(5):
        selected = [r for r in rows if min(4, int(signal_score(r["signals"]) * 5)) == index]
        count = len(selected)
        successes = sum(r["correct"] for r in selected)
        bins.append({"lower": index / 5, "upper": (index + 1) / 5, "count": count,
                     "probability": successes / count if count >= minimum_per_bin else None})
    return {"version": "empirical-bins-v1", "dataset_version": dataset_version,
            "minimum_per_bin": minimum_per_bin, "bins": bins}


def calibrated_confidence(signals, calibration):
    index = min(4, int(signal_score(signals) * 5))
    return calibration["bins"][index]["probability"]
