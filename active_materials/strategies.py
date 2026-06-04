"""Sampling strategies for active learning."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    regressor_,
    x: np.ndarray,
    *,
    number: int = 30,
    y_training: np.ndarray | None = None,
    **_: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the points with the largest expected improvement."""

    mu, std = regressor_.predict(x, return_std=True)
    std = np.maximum(std, 1e-9)
    best = float(np.min(y_training)) if y_training is not None else float(np.min(mu))
    z = (best - mu) / std
    score = (best - mu) * norm.cdf(z) + std * norm.pdf(z)
    query_idx = np.argsort(score)[::-1][:number]
    return query_idx, x[query_idx]


def threshold_expected_improvement(
    regressor_,
    x: np.ndarray,
    *,
    number: int = 30,
    y_training: np.ndarray | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classic EI with probability of being below a useful threshold."""

    mu, std = regressor_.predict(x, return_std=True)
    std = np.maximum(std, 1e-9)
    best = float(np.min(y_training)) if y_training is not None else float(np.min(mu))
    threshold = best if threshold is None else threshold

    z_best = (best - mu) / std
    classic_ei = (best - mu) * norm.cdf(z_best) + std * norm.pdf(z_best)

    z_threshold = (threshold - mu) / std
    threshold_ei = (threshold - mu) * norm.cdf(z_threshold) + std * norm.pdf(z_threshold)
    hit_probability = norm.cdf(z_threshold)

    score = 0.45 * _minmax(classic_ei) + 0.35 * _minmax(threshold_ei) + 0.20 * _minmax(hit_probability)
    query_idx = np.argsort(score)[::-1][:number]
    return query_idx, x[query_idx]


def get_strategy(name: str):
    """Return a strategy function by name."""

    strategies = {
        "expected_improvement": expected_improvement,
        "threshold_expected_improvement": threshold_expected_improvement,
    }
    try:
        return strategies[name]
    except KeyError as exc:
        available = ", ".join(sorted(strategies))
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available}") from exc


def _minmax(values: np.ndarray) -> np.ndarray:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper <= lower + 1e-12:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)
