"""Probabilistic forecast metrics, conformal calibration and DM tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def pinball_loss(actual: np.ndarray, quantile: np.ndarray, alpha: float) -> float:
    """Mean pinball loss for a single quantile level."""
    actual = np.asarray(actual, dtype=float)
    quantile = np.asarray(quantile, dtype=float)
    error = actual - quantile
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def conformal_adjustment(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample (n+1) correction."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_quantile(residuals: np.ndarray, tau: float) -> float:
    """Empirical residual quantile with the same finite-sample correction."""
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * tau) / n)
    return float(np.quantile(residuals, level, method="higher"))


def dm_test(errors_a: np.ndarray, errors_b: np.ndarray, power: float = 1.0) -> tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy (absolute loss)."""
    a = np.abs(np.asarray(errors_a, dtype=float)) ** power
    b = np.abs(np.asarray(errors_b, dtype=float)) ** power
    diff = a - b
    n = len(diff)
    if n < 2 or diff.std(ddof=1) == 0:
        return 0.0, 1.0
    statistic = float(diff.mean() / np.sqrt(diff.var(ddof=1) / n))
    p_value = float(2 * (1 - stats.norm.cdf(abs(statistic))))
    return statistic, p_value
