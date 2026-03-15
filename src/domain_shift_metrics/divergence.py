"""KL and Jensen-Shannon divergence estimators."""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


def _kde_density(X: np.ndarray, x_eval: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian KDE evaluated at x_eval."""
    diff = x_eval[:, None] - X[None, :]
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
    return weights.sum(axis=1) / (len(X) * bandwidth * np.sqrt(2 * np.pi))


def compute_kl(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 50,
    eps: float = 1e-10,
    per_feature: bool = False,
) -> float | np.ndarray:
    """Estimate KL(X || Y) via histogram binning.

    Args:
        X:           Source samples (n,) or (n, d).
        Y:           Target samples (m,) or (m, d).
        n_bins:      Histogram resolution.
        eps:         Floor to avoid log(0).
        per_feature: Return per-feature divergences.

    Returns:
        Scalar KL divergence (mean across features), or (d,) array.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    d = X.shape[1]
    kls = np.empty(d)
    for i in range(d):
        lo = min(X[:, i].min(), Y[:, i].min())
        hi = max(X[:, i].max(), Y[:, i].max())
        bins = np.linspace(lo, hi, n_bins + 1)
        px, _ = np.histogram(X[:, i], bins=bins, density=True)
        py, _ = np.histogram(Y[:, i], bins=bins, density=True)
        px = px + eps
        py = py + eps
        px /= px.sum()
        py /= py.sum()
        kls[i] = float(scipy_entropy(px, py))

    return kls if per_feature else float(kls.mean())


def compute_js(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 50,
    per_feature: bool = False,
) -> float | np.ndarray:
    """Estimate Jensen-Shannon divergence (symmetric, bounded [0, log 2]).

    JS(X, Y) = 0.5 * KL(X || M) + 0.5 * KL(Y || M)  where M = 0.5*(X+Y).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    d = X.shape[1]
    eps = 1e-10
    jss = np.empty(d)
    for i in range(d):
        lo = min(X[:, i].min(), Y[:, i].min())
        hi = max(X[:, i].max(), Y[:, i].max())
        bins = np.linspace(lo, hi, 51)
        px, _ = np.histogram(X[:, i], bins=bins, density=True)
        py, _ = np.histogram(Y[:, i], bins=bins, density=True)
        px = (px + eps) / (px + eps).sum()
        py = (py + eps) / (py + eps).sum()
        pm = 0.5 * (px + py)
        jss[i] = 0.5 * float(scipy_entropy(px, pm)) + 0.5 * float(scipy_entropy(py, pm))

    return jss if per_feature else float(jss.mean())
