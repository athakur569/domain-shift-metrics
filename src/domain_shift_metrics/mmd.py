"""Maximum Mean Discrepancy with RBF kernel."""

from __future__ import annotations

import numpy as np


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel matrix K(X, Y) = exp(-||x-y||^2 / (2 sigma^2))."""
    XY = X @ Y.T
    XX = (X ** 2).sum(axis=1, keepdims=True)
    YY = (Y ** 2).sum(axis=1, keepdims=True)
    sq_dist = XX - 2 * XY + YY.T
    return np.exp(-sq_dist / (2 * sigma ** 2))


def _median_heuristic(X: np.ndarray, Y: np.ndarray, subsample: int = 1000) -> float:
    """Estimate bandwidth as median pairwise distance."""
    Z = np.vstack([X, Y])
    if len(Z) > subsample:
        idx = np.random.default_rng(0).choice(len(Z), subsample, replace=False)
        Z = Z[idx]
    dists = np.sqrt(np.maximum(
        ((Z[:, None] - Z[None, :]) ** 2).sum(axis=-1), 0
    ))
    median = np.median(dists[dists > 0])
    return float(median) if median > 0 else 1.0


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float | None = None,
    biased: bool = False,
) -> float:
    """Compute MMD² between two sample sets.

    Args:
        X:       Source samples (n, d).
        Y:       Target samples (m, d).
        sigma:   RBF bandwidth. If None, uses median heuristic.
        biased:  Use biased estimator (includes diagonal terms).

    Returns:
        MMD² estimate (float). Zero means identical distributions.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if sigma is None:
        sigma = _median_heuristic(X, Y)

    n, m = len(X), len(Y)
    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    if biased:
        return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())

    # Unbiased: exclude diagonal
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return float(
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2 * Kxy.mean()
    )


class MMDEstimator:
    """Stateful MMD estimator that stores a reference distribution."""

    def __init__(self, sigma: float | None = None) -> None:
        self.sigma = sigma
        self._reference: np.ndarray | None = None

    def fit(self, X_reference: np.ndarray) -> "MMDEstimator":
        self._reference = np.asarray(X_reference, dtype=np.float64)
        if self._reference.ndim == 1:
            self._reference = self._reference[:, None]
        if self.sigma is None:
            self.sigma = _median_heuristic(self._reference, self._reference)
        return self

    def score(self, X_deployment: np.ndarray) -> float:
        if self._reference is None:
            raise RuntimeError("Call fit() before score()")
        return compute_mmd(self._reference, X_deployment, sigma=self.sigma)
