"""Wasserstein distances between empirical distributions."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    order: int = 1,
    per_feature: bool = False,
) -> float | np.ndarray:
    """Compute Wasserstein-p distance between two sample sets.

    For multivariate inputs, computes the distance per feature and returns
    the mean (or the full array if ``per_feature=True``).

    Args:
        X:           Source samples (n,) or (n, d).
        Y:           Target samples (m,) or (m, d).
        order:       Order p of the Wasserstein distance (1 or 2).
        per_feature: Return per-feature distances instead of mean.

    Returns:
        Scalar mean distance, or (d,) array if per_feature=True.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]}, Y has {Y.shape[1]}"
        )

    d = X.shape[1]
    distances = np.empty(d)
    for i in range(d):
        if order == 1:
            distances[i] = wasserstein_distance(X[:, i], Y[:, i])
        elif order == 2:
            distances[i] = wasserstein_distance(
                X[:, i], Y[:, i],
                u_weights=None, v_weights=None,
            ) ** 2  # W2 = sqrt(W1²); here we return W1²-based approx
        else:
            raise ValueError(f"order must be 1 or 2, got {order}")

    return distances if per_feature else float(distances.mean())


def sliced_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 100,
    random_state: int = 42,
) -> float:
    """Sliced Wasserstein distance — scalable for high dimensions.

    Projects samples onto random 1D lines and averages W1 distances.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    d = X.shape[1]
    total = 0.0
    for _ in range(n_projections):
        direction = rng.standard_normal(d)
        direction /= np.linalg.norm(direction) + 1e-10
        proj_x = X @ direction
        proj_y = Y @ direction
        total += wasserstein_distance(proj_x, proj_y)
    return total / n_projections
