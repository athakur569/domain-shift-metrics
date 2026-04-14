"""Combined shift report: runs all metrics and produces a summary."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .mmd import compute_mmd
from .wasserstein import compute_wasserstein, sliced_wasserstein
from .divergence import compute_kl, compute_js


@dataclass
class ShiftSummary:
    mmd_squared: float
    wasserstein_1: float
    sliced_wasserstein: float
    kl_divergence: float
    js_divergence: float
    composed_score: float  # normalized average
    flagged: bool          # True if composed_score > threshold

    def to_dict(self) -> dict:
        return asdict(self)


class ShiftReport:
    """Run all shift metrics between source and target distributions.

    Args:
        X_source:   Reference (training) features (n, d).
        X_target:   Deployment features (m, d).
        threshold:  Composed score above which the batch is flagged as shifted.
    """

    def __init__(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        threshold: float = 0.5,
    ) -> None:
        self.X_source = np.asarray(X_source, dtype=np.float64)
        self.X_target = np.asarray(X_target, dtype=np.float64)
        self.threshold = threshold
        self._summary: ShiftSummary | None = None

    def run_all(self) -> ShiftSummary:
        mmd2 = compute_mmd(self.X_source, self.X_target)
        w1 = compute_wasserstein(self.X_source, self.X_target, order=1)
        sw = sliced_wasserstein(self.X_source, self.X_target)
        kl = compute_kl(self.X_source, self.X_target)
        js = compute_js(self.X_source, self.X_target)

        # Composed score: sigmoid-normalized mean of log-scaled metrics
        raw = np.array([mmd2, w1, sw, kl, js])
        log_raw = np.log1p(raw)
        composed = float(np.mean(log_raw) / (1 + np.mean(log_raw)))

        self._summary = ShiftSummary(
            mmd_squared=mmd2,
            wasserstein_1=w1,
            sliced_wasserstein=sw,
            kl_divergence=kl,
            js_divergence=js,
            composed_score=composed,
            flagged=composed > self.threshold,
        )
        return self._summary

    def print_summary(self) -> None:
        if self._summary is None:
            self.run_all()
        s = self._summary
        flag_str = "FLAGGED" if s.flagged else "OK"
        print(f"\n{'='*50}")
        print(f"  Shift Report  [{flag_str}]")
        print(f"{'='*50}")
        print(f"  MMD²              : {s.mmd_squared:.4f}")
        print(f"  Wasserstein-1     : {s.wasserstein_1:.4f}")
        print(f"  Sliced Wasserstein: {s.sliced_wasserstein:.4f}")
        print(f"  KL divergence     : {s.kl_divergence:.4f}")
        print(f"  JS divergence     : {s.js_divergence:.4f}")
        print(f"  Composed score    : {s.composed_score:.4f}  (threshold {self.threshold})")
        print(f"{'='*50}\n")

    def save(self, path: str) -> None:
        if self._summary is None:
            self.run_all()
        with open(path, "w") as f:
            json.dump(self._summary.to_dict(), f, indent=2)
