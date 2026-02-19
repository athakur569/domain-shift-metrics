"""domain-shift-metrics: distribution divergence for covariate shift detection."""

from .mmd import compute_mmd, MMDEstimator
from .wasserstein import compute_wasserstein
from .divergence import compute_kl, compute_js
from .report import ShiftReport

__version__ = "0.1.0"
__all__ = [
    "compute_mmd", "MMDEstimator",
    "compute_wasserstein",
    "compute_kl", "compute_js",
    "ShiftReport",
]
