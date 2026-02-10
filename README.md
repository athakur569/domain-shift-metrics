# domain-shift-metrics

Distribution divergence metrics for detecting and quantifying covariate shift between datasets. Designed for computational pathology but applicable anywhere you need to compare two feature distributions.

## Metrics
- **MMD** — Maximum Mean Discrepancy with RBF kernel (median heuristic bandwidth)
- **Wasserstein-1 / W-2** — Earth mover's distances via SciPy linear programming
- **KL / JS divergence** — Estimated from kernel density or histogram binning
- **Composed shift report** — Single summary score with per-metric breakdown

## Installation
```bash
pip install domain-shift-metrics
```

## Quick start
```python
from domain_shift_metrics import compute_mmd, compute_wasserstein, ShiftReport

X_source = ...  # (n_samples, n_features) numpy array
X_target = ...

mmd = compute_mmd(X_source, X_target)
w1  = compute_wasserstein(X_source, X_target, order=1)

report = ShiftReport(X_source, X_target)
report.run_all()
report.print_summary()
```

## Use case
Detect when a deployed pathology model's input distribution has shifted from its training reference — Tier 1 of a label-free validity monitor.

## License
MIT
