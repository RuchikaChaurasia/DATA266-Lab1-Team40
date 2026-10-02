"""Recompute the gradient-norm statistics of a FINISHED Task 3 run without the fp16-overflow steps.

    python scripts/fix_gradnorm_report.py task3_gan/ruchika_chaurasia

Runs trained before this fix logged a whole 50-step window as inf/nan whenever one step overflowed in
fp16 (GradScaler skips those steps, so training itself was unaffected). This recomputes mean/max over the
finite windows of outputs/loss_log.csv, records how many windows were excluded, and updates
full_metrics_report.csv, metrics_report.csv and results.md in place. The original values are logged.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1] if len(sys.argv) > 1 else "task3_gan/ruchika_chaurasia")
L = pd.read_csv(d / "outputs" / "loss_log.csv")
R = pd.read_csv(d / "full_metrics_report.csv")
new = {}
for k in ("gradnorm_G", "gradnorm_D"):
    v = L[k].replace([np.inf, -np.inf], np.nan)
    new[f"{k}_mean"] = float(v.mean())
    new[f"{k}_max"] = float(v.max())
    new[f"{k}_nonfinite_windows"] = int(v.isna().sum())
new["gradnorm_note"] = "mean/max over 50-step logging windows without an fp16-overflow step (recomputed)"
print("old:", R[R.metric.str.startswith("gradnorm")].to_dict("records"))
R = R[~R.metric.isin(new)]
R = pd.concat([R, pd.DataFrame([{"metric": k, "direction": "both", "value": v} for k, v in new.items()])], ignore_index=True)
R.to_csv(d / "full_metrics_report.csv", index=False)
R.to_csv(d / "metrics_report.csv", index=False)
res = d / "results.md"
if res.exists():
    t = res.read_text(encoding="utf-8")
    for k, v in new.items():
        if isinstance(v, float):
            t = re.sub(rf"\| {k} \| [^|]+ \|", f"| {k} | {v:.4g} |", t)
    if "gradnorm_G_nonfinite_windows" not in t:
        t = t.replace("## Kaggle", f"Gradient norms recomputed without fp16-overflow windows: "
                      f"G non-finite windows {new['gradnorm_G_nonfinite_windows']}, D {new['gradnorm_D_nonfinite_windows']} "
                      f"(of {len(L)}).\n\n## Kaggle", 1)
    res.write_text(t, encoding="utf-8")
print("new:", new)
