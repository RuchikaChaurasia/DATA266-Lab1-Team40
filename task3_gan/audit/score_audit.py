"""Score the blinded audit:  python task3_gan/audit/score_audit.py

Reads ratings_rater1.csv, ratings_rater2.csv and key/key.csv, then writes
    audit_results.csv    mean (both raters) style/content/artifacts per member
    audit_agreement.csv  Cohen's kappa (quadratic-weighted and unweighted), exact and within-1 agreement
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
CRIT = ["style", "content", "artifacts"]
key = pd.read_csv(HERE / "key" / "key.csv")
r1 = pd.read_csv(HERE / "ratings_rater1.csv")
r2 = pd.read_csv(HERE / "ratings_rater2.csv")
for name, r in [("rater1", r1), ("rater2", r2)]:
    for c in CRIT:
        bad = ~r[c].isin([1, 2, 3, 4, 5])
        if bad.any():
            raise SystemExit(f"{name}: '{c}' must be an integer 1-5 for every item; check rows {r.loc[bad, 'item_id'].tolist()}")

df = key.merge(r1[["item_id"] + CRIT], on="item_id").merge(r2[["item_id"] + CRIT], on="item_id", suffixes=("_r1", "_r2"))
agree = []
for c in CRIT:
    a, b = df[f"{c}_r1"].astype(int), df[f"{c}_r2"].astype(int)
    agree.append({"criterion": c, "n_items": len(df),
                  "cohen_kappa_quadratic": cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5]),
                  "cohen_kappa_unweighted": cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5]),
                  "exact_agreement": float((a == b).mean()), "within_1_agreement": float(((a - b).abs() <= 1).mean())})
agree = pd.DataFrame(agree)

rows = []
for m, g in df.groupby("member"):
    row = {"member": m, "n_items": len(g)}
    for c in CRIT:
        both = (g[f"{c}_r1"] + g[f"{c}_r2"]) / 2
        row[f"{c}_mean"], row[f"{c}_std"] = both.mean(), both.std()
    row["overall_mean"] = np.mean([row[f"{c}_mean"] for c in CRIT])
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(HERE / "audit_results.csv", index=False)
agree.to_csv(HERE / "audit_agreement.csv", index=False)
print(res.round(3).to_string(index=False)); print(); print(agree.round(3).to_string(index=False))
