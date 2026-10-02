"""Task 2 shared evaluation — agreed by the team before training.

Every model is evaluated on the OFFICIAL Yelp Polarity test split (38,000 reviews)
with positive = label 1, a 0.5 threshold on P(positive), and the same slice definitions.
"""
from __future__ import annotations

import re

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

THRESHOLD = 0.5
N_BOOT = 1000
ECE_BINS = 15


# --------------------------------------------------------------------------- core metrics
def expected_calibration_error(y, p, n_bins=ECE_BINS):
    """ECE for P(positive): weighted mean |accuracy - confidence| over equal-width bins of the
    predicted-class confidence max(p, 1-p)."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= THRESHOLD).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_table(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                         "frac_pos": float(y[m].mean())})
    return rows


def classification_metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yhat = (p >= THRESHOLD).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    out = {
        "accuracy": accuracy_score(y, yhat),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "mcc": matthews_corrcoef(y, yhat),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    for avg in ("macro", "micro", "weighted"):
        out[f"precision_{avg}"] = precision_score(y, yhat, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y, yhat, average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(y, yhat, average=avg, zero_division=0)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def bootstrap_ci(y, p, n_boot=N_BOOT, seed=0, alpha=0.05) -> dict:
    """Percentile bootstrap 95% CI for accuracy, macro-F1 and MCC (test set resampled with replacement)."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    yhat = (p >= THRESHOLD).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y)
    acc, f1m, mcc = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        acc.append((yhat[i] == y[i]).mean())
        f1m.append(f1_score(y[i], yhat[i], average="macro", zero_division=0))
        mcc.append(matthews_corrcoef(y[i], yhat[i]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out = {}
    for name, arr in [("accuracy", acc), ("f1_macro", f1m), ("mcc", mcc)]:
        lo, hi = np.percentile(arr, q)
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = float(lo), float(hi)
    return out


def mcnemar(y, p_base, p_other) -> dict:
    """Paired McNemar test on the same test examples.
    b = baseline correct & other wrong, c = baseline wrong & other correct.
    Exact binomial test if b + c < 25, otherwise chi-square with continuity correction."""
    y = np.asarray(y).astype(int)
    a_ok = (np.asarray(p_base) >= THRESHOLD).astype(int) == y
    b_ok = (np.asarray(p_other) >= THRESHOLD).astype(int) == y
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return {"b_base_only_correct": b, "c_other_only_correct": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if b + c < 25:
        pval = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        return {"b_base_only_correct": b, "c_other_only_correct": c, "statistic": float(min(b, c)),
                "p_value": float(pval), "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b_base_only_correct": b, "c_other_only_correct": c, "statistic": float(stat),
            "p_value": float(stats.chi2.sf(stat, 1)), "method": "chi2 (continuity corrected)"}


# --------------------------------------------------------------------------- slices (robustness)
NEG_RE = re.compile(r"\b(not|no|never|nothing|nobody|none|nor|neither|cannot|can't|won't|don't|doesn't|didn't|isn't|wasn't|aren't|weren't|shouldn't|wouldn't|couldn't)\b", re.I)
CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet|except|unfortunately|despite)\b", re.I)


def slice_masks(raw_texts: list[str], length_edges=(50, 120, 250)) -> dict:
    """Slices defined on the RAW review text (before preprocessing), identical for every member."""
    n_words = np.array([len(t.split()) for t in raw_texts])
    a, b, c = length_edges
    return {
        f"len<= {a} words": n_words <= a,
        f"len {a + 1}-{b}": (n_words > a) & (n_words <= b),
        f"len {b + 1}-{c}": (n_words > b) & (n_words <= c),
        f"len > {c}": n_words > c,
        "has negation": np.array([bool(NEG_RE.search(t)) for t in raw_texts]),
        "has contrast word": np.array([bool(CONTRAST_RE.search(t)) for t in raw_texts]),
        "has '!'": np.array(["!" in t for t in raw_texts]),
        "no negation & no contrast": np.array([not NEG_RE.search(t) and not CONTRAST_RE.search(t) for t in raw_texts]),
    }


def slice_metrics(y, p, masks: dict) -> list[dict]:
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= THRESHOLD).astype(int)
    rows = []
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        rows.append({"slice": name, "n": int(m.sum()),
                     "f1_macro": float(f1_score(y[m], yhat[m], average="macro", zero_division=0)),
                     "error_rate": float((yhat[m] != y[m]).mean())})
    return rows
