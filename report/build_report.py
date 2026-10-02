"""Build the team report from every member's committed results.

    python report/build_report.py

1. Edit report/report_info.yaml (team number, names, repo link, ownership statement).
2. Write the joint-analysis paragraphs in report/joint_analysis.md (your own words).
3. Run this script. It writes report/DATA266_Lab1_Report_Team_<N>.md and .html.
4. Open the .html in Chrome/Safari -> Print -> "Save as PDF" -> save as report/DATA266_Lab1_Report_Team_<N>.pdf
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
INFO = yaml.safe_load((REPO / "report" / "report_info.yaml").read_text())
MEMBERS = INFO["member_folders"]


def fmt(v, nd=4):
    if isinstance(v, float):
        if v != v:
            return "–"
        return f"{v:,.{nd}g}" if abs(v) < 1e5 else f"{v:,.0f}"
    return "–" if v is None else str(v)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results committed yet._"
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(v) for v in r.values) + " |")
    return "\n".join(out)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def owner(m):
    return INFO["owners"].get(m, m)


# --------------------------------------------------------------------------- Task 1
def task1():
    keep = ["val_cross_entropy", "perplexity", "bits_per_character", "generalization_gap", "top1_next_char_accuracy",
            "distinct_2", "repeated_4gram_rate", "loss_spikes", "nan_losses", "parameter_count",
            "train_tokens_per_sec", "generation_tokens_per_sec", "peak_memory_mb", "total_training_time_min",
            "checkpoint_id"]
    cols = {}
    for m in MEMBERS:
        p = REPO / "task1_llm" / m / "metrics_report.csv"
        if p.exists():
            d = pd.read_csv(p).set_index("metric")["value"]
            cols[f"{owner(m)} ({m})"] = [_num(d.get(k)) for k in keep]
    return md_table(pd.DataFrame(cols, index=keep).reset_index().rename(columns={"index": "metric"})) if cols else md_table(pd.DataFrame())


# --------------------------------------------------------------------------- Task 2
def task2():
    frames = []
    for m in MEMBERS:
        p = REPO / "task2_sentiment" / m / "metrics_report.csv"
        if p.exists():
            d = pd.read_csv(p)
            d.insert(0, "member", owner(m))
            frames.append(d)
    if not frames:
        return md_table(pd.DataFrame()), md_table(pd.DataFrame())
    d = pd.concat(frames, ignore_index=True)
    ci = lambda r, k: f"{r[k]:.4f} [{r[k + '_ci_low']:.4f}, {r[k + '_ci_high']:.4f}]"
    main = pd.DataFrame({
        "member": d.member, "model": d.model, "role": d.role,
        "accuracy [95% CI]": [ci(r, "accuracy") for _, r in d.iterrows()],
        "macro-F1 [95% CI]": [ci(r, "f1_macro") for _, r in d.iterrows()],
        "MCC [95% CI]": [ci(r, "mcc") for _, r in d.iterrows()],
        "ROC-AUC": d.roc_auc, "PR-AUC": d.pr_auc, "Brier": d.brier, "ECE": d.ece,
        "McNemar p vs baseline": d.get("mcnemar_p_vs_baseline"),
    })
    cost = d[["member", "model", "params", "train_time_s", "train_examples_per_s", "infer_examples_per_s",
              "peak_memory_mb", "hardware", "checkpoint_id"]]
    return md_table(main), md_table(cost)


# --------------------------------------------------------------------------- Task 3
def task3():
    audit = REPO / "task3_gan" / "audit" / "audit_results.csv"
    audit = pd.read_csv(audit).set_index("member") if audit.exists() else None
    rows = []
    for m in MEMBERS:
        p = REPO / "task3_gan" / m / "full_metrics_report.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        def g(k, dr):
            v = pd.to_numeric(d[(d.metric == k) & (d.direction == dr)]["value"], errors="coerce")
            return float(v.iloc[0]) if len(v) else float("nan")
        kg = yaml.safe_load((REPO / "task3_gan" / m / "config.yaml").read_text()).get("kaggle", {})
        row = {"member": owner(m),
               "official FID / MiFID (submission.csv)": f"{g('official_submission_fid', 'both'):.3f} / {g('official_submission_mifid', 'both'):.4f}",
               "FID photo→Monet (B2A)": g("fid", "B2A"), "FID Monet→photo (A2B)": g("fid", "A2B"),
               "KID B2A / A2B": f"{g('kid_mean', 'B2A'):.4f} / {g('kid_mean', 'A2B'):.4f}",
               "prec/rec B2A": f"{g('gen_precision', 'B2A'):.3f}/{g('gen_recall', 'B2A'):.3f}",
               "density/coverage B2A": f"{g('gen_density', 'B2A'):.3f}/{g('gen_coverage', 'B2A'):.3f}",
               "cycle L1 B2A / A2B": f"{g('cycle_l1', 'B2A'):.4f}/{g('cycle_l1', 'A2B'):.4f}",
               "LPIPS B2A": g("lpips_input_vs_translation", "B2A"),
               "content cos B2A": g("content_cosine", "B2A"),
               "params (total)": g("params_total", "both"), "train h": g("training_time_min", "both") / 60,
               "img/s": g("inference_images_per_sec", "both"),
               "Kaggle public": kg.get("public_score"), "Kaggle rank": kg.get("rank")}
        if audit is not None and m in audit.index:
            a = audit.loc[m]
            row["audit style/content/artifacts"] = f"{a.style_mean:.2f}/{a.content_mean:.2f}/{a.artifacts_mean:.2f}"
        rows.append(row)
    agree = REPO / "task3_gan" / "audit" / "audit_agreement.csv"
    return md_table(pd.DataFrame(rows)), (md_table(pd.read_csv(agree).round(3)) if agree.exists() else "_Audit not scored yet._")


def evidence():
    lines = []
    for sub in ("manifests", "raw_logs"):
        for p in sorted((REPO / "reproducibility" / sub).glob("*")):
            if "_smoke_" in p.name or p.name.startswith("."):
                continue
            lines.append(f"* `{p.relative_to(REPO)}`")
    return "\n".join(lines) or "_none yet_"


def read(p):
    p = REPO / p
    return p.read_text(encoding="utf-8") if p.exists() else f"_{p.relative_to(REPO)} not written yet._"


t2_main, t2_cost = task2()
t3_main, t3_agree = task3()
fail = []
for t in ("task1_llm", "task2_sentiment", "task3_gan"):
    for m in MEMBERS:
        if (REPO / t / m / "failure_analysis.md").exists():
            fail.append(f"* {t} · {owner(m)}: [`{t}/{m}/failure_analysis.md`]({INFO['repo_url']}/blob/main/{t}/{m}/failure_analysis.md)")

report = f"""# DATA 266 — Lab 1 Report · Team {INFO['team_number']}

**Members:** {', '.join(INFO['member_names'])}  
**Repository:** {INFO['repo_url']}  
**Date:** {INFO.get('date', '')}

## Ownership statement
{INFO['ownership_statement'].strip()}

## Task 1 — GPT-style character-level LLM (TinyStories)
Hand-written Transformer (no prebuilt attention modules), 100K/10K split per member, ≥ 10 epochs, warm-up + LR schedule.
All metrics computed with the shared `task1_llm/common/eval_task1.py`.

{task1()}

## Task 2 — Yelp Polarity sentiment (embeddings learned from scratch)
Official 38K test split; shared `task2_sentiment/common/eval_task2.py`; 95% bootstrap CIs (1,000 resamples);
McNemar is paired, within each member (baseline vs experimental).

{t2_main}

**Cost and hardware**

{t2_cost}

## Task 3 — CycleGAN photo ↔ Monet
Course convention: A = Monet, B = photo (B2A = photo→Monet). "Official" FID/MiFID come from the ISA's unchanged
`Part3_Evaluation_Script.ipynb` (the Kaggle `submission.csv`); the other metrics use `task3_gan/common/metrics_task3.py`
on 500 team-fixed held-out photos and all 300 Monets.

{t3_main}

**Human audit — inter-rater agreement (30 fixed samples per model, 2 blinded raters)**

{t3_agree}

{read('report/joint_analysis.md')}

## Failure / error analyses
{chr(10).join(fail) or '_not written yet_'}

## Evidence (manifests and raw logs)
{evidence()}

## References
{chr(10).join('* ' + c for c in INFO['citations'])}
"""
n = INFO["team_number"]
md_path = REPO / "report" / f"DATA266_Lab1_Report_Team_{n}.md"
md_path.write_text(report, encoding="utf-8")
if importlib.util.find_spec("markdown") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "markdown"])
import markdown  # noqa: E402

css = ("body{font-family:Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;font-size:12px;line-height:1.45}"
       "table{border-collapse:collapse;margin:8px 0;font-size:10.5px}td,th{border:1px solid #bbb;padding:3px 6px}"
       "th{background:#eee}h1{font-size:22px}h2{border-bottom:1px solid #ccc;margin-top:26px}code{font-size:10.5px}"
       "@media print{h2{page-break-after:avoid}table{page-break-inside:auto}}")
html = markdown.markdown(report, extensions=["tables"])
(REPO / "report" / f"DATA266_Lab1_Report_Team_{n}.html").write_text(
    f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>", encoding="utf-8")
print(f"wrote {md_path.relative_to(REPO)} and .html — open the .html and Print -> Save as PDF "
      f"as report/DATA266_Lab1_Report_Team_{n}.pdf")
