# DATA 266 · Lab 1 — Team 40

**Members:** Ruchika Chaurasia (`ruchika_chaurasia/`) · Akhilesh Tandur (`akhilesh_tandur/`)
**Kaggle team:** PairProgramming_Team_40

Three generative-AI tasks. Each member designs, trains and evaluates **their own models** for every task in their own
member folder. Both members use one shared, agreed evaluation script per task, so the results are directly comparable.

| Task | What | Ruchika Chaurasia (`ruchika_chaurasia/`) | Akhilesh Tandur (`akhilesh_tandur/`) |
|---|---|---|---|
| 1 | Character-level GPT from scratch on TinyStories | 4 layers × 256d, 4 heads, context 256, 3.2M params · val loss 0.615 | 6 layers × 384d, 6 heads, context 256, 10.8M params · val loss 0.557 |
| 2 | Yelp Polarity sentiment, embeddings learned from scratch | lemmatised text · EmbBag → TextCNN → BiLSTM + attention · best accuracy 94.25% | stemmed text · bag of words + MLP → CNN → BiLSTM + max pooling · best accuracy 93.09% |
| 3 | CycleGAN Monet ↔ photo + Kaggle | ResNet-9 + transposed-conv upsampling, LSGAN; 8 runs, final `runs/final200_noid` (200 epochs, no identity loss) · Kaggle −50.7560 | ResNet-9 + resize-conv upsampling, LSGAN, λ_id 0.3, 30 epochs · Kaggle −51.9920 |

**Team report:** `report/DATA266_Lab1_Report_Team_40.pdf` (ownership statement, comparison tables for all tasks, joint analysis,
human audit, references). AI assistance is disclosed in `AI_USE.md`.

---

## Repository layout

```
common/runtime.py                 seeds, raw log, hardware info, manifest (used by every notebook)
task1_llm/
  data/                           TinyStories (downloaded automatically; see data/README.md)
  common/eval_task1.py            shared Task 1 metrics
  <member>/config.yaml            architecture + hyper-parameters + design rationale
  <member>/src/task1_llm.ipynb    executed notebook
  <member>/data_processed/        split indices, vocabulary (char_to_idx / idx_to_char)
  <member>/checkpoints/best.pt    trained weights
  <member>/outputs/               loss curves, LR schedule, generated samples, logs
  <member>/metrics_report.csv · failure_analysis.md · results.md
task2_sentiment/                  same pattern; common/eval_task2.py; 3 checkpoints per member
task3_gan/
  data/monet_jpg, data/photo_jpg  competition images (not in git; see data/README.md)
  common/Part3_Evaluation_Script.ipynb  the ISA's official FID/MiFID script, unchanged → Kaggle submission.csv
  common/metrics_task3.py         extra metrics: FID, KID, precision/recall/density/coverage, cycle L1, LPIPS, content cosine
  common/data_task3.py            data loading + team-fixed held-out/audit photo list
  common/team_eval_split.json     500 held-out photos + 30 audit photos (same for both members)
  <member>/src/                   training notebook(s) and model code
  <member>/evaluate_local.py      reproduces full_metrics_report.csv, prediction folders and submission.csv
  <member>/full_metrics_report.csv · submission.csv · outputs/pred_A2B (Monet→photo) · outputs/pred_B2A (photo→Monet)
  ruchika_chaurasia/runs/<run>/   follow-up experiments, each with its own config_<run>.yaml, notebook, metrics, submission.csv
  audit/                          blinded 2-rater human audit tooling and results
reproducibility/
  manifests/                      per-run JSON (hardware, packages, config, checkpoint IDs) + pip freeze
  raw_logs/                       unedited, timestamped logs of every run
  smoke_test.sh                   ONE-COMMAND smoke test
report/                           report builder and team report (PDF)
scripts/                          notebook sources (scripts/nb_src), notebook builder, extra-run launcher, helpers
```

All log timestamps are **UTC**. No personal paths are hard-coded: every notebook finds the repository root by walking up from
the folder it runs in (or from `$REPO_ROOT`).

---

## Smoke test (one command)

```bash
bash reproducibility/smoke_test.sh
```

This installs `requirements.txt` and runs Ruchika's Task 1 notebook end to end on a tiny setting (1,500 stories, 2-layer
model, 2 epochs) in a few minutes on CPU or GPU. It needs internet access for the TinyStories validation split. Outputs go to
`_smoke/`, so committed results are never touched. Any notebook can be smoke-tested the same way:

```bash
SMOKE=1 jupyter nbconvert --to notebook --execute <task>/<member>/src/<notebook>.ipynb --output-dir _smoke
```

---

## How Ruchika's results were produced

* **Hardware:** SJSU GPU lab, NVIDIA GeForce RTX 5090 (32 GB), Intel Core Ultra 9 285K, PyTorch 2.11.0+cu128, inside Docker.
* **Setup:** `pip install -r requirements.txt` (the lab image already provided a CUDA 12.8 PyTorch build, which was kept).
* **Runs:** each notebook was executed from a terminal with
  `jupyter nbconvert --to notebook --execute --inplace <notebook> --ExecutePreprocessor.timeout=-1`,
  so the committed notebooks contain all outputs.
* **Measured times on the RTX 5090:** Task 1 11 min (10 epochs) · Task 2 about 4 min (3 models) · Task 3 main run 37 min (30 epochs).
* **Extra Task 3 runs:** `bash scripts/run_task3_extra.sh <run>` trains a run defined by `task3_gan/ruchika_chaurasia/config_<run>.yaml`
  and writes everything to `runs/<run>/`. Re-running the same command after an interruption resumes from the last saved epoch.
  Several runs shared the GPU, so their wall-clock times are not comparable with the main run.
* **Final Task 3 model:** `runs/final200_noid`, chosen by a rule fixed in advance (best held-out photo→Monet FID and precision).

### Kaggle (Task 3)
Every `submission.csv` is produced by running the ISA's evaluation script **unchanged** on the member's own `pred_A2B/` and
`pred_B2A/` folders (`evaluate_local.py` does this automatically). Each member submits from their own Kaggle account into the
team `PairProgramming_Team_40`. Scores and ranks are recorded in each member's config (`kaggle:` section) and in the report.

### Human audit (Task 3)
Both members' final models were rated on 60 blinded items (30 per model) by two raters for style, content and artifacts (1–5).
Rating sheets, sealed key, results and agreement (Cohen's κ) are in `task3_gan/audit/`; the analysis is in the team report.
The tooling (`make_audit_pack.py`, `score_audit.py`) is described in `task3_gan/audit/README.md`.

### Kaggle selections
Final selected submissions: −50.7560 (Ruchika, final200_noid), −51.9920 (Akhilesh), −52.5597 (Ruchika, long100).

---

## Where each requirement is met (paths relative to each member folder)

| Requirement | Location |
|---|---|
| Hand-written attention, LayerNorm, FFN, residuals, embeddings, LM head (no prebuilt attention) | Task 1 notebook §1.2 (+ causal-mask unit test) |
| Character tokenisation, `char_to_idx` / `idx_to_char`, 100K/10K split, fixed-length input → target | Task 1 §1.1, `data_processed/` |
| ≥ 10 epochs, warm-up + schedule, cross-entropy | Task 1 §1.3, `outputs/lr_schedule.png` |
| All Task 1 metrics | `task1_llm/<member>/metrics_report.csv`, `results.md` |
| Loss curves, samples, 3 failure cases | `outputs/loss_curves.png`, `outputs/samples.txt`, `failure_analysis.md` |
| EDA, missing/malformed handling, cleaning, stopwords, lemmatise/stem, tokenise, embeddings from scratch | Task 2 §2.1 |
| 3 models, justification, hardware recorded per model | Task 2 §2.2, `config.yaml`, `results.md` |
| All Task 2 metrics incl. bootstrap CIs, McNemar, slices | `task2_sentiment/<member>/metrics_report.csv`, `outputs/` |
| 20-error review + testable fix | `failure_analysis.md`, `outputs/error_review_20.csv` |
| CycleGAN (2 generators, 2 discriminators), both directions, cycle consistency verified | Task 3 notebook, `outputs/figures/cycle_*.png`, `results.md` |
| All Task 3 metrics, both directions | `task3_gan/<member>/full_metrics_report.csv` (and `runs/<run>/` for each experiment) |
| Kaggle submission (official script) + score/rank | `submission.csv`, `config*.yaml → kaggle`, `results.md` |
| Blinded 30-sample audit, 2 raters, Cohen's κ | `task3_gan/audit/` |
| Manifests, raw logs, checkpoint IDs | `reproducibility/` (checkpoint ID = first 12 hex characters of the file's SHA-256) |

---

## Datasets and large files (Google Drive, read access)

Datasets are not pushed to GitHub.
* **Task 1 TinyStories / Task 2 Yelp Polarity:** downloaded automatically from Hugging Face (see `task*/data/README.md`).
* **Task 3 Monet + photo images (`dataset.zip`):** https://drive.google.com/file/d/17TEwtkjtcYiiqnPxmNOHHeoixGIOFtiL/view?usp=sharing
* **Checkpoints of Ruchika's non-submitted Task 3 runs** (final200, long100_diffaug, lc5_100, lc5_d4_100, lc5_ema_100):
  https://drive.google.com/file/d/1tMw3kpL9CQrydxSG9E8pvb1J56ROI3j0/view?usp=sharing. The three submitted models (main run,
  `runs/long100`, `runs/final200_noid`) are in the repository.

## Data and licences
TinyStories (Eldan & Li, 2023; Hugging Face) · Yelp Polarity (Zhang et al., 2015) · Monet and photo images from the class
Kaggle competition "DATA 266 Fall 2026 – GAN Image Style Transfer" (based on "I'm Something of a Painter Myself").
