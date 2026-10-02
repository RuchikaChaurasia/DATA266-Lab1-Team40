# DATA 266 · Lab 1 — Team repository

Three generative-AI tasks, each done as **two independent model lineups** (A and B) that share one agreed
evaluation script per task:

| Task | What | Lineup A (`ruchika_chaurasia/`) | Lineup B (`member_b/`) |
|---|---|---|---|
| 1 | Char-level GPT from scratch on TinyStories | 4 layers × 256d, ctx 256, tied head, cosine LR | 6 layers × 288d, ctx 128, untied head, linear LR |
| 2 | Yelp Polarity sentiment, embeddings from scratch | lemmatised · EmbBag → TextCNN → BiLSTM + attention | stemmed · EmbBag + bigrams → BiGRU-max → Transformer |
| 3 | CycleGAN Monet ↔ photo + Kaggle | ResNet-9 G, 70×70 PatchGAN, LSGAN, λ_id 0.5 | U-Net G, 4-layer PatchGAN, BCE GAN, λ_id 0.1 |

Who built what is stated in the report's **ownership statement** (`report/report_info.yaml`), and AI
assistance is disclosed in `AI_USE.md`.

---

## Repository layout

```
common/runtime.py                 seeds, raw log, hardware, manifest (used by every notebook)
task1_llm/
  data/                           TinyStories (auto-downloaded; see data/README.md)
  common/eval_task1.py            shared Task 1 metrics
  <member>/config.yaml            architecture + hyper-parameters + design rationale
  <member>/src/task1_llm.ipynb    executed notebook
  <member>/data_processed/        split indices, vocab (char_to_idx / idx_to_char)
  <member>/checkpoints/best.pt    trained weights
  <member>/outputs/               loss curves, LR schedule, samples, logs
  <member>/metrics_report.csv · failure_analysis.md · results.md
task2_sentiment/  (same pattern; common/eval_task2.py; 3 checkpoints per member)
task3_gan/
  data/monet_jpg, data/photo_jpg  Kaggle data (download once; see data/README.md)
  common/Part3_Evaluation_Script.ipynb  ISA's official FID/MiFID script (unchanged) → Kaggle submission.csv
  common/metrics_task3.py         extra metrics: FID, KID, precision/recall/density/coverage, cycle L1, LPIPS, content cosine
  common/data_task3.py            loading + team-fixed held-out/audit photo list
  common/team_eval_split.json     500 held-out photos + 30 audit photos (created on first run, then committed)
  <member>/src/task3_gan.ipynb    training notebook;  <member>/src/cyclegan_models.py
  <member>/evaluate_local.py      reproduces full_metrics_report.csv, pred folders and submission.csv
  <member>/full_metrics_report.csv · submission.csv · outputs/pred_A2B (Monet→photo) · outputs/pred_B2A (photo→Monet)
  audit/                          blinded 2-rater audit tooling (see audit/README.md)
reproducibility/
  manifests/                      per-run JSON (hardware, packages, config, checkpoint IDs) + pip freeze
  raw_logs/                       unedited timestamped logs of every run
  smoke_test.sh                   ONE-COMMAND smoke test
report/                           build_report.py → DATA266_Lab1_Report_Team_<N>.pdf
scripts/                          notebook sources (scripts/nb_src) + build_notebooks.sh
```

All log timestamps are UTC. No personal paths are hard-coded. Every notebook finds the repo root by walking up from where it is
run (or from `$REPO_ROOT`).

---

## Smoke test (one command)

```bash
bash reproducibility/smoke_test.sh
```

This installs `requirements.txt` and executes the Task 1 notebook of lineup A on a tiny setting (1,500 stories,
2-layer model, 2 epochs). It takes a few minutes on CPU and needs internet for the TinyStories validation split.
Outputs go to `_smoke/`, so committed results are never touched. Any notebook can be smoke-run the same way:

```bash
SMOKE=1 jupyter nbconvert --to notebook --execute <task>/<member>/src/<notebook>.ipynb --output-dir _smoke
```

---

## Full runs

### Setup
* **Google Colab (T4):** put the repo in Google Drive, open a notebook, choose *Runtime → Change runtime type → T4 GPU*,
  then run this first cell:
  ```python
  from google.colab import drive; drive.mount('/content/drive')
  %cd /content/drive/MyDrive/<repo-folder>
  ```
  Checkpoints and outputs are then written straight to Drive. Task 3 copies the images to local disk automatically.
* **Local / SJSU lab GPU:** `pip install -r requirements.txt`, then `jupyter lab` from the repo root.

### Run order and rough T4 times (with AMP)
| Notebook | Lineup A | Lineup B |
|---|---|---|
| `task3_gan/<member>/src/task3_gan.ipynb` | ~3–5 h (30 × 1,000 steps) | ~3–5 h |
| `task1_llm/<member>/src/task1_llm.ipynb` | ~1–1.5 h (10 epochs) | ~2–3 h |
| `task2_sentiment/<member>/src/task2_sentiment.ipynb` | ~30–60 min (3 models) | ~30–60 min |

These times are estimates; the real ones are recorded in each run's metrics. Task 1 and Task 3 save `last.pt`
every epoch. If Colab disconnects, re-run the notebook from the top and it **resumes** from the last finished epoch.

### Extra Task 3 runs (training-length / DiffAugment experiments)
`bash scripts/run_task3_extra.sh long100 long100_diffaug` trains the same member's model with
`config_long100.yaml` / `config_long100_diffaug.yaml` in the background and writes everything (executed notebook,
checkpoints, metrics, `submission.csv`) to `task3_gan/<member>/runs/<run>/`. The main run is never modified.
Runs trained before the gradient-norm fix can be corrected with `python scripts/fix_gradnorm_report.py task3_gan/<member>`.

### After Task 3
1. The notebook (or `python task3_gan/<member>/evaluate_local.py`) runs the ISA's official evaluation script unchanged
   and writes `task3_gan/<member>/submission.csv` (ID, FID, MiFID). **Each member uploads their own** file to Kaggle
   (team `PairProgramming_Team_##`) and records the score/rank in `task3_gan/<member>/config.yaml → kaggle`.
2. Human audit: follow `task3_gan/audit/README.md` (build pack → 2 blinded raters → `score_audit.py`).
3. Re-run the last cell of each Task 3 notebook to refresh `results.md`.

### Report
Edit `report/report_info.yaml` and `report/joint_analysis.md`, then run `python report/build_report.py`.
Open the generated `.html` → Print → Save as PDF → `report/DATA266_Lab1_Report_Team_<N>.pdf`.

---

## Where each requirement is met

| Requirement | Location |
|---|---|
| Hand-written attention, LayerNorm, FFN, residuals, embeddings, LM head (no prebuilt attention) | Task 1 notebook §1.2 (+ causal-mask unit test) |
| Char tokenisation, `char_to_idx`/`idx_to_char`, 100K/10K split, fixed-length input→target | Task 1 §1.1, `data_processed/` |
| ≥10 epochs, warm-up + schedule, cross-entropy | Task 1 §1.3, `outputs/lr_schedule.png` |
| Train/val CE, perplexity, BPC, gap, top-1 acc, distinct-n, repeated 4-gram, grad norms/spikes/NaNs, params, tokens/s (train + gen), peak memory, time | `task1_llm/<member>/metrics_report.csv` |
| Loss curves, samples, 3 failures | `outputs/loss_curves.png`, `outputs/samples.txt`, `failure_analysis.md` |
| EDA (length, class distribution, balance), missing/malformed handling, cleaning, stopwords, lemmatise/stem, tokenise | Task 2 §2.1 |
| 3 models, embeddings from scratch, justification, hardware recorded | Task 2 §2.2, `config.yaml → why`, `results.md` |
| Accuracy, P/R/F1 (macro/micro/weighted), CM, ROC-AUC, PR-AUC, MCC, Brier, ECE, bootstrap CIs, McNemar, slices, params, time, ex/s, peak memory | `task2_sentiment/<member>/metrics_report.csv`, `outputs/slice_metrics.csv`, `outputs/mcnemar.csv` |
| 20-error review (5 FP, 5 FN, 5 near-threshold, 5 slice) + testable fix | `failure_analysis.md`, `outputs/error_review_20.csv` |
| CycleGAN both directions (A = Monet, B = photo, as in the course script), cycle consistency verified | Task 3 notebook, `outputs/figures/cycle_*.png` |
| FID, KID, precision/recall/density/coverage, cycle L1, LPIPS, content cosine, losses, grad norms, NaNs, params, time, img/s, peak memory | `task3_gan/<member>/full_metrics_report.csv` |
| Kaggle `submission.csv` from the official script, score + rank | `task3_gan/<member>/submission.csv`, `config.yaml → kaggle`, `results.md` |
| Blinded 30-sample audit, 2 raters, Cohen's κ | `task3_gan/audit/` |
| Manifests, raw logs, checkpoint IDs | `reproducibility/` (checkpoint ID = first 12 hex chars of the file's SHA-256) |

## Datasets (Google Drive, read access)
Datasets are not pushed to GitHub. Zipped copies with read access enabled:
* Task 1 TinyStories subset / Task 2 Yelp: downloaded automatically from Hugging Face (see `task*/data/README.md`)
* Task 3 Monet + photos: TODO — paste the Google Drive link to `dataset.zip` here (Share → "Anyone with the link" → Viewer)

## Data and licences
TinyStories (Eldan & Li, 2023; Hugging Face) · Yelp Polarity (Zhang et al., 2015) · Monet/photo images from the
Kaggle "I'm Something of a Painter Myself" competition. Raw data is not committed (see `.gitignore`).
