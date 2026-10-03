# Task 2 — Yelp Polarity sentiment classification · ruchika_chaurasia (lineup A)

## Data and preprocessing
* **Dataset:** Yelp Polarity (Hugging Face `fancyzhx/yelp_polarity`): 560,000 training and 38,000 test reviews;
  label 0 = negative (1–2 stars), 1 = positive (4–5 stars).
* **Cleaning:** replaced escaped line breaks (`\n` stored as text) with spaces, removed **49 duplicate** training reviews and
  **10 training reviews that also appear in the test set** (to avoid leakage), leaving 559,941 training reviews. No missing
  labels or empty reviews remained.
* **EDA:** classes are perfectly balanced (balance ratio 1.000), so accuracy is meaningful; macro-F1 and MCC are reported
  as well. Review length (words) percentiles 50 / 90 / 95 / 99: 98 / 284 / 375 / 613 (`outputs/eda_length_class.png`).
* **Split:** stratified random subset of **200,000 training** and **20,000 validation** reviews (seed 8283). The **official
  38,000-review test set** is used only for the final evaluation.
* **Text preprocessing:** lowercasing → expanding contractions (didn't → did not) **before** removing punctuation, so negations
  survive → removing URLs, punctuation, digits and special characters → removing English stopwords **except** negation and
  contrast words (not, no, never, but, very, too, only …) → WordNet lemmatisation (verb, then noun) → whitespace tokenisation.
* **Vocabulary and encoding:** the 40,000 most frequent tokens with frequency ≥ 3, built from the training subset only
  (final size 39,449 incl. `<pad>`/`<unk>`). Reviews are cut or padded to 256 tokens, keeping the **first and last 128
  tokens** of long reviews, because Yelp reviewers often give their verdict at the end. After preprocessing the median review
  has 52 tokens (95th percentile 196); only 2.2% of reviews are truncated; the test-set out-of-vocabulary rate is 0.65%.
* **Embeddings:** every model starts with an `nn.Embedding` layer initialised randomly and trained together with the
  classifier, so all embeddings are **learned from scratch** (no pretrained vectors or language models).

## My three models

| Model | Role | Architecture | Main hyper-parameters |
|---|---|---|---|
| A_baseline_embbag | baseline | average of word embeddings → linear layer | emb 100, dropout 0.2, lr 5e-3, batch 512 |
| A_exp1_textcnn | experimental | 1-D convolutions (widths 3, 4, 5; 128 filters each) + max-over-time pooling | emb 128, dropout 0.5, lr 1e-3, wd 1e-4, batch 256 |
| A_exp2_bilstm_attn | experimental | bidirectional LSTM (hidden 128) + additive attention pooling | emb 128, dropout 0.3, lr 1e-3, wd 1e-4, batch 256 |

All three use AdamW with 5% linear warm-up and cosine decay, gradient clipping 1.0, fp16 mixed precision and binary
cross-entropy, and the epoch with the best validation macro-F1 is kept (max 4–5 epochs, early-stopping patience 2).

## Why these models
* **Baseline, bag of learned embeddings (fastText style):** the simplest model that still learns word embeddings. It ignores
  word order, trains in seconds and gives a lower bound that the other models must beat.
* **Experimental 1, TextCNN (changes the architecture):** filters of width 3–5 act as phrase detectors ("not worth it",
  "will be back"), and max-pooling lets a phrase count wherever it appears in the review. Dropout 0.5 follows Kim (2014).
* **Experimental 2, BiLSTM + attention (changes the architecture again):** reads the whole review in order in both
  directions, so it can carry negation and contrast ("good food but awful service") across long spans. Attention learns which
  words matter most, and its weights can be inspected in the error review.
* **Embedding sizes** of 100–128 are enough for a 40K-word vocabulary and keep each model under 5.5M parameters.

## Results on the official test set (38,000 reviews)

| | Baseline: EmbBag | Exp 1: TextCNN | Exp 2: BiLSTM + attention |
|---|---|---|---|
| Accuracy [95% CI] | 0.9289 [0.9264, 0.9312] | 0.9343 [0.9317, 0.9368] | **0.9425** [0.9402, 0.9449] |
| Macro-F1 [95% CI] | 0.9289 [0.9264, 0.9312] | 0.9343 [0.9317, 0.9368] | **0.9425** [0.9402, 0.9449] |
| Micro-F1 / weighted-F1 | 0.9289 / 0.9289 | 0.9343 / 0.9343 | **0.9425 / 0.9425** |
| Precision / recall (macro) | 0.9289 / 0.9289 | 0.9343 / 0.9343 | **0.9425 / 0.9425** |
| MCC [95% CI] | 0.8577 [0.8527, 0.8624] | 0.8687 [0.8633, 0.8737] | **0.8850** [0.8804, 0.8899] |
| ROC-AUC / PR-AUC | 0.9780 / 0.9775 | 0.9833 / 0.9835 | **0.9865 / 0.9869** |
| Brier score (lower = better) | 0.0535 | 0.0480 | **0.0443** |
| ECE (lower = better) | 0.0108 | **0.0068** | 0.0213 |
| Confusion matrix (TN / FP / FN / TP) | 17,655 / 1,345 / 1,358 / 17,642 | 17,673 / 1,327 / 1,169 / 17,831 | 17,850 / 1,150 / 1,035 / 17,965 |
| McNemar p vs baseline | — | 4.5 × 10⁻⁶ | 5.3 × 10⁻²⁹ |
| Best validation macro-F1 | 0.9254 | 0.9339 | 0.9381 |
| Parameters | 3.95 M | 5.25 M | 5.38 M |
| Training time (4 epochs) | **4.5 s** | 13.9 s | 146.8 s |
| Training / inference examples per second | **177,661 / 1,350,547** | 57,465 / 267,338 | 5,448 / 50,158 |
| Peak GPU memory | **296 MB** | 664 MB | 742 MB |
| Checkpoint ID | `4cf09316ac72` | `977844e053f5` | `ff1c3543fe50` |

Precision, recall and F1 are the same for macro, micro and weighted averaging because the test set is perfectly balanced.
Confidence intervals: 1,000 bootstrap resamples of the test set. Hardware for every model: NVIDIA GeForce RTX 5090 ·
Intel Core Ultra 9 285K (24 cores).

**Paired McNemar test** (same 38,000 test reviews): TextCNN fixed 1,113 of the baseline's errors and made 906 new ones
(χ² = 21.0, p = 4.5 × 10⁻⁶); the BiLSTM fixed 1,329 and made 811 new ones (χ² = 124.9, p = 5.3 × 10⁻²⁹).

### Robustness: error rate per data slice

| Slice | n | Baseline | TextCNN | BiLSTM |
|---|---|---|---|---|
| has contrast word (but, however, although …) | 22,846 | 7.97% | 7.30% | 6.41% |
| has negation (not, never, don't …) | 28,198 | 7.67% | 6.71% | 5.94% |
| ≤ 50 words | 9,287 | 7.34% | 6.59% | 6.04% |
| 51–120 words | 13,284 | 7.05% | 6.26% | 5.35% |
| 121–250 words | 10,444 | 7.17% | 6.84% | 5.97% |
| > 250 words | 4,985 | 6.72% | 6.80% | 5.80% |
| has "!" | 17,636 | 5.85% | 5.34% | 4.52% |
| no negation and no contrast | 6,841 | 5.09% | 5.66% | 4.78% |

Macro-F1 per slice is in `outputs/slice_metrics.csv`.

## Comparative analysis and observations

* **Each experimental model beats the baseline, and the differences are real.** Accuracy goes 92.9% → 93.4% → 94.3%. The
  95% bootstrap confidence intervals do not overlap and both McNemar tests are highly significant.
* **Word order matters.** The baseline averages word embeddings, so "not good" and "good, not bad" look similar to it.
  TextCNN looks at short phrases and the BiLSTM reads the whole review in order, and each step adds accuracy.
* **The best model is not the best calibrated.** The BiLSTM has the best Brier score but the worst ECE (2.1%): its
  probabilities are a bit over-confident. TextCNN gives the most trustworthy probabilities (ECE 0.7%). If calibrated
  probabilities were needed, I would apply temperature scaling to the BiLSTM.
* **Accuracy costs time.** The BiLSTM is 1.4 points more accurate than the baseline but trains about **33× slower** and
  predicts about **27× slower**, because it reads words one after another, while the other two models process all words in
  parallel. TextCNN is a good middle ground: +0.5 points for about 3× the training time.
* **Contrast and negation are the hardest slices for all three models,** and the 20-error review agrees: 12 of 20 errors are
  reviews whose opinion changes inside the text. The BiLSTM improves most on exactly these slices (negation 7.67% → 5.94%,
  contrast 7.97% → 6.41%), which fits the idea that reading in order helps with "not" and "but".
* **TextCNN is slightly worse than the baseline** on reviews longer than 250 words and on the "no negation / no contrast"
  slice. Its max-pooling keeps only the single strongest phrase per filter, which can be misleading in long or neutral reviews.

**Strengths:** about 93–94% accuracy without any pretrained embeddings or language models; training takes seconds to a few
minutes; the BiLSTM's attention weights show which words drove a decision.
**Weaknesses:** mixed and "changed over time" reviews (edited reviews, "it used to be good") remain the main problem, and the
BiLSTM is over-confident.
**Limitations:** I trained on 200K of the 560K training reviews with a 40K-word vocabulary; long reviews are cut to 256 tokens;
removing punctuation deleted useful signals such as ":(" and "!!!"; some labels are noisy (the star rating contradicts the
text), which limits the best possible accuracy; all results come from a single seed.

**Possible improvements / future work:** a sentence-level (hierarchical) attention model for reviews whose opinion changes
(the testable fix in `failure_analysis.md`); training on all 560K reviews; keeping emoticons and "!" as tokens; temperature
scaling for calibration; several seeds per model; and comparison with my teammate's models in the team report.

## Evidence
* Training curves: `outputs/training_curves.png` · confusion matrices and reliability diagrams: `outputs/confusion_and_calibration.png`
* ROC and PR curves: `outputs/roc_pr_curves.png` · model comparison: `outputs/model_comparison.png` · slices: `outputs/slice_error_rates.png`, `outputs/slice_metrics.csv`
* McNemar: `outputs/mcnemar.csv` · test-set probabilities: `outputs/test_probs_*.npy` · EDA: `outputs/eda_length_class.png`
* Error review: `failure_analysis.md`, `outputs/error_review_20.csv`
* Checkpoints: `checkpoints/A_baseline_embbag.pt`, `A_exp1_textcnn.pt`, `A_exp2_bilstm_attn.pt` · vocabulary: `data_processed/vocab.json`
* Raw log: `reproducibility/raw_logs/task2_sentiment_ruchika_chaurasia_20261002_000303.log`
* Manifest: `reproducibility/manifests/task2_sentiment_ruchika_chaurasia_20261002_000655.json`
