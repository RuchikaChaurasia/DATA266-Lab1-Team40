## Comparative analysis and observations (add to the end of task2_sentiment/ruchika_chaurasia/results.md)

All three models were trained on the same 200,000 cleaned training reviews (20,000 for validation) and tested on the full
official test set of **38,000 reviews**, on an NVIDIA RTX 5090 with an Intel Core Ultra 9 285K CPU. All word embeddings were
learned from scratch.

### Results

| | Baseline: EmbBag | Exp 1: TextCNN | Exp 2: BiLSTM + attention |
|---|---|---|---|
| Accuracy [95% CI] | 0.9289 [0.9264, 0.9312] | 0.9343 [0.9317, 0.9368] | **0.9425** [0.9402, 0.9449] |
| Macro-F1 [95% CI] | 0.9289 [0.9264, 0.9312] | 0.9343 [0.9317, 0.9368] | **0.9425** [0.9402, 0.9449] |
| MCC [95% CI] | 0.8577 [0.8527, 0.8624] | 0.8687 [0.8633, 0.8737] | **0.8850** [0.8804, 0.8899] |
| ROC-AUC / PR-AUC | 0.9780 / 0.9775 | 0.9833 / 0.9835 | **0.9865 / 0.9869** |
| Brier score (lower = better) | 0.0535 | 0.0480 | **0.0443** |
| ECE (lower = better) | 0.0108 | **0.0068** | 0.0213 |
| False positives / false negatives | 1,345 / 1,358 | 1,327 / 1,169 | **1,150 / 1,035** |
| McNemar p vs baseline | — | 4.5 × 10⁻⁶ | 5.3 × 10⁻²⁹ |
| Parameters | 3.95 M | 5.25 M | 5.38 M |
| Training time (4 epochs) | **4.5 s** | 13.9 s | 146.8 s |
| Training / inference examples per second | **177,661 / 1,350,547** | 57,465 / 267,338 | 5,448 / 50,158 |
| Peak GPU memory | **296 MB** | 664 MB | 742 MB |

(Precision, recall and F1 are the same for macro, micro and weighted averaging because the test set is perfectly balanced.)

### Comparing my models

* **Each experimental model is better than the baseline, and the differences are real.** Accuracy goes 92.9% → 93.4% →
  94.3%. The 95% bootstrap confidence intervals do not overlap, and the paired McNemar tests against the baseline are highly
  significant (TextCNN p = 4.5 × 10⁻⁶; BiLSTM p = 5.3 × 10⁻²⁹). On the same reviews, the BiLSTM fixed 1,329 of the baseline's
  mistakes and made only 811 new ones.
* **Word order matters.** The baseline averages word embeddings, so "not good" and "good, not bad" look similar to it.
  TextCNN looks at short phrases (3–5 words) and the BiLSTM reads the whole review in order, and each step adds accuracy.
* **The best model is not the best calibrated.** The BiLSTM has the best Brier score but the worst ECE (2.1%): its
  probabilities are a bit over-confident. TextCNN has the most trustworthy probabilities (ECE 0.7%). If the probabilities
  themselves were needed (for example to rank reviews), I would apply temperature scaling to the BiLSTM.
* **Accuracy costs time.** The BiLSTM is 1.4 points more accurate than the baseline but trains about **33× slower** and
  predicts about **27× slower**, because it reads the words one after another, while the other two models process all words
  in parallel. TextCNN is a good middle ground: +0.5 points for only 3× the training time.

### Robustness (per-slice error rates)

| Slice | n | Baseline | TextCNN | BiLSTM |
|---|---|---|---|---|
| has contrast word (but, however, although ...) | 22,846 | 7.97% | 7.30% | 6.41% |
| has negation (not, never, don't ...) | 28,198 | 7.67% | 6.71% | 5.94% |
| has "!" | 17,636 | 5.85% | 5.34% | 4.52% |
| no negation and no contrast | 6,841 | 5.09% | 5.66% | 4.78% |
| ≤ 50 words | 9,287 | 7.34% | 6.59% | 6.04% |
| > 250 words | 4,985 | 6.72% | 6.80% | 5.80% |

* **Contrast and negation are the hardest slices for all three models.** They have the highest error rates, and the 20-error
  review agrees: 12 of 20 errors were reviews whose opinion changes inside the text.
* The BiLSTM improves most on exactly these slices (negation 7.67% → 5.94%, contrast 7.97% → 6.41%), which fits the idea that
  reading in order helps with "not" and "but".
* Simple reviews without negation or contrast are the easiest for every model.
* TextCNN is slightly **worse** than the baseline on reviews longer than 250 words and on the "no negation / no contrast" slice.
  Its max-pooling keeps only the single strongest phrase per filter, which can be misleading in long or neutral reviews.

### Strengths, weaknesses and limitations

* **Strengths:** all three models reach about 93–94% accuracy without any pretrained embeddings or language models, and they
  train in seconds to a few minutes. The attention weights of the BiLSTM can be inspected to see which words drove a decision.
* **Weaknesses:** mixed and "changed over time" reviews (edited reviews, "it used to be good") are still the biggest problem,
  and the BiLSTM is over-confident.
* **Limitations:** I trained on 200K of the 560K training reviews and kept the 40K most frequent words, and long reviews are cut
  to 256 tokens (head + tail). Removing punctuation also deleted useful signals like ":(" and "!!!". Some test labels are noisy
  (the star rating does not match the text), which puts a ceiling on achievable accuracy. All results come from a single seed.

### Possible improvements / future work

* A sentence-level (hierarchical) attention model for reviews whose opinion changes (the testable fix in `failure_analysis.md`).
* Train on all 560K reviews, keep emoticons and "!" as tokens, and add hashed bigrams to the baseline.
* Temperature scaling to fix the BiLSTM's calibration.
* Repeat each model with several seeds to measure run-to-run variation.
* Compare with my teammate's three models (team comparison in the report).
