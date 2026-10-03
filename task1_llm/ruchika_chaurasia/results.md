# Task 1 — GPT-style character-level LLM from scratch · ruchika_chaurasia (lineup A)

## What I built
A GPT-style language model written from scratch in PyTorch: multi-head causal self-attention, layer normalisation,
feed-forward network, residual connections, token + positional embeddings and a language-modelling head are all my own
code (no `nn.MultiheadAttention`, `nn.Transformer*` or `scaled_dot_product_attention`). A unit test in the notebook checks
that the causal mask works: changing the last character does not change the model's output at any earlier position.

| Setting | Value |
|---|---|
| Architecture | 4 pre-norm Transformer blocks, 4 attention heads (head size 64), d_model 256, FFN 4× with GELU |
| Context length | 256 characters |
| Vocabulary | 94 characters (incl. `<unk>` and end-of-story) |
| Embeddings | learned token + positional embeddings; output head tied to the token embedding |
| Dropout | 0.1 |
| Parameters | 3,249,152 |
| Data | TinyStories, my own random split of 100,000 training / 10,000 validation stories (seed 8283) |
| Tokenisation | character level, own `char_to_idx` / `idx_to_char`; characters seen < 5 times → `<unk>`; stories joined with an end-of-story token |
| Optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 (weight matrices only), gradient clipping 1.0 |
| Learning rate | linear warm-up for 500 steps to 1e-3, then cosine decay to 1e-4 |
| Training | 10 epochs × 2,739 steps (27,390 steps), batch 128 × 256 = 32,768 characters per step, fp16 mixed precision |
| Hardware | NVIDIA GeForce RTX 5090 (32 GB) · Intel Core Ultra 9 285K (24 cores) · PyTorch 2.11.0+cu128 |

## Why I made these choices
* **Small but wide model:** TinyStories uses a very small vocabulary, and the TinyStories paper shows that small models
  can already write fluent stories. 4 layers × 256 dims (about 3.2M parameters) is enough capacity for that and lets me train
  10 full epochs in about 11 minutes.
* **Context of 256 characters** (about two sentences) so the model can keep names and subjects consistent within a scene.
* **Weight tying** shares the input and output character representations and saves parameters, which usually helps small
  models with small vocabularies.
* **Pre-norm blocks** (LayerNorm before each sub-layer, as in GPT-2) train more stably than the original post-norm design.
* **High peak learning rate (1e-3) with warm-up and cosine decay:** small models tolerate a high learning rate. Warm-up
  avoids unstable early Adam steps, and cosine decay ends with a small learning rate for a good final checkpoint.
* **Dropout 0.1** only, because 100K stories × 10 epochs is a lot of data for a 3M-parameter model, so heavy regularisation
  is not needed.

## Metrics (best checkpoint `bceaa3cfc2f2`, epoch 10)

| Metric | Value | Notes |
|---|---|---|
| Training cross-entropy | 0.6077 | nats/char, eval mode |
| Validation cross-entropy | 0.6150 | nats/char, full validation set |
| Perplexity | 1.850 | exp(validation CE), character level |
| Bits per character | 0.887 | validation CE / ln 2 |
| Generalisation gap | 0.0073 | validation CE − training CE |
| Top-1 next-character accuracy | 0.8042 | validation |
| Distinct-1 / 2 / 3 | 0.298 / 0.712 / 0.877 | word level, temperature 0.7 |
| Repeated 4-gram rate | 0.0185 | word level, temperature 0.7 |
| Repeated 4-gram rate (greedy) | 0.3429 | greedy decoding |
| Gradient norm mean / max | 0.227 / 8.94 | before clipping, all epochs |
| Loss spikes / NaN losses | 0 / 0 | spike = loss > 1.5 × running average after warm-up |
| AMP overflow skips | 7 | fp16 steps skipped by the GradScaler |
| Parameter count | 3,249,152 | |
| Training speed | 1,365,667 characters/s | excludes evaluation time |
| Generation speed | 307 characters/s | batch 1, 400 characters |
| Peak GPU memory | 3,055 MB | |
| Total training time | 10.95 min | 10 epochs |

### Generation metrics per decoding temperature (word level)

| Temperature | Samples | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram rate | Non-word rate |
|---|---|---|---|---|---|---|
| 0 (greedy) | 5 | 0.254 | 0.466 | 0.558 | 0.343 | 0.000 |
| 0.7 | 10 | 0.298 | 0.712 | 0.877 | 0.019 | 0.004 |
| 1.0 | 10 | 0.339 | 0.776 | 0.929 | 0.005 | 0.008 |

Higher temperature gives more varied text and almost no repetition, but more invented words (non-word rate doubles from
0.7 to 1.0). Greedy decoding has no invented words but repeats itself heavily. I use temperature 0.7 as the headline
setting because it balances the two.

## Discussion of results

**How good is the model?** My best checkpoint has a validation loss of **0.615**, which is a perplexity of **1.85** and
**0.887 bits per character**. It guesses the next character correctly **80.4%** of the time. A perplexity under 2 means the
model is usually choosing between only one or two likely next characters, which makes sense for TinyStories' simple and
repetitive language.

**Overfitting?** No. The training loss (0.608) and validation loss (0.615) are almost the same, a gap of only **0.0073**.
With 100K stories, dropout 0.1 and only 3.25M parameters, the model cannot memorise the data. The validation loss was still
going down in the last epoch (0.6163 → 0.6131), and the best epoch was the last one, so my model is slightly
**under-trained** rather than overfitted. More epochs or a bigger model would probably help a little, but the improvement
per epoch had already become much smaller: from 0.056 (epoch 1 → 2) down to 0.003 (epoch 9 → 10).

**Was training stable?** Yes. I had **0 loss spikes** and **0 NaN losses**. The biggest gradient norm (**8.94**) happened in
the first epoch, when the weights were still random and the learning rate was warming up. After that, the gradient norms
stayed around 0.2, well below my clipping value of 1.0. Mixed precision skipped **7 of about 27,000 steps** because of fp16
overflow, which is normal and corrects itself. The learning rate warmed up for 500 steps to 1e-3 and then followed a cosine
curve down to 1e-4 (`outputs/lr_schedule.png`).

**Diversity of the generated text (temperature 0.7):** distinct-1/2/3 are **0.298 / 0.712 / 0.877**, so most word pairs and
triples in the samples are different and the model is not just copying the same text. Distinct-1 is lower because children's
stories use a small set of words. The repeated 4-gram rate is **0.019** at temperature 0.7 but **0.343** with greedy decoding,
about 18 times higher; failure 1 in `failure_analysis.md` shows what that looks like.

**Cost:** training took **11 minutes** for 10 epochs (about 66 seconds per epoch) at **1.37 million characters per second**,
with a peak GPU memory of **3.05 GB**. Generation runs at only **307 characters per second**, because the model writes one
character at a time and has to run on the whole context again for every new character, while training processes 32,768
characters in parallel in each step.

**What I would try next:** a longer context (512 characters) and a deeper model for better story coherence, nucleus sampling
or a repetition penalty to stop the greedy loops, and more epochs, since the validation loss had not fully levelled off.

## Evidence
* Loss curves: `outputs/loss_curves.png` · learning-rate schedule: `outputs/lr_schedule.png`
* Per-epoch history: `outputs/epoch_history.csv` · per-step log: `outputs/step_log.csv`
* Generated samples: `outputs/samples.txt`, `outputs/samples.json` · generation metrics: `outputs/generation_metrics.csv`
* Split and vocabulary: `data_processed/split_indices.json`, `data_processed/vocab.json`
* Checkpoint: `checkpoints/best.pt` (ID `bceaa3cfc2f2`) · failure analysis: `failure_analysis.md`
* Raw log: `reproducibility/raw_logs/task1_llm_ruchika_chaurasia_20261001_235055.log`
* Manifest: `reproducibility/manifests/task1_llm_ruchika_chaurasia_20261002_000259.json`
