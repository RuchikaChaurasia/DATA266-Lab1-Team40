## Discussion of results (add to the end of task1_llm/ruchika_chaurasia/results.md)

**How good is the model?** My best checkpoint (`bceaa3cfc2f2`, epoch 10) has a validation loss of **0.615**, which is a
perplexity of **1.85** and **0.887 bits per character**. It guesses the next character correctly **80.4%** of the time.
A perplexity under 2 means the model is usually choosing between only one or two likely next characters. That makes sense
for TinyStories, which uses simple and repetitive language.

**Overfitting?** No. The training loss (0.608) and validation loss (0.615) are almost the same, a gap of only **0.0073**.
With 100K stories, dropout 0.1 and only 3.25M parameters, the model cannot memorise the data. The validation loss was still
going down in the last epoch (0.6163 → 0.6131), and the best epoch was the last one. So my model is slightly
**under-trained** rather than overfitted, and more epochs or a bigger model would probably help a little. The improvement
per epoch did get much smaller over time, though: from 0.056 between epochs 1 and 2 down to 0.003 between epochs 9 and 10.

**Was training stable?** Yes. I had **0 loss spikes** and **0 NaN losses**. The biggest gradient norm (**8.94**) happened in
the first epoch, when the weights were still random and the learning rate was warming up. After that, the gradient norms
stayed around 0.2, well below my clipping value of 1.0. Mixed precision skipped **7 steps out of about 27,000** because of
fp16 overflow, which is normal and fixes itself. The learning rate warmed up for 500 steps to 1e-3 and then followed a
cosine curve down to 1e-4 (see `outputs/lr_schedule.png`).

**Diversity of the generated text (temperature 0.7):** distinct-1/2/3 are **0.298 / 0.712 / 0.877**, so most word pairs
and triples in the samples are different, meaning the model is not just copying the same text. Distinct-1 is lower because
children's stories use a small set of words. The repeated 4-gram rate is **0.019** at temperature 0.7 but **0.343** with
greedy decoding, about 18 times higher. Failure 1 in my failure analysis shows what that looks like.

**Cost:** **3,249,152** parameters · **11 minutes** of training for 10 epochs on an NVIDIA RTX 5090 (about 66 seconds per
epoch) · **1.37 million characters per second** during training · peak GPU memory **3.05 GB** · generation **307 characters
per second**. Generation is much slower than training because the model writes one character at a time and has to run on the
whole context again for every new character, while training processes 32,768 characters in parallel in each step.

**What I would try next:** a longer context (512 characters) and a deeper model for better story coherence, nucleus sampling
or a repetition penalty to stop the greedy loops, and more epochs, since the validation loss had not fully levelled off.
