"""Task 1 shared evaluation — every member computes metrics with THIS file,
so numbers in the comparison table are directly comparable.

Conventions (agreed once, used by everyone):
  * loss      = mean next-character cross-entropy in nats, over ALL positions
  * perplexity = exp(loss)            (character-level perplexity)
  * bpc        = loss / ln(2)         (bits per character)
  * gap        = val_loss - train_loss, both measured in eval mode (dropout off)
                 on the same number of sequences
  * top-1 acc  = argmax(logits) == next character
  * distinct-n, repeated 4-gram rate: WORD-level, on generated samples
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def loss_to_ppl(loss: float) -> float:
    return float(math.exp(loss))


def loss_to_bpc(loss: float) -> float:
    return float(loss / LN2)


@torch.no_grad()
def evaluate_lm(model, data: torch.Tensor, block_size: int, batch_size: int,
                max_seqs: int | None = None, device=None, amp: bool = False) -> dict:
    """Loss and top-1 accuracy over non-overlapping windows of a 1-D token tensor.

    `model(x)` must return logits of shape (B, T, V).
    """
    model.eval()
    device = device or next(model.parameters()).device
    n_seq = (len(data) - 1) // block_size
    if max_seqs is not None:
        n_seq = min(n_seq, max_seqs)
    starts = torch.arange(n_seq, device=data.device) * block_size
    offs = torch.arange(block_size, device=data.device)
    tot_loss, tot_correct, tot_tok = 0.0, 0, 0
    for i in range(0, n_seq, batch_size):
        s = starts[i:i + batch_size]
        idx = s[:, None] + offs[None, :]
        x = data[idx].long().to(device, non_blocking=True)
        y = data[idx + 1].long().to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        tot_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                    reduction="sum").item()
        tot_correct += (logits.argmax(-1) == y).sum().item()
        tot_tok += y.numel()
    loss = tot_loss / max(tot_tok, 1)
    return {"loss": loss, "acc": tot_correct / max(tot_tok, 1), "ppl": loss_to_ppl(loss),
            "bpc": loss_to_bpc(loss), "tokens": tot_tok}


# --------------------------------------------------------------------------- generation metrics
_WORD = re.compile(r"[a-z']+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples: list[str], n: int) -> float:
    """#unique n-grams / #total n-grams, pooled over all samples (Li et al., 2016)."""
    all_ng = []
    for s in samples:
        all_ng.extend(ngrams(words(s), n))
    return len(set(all_ng)) / len(all_ng) if all_ng else 0.0


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams in ONE sample that already appeared earlier in that sample."""
    ng = ngrams(words(text), n)
    if not ng:
        return 0.0
    return 1.0 - len(set(ng)) / len(ng)


def nonword_rate(text: str, vocab_words: set[str]) -> float:
    """Fraction of generated words never seen in the training text (invented/misspelled words)."""
    w = words(text)
    if not w:
        return 0.0
    return sum(1 for t in w if t not in vocab_words) / len(w)


def generation_metrics(samples: list[str], vocab_words: set[str] | None = None) -> dict:
    out = {f"distinct_{n}": distinct_n(samples, n) for n in (1, 2, 3)}
    rates = [repeated_ngram_rate(s, 4) for s in samples]
    out["repeated_4gram_rate"] = float(np.mean(rates)) if rates else 0.0
    if vocab_words is not None:
        out["nonword_rate"] = float(np.mean([nonword_rate(s, vocab_words) for s in samples]))
    return out


def training_word_vocab(texts: list[str], min_count: int = 1) -> set[str]:
    c = Counter()
    for t in texts:
        c.update(words(t))
    return {w for w, k in c.items() if k >= min_count}
