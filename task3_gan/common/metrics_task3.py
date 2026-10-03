"""Task 3 shared evaluation — every member uses THIS file, on the SAME held-out photos
(team-fixed list in task3_gan/common/team_eval_split.json), so numbers are comparable.

Direction convention = the course's Part3_Evaluation_Script.ipynb:  A = Monet,  B = photo
    A2B = Monet -> photo   (pred_A2B/ holds generated photos)
    B2A = photo -> Monet   (pred_B2A/ holds generated Monets — the painting direction)

The official Kaggle numbers (FID and the script's "MiFID") come ONLY from that unchanged script.
The metrics here are the extra metrics the lab brief asks for.

Feature extractor: the standard FID Inception-v3 (pool3, 2048-d) from torch-fidelity / torchmetrics.
It is used ONLY to *measure* images, never to generate them.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

_TEST_ONLY = os.environ.get("_TEST_ONLY_RANDOM_FEATURES") == "1"   # offline CI hook, never used for real numbers


# --------------------------------------------------------------------------- feature extractor
class InceptionFeatures:
    def __init__(self, device):
        self.device = device
        if _TEST_ONLY:
            torch.manual_seed(0)
            self.net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 8, 8), torch.nn.ReLU(),
                                           torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                           torch.nn.Linear(16, 64)).to(device).eval()
            self.uint8 = False
        else:
            from torchmetrics.image.fid import NoTrainInceptionV3

            self.net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
            self.uint8 = True

    @torch.no_grad()
    def __call__(self, imgs_m11: torch.Tensor, batch_size: int = 50) -> np.ndarray:
        """imgs in [-1, 1], shape (N, 3, H, W) -> (N, D) float64 features."""
        out = []
        for i in range(0, len(imgs_m11), batch_size):
            x = imgs_m11[i:i + batch_size].to(self.device)
            if self.uint8:
                x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            out.append(self.net(x).double().cpu())
        return torch.cat(out).numpy()


# --------------------------------------------------------------------------- distribution metrics
def fid(f_real: np.ndarray, f_fake: np.ndarray) -> float:
    mu1, mu2 = f_real.mean(0), f_fake.mean(0)
    s1, s2 = np.cov(f_real, rowvar=False), np.cov(f_fake, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kid(f_real: np.ndarray, f_fake: np.ndarray, n_subsets: int = 100, subset_size: int = 100, seed: int = 0):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al., 2018). Returns (mean, std)."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f_real), len(f_fake))
    d = f_real.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f_real[rng.choice(len(f_real), m, replace=False)]
        y = f_fake[rng.choice(len(f_fake), m, replace=False)]
        kxx, kyy, kxy = ((x @ x.T) / d + 1) ** 3, ((y @ y.T) / d + 1) ** 3, ((x @ y.T) / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def _pairwise(a, b):
    a2, b2 = (a ** 2).sum(1)[:, None], (b ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2 * a @ b.T, 0))


def prdc(f_real: np.ndarray, f_fake: np.ndarray, k: int = 5) -> dict:
    """Precision/recall (Kynkäänniemi et al., 2019) and density/coverage (Naeem et al., 2020)."""
    k = min(k, len(f_real) - 1, len(f_fake) - 1)
    rr, ff, rf = _pairwise(f_real, f_real), _pairwise(f_fake, f_fake), _pairwise(f_real, f_fake)
    r_rad = np.sort(rr, axis=1)[:, k]          # k-th NN radius (index 0 is the point itself)
    f_rad = np.sort(ff, axis=1)[:, k]
    precision = (rf < r_rad[:, None]).any(0).mean()
    recall = (rf < f_rad[None, :]).any(1).mean()
    density = (1.0 / k) * (rf < r_rad[:, None]).sum(0).mean()
    coverage = (rf.min(1) < r_rad).mean()
    return {"precision": float(precision), "recall": float(recall), "density": float(density), "coverage": float(coverage)}


# --------------------------------------------------------------------------- paired metrics
def cycle_l1(x: torch.Tensor, x_rec: torch.Tensor) -> float:
    """Mean absolute error in [-1, 1] pixel space between input and its reconstruction F(G(x))."""
    return float((x - x_rec).abs().mean())


def content_cosine(f_in: np.ndarray, f_out: np.ndarray) -> float:
    """Mean cosine similarity between Inception features of each input and ITS translation."""
    a = f_in / np.linalg.norm(f_in, axis=1, keepdims=True)
    b = f_out / np.linalg.norm(f_out, axis=1, keepdims=True)
    return float((a * b).sum(1).mean())


class LPIPS:
    """LPIPS (AlexNet) between each input and its translation — lower = more perceptually similar."""

    def __init__(self, device):
        self.device = device
        self.metric = None
        if not _TEST_ONLY:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            self.metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)

    @torch.no_grad()
    def __call__(self, a_m11: torch.Tensor, b_m11: torch.Tensor, batch_size: int = 25) -> float:
        if self.metric is None:
            return float("nan")
        vals = []
        for i in range(0, len(a_m11), batch_size):
            self.metric.reset()
            a = a_m11[i:i + batch_size].to(self.device).clamp(-1, 1)
            b = b_m11[i:i + batch_size].to(self.device).clamp(-1, 1)
            vals.append(float(self.metric(a, b)) * len(a))
        return float(np.sum(vals) / len(a_m11))


def evaluate_direction(name, inputs, translations, reconstructions, real_target, feats: InceptionFeatures,
                       lp: LPIPS, seed: int = 0) -> dict:
    """All per-direction metrics. Tensors are (N, 3, H, W) in [-1, 1]."""
    f_real, f_fake, f_in = feats(real_target), feats(translations), feats(inputs)
    k_mean, k_std = kid(f_real, f_fake, subset_size=min(100, len(f_real), len(f_fake)), seed=seed)
    out = {"direction": name, "n_translated": len(translations), "n_real_reference": len(real_target),
           "fid": fid(f_real, f_fake), "kid_mean": k_mean, "kid_std": k_std,
           **{f"gen_{k}": v for k, v in prdc(f_real, f_fake).items()},
           "cycle_l1": cycle_l1(inputs, reconstructions),
           "lpips_input_vs_translation": lp(inputs, translations),
           "content_cosine": content_cosine(f_in, f_fake)}
    return out
