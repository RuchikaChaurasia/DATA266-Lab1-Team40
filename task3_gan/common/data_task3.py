"""Task 3 shared data helpers.

The TEAM eval split (held-out photos for metrics + the 30 fixed audit photos) is created once with a
team seed and committed as team_eval_split.json, so every member is measured on the same images.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEAM_SEED = 266
N_EVAL_PHOTOS = 500
N_AUDIT = 30
EXTS = {".jpg", ".jpeg", ".png"}


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS and not p.name.startswith("."))


def team_split(repo_root: Path, photo_dir: Path) -> dict:
    """Load (or create once) the team-fixed held-out photo list and the 30 audit photos."""
    path = repo_root / "task3_gan" / "common" / "team_eval_split.json"
    if path.exists():
        return json.loads(path.read_text())
    names = [p.name for p in list_images(photo_dir)]
    rng = np.random.default_rng(TEAM_SEED)
    n_eval = min(N_EVAL_PHOTOS, len(names) // 5)
    eval_names = sorted(rng.choice(names, n_eval, replace=False).tolist())
    audit = sorted(rng.choice(eval_names, min(N_AUDIT, n_eval), replace=False).tolist())
    split = {"team_seed": TEAM_SEED, "eval_photos": eval_names, "audit_photos": audit,
             "note": "A=Monet, B=photo (course convention). eval_photos are never used for training by any member."}
    path.write_text(json.dumps(split, indent=1))
    return split


def maybe_local_copy(src: Path, scratch: Path | None) -> Path:
    """Reading thousands of small files from Google Drive is slow; copy once to local disk."""
    if scratch is None or not str(src).startswith("/content/drive"):
        return src
    dst = scratch / src.name
    if not dst.exists():
        shutil.copytree(src, dst)
    return dst


def load_uint8(paths: list[Path], size: int) -> torch.Tensor:
    """Load images resized to size x size as a uint8 tensor (N, 3, size, size) kept in RAM."""
    out = torch.empty((len(paths), 3, size, size), dtype=torch.uint8)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        out[i] = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
    return out


def to_m11(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float() / 127.5 - 1.0


def to_uint8(x_m11: torch.Tensor) -> torch.Tensor:
    return ((x_m11.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def random_crop_flip(batch_uint8: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Random jitter used by CycleGAN: crop size x size from the (larger) stored image + random flip."""
    n, _, h, w = batch_uint8.shape
    out = torch.empty((n, 3, size, size), dtype=batch_uint8.dtype)
    for i in range(n):
        top = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        left = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        img = batch_uint8[i, :, top:top + size, left:left + size]
        if torch.rand(1, generator=gen).item() < 0.5:
            img = img.flip(-1)
        out[i] = img
    return out


def save_jpg(x_m11: torch.Tensor, path: Path, quality: int = 95) -> None:
    arr = to_uint8(x_m11).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path, quality=quality)
