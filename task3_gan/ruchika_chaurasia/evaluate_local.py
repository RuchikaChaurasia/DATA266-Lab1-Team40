"""Task 3 local evaluation for ONE member (this file lives in task3_gan/<member>/).

    python task3_gan/<member>/evaluate_local.py              # metrics + pred folders + official submission.csv
    python task3_gan/<member>/evaluate_local.py --no-official

Convention (course script): A = Monet, B = photo. G_AB = Monet->photo, G_BA = photo->Monet.
Uses only this member's trained generators (checkpoints/cyclegan_final.pt) and the team's shared
metrics in task3_gan/common/. Generated images are raw generator outputs — nothing is edited or hand-picked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MEMBER_DIR = Path(__file__).resolve().parent
REPO_ROOT = MEMBER_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "task3_gan" / "common"))
sys.path.insert(0, str(MEMBER_DIR / "src"))

import data_task3 as D            # noqa: E402
import metrics_task3 as M         # noqa: E402
from cyclegan_models import build_models, count_params   # noqa: E402


def load_final(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    G_AB, G_BA, D_A, D_B = build_models(ck["config"]["model"])
    for m, k in [(G_AB, "G_AB"), (G_BA, "G_BA"), (D_A, "D_A"), (D_B, "D_B")]:
        m.load_state_dict({n: t.float() for n, t in ck[k].items()})
        m.to(device).eval()
    return ck, G_AB, G_BA, D_A, D_B


@torch.no_grad()
def translate(G, x_uint8: torch.Tensor, device, bs: int = 8) -> torch.Tensor:
    out = []
    for i in range(0, len(x_uint8), bs):
        out.append(G(D.to_m11(x_uint8[i:i + bs]).to(device)).float().cpu())
    return torch.cat(out)


def images_per_sec(G, device, size=256, bs=8, reps=10) -> float:
    x = torch.randn(bs, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(2):
            G(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            G(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return bs * reps / (time.perf_counter() - t0)


def save_grid(rows, titles, path):
    import matplotlib.pyplot as plt

    n = len(rows[0])
    fig, ax = plt.subplots(len(rows), n, figsize=(2.2 * n, 2.3 * len(rows)))
    for r, (row, title) in enumerate(zip(rows, titles)):
        for c in range(n):
            ax[r, c].imshow(D.to_uint8(row[c]).permute(1, 2, 0).numpy()); ax[r, c].axis("off")
        ax[r, 0].set_title(title, loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def run_evaluation(cfg: dict, G_AB, G_BA, D_A, D_B, train_summary: dict, device, out_root: Path,
                   log=print, seed: int = 0) -> pd.DataFrame:
    dc = cfg["data"]
    photo_dir, monet_dir = REPO_ROOT / dc["photo_dir"], REPO_ROOT / dc["monet_dir"]
    split = D.team_split(REPO_ROOT, photo_dir)
    size = dc["image_size"]
    monet_paths = D.list_images(monet_dir)
    monets = D.load_uint8(monet_paths, size)                                          # domain A
    eval_photos = D.load_uint8([photo_dir / n for n in split["eval_photos"]], size)   # domain B (held out)
    log(f"evaluating on {len(monets)} Monet paintings (A) and {len(eval_photos)} held-out photos (B, team split)")

    x_A, x_B = D.to_m11(monets), D.to_m11(eval_photos)
    fake_B = translate(G_AB, monets, device);      rec_A = translate(G_BA, D.to_uint8(fake_B), device)   # Monet->photo->Monet
    fake_A = translate(G_BA, eval_photos, device); rec_B = translate(G_AB, D.to_uint8(fake_A), device)   # photo->Monet->photo

    feats, lp = M.InceptionFeatures(device), M.LPIPS(device)
    a2b = M.evaluate_direction("A2B (Monet->photo)", x_A, fake_B, rec_A, x_B, feats, lp, seed)
    b2a = M.evaluate_direction("B2A (photo->Monet)", x_B, fake_A, rec_B, x_A, feats, lp, seed)
    # cycle-consistency sanity check: reconstruction must be much closer to the input than the translation is
    a2b["l1_input_vs_translation"] = M.cycle_l1(x_A, fake_B)
    b2a["l1_input_vs_translation"] = M.cycle_l1(x_B, fake_A)

    # ---- prediction folders in the layout the official script expects (all images, stale files removed)
    for sub in ("pred_A2B", "pred_B2A", "audit"):
        shutil.rmtree(out_root / "outputs" / sub, ignore_errors=True)
    for sub in ("pred_A2B", "pred_B2A", "audit", "figures"):
        (out_root / "outputs" / sub).mkdir(parents=True, exist_ok=True)
    for p, img in zip(monet_paths, fake_B):                       # generated photos, one per Monet
        D.save_jpg(img, out_root / "outputs" / "pred_A2B" / f"{p.stem}.jpg")
    for n, img in zip(split["eval_photos"], fake_A):              # generated Monets, one per held-out photo
        D.save_jpg(img, out_root / "outputs" / "pred_B2A" / f"{Path(n).stem}.jpg")
    name_to_i = {n: i for i, n in enumerate(split["eval_photos"])}
    for n in split["audit_photos"]:
        D.save_jpg(fake_A[name_to_i[n]], out_root / "outputs" / "audit" / f"{Path(n).stem}.jpg")
    k = 6
    save_grid([x_B[:k], fake_A[:k], rec_B[:k]], ["photo (input)", "G_BA: Monet-style", "G_AB(G_BA(x)): reconstruction"],
              out_root / "outputs" / "figures" / "cycle_B2A_photo2monet.png")
    save_grid([x_A[:k], fake_B[:k], rec_A[:k]], ["Monet (input)", "G_AB: photo-style", "G_BA(G_AB(y)): reconstruction"],
              out_root / "outputs" / "figures" / "cycle_A2B_monet2photo.png")
    per_img = (x_B - rec_B).abs().mean((1, 2, 3)).numpy()
    worst = np.argsort(-per_img)[:k]
    save_grid([x_B[worst], fake_A[worst], rec_B[worst]], ["worst cycle-L1 photos", "Monet-style", "reconstruction"],
              out_root / "outputs" / "figures" / "worst_cycle_B2A.png")

    # ---- long-format report
    rows = []
    for d in (a2b, b2a):
        tag = d["direction"].split()[0]
        for key, val in d.items():
            if key != "direction":
                rows.append({"metric": key, "direction": tag, "value": val})
    ips = images_per_sec(G_BA, device, size)
    common = {
        "params_G_AB": count_params(G_AB), "params_G_BA": count_params(G_BA), "params_D_A": count_params(D_A),
        "params_D_B": count_params(D_B), "params_total": sum(count_params(m) for m in (G_AB, G_BA, D_A, D_B)),
        "inference_images_per_sec": ips, **train_summary,
    }
    for key, val in common.items():
        rows.append({"metric": key, "direction": "both", "value": val})
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "full_metrics_report.csv", index=False)
    df.to_csv(out_root / "metrics_report.csv", index=False)
    log(df.to_string(index=False))
    return df


def official_submission(cfg: dict, out_root: Path, log=print) -> pd.DataFrame:
    """Run the course's Part3_Evaluation_Script.ipynb UNCHANGED on this member's pred folders.

    The script reads the relative path "Part 3/Data/{monet_jpg,photo_jpg,pred_A2B,pred_B2A}", so a
    scratch folder with that layout (symlinks, local disk) is built and the notebook executed there.
    Its submission.csv (ID, FID, MiFID) is copied to task3_gan/<member>/submission.csv, and the
    per-direction values it prints are appended to full_metrics_report.csv.
    """
    dc = cfg["data"]
    work = Path(tempfile.mkdtemp(prefix="official_eval_"))
    data = work / "Part 3" / "Data"
    data.mkdir(parents=True)
    links = {"monet_jpg": REPO_ROOT / dc["monet_dir"], "photo_jpg": REPO_ROOT / dc["photo_dir"],
             "pred_A2B": out_root / "outputs" / "pred_A2B", "pred_B2A": out_root / "outputs" / "pred_B2A"}
    for name, target in links.items():
        os.symlink(target.resolve(), data / name, target_is_directory=True)
    nb = REPO_ROOT / "task3_gan" / "common" / "Part3_Evaluation_Script.ipynb"
    shutil.copy(nb, work / nb.name)
    log("running the official evaluation script (unchanged) ...")
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", nb.name,
                    "--output", "official_eval_executed.ipynb", "--ExecutePreprocessor.timeout=3600"],
                   cwd=work, check=True)
    shutil.copy(work / "submission.csv", out_root / "submission.csv")
    shutil.copy(work / "official_eval_executed.ipynb", out_root / "outputs" / "official_eval_executed.ipynb")
    sub = pd.read_csv(out_root / "submission.csv")
    # per-direction numbers printed by the official notebook
    text = "".join("".join(o.get("text", "")) for c in json.load(open(work / "official_eval_executed.ipynb"))["cells"]
                   for o in c.get("outputs", []))
    rows = []
    for tag, label in [("B2A", r"\[Photo->Monet\]"), ("A2B", r"\[Monet->Photo\]")]:
        m = re.search(label + r" FID=([\d.]+)\s+MiFID=([\d.]+)", text)
        if m:
            rows += [{"metric": "official_fid", "direction": tag, "value": float(m.group(1))},
                     {"metric": "official_mifid", "direction": tag, "value": float(m.group(2))}]
    rows += [{"metric": "official_submission_fid", "direction": "both", "value": float(sub["FID"].iloc[0])},
             {"metric": "official_submission_mifid", "direction": "both", "value": float(sub["MiFID"].iloc[0])}]
    rep = out_root / "full_metrics_report.csv"
    if rep.exists():
        df = pd.read_csv(rep)
        df = pd.concat([df[~df.metric.str.startswith("official_")], pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(rep, index=False); df.to_csv(out_root / "metrics_report.csv", index=False)
    log(f"official submission.csv: {sub.to_dict('records')}  -> {out_root / 'submission.csv'}")
    shutil.rmtree(work, ignore_errors=True)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-official", action="store_true", help="skip the official script / submission.csv")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--run", default="", help="extra run name, e.g. long100 -> runs/long100/")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = MEMBER_DIR / "runs" / args.run if args.run else MEMBER_DIR
    ck, G_AB, G_BA, D_A, D_B = load_final(out / "checkpoints" / "cyclegan_final.pt", device)
    if not args.skip_metrics:
        run_evaluation(ck["config"], G_AB, G_BA, D_A, D_B, ck.get("train_summary", {}), device, out,
                       seed=ck["config"]["seed"])
    if not args.no_official:
        official_submission(ck["config"], out)


if __name__ == "__main__":
    main()
