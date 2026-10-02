"""Shared runtime helpers used by every notebook in the repo.

Nothing in here is task-specific: finding the repo root (so no personal
paths are ever hard-coded), loading YAML configs, seeding, writing an
append-only raw log, recording hardware, and writing a reproducibility
manifest that maps checkpoints to the numbers in the report.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path


# --------------------------------------------------------------------------- paths
def find_repo_root(start: str | os.PathLike | None = None) -> Path:
    """Walk upwards from `start` (default: cwd) until the repo root is found.

    The repo root is the folder that contains README.md and task1_llm/.
    You can also force it with the REPO_ROOT environment variable.
    """
    if os.environ.get("REPO_ROOT"):
        return Path(os.environ["REPO_ROOT"]).resolve()
    p = Path(start or os.getcwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "README.md").exists() and (q / "task1_llm").is_dir():
            return q
    raise FileNotFoundError(
        "Could not find the repo root. `cd` into the repo (or set REPO_ROOT) before running."
    )


def is_smoke() -> bool:
    """SMOKE=1 runs a tiny version of a notebook (used by the one-command smoke test)."""
    return os.environ.get("SMOKE", "0") == "1"


def load_config(path: str | os.PathLike) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def personal_params(sid4: int) -> dict:
    """Course standing requirement 0.1 (derived once from the last 4 SID digits)."""
    cls_a = sid4 % 10
    return {
        "SID4": sid4,
        "SEED": sid4,
        "SLICE": sid4 % 1000,
        "HP_ID": sid4 % 6,
        "CLS_A": cls_a,
        "CLS_B": (cls_a + 1 + ((sid4 // 10) % 9)) % 10,
    }


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- raw log
class RawLogger:
    """print() + append the same line (with a timestamp) to an unedited raw log file."""

    def __init__(self, repo_root: Path, task: str, member: str, tag: str = ""):
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        sub = "_smoke" if is_smoke() else ""
        name = f"{task}_{member}{('_' + tag) if tag else ''}{sub}_{ts}.log"
        self.path = repo_root / "reproducibility" / "raw_logs" / name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)
        self("raw log started:", self.path.relative_to(repo_root))

    def __call__(self, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(msg, **kwargs)
        self._f.write(f"[{stamp}] {msg}\n")

    def exception(self, err: BaseException):
        tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        self("EXCEPTION:\n" + tb)

    def close(self):
        self._f.close()


# --------------------------------------------------------------------------- hardware
def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:  # macOS
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        return platform.processor() or platform.machine()


def hardware_info() -> dict:
    info = {
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_memory_gb"] = round(p.total_memory / 1024**3, 2)
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
        else:
            info["gpu"] = "none (CPU only)"
    except ImportError:
        pass
    return info


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_memory_mb(device) -> float:
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    try:  # CPU fallback: resident set size of this process
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


def reset_peak_memory(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


class Timer:
    def __enter__(self):
        self._sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self.t0

    @staticmethod
    def _sync():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass


# --------------------------------------------------------------------------- manifest
def file_sha(path: str | os.PathLike, n: int = 12) -> str:
    """Short content hash = the 'checkpoint ID' quoted in the report."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown (not a git checkout)"


def package_versions(names=None) -> dict:
    import importlib.metadata as md

    names = names or [
        "torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
        "datasets", "nltk", "torchmetrics", "torch-fidelity", "pyyaml",
    ]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not installed"
    return out


def write_manifest(repo_root: Path, task: str, member: str, config: dict,
                   checkpoints: dict, results_files: list, extra: dict | None = None) -> Path:
    """checkpoints: {label: path}. Records sha IDs so every report number is traceable."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = "_smoke" if is_smoke() else ""
    out_dir = repo_root / "reproducibility" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = {}
    for label, p in checkpoints.items():
        p = Path(p)
        if p.exists():
            ck[label] = {"path": str(p.relative_to(repo_root)), "checkpoint_id": file_sha(p),
                         "size_mb": round(p.stat().st_size / 1024**2, 2)}
    manifest = {
        "task": task,
        "member": member,
        "created": ts,
        "smoke_run": is_smoke(),
        "git_commit": git_commit(repo_root),
        "hardware": hardware_info(),
        "packages": package_versions(),
        "config": config,
        "checkpoints": ck,
        "results_files": [str(Path(r).relative_to(repo_root)) for r in results_files if Path(r).exists()],
    }
    if extra:
        manifest.update(extra)
    path = out_dir / f"{task}_{member}{sub}_{ts}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    # full environment snapshot next to it
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        (out_dir / f"{task}_{member}{sub}_{ts}_pip_freeze.txt").write_text(freeze)
    except Exception:
        pass
    return path
