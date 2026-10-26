"""随机种子、运行配置记录和checkpoint读写。"""

import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

from common.identifiers import EXPERIMENT_IDS


def json_write(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def seed_all(seed, cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32


def device_for(cfg):
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )


def fingerprint(cfg, dm, seed, task, parents):
    from common.config import VERSION

    return {
        "version": VERSION,
        "code_sha256": code_hash(),
        "data_sha256": dm.data_hash,
        "config": cfg.dict(),
        "seed": seed,
        "task": task,
        "parents": parents,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "synthetic": dm.synthetic,
        "result_origin": "training_run",
        "data_card_ids": list(EXPERIMENT_IDS.get(task, ())),
    }


def save_checkpoint(model, path, meta, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": state, "meta": meta, **extra}, tmp)
    os.replace(tmp, path)


def read_checkpoint(path):
    # 按state_dict格式加载checkpoint。
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "meta" not in obj or "state_dict" not in obj:
        raise ValueError("Expected checkpoint keys: state_dict and meta")
    return obj


def prepare_output(path, meta, reuse=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not reuse:
            raise FileExistsError(f"Output directory already contains a run: {path}")
        old = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        if old != meta:
            raise ValueError(f"Code/data/config/parent mismatch in existing run: {path}")
        if not (path / "COMPLETED.json").exists():
            raise RuntimeError("Incomplete run: choose a new output directory")
        return False
    path.mkdir(parents=True, exist_ok=True)
    json_write(meta, path / "run_manifest.json")
    return True
