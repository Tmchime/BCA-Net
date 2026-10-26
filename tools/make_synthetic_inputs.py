"""生成随机图像、光谱和VOC，用于训练流程测试。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    rng = np.random.default_rng(42)
    rows = []
    spectra = []
    vocs = []
    old = {"XL": "M", "DT": "H", "BPP": "M", "FD": "H", "HPP": "M", "YTSK": "L"}
    new = {"DX": "H", "LM": "H", "MMK": "H", "PLBZ": "L", "XCX": "M", "XHL": "L"}
    sites = {
        "DX": "Nanjing",
        "PLBZ": "Nanjing",
        "LM": "Kaiyuan",
        "MMK": "Kaiyuan",
        "XCX": "Quanjiao",
        "XHL": "Quanjiao",
    }
    for b, cvs in [("B1", old), ("B2", old), ("B3", new)]:
        for cv, label in cvs.items():
            for j in range(4):
                sid = f"{b}_{cv}_{j}"
                path = f"images/{sid}.png"
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / path)
                rows.append(
                    dict(
                        sample_id=sid,
                        plant_id=sid,
                        batch=b,
                        cultivar=cv,
                        site=sites[cv] if b == "B3" else "Nanjing",
                        nitrogen="NN",
                        label=label,
                        image_path=path,
                    )
                )
                spectra.append([sid, *rng.normal(size=499)])
                if b == "B1":
                    vocs.append([sid, *rng.normal(size=4)])
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame(spectra, columns=["sample_id", *[f"band_{i:03d}" for i in range(499)]]).to_csv(
        root / "nir.csv", index=False
    )
    pd.DataFrame(
        vocs,
        columns=[
            "sample_id",
            "beta_caryophyllene",
            "camphene",
            "gamma_elemene",
            "propyl_2_butenoate",
        ],
    ).to_csv(root / "voc.csv", index=False)
    pd.DataFrame({"wavelength_nm": np.linspace(1550, 1950, 499)}).to_csv(
        root / "wavelengths.csv", index=False
    )
    (root / "SYNTHETIC_ONLY").write_text("Random inputs for software testing.\n", encoding="utf-8")
    cfg = {
        "data_dir": str(root.resolve()),
        "output_dir": str((root.parent / "synthetic_runs").resolve()),
        "workers": 0,
        "pretrained": False,
        "strict_paper_counts": False,
        "teacher_epochs": 1,
        "student_epochs": 1,
        "adapt_epochs": 1,
        "backbone": "convnext_atto",
        "embed_dim": 32,
        "image_size": 32,
        "loco_epochs": 1,
        "loco_lr": 1e-4,
        "loco_weight_decay": 1e-4,
        "device": "cpu",
    }
    (root / "synthetic_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return root / "synthetic_config.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(make(a.out))
