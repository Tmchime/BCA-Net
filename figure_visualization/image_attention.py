"""Student-Image辅助空间注意力可视化。"""

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import load_config
from common.data import DataModule
from common.engine import load_model
from common.runtime import device_for, json_write, sha256


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    dm = DataModule(cfg)
    device = device_for(cfg)
    model, obj = load_model(a.checkpoint, dm, cfg, device)
    model.eval()
    if obj["model_task"] != "student_image" or obj["meta"]["seed"] != 42:
        raise ValueError("Spatial attention requires student_image checkpoint with seed 42")
    ids = np.where(dm.frame.sample_id.to_numpy() == a.sample_id)[0]
    if len(ids) != 1:
        raise ValueError("Unknown sample ID")
    index = int(ids[0])
    row = dm.loader(ids).dataset[0]
    with Image.open(dm.images[index]) as im:
        rgb = np.asarray(im.convert("RGB")).copy()
    with torch.inference_mode():
        _, weights, _, _ = model(
            row["image"].unsqueeze(0).to(device), row["nir"].unsqueeze(0).to(device)
        )
        side = math.isqrt(weights.shape[-1])
        if side * side != weights.shape[-1]:
            raise ValueError("Expected a square spatial attention map")
        heat = F.interpolate(
            weights.reshape(1, 1, side, side),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        heat = heat - heat.min()
        heat = heat / heat.max().clamp_min(1e-12)
        heat = heat.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original leaf")
    axes[1].imshow(rgb)
    layer = axes[1].imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Student-Image attention")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(layer, ax=axes[1], label="Normalized attention")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    json_write(
        {
            "checkpoint_sha256": sha256(a.checkpoint),
            "model_task": obj["model_task"],
            "seed": 42,
            "sample_id": a.sample_id,
            "method": "spatial attention",
            "interpolation": "bicubic",
            "normalization": "min-max for display",
            "synthetic": dm.synthetic,
        },
        str(a.out) + ".provenance.json",
    )


if __name__ == "__main__":
    main()
