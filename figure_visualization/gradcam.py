"""对所选类别logit反向传播，绘制Grad-CAM。"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import load_config
from common.data import DataModule
from common.engine import load_model
from common.runtime import device_for, json_write, sha256


def compute(model, img, nir, class_index=None):
    model.eval()
    model.zero_grad(set_to_none=True)
    if not hasattr(model, "forward_cam"):
        raise ValueError("CAM supported for image_only and BCA students")
    logits, features = model.forward_cam(img, nir)
    cls = int(logits.argmax(1)[0]) if class_index is None else int(class_index)
    if cls not in [0, 1, 2]:
        raise ValueError("class-index must be 0/1/2")
    logits[0, cls].backward()
    if features.grad is None:
        raise RuntimeError("Target feature has no gradients")
    weights = features.grad.mean((-2, -1), keepdim=True)
    cam = F.relu((weights * features).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-12)
    return cam.detach().cpu().numpy(), cls


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--class-index", type=int)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    dm = DataModule(cfg)
    device = device_for(cfg)
    model, obj = load_model(a.checkpoint, dm, cfg, device)
    matches = np.where(dm.frame.sample_id.to_numpy() == a.sample_id)[0]
    if len(matches) != 1:
        raise ValueError("Unknown sample ID")
    row = dm.loader(matches).dataset[0]
    im = row["image"].unsqueeze(0).to(device)
    nir = row["nir"].unsqueeze(0).to(device)
    cam, cls = compute(model, im, nir, a.class_index)
    rgb = (
        im[0].detach().cpu().numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
        + np.array([0.485, 0.456, 0.406])
    ).clip(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input leaf")
    # 色标显示归一化CAM值。
    heat = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    fig.colorbar(heat, ax=axes[1], label="Normalized CAM")
    axes[2].imshow(rgb)
    axes[2].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    axes[2].set_title(f"{obj['model_task']} / class {cls}")
    for ax in axes:
        ax.axis("off")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    plt.close(fig)
    json_write(
        {
            "checkpoint_sha256": sha256(a.checkpoint),
            "model_task": obj["model_task"],
            "seed": obj["meta"]["seed"],
            "sample_id": a.sample_id,
            "target_class": cls,
            "synthetic": dm.synthetic,
        },
        str(a.out) + ".provenance.json",
    )


if __name__ == "__main__":
    main()
