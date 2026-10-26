"""读取已有npz，绘制光谱与注意力权重。"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source, out):
    with np.load(source, allow_pickle=False) as z:
        wl, x, a = [np.asarray(z[k]) for k in ["wavelengths", "nir_values", "attn_weights"]]
        labels = np.asarray(z["label"]).reshape(-1)
        if not len(labels):
            raise ValueError("Expected a class label")
        label = str(labels[0])
        sid = str(z["sample_id"].item()) if "sample_id" in z else None
    if (
        any(v.ndim != 1 for v in [wl, x, a])
        or not len(wl) == len(x) == len(a)
        or not all(np.isfinite(v).all() for v in [wl, x, a])
    ):
        raise ValueError("Invalid spectral arrays")
    if not (np.diff(wl) > 0).all():
        raise ValueError("Expected increasing wavelengths")
    a = np.clip(a, 0, None)
    fig, ax = plt.subplots(figsize=(10, 5))
    other = ax.twinx()
    line1 = ax.plot(wl, x, color="#1f77b4", linewidth=2.5, label="SNV transformed absorbance")
    line2 = other.plot(wl, a, color="#d62728", linewidth=1.5, label="Attention weight")
    other.fill_between(wl, 0, a, color="#d62728", alpha=0.15)
    other.set_ylim(0, max(1e-6, float(a.max()) * 1.1))
    title = f"Spectral importance (Class: {label})" + (f" / {sid}" if sid else "")
    ax.set(
        xlabel="Wavelength (nm)",
        ylabel="SNV transformed absorbance",
        title=title,
        xlim=(wl[0], wl[-1]),
    )
    other.set_ylabel("Attention weight")
    ax.set_zorder(other.get_zorder() + 1)
    ax.patch.set_visible(False)
    lines = line1 + line2
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plot(a.input, a.out)
