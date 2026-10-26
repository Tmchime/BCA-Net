"""t-SNE散点、2σ协方差椭圆和边缘核密度曲线。"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot(folder, out, perplexity=30):
    folder = Path(folder)
    x = np.load(folder / "features.npy", allow_pickle=False)
    meta = pd.read_csv(folder / "samples.csv")
    if (
        x.ndim != 2
        or len(x) != len(meta)
        or not np.isfinite(x).all()
        or meta.sample_id.duplicated().any()
    ):
        raise ValueError("Invalid feature export")
    if not set(meta.label).issubset({"H", "M", "L"}) or not set(meta.domain).issubset({"B1", "B2"}):
        raise ValueError("Unknown class/domain")
    if not 0 < perplexity < len(x):
        raise ValueError("perplexity must be positive and smaller than n_samples")
    x = StandardScaler().fit_transform(x)
    x = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=42).fit_transform(x)
    xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
        n_jobs=1,
    ).fit_transform(x)
    fig = plt.figure(figsize=(9, 9))
    grid = fig.add_gridspec(4, 4, wspace=0.08, hspace=0.08)
    ax = fig.add_subplot(grid[1:4, 0:3])
    top = fig.add_subplot(grid[0, 0:3], sharex=ax)
    right = fig.add_subplot(grid[1:4, 3], sharey=ax)
    palette = {"H": "#d62728", "M": "#2ca02c", "L": "#1f77b4"}
    for cls, color in palette.items():
        z = xy[meta.label == cls]
        if len(z) >= 3:
            val, vec = np.linalg.eigh(np.cov(z.T))
            order = val.argsort()[::-1]
            val = val[order]
            vec = vec[:, order]
            angle = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
            w, h = 4 * np.sqrt(np.clip(val, 0, None))
            ax.add_patch(
                Ellipse(
                    z.mean(0), w, h, angle=angle, fill=False, color=color, linestyle="--", alpha=0.5
                )
            )
        # 各类别在两个坐标轴上的边缘核密度。
        if len(z) >= 2 and np.std(z[:, 0]) > 1e-10:
            sns.kdeplot(
                x=z[:, 0],
                color=color,
                ax=top,
                fill=True,
                alpha=0.25,
                linewidth=1.5,
                warn_singular=False,
            )
        if len(z) >= 2 and np.std(z[:, 1]) > 1e-10:
            sns.kdeplot(
                y=z[:, 1],
                color=color,
                ax=right,
                fill=True,
                alpha=0.25,
                linewidth=1.5,
                warn_singular=False,
            )
    for domain, marker in [("B2", "^"), ("B1", "o")]:
        for cls, color in palette.items():
            mask = (meta.label == cls) & (meta.domain == domain)
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                color=color,
                marker=marker,
                alpha=0.6,
                s=24,
                label=f"{cls} / {domain}",
            )
    ax.set(xlabel="t-SNE component 1", ylabel="t-SNE component 2")
    ax.legend(fontsize=8, loc="best")
    for marginal in [top, right]:
        marginal.set(xlabel="", ylabel="")
        marginal.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in marginal.spines.values():
            spine.set_visible(False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    meta = meta.copy()
    meta["tsne1"] = xy[:, 0]
    meta["tsne2"] = xy[:, 1]
    meta.to_csv(str(out) + ".coordinates.csv", index=False)
    Path(str(out) + ".settings.json").write_text(
        json.dumps(
            {
                "random_state": 42,
                "perplexity": perplexity,
                "ellipse": "2-sigma covariance ellipse",
                "marginals": "Gaussian KDE by class",
                "embedding_scope": "input folder",
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--perplexity", type=float, default=30)
    a = p.parse_args()
    plot(a.input, a.out, a.perplexity)
