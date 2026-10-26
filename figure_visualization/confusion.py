"""读取逐样本预测并绘制混淆矩阵。"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--evaluation", choices=["B1-Test", "B2-All"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = pd.read_csv(a.predictions)
    d = d[d.evaluation == a.evaluation]
    if d.empty or d.sample_id.duplicated().any():
        raise ValueError("Missing or duplicate selected evaluation predictions")
    cm = confusion_matrix(d.true, d.pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=["H", "M", "L"],
        yticklabels=["H", "M", "L"],
        xlabel="Predicted",
        ylabel="True",
        title=a.evaluation,
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
