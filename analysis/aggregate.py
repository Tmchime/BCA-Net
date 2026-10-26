"""汇总三个随机种子的指标，计算均值、样本标准差和Welch检验。"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

from common.identifiers import result_ids


def aggregate(root, out, allow_synthetic=False):
    rows = []
    for path in Path(root).glob("seed_*/*/metrics.json"):
        if not (path.parent / "COMPLETED.json").exists():
            raise ValueError(f"Incomplete task: {path.parent}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        task = obj["task"]
        if obj["synthetic"] and not allow_synthetic:
            raise ValueError("Synthetic results: pass --allow-synthetic to summarize test runs")
        for scope, m in obj["metrics"].items():
            rows.append(
                {
                    "experiment": task,
                    "synthetic": bool(obj["synthetic"]),
                    "seed": obj["seed"],
                    "evaluation": scope,
                    "data_card_ids": ";".join(result_ids(task, scope)),
                    **{
                        k: m.get(k)
                        for k in [
                            "n",
                            "accuracy",
                            "macro_precision",
                            "macro_recall",
                            "macro_f1",
                            "kappa",
                        ]
                    },
                }
            )
    if not rows:
        raise ValueError("No completed metrics found")
    data = pd.DataFrame(rows)
    if data.duplicated(["experiment", "seed", "evaluation"]).any():
        raise ValueError("Duplicate task/seed/evaluation")
    outputs = []
    for (task, scope), g in data.groupby(["experiment", "evaluation"]):
        if set(g.seed) != {6, 7, 42} or len(g) != 3:
            raise ValueError(f"Need exactly seeds 6/7/42 for {task}/{scope}")
        if g["synthetic"].nunique() != 1:
            raise ValueError(f"Mixed real and synthetic results: {task}/{scope}")
        if g.n.nunique() != 1:
            raise ValueError("Evaluation sample count differs between seeds")
        r = {
            "experiment": task,
            "evaluation": scope,
            "n": int(g.n.iloc[0]),
            "n_seeds": 3,
            "synthetic": bool(g["synthetic"].iloc[0]),
        }
        r["data_card_ids"] = ";".join(result_ids(task, scope))
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "kappa"]:
            r[metric + "_mean"] = float(g[metric].mean())
            r[metric + "_sd"] = float(g[metric].std(ddof=1))
        outputs.append(r)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "seed_metrics.csv", index=False)
    pd.DataFrame(outputs).to_csv(out / "mean_sd.csv", index=False)
    tests = []
    for baseline in ["deepcoral", "dann"]:
        a = data[(data.experiment == "bca_full") & (data.evaluation == "B2-All")].macro_f1
        b = data[(data.experiment == baseline) & (data.evaluation == "B2-All")].macro_f1
        if len(a) == len(b) == 3:
            t, p = ttest_ind(a, b, equal_var=False)
            tests.append(
                {
                    "baseline": baseline,
                    "baseline_id": ";".join(result_ids(baseline, "B2-All")),
                    "reference_ids": ";".join(result_ids("bca_full", "B2-All")),
                    "test": "Welch independent two-sample t-test across seeds",
                    "t": float(t),
                    "p": float(p),
                    "mean_difference": float(a.mean() - b.mean()),
                }
            )
    pd.DataFrame(tests).to_csv(out / "welch_tests.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-synthetic", action="store_true")
    a = p.parse_args()
    aggregate(a.runs, a.out, a.allow_synthetic)
