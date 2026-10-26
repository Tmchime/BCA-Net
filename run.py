"""训练入口。使用--experiment选择实验，--config指定配置文件。"""

import argparse
from pathlib import Path

from common.config import load_config
from common.data import DataModule
from common.engine import fit
from common.identifiers import ALIASES, resolve_experiment
from common.registry import TASKS
from table4_protocols.protocols import loco5, partial_adaptation


def parser():
    p = argparse.ArgumentParser(description="BCA-Net training and evaluation")
    p.add_argument("--config", default=str(Path(__file__).parent / "configs/paper.json"))
    p.add_argument(
        "--experiment",
        choices=[
            *ALIASES,
            *TASKS,
            "transductive_B2_B3",
            "partial_B2",
            "partial_B2_B3",
            "loco5",
            "table2",
            "table3",
            "table4",
            "all",
        ],
    )
    p.add_argument("--seed", type=int, choices=[6, 7, 42])
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--check-data", action="store_true")
    return p


def main(argv=None, default_experiment=None):
    p = parser()
    args = p.parse_args(argv)
    args.experiment = resolve_experiment(args.experiment or default_experiment)
    if not args.experiment and not args.check_data:
        p.print_help()
        return
    cfg = load_config(args.config)
    dm = DataModule(cfg)
    if args.check_data:
        print(
            "Data OK. B1 split:",
            {k: len(v) for k, v in dm.splits.items()},
            "synthetic=",
            dm.synthetic,
        )
        return
    table2 = [
        "image_only",
        "nir_only",
        "early_concat",
        "late_fusion",
        "cross_modal_transformer",
        "vanilla_kd",
        "deepcoral",
        "dann",
        "bca_source",
        "bca_full",
    ]
    table3 = [
        "cross_attention_only",
        "bca_source",
        "without_all_kd",
        "without_kl",
        "without_mse_img",
        "without_mse_nir",
        "activation_swap",
        "bca_full",
    ]
    table4 = [
        "teacher",
        "bca_source",
        "bca_full",
        "student_nir",
        "student_image",
        "transductive_B2_B3",
        "partial_B2_B3",
        "loco5",
    ]
    selections = {
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "all": list(dict.fromkeys(table2 + table3 + table4)),
    }
    tasks = selections.get(args.experiment, [args.experiment])
    # 训练前检查LOCO配置。
    if "loco5" in tasks and any(
        x is None for x in [cfg.loco_epochs, cfg.loco_lr, cfg.loco_weight_decay]
    ):
        raise ValueError(
            "Incomplete LOCO configuration: set loco_epochs, loco_lr and loco_weight_decay"
        )
    for seed in [args.seed] if args.seed else cfg.seeds:
        run_seed(tasks, dm, cfg, seed, args.reuse)


def run_seed(tasks, dm, cfg, seed, reuse):
    root = Path(cfg.output_dir) / f"seed_{seed}"
    done = {}

    def ensure(task):
        if task in done:
            return done[task]
        spec = TASKS[task]
        teacher = None
        if spec.kd:
            teacher = ensure("teacher_swapped" if spec.swapped else "teacher")
        ckpt = fit(
            task,
            dm,
            cfg,
            seed,
            root / task,
            teacher_path=teacher,
            target_batches=("B2",) if (spec.coral or spec.dann) else (),
            reuse=reuse or (root / task / "COMPLETED.json").exists(),
        )
        done[task] = ckpt
        return ckpt

    for task in tasks:
        if task == "transductive_B2_B3":
            fit(
                "bca_full",
                dm,
                cfg,
                seed,
                root / task,
                teacher_path=ensure("teacher"),
                target_batches=("B2", "B3"),
                reuse=reuse,
                experiment=task,
            )
        elif task in {"partial_B2", "partial_B2_B3"}:
            partial_adaptation(
                ensure("bca_source"),
                dm,
                cfg,
                seed,
                root / task,
                batches=("B2",) if task == "partial_B2" else ("B2", "B3"),
                reuse=reuse,
            )
        elif task == "loco5":
            loco5(ensure("bca_full"), dm, cfg, seed, root / task, reuse=reuse)
        else:
            ensure(task)


if __name__ == "__main__":
    main()
