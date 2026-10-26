"""Table 4：部分域适应和五折留一品种验证。

部分适应从B1学生初始化，联合使用B2/B3适应子集。
LOCO每折从B1/B2学生初始化，在其余品种上微调。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from common.engine import load_model, optimizer, predict, step, tensors
from common.identifiers import result_ids
from common.losses import coral, mixed_ce, mixup
from common.metrics import classification
from common.runtime import (
    device_for,
    fingerprint,
    json_write,
    prepare_output,
    save_checkpoint,
    seed_all,
    sha256,
)


def partial_adaptation(source_path, dm, cfg, seed, out, batches=("B2", "B3"), reuse=False):
    """联合B2 Adapt70与B3 Adapt300进行适应。batches指定参与适应的批次。"""
    device = device_for(cfg)
    seed_all(seed, cfg)
    model, parent = load_model(source_path, dm, cfg, device)
    if (
        parent["model_task"] != "bca_source"
        or parent["meta"]["target_train_ids"]
        or parent["meta"]["seed"] != seed
    ):
        raise ValueError("Partial must start from same-seed B1-only BCA source model")
    ids = np.concatenate([dm.target_splits[b][0] for b in batches]).astype(int)
    hold = np.concatenate([dm.target_splits[b][1] for b in batches]).astype(int)
    if set(ids) & set(hold):
        raise RuntimeError("Adapt/Holdout leakage")
    name = "partial_" + "_".join(batches)
    meta = fingerprint(cfg, dm, seed, name, {"source_sha256": sha256(source_path)})
    meta.update(
        {
            "phase": "partial",
            "source_ids": dm.ids(dm.splits["train"]),
            "target_train_ids": dm.ids(ids),
            "holdout_ids": dm.ids(hold),
            "parent_task": parent["meta"]["task"],
            "selection": "fixed_final_adaptation_epoch",
        }
    )
    out = Path(out)
    if not prepare_output(out, meta, reuse):
        return out / "best.pt"
    src = dm.loader(dm.splits["train"], True, seed=seed)
    tar = dm.loader(ids, True, True, seed + 1000)
    opt = optimizer(model, cfg, cfg.adapt_lr, cfg.adapt_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.adapt_epochs, eta_min=cfg.min_lr
    )
    history = []
    for epoch in range(1, cfg.adapt_epochs + 1):
        model.train()
        # 冻结BN运行均值与方差，更新仿射参数。
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
        it = iter(tar)
        losses = []
        for batch in src:
            try:
                tb = next(it)
            except StopIteration:
                it = iter(tar)
                tb = next(it)
            im, nir, _, y = tensors(batch, device)
            ti, tn, _, ty = tensors(tb, device)
            if not (ty == -1).all():
                raise RuntimeError("Unlabeled target loader exposed labels")
            opt.zero_grad(set_to_none=True)
            logits, _, _, _ = model(im, nir)
            sf = model.forward_features(im, nir)
            tf = model.forward_features(ti, tn)
            # 适应损失：源域CE + lambda × CORAL。
            loss = F.cross_entropy(logits, y) + cfg.coral_weight * coral(sf, tf)
            step(loss, model, opt, cfg)
            losses.append(float(loss.detach()))
        sched.step()
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(
            f"{name} seed={seed} epoch={epoch}/{cfg.adapt_epochs} loss={np.mean(losses):.5f}",
            flush=True,
        )
    save_checkpoint(
        model,
        out / "best.pt",
        meta,
        model_task="bca_source",
        best_epoch=cfg.adapt_epochs,
        preprocessing=dm.metadata(),
        selection="final_epoch_not_holdout_selected",
    )
    metrics = {}
    frames = []
    for key, ix in [
        ("B1-Test", dm.splits["test"]),
        *[(b + "-Holdout", dm.target_splits[b][1]) for b in batches],
    ]:
        m, p = predict(model, dm.loader(ix), device)
        m["evaluated_samples_used_unlabeled"] = False
        m["data_card_ids"] = result_ids(name, key)
        p["data_card_ids"] = ";".join(m["data_card_ids"])
        metrics[key] = m
        p["evaluation"] = key
        frames.append(p)
    pd.concat(frames).to_csv(out / "predictions.csv", index=False)
    json_write(
        {"task": name, "seed": seed, "synthetic": dm.synthetic, "metrics": metrics},
        out / "metrics.json",
    )
    json_write(history, out / "history.json")
    json_write({"checkpoint_sha256": sha256(out / "best.pt")}, out / "COMPLETED.json")
    return out / "best.pt"


def loco5(initial_path, dm, cfg, seed, out, reuse=False):
    if any(x is None for x in [cfg.loco_epochs, cfg.loco_lr, cfg.loco_weight_decay]):
        raise ValueError(
            "Incomplete LOCO configuration: set loco_epochs, loco_lr and loco_weight_decay"
        )
    if cfg.loco_epochs < 1 or cfg.loco_lr <= 0 or cfg.loco_weight_decay < 0:
        raise ValueError("Invalid LOCO settings")
    device = device_for(cfg)
    initial, parent = load_model(initial_path, dm, cfg, device)
    if parent["model_task"] != "bca_full" or parent["meta"]["seed"] != seed:
        raise ValueError("LOCO must initialize from same-seed B1/B2 trained bimodal student")
    used_ids = set(parent["meta"]["source_ids"]) | set(parent["meta"]["target_train_ids"])
    if used_ids & set(dm.ids(dm.batch["B3"])):
        raise ValueError("LOCO initialization contains B3 training samples")
    del initial
    meta = fingerprint(cfg, dm, seed, "loco5", {"initial_sha256": sha256(initial_path)})
    meta.update(
        {
            "phase": "loco5",
            "folds": ["DX", "LM", "MMK", "PLBZ", "XHL"],
            "always_in_training": "XCX",
            "macro_labels": ["H", "L"],
            "selection": "B1-Val only",
            "train_policy": "B1 train + five B3 cultivars",
        }
    )
    out = Path(out)
    if not prepare_output(out, meta, reuse):
        return out / "metrics.json"
    pooled = []
    for cultivar in meta["folds"]:
        seed_all(seed, cfg)
        model, parent = load_model(initial_path, dm, cfg, device)
        b3 = dm.batch["B3"]
        held = b3[dm.frame.iloc[b3].cultivar.to_numpy() == cultivar]
        donor = b3[dm.frame.iloc[b3].cultivar.to_numpy() != cultivar]
        train = np.concatenate([dm.splits["train"], donor]).astype(int)
        if not {0, 1, 2}.issubset(set(dm.y[train])):
            raise RuntimeError("LOCO training must include all three classes")
        fold = out / cultivar
        fold.mkdir()
        fm = dict(
            meta,
            held_cultivar=cultivar,
            source_ids=dm.ids(train),
            target_train_ids=parent["meta"]["target_train_ids"],
            held_ids=dm.ids(held),
        )
        opt = optimizer(model, cfg, cfg.loco_lr, cfg.loco_weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cfg.loco_epochs, eta_min=cfg.min_lr
        )
        tr = dm.loader(train, True, seed=seed)
        va = dm.loader(dm.splits["val"])
        best = -1.0
        bestce = float("inf")
        wait = 0
        history = []
        for epoch in range(1, cfg.loco_epochs + 1):
            model.train()
            losses = []
            for batch in tr:
                im, nir, _, y = tensors(batch, device)
                mi, mn, ya, yb, lam, _ = mixup(im, nir, y, cfg.mixup_alpha)
                opt.zero_grad(set_to_none=True)
                loss = mixed_ce(model(mi, mn)[0], ya, yb, lam)
                step(loss, model, opt, cfg)
                losses.append(float(loss.detach()))
            sch.step()
            v, _ = predict(model, va, device)
            if v["macro_f1"] > best + 1e-8 or (
                abs(v["macro_f1"] - best) <= 1e-8 and v["ce"] < bestce
            ):
                best, bestce = v["macro_f1"], v["ce"]
                wait = 0
                save_checkpoint(
                    model,
                    fold / "best.pt",
                    fm,
                    model_task="bca_full",
                    best_epoch=epoch,
                    preprocessing=dm.metadata(),
                )
            else:
                wait += 1
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": v})
            if wait >= cfg.patience:
                break
        model, _ = load_model(fold / "best.pt", dm, cfg, device)
        _, pred = predict(model, dm.loader(held), device)
        pred["fold"] = cultivar
        pred["evaluation"] = "B3-LOCO5"
        pred["data_card_ids"] = "C11"
        pooled.append(pred)
        pred.to_csv(fold / "predictions.csv", index=False)
        json_write(history, fold / "history.json")
        del model
    pred = pd.concat(pooled, ignore_index=True)
    if pred.sample_id.duplicated().any() or set(pred.cultivar) != set(meta["folds"]):
        raise RuntimeError("Invalid pooled OOF predictions")
    m = classification(pred.true, pred.pred, loco=True)
    pred.to_csv(out / "predictions.csv", index=False)
    m["data_card_ids"] = result_ids("loco5", "B3-LOCO5")
    json_write(
        {"task": "loco5", "seed": seed, "synthetic": dm.synthetic, "metrics": {"B3-LOCO5": m}},
        out / "metrics.json",
    )
    json_write({"pooled_n": len(pred)}, out / "COMPLETED.json")
    return out / "metrics.json"
