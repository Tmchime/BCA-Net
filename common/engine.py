"""各实验共用的训练、验证与评估流程。
实验配置由common.registry中的Task定义。
使用FP32训练，以B1验证集macro-F1选择模型，并将结果写入output_dir。"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from common.identifiers import result_ids
from common.losses import coral, kl_to_prob, mixed_ce, mixup, teacher_targets
from common.metrics import classification
from common.registry import TASKS, build_model
from common.runtime import (
    device_for,
    fingerprint,
    json_write,
    prepare_output,
    read_checkpoint,
    save_checkpoint,
    seed_all,
    sha256,
)


def optimizer(model, cfg, lr=None, weight_decay=None):
    base, other = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (base if "backbone" in name else other).append(p)
    rate = cfg.lr if lr is None else lr
    groups = []
    if base:
        groups.append({"params": base, "lr": rate * cfg.backbone_lr_mult})
    if other:
        groups.append({"params": other, "lr": rate})
    return torch.optim.AdamW(
        groups, weight_decay=cfg.weight_decay if weight_decay is None else weight_decay
    )


def tensors(batch, device):
    return tuple(batch[k].to(device, non_blocking=True) for k in ["image", "nir", "voc", "label"])


def forward(model, batch, device, teacher=False):
    img, nir, voc, y = tensors(batch, device)
    return (model(img, nir, voc) if teacher else model(img, nir)), y


@torch.inference_mode()
def predict(model, loader, device, teacher=False):
    model.eval()
    rows = []
    ce = 0.0
    n = 0
    for batch in loader:
        bs = len(batch["label"])
        chunk = bs
        start = 0
        while start < bs:
            part = {
                k: (v[start : start + chunk] if torch.is_tensor(v) or isinstance(v, list) else v)
                for k, v in batch.items()
            }
            try:
                out, y = forward(model, part, device, teacher)
            except torch.OutOfMemoryError:
                if device.type != "cuda" or chunk <= 1:
                    raise
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                continue
            logits = out[0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite prediction")
            if (y < 0).any():
                raise ValueError("Evaluation requires labeled samples")
            ce += float(F.cross_entropy(logits, y, reduction="sum"))
            n += len(y)
            p = logits.softmax(1).cpu().numpy()
            yt = y.cpu().numpy()
            for j in range(len(y)):
                rows.append(
                    {
                        "sample_id": part["sample_id"][j],
                        "batch": part["batch"][j],
                        "cultivar": part["cultivar"][j],
                        "site": part["site"][j],
                        "true": int(yt[j]),
                        "pred": int(p[j].argmax()),
                        "p_H": float(p[j, 0]),
                        "p_M": float(p[j, 1]),
                        "p_L": float(p[j, 2]),
                    }
                )
            start += len(y)
    df = pd.DataFrame(rows)
    m = classification(df.true, df.pred)
    m["ce"] = ce / n
    return m, df


def step(loss, model, opt, cfg):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
    opt.step()


def load_model(path, dm, cfg, device):
    obj = read_checkpoint(path)
    if obj["meta"]["data_sha256"] != dm.data_hash:
        raise ValueError("Checkpoint data hash mismatch")
    if obj["meta"]["config"] != cfg.dict():
        raise ValueError("Checkpoint configuration mismatch")
    if (
        obj["meta"]["code_sha256"]
        != fingerprint(cfg, dm, obj["meta"]["seed"], obj["meta"]["task"], obj["meta"]["parents"])[
            "code_sha256"
        ]
    ):
        raise ValueError("Checkpoint source code hash mismatch")
    model = build_model(obj["model_task"], cfg, pretrained=False).to(device)
    model.load_state_dict(obj["state_dict"], strict=True)
    return model, obj


def fit(
    task, dm, cfg, seed, out, teacher_path=None, target_batches=(), reuse=False, experiment=None
):
    spec = TASKS[task]
    device = device_for(cfg)
    seed_all(seed, cfg)
    parents = {}
    if spec.kd:
        if teacher_path is None:
            raise ValueError("KD requires an explicit teacher checkpoint")
        parents["teacher_sha256"] = sha256(teacher_path)
    target_ids = (
        np.concatenate([dm.batch[b] for b in target_batches]).astype(int)
        if target_batches
        else np.array([], dtype=int)
    )
    if (spec.coral or spec.dann) != bool(len(target_ids)):
        raise ValueError("Target loader configuration mismatch for the selected experiment")
    meta = fingerprint(cfg, dm, seed, experiment or task, parents)
    meta.update(
        {
            "phase": "transductive" if len(target_ids) else "source_only",
            "source_ids": dm.ids(dm.splits["train"]),
            "target_train_ids": dm.ids(target_ids),
            "teacher_architecture": f"4-{cfg.embed_dim} Linear" if spec.teacher else None,
        }
    )
    out = Path(out)
    if not prepare_output(out, meta, reuse):
        return out / "best.pt"
    json_write(dm.metadata(), out / "data_protocol.json")
    model = build_model(task, cfg).to(device)
    teacher = None
    if spec.kd:
        teacher, tobj = load_model(teacher_path, dm, cfg, device)
        if (
            not TASKS[tobj["model_task"]].teacher
            or tobj["meta"]["seed"] != seed
            or tobj["meta"]["target_train_ids"]
        ):
            raise ValueError("Teacher must be same-seed, B1-only teacher")
        if TASKS[tobj["model_task"]].swapped != spec.swapped:
            raise ValueError("Teacher/student attention activation mismatch")
        teacher.eval()
        teacher.requires_grad_(False)
    tr = dm.loader(dm.splits["train"], True, seed=seed)
    va = dm.loader(dm.splits["val"])
    target = dm.loader(target_ids, True, True, seed + 1000) if len(target_ids) else None
    opt = optimizer(model, cfg)
    epochs = cfg.teacher_epochs if spec.teacher else cfg.student_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=epochs, eta_min=cfg.min_lr
    )
    best_f1 = -1.0
    best_ce = float("inf")
    noimp = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        terms = []
        target_iter = iter(target) if target is not None else None
        for bi, batch in enumerate(tr):
            img, nir, voc, y = tensors(batch, device)
            opt.zero_grad(set_to_none=True)
            if spec.teacher:
                loss = F.cross_entropy(model(img, nir, voc)[0], y)
                record = {"ce": float(loss.detach())}
            else:
                mi, mn, ya, yb, lam, _ = mixup(img, nir, y, cfg.mixup_alpha)
                slog, sai, san, _ = model(mi, mn)
                ce = mixed_ce(slog, ya, yb, lam)
                loss = (cfg.ce_weight if spec.kd else 1.0) * ce
                record = {"ce": float(ce.detach())}
                if spec.kd:
                    with torch.no_grad():
                        tl, tai, tan, _ = teacher(img, nir, voc)
                    prob, ai, an = teacher_targets(tl, tai, tan, cfg.temperature)
                    if spec.kl:
                        kl = kl_to_prob(slog, prob, cfg.temperature)
                        loss = loss + cfg.kl_weight * kl
                        record["kl"] = float(kl.detach())
                    if spec.mse_img:
                        mse = F.mse_loss(sai.float(), ai.float())
                        loss = loss + cfg.attention_weight * mse
                        record["mse_img"] = float(mse.detach())
                    if spec.mse_nir:
                        mse = F.mse_loss(san.float(), an.float())
                        loss = loss + cfg.attention_weight * mse
                        record["mse_nir"] = float(mse.detach())
                if target_iter is not None:
                    try:
                        tb = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target)
                        tb = next(target_iter)
                    ti, tn, _, ty = tensors(tb, device)
                    if not (ty == -1).all():
                        raise RuntimeError("Target labels must be hidden")
                    sf = model.forward_features(img, nir)
                    tf = model.forward_features(ti, tn)
                    if spec.coral:
                        cor = coral(sf, tf)
                        loss = loss + cfg.coral_weight * cor
                        record["coral"] = float(cor.detach())
                    if spec.dann:
                        p = ((epoch - 1) * len(tr) + bi) / max(1, epochs * len(tr) - 1)
                        grl = 2 / (1 + math.exp(-10 * p)) - 1
                        ds = model.domain(sf, grl)
                        dt = model.domain(tf, grl)
                        domain = 0.5 * (
                            F.binary_cross_entropy_with_logits(ds, torch.zeros_like(ds))
                            + F.binary_cross_entropy_with_logits(dt, torch.ones_like(dt))
                        )
                        loss = loss + domain
                        record["domain"] = float(domain.detach())
            record["total"] = float(loss.detach())
            step(loss, model, opt, cfg)
            terms.append(record)
        scheduler.step()
        vm, _ = predict(model, va, device, spec.teacher)
        improved = vm["macro_f1"] > best_f1 + 1e-8 or (
            abs(vm["macro_f1"] - best_f1) <= 1e-8 and vm["ce"] < best_ce
        )
        if improved:
            best_f1, best_ce = vm["macro_f1"], vm["ce"]
            noimp = 0
            save_checkpoint(
                model,
                out / "best.pt",
                meta,
                model_task=task,
                best_epoch=epoch,
                preprocessing=dm.metadata(),
                validation=vm,
            )
        else:
            noimp += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": {
                    k: float(np.mean([t.get(k, 0) for t in terms])) for k in set().union(*terms)
                },
                "validation": vm,
            }
        )
        json_write(history, out / "history.json")
        print(
            f"{task} seed={seed} epoch={epoch}/{epochs} B1-Val F1={vm['macro_f1']:.4f} CE={vm['ce']:.4f}",
            flush=True,
        )
        if not spec.teacher and noimp >= cfg.patience:
            break
    del model
    model, obj = load_model(out / "best.pt", dm, cfg, device)
    evaluate(model, obj, dm, cfg, out, teacher=spec.teacher)
    json_write(
        {"checkpoint_sha256": sha256(out / "best.pt"), "best_epoch": obj["best_epoch"]},
        out / "COMPLETED.json",
    )
    return out / "best.pt"


def evaluate(model, obj, dm, cfg, out, teacher=False):
    out = Path(out)
    device = device_for(cfg)
    metrics = {}
    frames = []
    scopes = {"B1-Test": dm.splits["test"]}
    if not teacher:
        scopes.update({"B2-All": dm.batch["B2"], "B3-All": dm.batch["B3"]})
    exposed = set(obj["meta"]["target_train_ids"])
    for name, ids in scopes.items():
        m, p = predict(model, dm.loader(ids), device, teacher)
        m["data_card_ids"] = result_ids(obj["meta"]["task"], name)
        p["data_card_ids"] = ";".join(m["data_card_ids"])
        p["evaluation"] = name
        m["evaluated_samples_used_unlabeled"] = bool(set(dm.ids(ids)) & exposed)
        metrics[name] = m
        frames.append(p)
    pd.concat(frames).to_csv(out / "predictions.csv", index=False)
    json_write(
        {
            "task": obj["meta"]["task"],
            "seed": obj["meta"]["seed"],
            "synthetic": dm.synthetic,
            "metrics": metrics,
        },
        out / "metrics.json",
    )
