"""CORAL、Mixup及知识蒸馏损失。"""

import numpy as np
import torch
from torch.nn import functional as F


def coral(source, target):
    source, target = source.float(), target.float()
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] != target.shape[1]
        or min(len(source), len(target)) < 2
    ):
        raise ValueError(
            "CORAL needs two 2-D batches with same feature dimension and >=2 samples each"
        )
    s = source - source.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)
    cs = s.T @ s / (len(s) - 1)
    ct = t.T @ t / (len(t) - 1)
    # 协方差元素平方差取均值，再除以4d²。
    return (cs - ct).square().mean() / (4 * source.shape[1] ** 2)


def mixup(image, nir, labels, alpha):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    permutation = torch.randperm(len(image), device=image.device)
    return (
        lam * image + (1 - lam) * image[permutation],
        lam * nir + (1 - lam) * nir[permutation],
        labels,
        labels[permutation],
        lam,
        permutation,
    )


def mixed_ce(logits, ya, yb, lam):
    return lam * F.cross_entropy(logits, ya) + (1 - lam) * F.cross_entropy(logits, yb)


def teacher_targets(logits, ai, an, temperature):
    # 教师使用原始输入，输出概率和注意力作为蒸馏目标。
    p = F.softmax(logits.float() / temperature, dim=-1).detach()
    return p, None if ai is None else ai.detach(), None if an is None else an.detach()


def kl_to_prob(student_logits, target_prob, temperature):
    return (
        F.kl_div(
            F.log_softmax(student_logits.float() / temperature, dim=-1),
            target_prob.float(),
            reduction="batchmean",
        )
        * temperature**2
    )
