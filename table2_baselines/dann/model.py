"""A08：Table2 DANN：与EarlyConcat相同特征提取器，附加GRL及域判别器。"""

import torch
from torch import nn

from table2_baselines.early_concat.model import Model as EarlyConcat


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


class Model(EarlyConcat):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.domain_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def domain(self, features, weight):
        return self.domain_head(Reverse.apply(features, weight)).squeeze(1)
