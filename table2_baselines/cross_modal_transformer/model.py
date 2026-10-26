"""Table 2 Cross-modal Transformer：两层双向跨模态注意力，每层8头。"""

import timm
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.i1 = nn.LayerNorm(d)
        self.n1 = nn.LayerNorm(d)
        self.iatt = nn.MultiheadAttention(d, 8, 0.1, batch_first=True)
        self.natt = nn.MultiheadAttention(d, 8, 0.1, batch_first=True)
        self.i2 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

        def ff():
            return nn.Sequential(
                nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 4, d)
            )

        self.iff = ff()
        self.nff = ff()

    def forward(self, i, n):
        a, b = self.i1(i), self.n1(n)
        i = i + self.iatt(a, b, b, need_weights=False)[0]
        n = n + self.natt(b, a, a, need_weights=False)[0]
        return i + self.iff(self.i2(i)), n + self.nff(self.n2(n))


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, features_only=True, out_indices=(3,)
        )
        self.ip = nn.Linear(self.backbone.feature_info.channels()[0], d)
        self.np = nn.Conv1d(1, d, 7, stride=4, padding=3)
        self.blocks = nn.ModuleList([Block(d), Block(d)])
        self.head = nn.Sequential(
            nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.4), nn.Linear(d, 3)
        )

    def forward_features(self, img, nir):
        i = self.ip(self.backbone(img)[-1].flatten(2).transpose(1, 2))
        n = self.np(nir.unsqueeze(1)).transpose(1, 2)
        for block in self.blocks:
            i, n = block(i, n)
        return torch.cat([i.mean(1), n.mean(1)], 1)

    def forward(self, img, nir):
        f = self.forward_features(img, nir)
        return self.head(f), None, None, f
