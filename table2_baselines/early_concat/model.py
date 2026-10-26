"""A03：Table 2 Early concat：拼接图像与NIR特征后分类。"""

import torch
from torch import nn

from common.encoders import ImageEncoder, NIREncoder, classifier


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.image = ImageEncoder(cfg)
        self.nir = NIREncoder(cfg)
        self.fuse = nn.Sequential(
            nn.Linear(d * 2, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(0.4)
        )
        self.cls = classifier(d)

    def forward(self, img, nir):
        f = self.forward_features(img, nir)
        return self.cls(f), None, None, f

    def forward_features(self, img, nir):
        return self.fuse(torch.cat([self.image(img), self.nir(nir)], 1))
