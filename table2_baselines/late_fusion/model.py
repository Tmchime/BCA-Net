"""A04：Table 2 Late fusion：图像与NIR分类logits等权平均。"""

import torch
from torch import nn

from common.encoders import ImageEncoder, NIREncoder


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.image = ImageEncoder(cfg)
        self.nir = NIREncoder(cfg)

        def head():
            return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 3))

        self.img_cls = head()
        self.nir_cls = head()

    def forward(self, img, nir):
        fi, fn = self.image(img), self.nir(nir)
        return 0.5 * (self.img_cls(fi) + self.nir_cls(fn)), None, None, torch.cat([fi, fn], 1)

    def forward_features(self, img, nir):
        return torch.cat([self.image(img), self.nir(nir)], 1)
