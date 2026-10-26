"""A02：Table 2 NIR-only：采用三层一维卷积进行光谱分类。"""

from torch import nn

from common.encoders import NIREncoder, classifier


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nir = NIREncoder(cfg)
        self.cls = classifier(cfg.embed_dim)

    def forward(self, img, nir):
        f = self.nir(nir)
        return self.cls(f), None, None, f

    def forward_features(self, img, nir):
        return self.nir(nir)
