"""基线的图像编码器和一维光谱卷积编码器。"""

import timm
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, features_only=True, out_indices=(3,)
        )
        self.proj = nn.Linear(self.backbone.feature_info.channels()[0], cfg.embed_dim)

    def forward(self, x):
        return self.proj(self.backbone(x)[-1].mean((-2, -1)))


class NIREncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, cfg.embed_dim)

    def forward(self, x):
        return self.proj(self.enc(x.unsqueeze(1)).squeeze(-1))


def classifier(dim):
    return nn.Sequential(
        nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 3)
    )
