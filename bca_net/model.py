"""BCA-Net教师与学生网络，对应Fig. 1、方法2.6和Table S3。

教师通过VOC投影生成查询，学生使用共享的可学习查询。
图像分支采用两层FPN，两路注意力池化特征与查询相加融合。"""

import timm
import torch
from torch import nn
from torch.nn import functional as F


def attention_pool(q, kv, activation):
    q, kv = q.float(), kv.float()
    score = torch.bmm(q, kv) / q.shape[-1] ** 0.5
    a = F.softmax(score, dim=-1) if activation == "softmax" else torch.sigmoid(score)
    return a, torch.bmm(a, kv.transpose(1, 2)).squeeze(1)


class BCABase(nn.Module):
    def __init__(self, cfg, swapped=False, pretrained=None):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.embed_dim
        self.visual_backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained if pretrained is None else pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        c2, c3, c4 = self.visual_backbone.feature_info.channels()
        self.lat_c4 = nn.Conv2d(c4, self.d, 1)  # 顶层投影参数；两层融合使用下方的lat_c2和lat_c3。
        self.lat_c3 = nn.Conv2d(c3, self.d, 1)
        self.lat_c2 = nn.Conv2d(c2, self.d, 1)
        self.fpn_smooth = nn.Conv2d(self.d, self.d, 3, padding=1)
        self.nir_proj = nn.Linear(1, self.d)
        self.classifier = nn.Sequential(
            nn.Linear(self.d, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 3),
        )
        self.img_activation = "sigmoid" if swapped else "softmax"
        self.nir_activation = "softmax" if swapped else "sigmoid"

    def fuse(self, img, nir, q, capture=False):
        if nir.ndim != 2 or nir.shape[1] != self.cfg.nir_dim:
            raise ValueError("NIR must have the configured 499 ordered features")
        feats = self.visual_backbone(img)
        p3 = self.lat_c3(feats[1])
        p2 = self.lat_c2(feats[0]) + F.interpolate(
            p3, size=feats[0].shape[-2:], mode="bilinear", align_corners=False
        )
        if capture and p2.requires_grad:
            p2.retain_grad()
        kv_img = self.fpn_smooth(p2).flatten(2)
        ai, fi = attention_pool(q, kv_img, self.img_activation)
        kv_nir = self.nir_proj(nir.unsqueeze(-1)).transpose(1, 2)
        an, fn = attention_pool(q, kv_nir, self.nir_activation)
        feat = fi + fn + q.squeeze(1)
        return ai, an, feat, p2


class BCATeacher(BCABase):
    def __init__(self, cfg, swapped=False, pretrained=None):
        super().__init__(cfg, swapped, pretrained)
        self.enose_proj = nn.Linear(4, self.d)

    def forward(self, img, nir, voc):
        if voc.shape[-1] != 4:
            raise ValueError("Teacher requires exactly four VOC features")
        ai, an, feat, _ = self.fuse(img, nir, self.enose_proj(voc).unsqueeze(1))
        return self.classifier(feat), ai, an, feat


class BCAStudent(BCABase):
    def __init__(self, cfg, swapped=False, pretrained=None, mode="multi"):
        super().__init__(cfg, swapped, pretrained)
        if mode not in {"multi", "nir", "image"}:
            raise ValueError(mode)
        self.mode = mode
        self.surrogate_query = nn.Parameter(torch.randn(1, 1, self.d))

    def inputs(self, img, nir):
        return (
            torch.zeros_like(img) if self.mode == "nir" else img,
            torch.zeros_like(nir) if self.mode == "image" else nir,
        )

    def forward(self, img, nir):
        img, nir = self.inputs(img, nir)
        ai, an, feat, _ = self.fuse(img, nir, self.surrogate_query.expand(len(img), -1, -1))
        return self.classifier(feat), ai, an, feat

    def forward_features(self, img, nir):
        img, nir = self.inputs(img, nir)
        return self.fuse(img, nir, self.surrogate_query.expand(len(img), -1, -1))[2]

    def forward_cam(self, img, nir):
        img, nir = self.inputs(img, nir)
        out = self.fuse(img, nir, self.surrogate_query.expand(len(img), -1, -1), capture=True)
        return self.classifier(out[2]), out[3]


def build(cfg, teacher=False, **kwargs):
    return BCATeacher(cfg, **kwargs) if teacher else BCAStudent(cfg, **kwargs)
