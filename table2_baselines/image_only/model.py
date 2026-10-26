"""A01：Table 2 Image-only：采用ConvNeXt-Base进行图像分类。"""

from torch import nn

from common.encoders import ImageEncoder, classifier


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image = ImageEncoder(cfg)
        self.cls = classifier(cfg.embed_dim)

    def forward(self, img, nir=None):
        f = self.image(img)
        return self.cls(f), None, None, f

    def forward_features(self, img, nir=None):
        return self.image(img)

    def forward_cam(self, img, nir=None):
        # 在分类前向中保留骨干末层特征的梯度，用于Grad-CAM。
        fmap = self.image.backbone(img)[-1]
        fmap.retain_grad()
        f = self.image.proj(fmap.mean((-2, -1)))
        return self.cls(f), fmap
