import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from lightmnet3 import (
    RobustSemanticCrossFusion,
    LightweightHybridAttention,
)


class LightMNet(nn.Module):
    """
    Ablation variant without the parameter-free metric module.
    The semantic fusion and attention decoder are kept unchanged.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for name, param in self.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.prior_conv = nn.ModuleList([
            nn.Conv2d(64, 256, 1),
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(512, 256, 1),
        ])

        self.cross_fusion = RobustSemanticCrossFusion(in_channels=256)
        self.hybrid_attention = LightweightHybridAttention(in_ch=256)

    def extract_features(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]

    @staticmethod
    def build_zero_diff_features(features):
        return [
            torch.zeros(
                feat.shape[0],
                1,
                feat.shape[2],
                feat.shape[3],
                device=feat.device,
                dtype=feat.dtype,
            )
            for feat in features
        ]

    def forward(self, img_t1, img_t2):
        feat_t1 = self.extract_features(img_t1)
        feat_t2 = self.extract_features(img_t2)

        diff_feats = self.build_zero_diff_features(feat_t1)

        base_h, base_w = feat_t1[0].shape[2:]
        base_agg = None

        for i, (f1, f2) in enumerate(zip(feat_t1, feat_t2)):
            prior = self.prior_conv[i](f1 + f2)

            if prior.shape[2:] != (base_h, base_w):
                prior = F.interpolate(
                    prior,
                    size=(base_h, base_w),
                    mode="bilinear",
                    align_corners=False,
                )

            if base_agg is None:
                base_agg = prior
            else:
                base_agg = base_agg + prior

        fused = self.cross_fusion(base_agg, diff_feats)
        out = self.hybrid_attention(fused)

        out = F.interpolate(
            out,
            size=img_t1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return out
