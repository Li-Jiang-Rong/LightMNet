import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# =========================================================
# 无参数度量增强（论文原版）
# =========================================================
class ParameterFreeMetricEnhance(nn.Module):
    """
    基于余弦相似度的无参数差异度量
    输出:
        diff_features:
            4层差异图
            shape = [B,1,H,W]
    """

    def forward(self, feat_t1_list, feat_t2_list):

        diff_features = []

        for f1, f2 in zip(feat_t1_list, feat_t2_list):

            B, C, H, W = f1.shape

            f1_flat = f1.view(B, C, -1)
            f2_flat = f2.view(B, C, -1)

            f1_norm = F.normalize(f1_flat, p=2, dim=1)
            f2_norm = F.normalize(f2_flat, p=2, dim=1)

            cos_sim = (f1_norm * f2_norm).sum(dim=1)

            diff = 1.0 - cos_sim
            diff = diff.view(B, 1, H, W)

            diff_features.append(diff)

        return diff_features


# =========================================================
# 稳健性语义交叉融合（论文版）
# concat + lightweight conv
# =========================================================
class RobustSemanticCrossFusion(nn.Module):

    def __init__(self, in_channels=256):

        super().__init__()

        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + 1, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(4)
        ])

        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, base_feat, diff_feats):

        base_h, base_w = base_feat.shape[2:]

        outputs = []

        for i, diff in enumerate(diff_feats):

            if diff.shape[2:] != (base_h, base_w):

                diff = F.interpolate(
                    diff,
                    size=(base_h, base_w),
                    mode='bilinear',
                    align_corners=False
                )

            fusion = torch.cat([base_feat, diff], dim=1)

            fusion = self.fusion_blocks[i](fusion)

            outputs.append(fusion)

        out = torch.cat(outputs, dim=1)

        return self.final_fusion(out)


# =========================================================
# 轻量级混合注意力（论文版）
# 多尺度 + 分组卷积
# =========================================================
class LightweightHybridAttention(nn.Module):

    def __init__(self, in_ch=256):

        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                64,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                64,
                kernel_size=5,
                padding=2,
                groups=4,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                64,
                kernel_size=7,
                padding=3,
                groups=8,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                64,
                kernel_size=9,
                padding=4,
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 2, 1)
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)

        out = self.fusion(out)

        return out


# =========================================================
# LightMNet（论文第四章原版风格）
# =========================================================
class LightMNet(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        # -------------------------------------------------
        # ResNet18 Backbone
        # -------------------------------------------------
        resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1   # 64
        self.layer2 = resnet.layer2   # 128
        self.layer3 = resnet.layer3   # 256
        self.layer4 = resnet.layer4   # 512

        # -------------------------------------------------
        # 冻结 layer0~layer2
        # 解冻 layer3/layer4
        # -------------------------------------------------
        for name, param in self.named_parameters():

            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # -------------------------------------------------
        # 无参数度量增强
        # -------------------------------------------------
        self.metric_enhance = ParameterFreeMetricEnhance()

        # -------------------------------------------------
        # 基础语义投影
        # -------------------------------------------------
        self.prior_conv = nn.ModuleList([
            nn.Conv2d(64, 256, 1),
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(512, 256, 1),
        ])

        # -------------------------------------------------
        # 稳健性语义交叉融合
        # -------------------------------------------------
        self.cross_fusion = RobustSemanticCrossFusion(
            in_channels=256
        )

        # -------------------------------------------------
        # 轻量级混合注意力
        # -------------------------------------------------
        self.hybrid_attention = LightweightHybridAttention(
            in_ch=256
        )

    # =====================================================
    # 特征提取
    # =====================================================
    def extract_features(self, x):

        x0 = self.layer0(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, img_t1, img_t2):

        # -------------------------------------------------
        # 双时相特征提取
        # -------------------------------------------------
        feat_t1 = self.extract_features(img_t1)
        feat_t2 = self.extract_features(img_t2)

        # -------------------------------------------------
        # 无参数差异度量
        # 输出:
        # [B,1,H,W]
        # -------------------------------------------------
        diff_feats = self.metric_enhance(
            feat_t1,
            feat_t2
        )

        # -------------------------------------------------
        # 构建基础语义聚合特征
        # 统一到 layer1 尺寸
        # -------------------------------------------------
        base_h, base_w = feat_t1[0].shape[2:]

        base_agg = None

        for i, (f1, f2) in enumerate(zip(feat_t1, feat_t2)):

            prior = self.prior_conv[i](f1 + f2)

            if prior.shape[2:] != (base_h, base_w):

                prior = F.interpolate(
                    prior,
                    size=(base_h, base_w),
                    mode='bilinear',
                    align_corners=False
                )

            if base_agg is None:
                base_agg = prior
            else:
                base_agg = base_agg + prior

        # -------------------------------------------------
        # 稳健性语义交叉融合
        # -------------------------------------------------
        fused = self.cross_fusion(
            base_agg,
            diff_feats
        )

        # -------------------------------------------------
        # 轻量级混合注意力解码
        # -------------------------------------------------
        out = self.hybrid_attention(fused)

        # -------------------------------------------------
        # 上采样回原图大小
        # -------------------------------------------------
        out = F.interpolate(
            out,
            size=img_t1.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return out


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":

    model = LightMNet(pretrained=False)
    print(model)
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)

    y = model(x1, x2)

    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")