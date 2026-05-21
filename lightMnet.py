import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from PIL import Image
import random

# -------------------- 基础模块（保持不变） --------------------
def cosine_similarity(x1, x2, eps=1e-8):
    dot = (x1 * x2).sum(dim=1)
    norm1 = x1.norm(p=2, dim=1)
    norm2 = x2.norm(p=2, dim=1)
    cos = dot / (norm1 * norm2 + eps)
    return cos

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N, _ = query.shape
        Q = self.q_proj(query).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        return out

class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim, key_dim, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(key_dim)
        self.key_proj = nn.Linear(key_dim, d_model)
        self.value_proj = nn.Linear(key_dim, d_model)
        self.attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, base_feat, metric_feat):
        B, Cq, H, W = base_feat.shape
        base_flat = base_feat.flatten(2).transpose(1, 2)
        metric_flat = metric_feat.flatten(2).transpose(1, 2)

        q = self.norm_q(base_flat)
        k = self.key_proj(self.norm_kv(metric_flat))
        v = self.value_proj(metric_flat)

        attn_out = self.attn(q, k, v) + base_flat
        attn_out = self.norm_out(attn_out)
        out = self.mlp(attn_out) + attn_out
        out = out.transpose(1, 2).view(B, -1, H, W)
        return out

class PyramidSplitAttention(nn.Module):
    def __init__(self, in_channels, kernels=[3,5,7,9], groups=[1,4,8,16]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k, g in zip(kernels, groups):
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=g, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.eca = ECALayer(in_channels * len(kernels))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        concat = torch.cat(feats, dim=1)
        weights = self.eca(concat)
        B, C_total, H, W = concat.shape
        group_size = C_total // len(self.branches)
        weights = weights.view(B, len(self.branches), group_size, H, W)
        weights = self.softmax(weights)
        weights = weights.view(B, C_total, H, W)
        out = concat * weights
        out = out.view(B, len(self.branches), group_size, H, W).sum(dim=1)
        return out

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels, scales=[1,2,4]):
        super().__init__()
        self.scales = scales
        self.query_conv = nn.Conv2d(in_channels, in_channels//2, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//2, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        outputs = []
        for scale in self.scales:
            if scale != 1:
                scaled_x = F.adaptive_avg_pool2d(x, (H//scale, W//scale))
            else:
                scaled_x = x
            N = scaled_x.shape[2] * scaled_x.shape[3]
            Q = self.query_conv(scaled_x).view(B, -1, N).permute(0, 2, 1)
            K = self.key_conv(scaled_x).view(B, -1, N)
            V = self.value_conv(scaled_x).view(B, -1, N).permute(0, 2, 1)
            attn = F.softmax(torch.bmm(Q, K), dim=-1)
            out = torch.bmm(attn, V)
            out = out.transpose(1, 2).contiguous().view_as(scaled_x)
            if scale != 1:
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            outputs.append(out)
        out = sum(outputs) / len(outputs)
        return x + self.gamma * out

# ---------- 修改后的 LightMNet ----------
class LightMNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, freeze_backbone=True, fusion_size=32):
        super().__init__()
        self.fusion_size = fusion_size  # 固定融合分辨率
        # 骨干网络
        if backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            self.encoder_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            self.channels = [64, 128, 256, 512]
        else:
            raise ValueError("仅支持 resnet18")

        # ---------- 冻结整个骨干 ----------
        if freeze_backbone:
            # 完全冻结浅层
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
            # layer3、layer4 解冻
        else:
            # 全部可训练
            pass

        # ---------- 基础特征聚合 ----------
        self.agg_conv = nn.Sequential(
            nn.Conv2d(sum(self.channels), 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ---------- 稳健性语义交叉融合 ----------
        self.cross_fusions = nn.ModuleList()
        for ch in self.channels:
            self.cross_fusions.append(
                CrossAttentionFusion(
                    query_dim=256,
                    key_dim=ch,
                    d_model=256,
                    n_heads=8,
                    dropout=0.1
                )
            )
        self.fusion_out = nn.Conv2d(256 * len(self.channels), 256, 1)

        # ---------- 轻量级混合注意力 ----------
        self.pyramid_attn = PyramidSplitAttention(256)
        self.spatial_attn = MultiScaleSpatialAttention(256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)
        )

    def extract_features(self, x):
        feats = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        for layer_name in self.encoder_layers:
            x = getattr(self.backbone, layer_name)(x)
            feats.append(x)
        return feats

    def forward(self, img1, img2):
        # 1. 多尺度特征提取
        feats1 = self.extract_features(img1)
        feats2 = self.extract_features(img2)

        # 2. 无参数度量差异特征（扩散为原特征形状）
        diff_list = []
        for f1, f2 in zip(feats1, feats2):
            diff = 1.0 - cosine_similarity(f1, f2)   # [B, H, W]
            diff = diff.unsqueeze(1).repeat(1, f1.shape[1], 1, 1)  # [B, C, H, W]
            diff_list.append(diff)

        # 3. 所有特征统一到融合分辨率 (32x32)
        ref_size = (self.fusion_size, self.fusion_size)  # (32,32)
        agg_feats = []
        for f1, f2 in zip(feats1, feats2):
            fused = f1 + f2
            fused = F.interpolate(fused, size=ref_size, mode='bilinear', align_corners=False)
            agg_feats.append(fused)
        fused_concat = torch.cat(agg_feats, dim=1)        # [B, 960, 32, 32]
        base_agg = self.agg_conv(fused_concat)            # [B, 256, 32, 32]

        # 4. 交叉融合（均在 32x32 上进行）
        metric_list = [F.interpolate(m, size=ref_size, mode='bilinear', align_corners=False) for m in diff_list]
        cross_outs = []
        for i, fusion in enumerate(self.cross_fusions):
            out = fusion(base_agg, metric_list[i])        # [B, 256, 32, 32]
            cross_outs.append(out)

        cross_concat = torch.cat(cross_outs, dim=1)
        encoder_feat = self.fusion_out(cross_concat)     # [B, 256, 32, 32]

        # 5. 混合注意力（32x32）
        feat = self.pyramid_attn(encoder_feat)
        feat = self.spatial_attn(feat)                   # [B, 256, 32, 32]

        # 6. 分类并上采样到原图尺寸
        out = self.classifier(feat)                      # [B, 2, 32, 32]
        out = F.interpolate(out, size=img1.shape[2:], mode='bilinear', align_corners=False)
        return out

# -------------------- 损失函数（不变） --------------------
# -------------------- 损失函数 (改进：Focal Loss + Dice) --------------------
class FocalLoss(nn.Module):
    """Focal Loss 用于缓解类别不平衡中的假阳性问题"""
    def __init__(self, alpha=0.75, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class FocalDiceLoss(nn.Module):
    """Dice Loss + Focal Loss，不再使用固定类别权重"""
    def __init__(self, dice_weight=0.3, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.focal = FocalLoss(alpha=0.75, gamma=2.5)

    def forward(self, pred, target):
        # Dice Loss
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=2).permute(0,3,1,2).float()
        intersection = (pred_soft * target_one_hot).sum(dim=[2,3])
        union = (pred_soft + target_one_hot).sum(dim=[2,3])
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # Focal Loss
        focal_loss = self.focal(pred, target)

        return self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss

def compute_class_weights(dataset, device, num_classes=2):
    class_counts = torch.zeros(num_classes)
    for _, _, label in dataset:
        label = label.long()
        class_counts[0] += (label == 0).sum()
        class_counts[1] += (label == 1).sum()
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    return weights.to(device)

# -------------------- 数据集（不变） --------------------
class ChangeDetectionDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.split = split
        self.root = os.path.join(root, split)
        self.imgA_dir = os.path.join(self.root, 'A')
        self.imgB_dir = os.path.join(self.root, 'B')
        self.label_dir = os.path.join(self.root, 'label')
        self.filenames = sorted(os.listdir(self.imgA_dir))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        imgA = Image.open(os.path.join(self.imgA_dir, name)).convert('RGB')
        imgB = Image.open(os.path.join(self.imgB_dir, name)).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, name)).convert('L')

        if self.split == 'train':
            if random.random() > 0.5:
                imgA = transforms.functional.hflip(imgA)
                imgB = transforms.functional.hflip(imgB)
                label = transforms.functional.hflip(label)
            if random.random() > 0.5:
                imgA = transforms.functional.vflip(imgA)
                imgB = transforms.functional.vflip(imgB)
                label = transforms.functional.vflip(label)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                imgA = transforms.functional.rotate(imgA, angle, interpolation=Image.BILINEAR)
                imgB = transforms.functional.rotate(imgB, angle, interpolation=Image.BILINEAR)
                label = transforms.functional.rotate(label, angle, interpolation=Image.NEAREST)

        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        label = self.label_transform(label) * 255.0
        label = (label > 128).long().squeeze(0)
        return imgA, imgB, label

# -------------------- 训练与评估（添加速度统计） --------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for img1, img2, target in loader:
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(img1, img2)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()

    tp, fp, fn, tn = 0, 0, 0, 0
    for img1, img2, target in loader:
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        pred = model(img1, img2)
        pred_label = pred.argmax(dim=1)
        tp += ((pred_label == 1) & (target == 1)).sum().item()
        fp += ((pred_label == 1) & (target == 0)).sum().item()
        fn += ((pred_label == 0) & (target == 1)).sum().item()
        tn += ((pred_label == 0) & (target == 0)).sum().item()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.time()

    num_samples = len(loader.dataset)
    infer_speed = num_samples / (t_end - t_start + 1e-8)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    oa = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return precision, recall, f1, oa, infer_speed

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='E:/pyCharmProjects/LEVIR-CD')
    parser.add_argument('--batch_size', type=int, default=8)   # 提高 batch size 可能更充分利用 GPU
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fusion_size', type=int, default=32)  # 控制融合分辨率
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_set = ChangeDetectionDataset(args.data_root, 'train')
    val_set = ChangeDetectionDataset(args.data_root, 'val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_weights = compute_class_weights(train_set, device)
    print(f"Class weights: {class_weights.tolist()}")

    model = LightMNet(backbone='resnet18', pretrained=True, freeze_backbone=True, fusion_size=args.fusion_size).to(device)
    trainable_params = count_trainable_params(model)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalDiceLoss(dice_weight=0.3)

    best_f1 = 0
    for epoch in range(1, args.epochs + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_time = time.time() - t0
        train_speed = len(train_set) / (train_time + 1e-8)

        prec, rec, f1, oa, infer_speed = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Prec: {prec:.4f} Rec: {rec:.4f} F1: {f1:.4f} OA: {oa:.4f} | "
              f"Train: {train_speed:.2f} pairs/s | Infer: {infer_speed:.2f} pairs/s")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_lightmnet.pth')
            print("  best model saved.")

    print(f"Training finished. Best F1: {best_f1:.4f}")