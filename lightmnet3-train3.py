import os
import time
import random
import sys
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from lightmnet3 import LightMNet


# =========================================================
# 忽略 warning
# =========================================================
warnings.filterwarnings("ignore")


# =========================================================
# CPU线程优化（Windows非常重要）
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)


# =========================================================
# 随机种子
# =========================================================
def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    # 速度优化关键
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================================================
# Dataset
# =========================================================
class LEVIRCDDataset(Dataset):

    def __init__(self, root, split='train', transform=None):

        self.root = os.path.join(root, split)

        self.transform = transform
        self.label_dir = self._resolve_label_dir()

        self.img_names = sorted(
            os.listdir(os.path.join(self.root, 'A'))
        )

        b_set = set(os.listdir(os.path.join(self.root, 'B')))
        l_set = set(os.listdir(self.label_dir))

        self.img_names = [
            n for n in self.img_names
            if n in b_set and n in l_set
        ]

    def __len__(self):
        return len(self.img_names)

    def _resolve_label_dir(self):

        for dir_name in ('label', 'OUT', 'out', 'mask', 'masks'):
            candidate = os.path.join(self.root, dir_name)
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(
            f'No label directory found under: {self.root}'
        )

    def __getitem__(self, idx):

        name = self.img_names[idx]

        img_a = np.array(
            Image.open(
                os.path.join(self.root, 'A', name)
            ).convert('RGB')
        )

        img_b = np.array(
            Image.open(
                os.path.join(self.root, 'B', name)
            ).convert('RGB')
        )

        label = np.array(
            Image.open(
                os.path.join(self.label_dir, name)
            ).convert('L')
        )

        label = (label > 127).astype(np.uint8)

        if self.transform:

            augmented = self.transform(
                image=img_a,
                image_b=img_b,
                mask=label
            )

            img_a = augmented['image']
            img_b = augmented['image_b']
            label = augmented['mask']

        label = label.unsqueeze(0).float()

        return img_a, img_b, label


# =========================================================
# Dice + CE Loss
# =========================================================
class DiceCELoss(nn.Module):

    def __init__(self, pos_weight=4.0, smooth=1e-5):

        super().__init__()

        # 类别不平衡关键
        self.register_buffer(
            'class_weights',
            torch.tensor([1.0, pos_weight], dtype=torch.float32)
        )

        self.smooth = smooth

    def forward(self, pred, target):

        target = target.squeeze(1).long()

        ce_loss = F.cross_entropy(
            pred,
            target,
            weight=self.class_weights
        )

        prob = torch.softmax(pred, dim=1)[:, 1]

        target_float = target.float()

        inter = (prob * target_float).sum(dim=(1, 2))

        union = (
            prob.sum(dim=(1, 2))
            + target_float.sum(dim=(1, 2))
        )

        dice = 1 - (
            2 * inter + self.smooth
        ) / (
            union + self.smooth
        )

        dice = dice.mean()

        return ce_loss + dice


# =========================================================
# 冻结BN
# =========================================================
def freeze_bn(module):

    if isinstance(module, nn.BatchNorm2d):

        module.eval()

        for param in module.parameters():
            param.requires_grad = False


# =========================================================
# Metrics + 推理速度
# =========================================================
@torch.no_grad()
def compute_metrics(model, loader, device):

    model.eval()
    amp_enabled = (device.type == 'cuda')
    progress_enabled = sys.stdout.isatty()

    tp = fp = fn = tn = 0

    # ================= 推理速度统计 =================
    if device.type == 'cuda':
        torch.cuda.synchronize()

    infer_start = time.time()

    for img_a, img_b, label in tqdm(
            loader,
            desc='Metrics',
            leave=False,
            disable=not progress_enabled
    ):

        img_a = img_a.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last
        )

        img_b = img_b.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last
        )

        label = label.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type='cuda',
            enabled=amp_enabled
        ):

            pred = model(img_a, img_b)

        pred_map = torch.argmax(
            pred,
            dim=1,
            keepdim=True
        ).float()

        label = (label > 0.5).float()

        tp += (pred_map * label).sum().item()

        fp += (pred_map * (1 - label)).sum().item()

        fn += ((1 - pred_map) * label).sum().item()

        tn += (
            (1 - pred_map) * (1 - label)
        ).sum().item()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    infer_end = time.time()

    infer_time = infer_end - infer_start

    infer_speed = len(loader.dataset) / (infer_time + 1e-8)

    precision = tp / (tp + fp + 1e-7)

    recall = tp / (tp + fn + 1e-7)

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-7)
    )

    oa = (
        (tp + tn)
        / (tp + fp + fn + tn + 1e-7)
    )

    return precision, recall, f1, oa, infer_speed


# =========================================================
# Main
# =========================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        type=str,
        default=r'E:/pyCharmProjects/CDD/Real/subset'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='best_lightmnet.pth'
    )
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    parser.add_argument('--resume_path', type=str, default='best_lightmnet_tuned.pth')
    parser.add_argument('--target_f1', type=float, default=1)
    args = parser.parse_args()

    seed_everything(42)

    # =====================================================
    # Path
    # =====================================================
    data_root = args.data_root

    save_path = args.save_path

    # =====================================================
    # Device
    # =====================================================
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f'Device: {device}')

    # =====================================================
    # Hyper Parameters
    # =====================================================
    batch_size = args.batch_size

    epochs = args.epochs

    lr = args.lr

    weight_decay = args.weight_decay

    num_workers = args.num_workers

    # =====================================================
    # Transform
    # =====================================================
    train_transform = A.Compose([

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.Normalize(),

        ToTensorV2()

    ], additional_targets={'image_b': 'image'})

    val_transform = A.Compose([

        A.Normalize(),

        ToTensorV2()

    ], additional_targets={'image_b': 'image'})

    # =====================================================
    # Dataset
    # =====================================================
    train_set = LEVIRCDDataset(
        data_root,
        split='train',
        transform=train_transform
    )

    val_set = LEVIRCDDataset(
        data_root,
        split='val',
        transform=val_transform
    )

    # =====================================================
    # DataLoader
    # =====================================================
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # =====================================================
    # Model
    # =====================================================
    model = LightMNet(
        pretrained=True
    ).to(device)

    # channels_last加速
    model = model.to(
        memory_format=torch.channels_last
    )

    if args.resume_path:
        state_dict = torch.load(
            args.resume_path,
            map_location=device
        )
        model.load_state_dict(state_dict)
        print(f'Resumed from: {args.resume_path}')

    # =====================================================
    # 冻结layer1/layer2
    # =====================================================
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.prior_conv.parameters():
        param.requires_grad = True

    for param in model.cross_fusion.parameters():
        param.requires_grad = True

    for param in model.hybrid_attention.parameters():
        param.requires_grad = True

    # 冻结BN
    model.apply(freeze_bn)

    # =====================================================
    # 参数量统计
    # =====================================================
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f'Total Params      : {total_params / 1e6:.2f} M')

    print(f'Trainable Params  : {trainable_params / 1e6:.2f} M')

    # =====================================================
    # Optimizer
    # =====================================================
    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lr,
        weight_decay=weight_decay
    )

    # =====================================================
    # Loss
    # =====================================================
    criterion = DiceCELoss(
        pos_weight=args.pos_weight
    ).to(device)

    # =====================================================
    # Scheduler
    # =====================================================
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # =====================================================
    # AMP
    # =====================================================
    amp_enabled = (device.type == 'cuda')
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=amp_enabled
    )
    progress_enabled = sys.stdout.isatty()

    # =====================================================
    # Train
    # =====================================================
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):

        # =================================================
        # Train
        # =================================================
        model.train()

        total_loss = 0.0

        # ================= 训练速度统计 =================
        if device.type == 'cuda':
            torch.cuda.synchronize()

        train_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch:03d}/{epochs}',
            disable=not progress_enabled
        )

        for img_a, img_b, label in pbar:

            img_a = img_a.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last
            )

            img_b = img_b.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last
            )

            label = label.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            # =============================================
            # AMP
            # =============================================
            with torch.amp.autocast(
                device_type='cuda',
                enabled=amp_enabled
            ):

                pred = model(img_a, img_b)

                loss = criterion(pred, label)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()

            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                lr=f'{optimizer.param_groups[0]["lr"]:.6f}'
            )

        if device.type == 'cuda':
            torch.cuda.synchronize()

        train_end = time.time()

        train_time = train_end - train_start

        train_speed = len(train_loader.dataset) / (train_time + 1e-8)

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # =================================================
        # Metrics + 推理速度
        # =================================================
        precision, recall, f1, oa, infer_speed = compute_metrics(
            model,
            val_loader,
            device
        )

        print(
            f'\nEpoch {epoch:03d}'
            f' | Train Loss: {avg_loss:.4f}'
            f' | Precision: {precision:.4f}'
            f' | Recall: {recall:.4f}'
            f' | F1: {f1:.4f}'
            f' | OA: {oa:.4f}'
            f' | Train Speed: {train_speed:.2f} img/s'
            f' | Infer Speed: {infer_speed:.2f} img/s'
        )

        # =================================================
        # Save Best
        # =================================================
        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f'>>> New Best F1: '
                f'{best_f1:.4f}'
            )

            if best_f1 >= args.target_f1:
                print(
                    f'>>> Target F1 reached: '
                    f'{best_f1:.4f} >= {args.target_f1:.4f}'
                )
                break

    print(
        f'\nTraining Finished!'
        f'\nBest F1: {best_f1:.4f}'
    )


# =========================================================
# Run
# =========================================================
if __name__ == '__main__':

    main()
