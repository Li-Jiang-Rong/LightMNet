import argparse
import importlib
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class LEVIRCDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.label_dir = self._resolve_label_dir()

        self.img_names = sorted(os.listdir(os.path.join(self.root, "A")))
        b_set = set(os.listdir(os.path.join(self.root, "B")))
        l_set = set(os.listdir(self.label_dir))

        self.img_names = [
            name for name in self.img_names
            if name in b_set and name in l_set
        ]

    def __len__(self):
        return len(self.img_names)

    def _resolve_label_dir(self):
        for dir_name in ("label", "OUT", "out", "mask", "masks"):
            candidate = os.path.join(self.root, dir_name)
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(
            f"No label directory found under: {self.root}"
        )

    def __getitem__(self, idx):
        name = self.img_names[idx]

        img_a = np.array(
            Image.open(os.path.join(self.root, "A", name)).convert("RGB")
        )
        img_b = np.array(
            Image.open(os.path.join(self.root, "B", name)).convert("RGB")
        )
        label = np.array(
            Image.open(os.path.join(self.label_dir, name)).convert("L")
        )

        label = (label > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(
                image=img_a,
                image_b=img_b,
                mask=label,
            )
            img_a = augmented["image"]
            img_b = augmented["image_b"]
            label = augmented["mask"]

        label = label.unsqueeze(0).float()
        return img_a, img_b, label


class DiceCELoss(nn.Module):
    def __init__(self, pos_weight=3.0, smooth=1e-5):
        super().__init__()
        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, pos_weight], dtype=torch.float32),
        )
        self.smooth = smooth

    def forward(self, pred, target):
        target = target.squeeze(1).long()

        ce_loss = F.cross_entropy(
            pred,
            target,
            weight=self.class_weights,
        )

        prob = torch.softmax(pred, dim=1)[:, 1]
        target_float = target.float()

        inter = (prob * target_float).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))

        dice = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        return ce_loss + dice.mean()


def freeze_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False


@torch.no_grad()
def compute_metrics(model, loader, device):
    model.eval()
    amp_enabled = (device.type == "cuda")
    progress_enabled = sys.stdout.isatty()

    tp = fp = fn = tn = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_start = time.time()

    for img_a, img_b, label in tqdm(
        loader,
        desc="Metrics",
        leave=False,
        disable=not progress_enabled,
    ):
        img_a = img_a.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        img_b = img_b.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        label = label.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            pred = model(img_a, img_b)

        pred_map = torch.argmax(
            pred,
            dim=1,
            keepdim=True,
        ).float()

        label = (label > 0.5).float()

        tp += (pred_map * label).sum().item()
        fp += (pred_map * (1 - label)).sum().item()
        fn += ((1 - pred_map) * label).sum().item()
        tn += ((1 - pred_map) * (1 - label)).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.time() - infer_start
    infer_speed = len(loader.dataset) / (infer_time + 1e-8)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    oa = (tp + tn) / (tp + fp + fn + tn + 1e-7)

    return precision, recall, f1, oa, infer_speed


def save_result_json(result_path, payload):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(model_module_name, pretrained=True):
    module = importlib.import_module(model_module_name)
    model_class = getattr(module, "LightMNet")
    return model_class(pretrained=pretrained)


def run_training(args):
    seed_everything(args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Experiment: {args.experiment_name}", flush=True)
    print(f"Dataset Root: {args.data_root}", flush=True)
    print(f"Model Module: {args.model_module}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Start Time: {datetime.now().isoformat(timespec='seconds')}", flush=True)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ], additional_targets={"image_b": "image"})

    val_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ], additional_targets={"image_b": "image"})

    train_set = LEVIRCDDataset(
        args.data_root,
        split="train",
        transform=train_transform,
    )
    val_set = LEVIRCDDataset(
        args.data_root,
        split="val",
        transform=val_transform,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        **loader_kwargs,
    )

    model = load_model(
        args.model_module,
        pretrained=args.pretrained_backbone,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Resumed from: {args.resume_path}", flush=True)

    for param in model.parameters():
        param.requires_grad = False

    for layer_name in ("layer3", "layer4", "prior_conv", "cross_fusion", "hybrid_attention"):
        if hasattr(model, layer_name):
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = True

    model.apply(freeze_bn)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total Params      : {total_params / 1e6:.2f} M", flush=True)
    print(f"Trainable Params  : {trainable_params / 1e6:.2f} M", flush=True)
    print(f"Train Samples     : {len(train_set)}", flush=True)
    print(f"Val Samples       : {len(val_set)}", flush=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = DiceCELoss(pos_weight=args.pos_weight).to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    amp_enabled = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    progress_enabled = sys.stdout.isatty()

    best_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    last_metrics = {}
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs}",
            disable=not progress_enabled,
        )

        for img_a, img_b, label in pbar:
            img_a = img_a.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            img_b = img_b.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled,
            ):
                pred = model(img_a, img_b)
                loss = criterion(pred, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - train_start
        train_speed = len(train_loader.dataset) / (train_time + 1e-8)

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        precision, recall, f1, oa, infer_speed = compute_metrics(
            model,
            val_loader,
            device,
        )

        last_metrics = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "oa": round(oa, 4),
            "train_speed_img_s": round(train_speed, 2),
            "infer_speed_img_s": round(infer_speed, 2),
        }

        print(
            f"Epoch {epoch:03d}"
            f" | Train Loss: {avg_loss:.4f}"
            f" | Precision: {precision:.4f}"
            f" | Recall: {recall:.4f}"
            f" | F1: {f1:.4f}"
            f" | OA: {oa:.4f}"
            f" | Train Speed: {train_speed:.2f} img/s"
            f" | Infer Speed: {infer_speed:.2f} img/s"
        , flush=True)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_metrics = last_metrics.copy()

            torch.save(model.state_dict(), args.save_path)
            print(f">>> New Best F1: {best_f1:.4f}", flush=True)

            if best_f1 >= args.target_f1:
                stop_reason = "target_f1_reached"
                print(
                    f">>> Target F1 reached: "
                    f"{best_f1:.4f} >= {args.target_f1:.4f}"
                , flush=True)
                break

    result_payload = {
        "experiment_name": args.experiment_name,
        "dataset_root": args.data_root,
        "model_module": args.model_module,
        "device": str(device),
        "save_path": os.path.abspath(args.save_path),
        "result_path": os.path.abspath(args.result_path),
        "best_epoch": best_epoch,
        "best_f1": round(best_f1, 4),
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "stop_reason": stop_reason,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_result_json(args.result_path, result_payload)

    print("Training Finished!", flush=True)
    print(f"Best F1: {best_f1:.4f}", flush=True)
    print(f"Best Epoch: {best_epoch}", flush=True)
    print(f"Result Saved: {args.result_path}", flush=True)


def build_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=defaults["experiment_name"])
    parser.add_argument("--model_module", type=str, default=defaults["model_module"])
    parser.add_argument("--data_root", type=str, default=defaults["data_root"])
    parser.add_argument("--save_path", type=str, default=defaults["save_path"])
    parser.add_argument("--result_path", type=str, default=defaults["result_path"])
    parser.add_argument("--resume_path", type=str, default=defaults.get("resume_path", ""))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 4))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 5e-5))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 5e-3))
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 2))
    parser.add_argument("--pos_weight", type=float, default=defaults.get("pos_weight", 3.0))
    parser.add_argument("--target_f1", type=float, default=defaults.get("target_f1", 1.0))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument(
        "--pretrained_backbone",
        action="store_true",
        default=defaults.get("pretrained_backbone", True),
    )
    parser.add_argument(
        "--no_pretrained_backbone",
        action="store_false",
        dest="pretrained_backbone",
    )
    return parser


def main_with_defaults(defaults):
    args = build_parser(defaults).parse_args()
    run_training(args)
