import argparse
import json
import os
import time
from datetime import datetime

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from lightmnet3_train_ablation_common import LEVIRCDDataset, load_model


def resolve_cdd_root(root):
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, "train")):
        return root
    candidate = os.path.join(root, "Real", "subset")
    if os.path.isdir(os.path.join(candidate, "train")):
        return candidate
    return root


def build_loader(data_root, split, batch_size, num_workers, pin_memory):
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ], additional_targets={"image_b": "image"})

    dataset = LEVIRCDDataset(
        data_root,
        split=split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    return dataset, loader


@torch.no_grad()
def measure_infer_speed(model, loader, device):
    model.eval()
    amp_enabled = (device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for img_a, img_b, _ in loader:
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
        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            pred = model(img_a, img_b)
        _ = torch.argmax(pred, dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    speed = len(loader.dataset) / (elapsed + 1e-8)

    return round(float(speed), 2), round(float(elapsed), 2)


@torch.no_grad()
def compute_metrics_full(model, loader, device):
    model.eval()
    amp_enabled = (device.type == "cuda")

    tp = fp = fn = tn = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_start = time.time()

    for img_a, img_b, label in loader:
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
    iou = tp / (tp + fp + fn + 1e-7)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "oa": round(float(oa), 6),
        "iou": round(float(iou), 6),
        "infer_speed_img_s": round(float(infer_speed), 2),
        "infer_time_s": round(float(infer_time), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--model_module", type=str, default="lightmnet3")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"e:/pyCharmProjects/lightMnet/ablation_runs/test_eval/weight_train_val_speed.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight not found: {args.weight_path}")

    data_root = resolve_cdd_root(args.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    model = load_model(args.model_module, pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)
    state_dict = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(state_dict)

    train_set, train_loader = build_loader(
        data_root,
        "train",
        args.batch_size,
        args.num_workers,
        pin_memory,
    )
    val_set, val_loader = build_loader(
        data_root,
        "val",
        args.batch_size,
        args.num_workers,
        pin_memory,
    )
    test_set, test_loader = build_loader(
        data_root,
        "test",
        args.batch_size,
        args.num_workers,
        pin_memory,
    )

    train_speed, train_time = measure_infer_speed(model, train_loader, device)
    val_speed, val_time = measure_infer_speed(model, val_loader, device)
    test_metrics = compute_metrics_full(model, test_loader, device)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "data_root_input": args.data_root,
        "data_root_resolved": data_root,
        "weight_path": os.path.abspath(args.weight_path),
        "model_module": args.model_module,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "train": {
            "num_samples": len(train_set),
            "infer_speed_img_s": train_speed,
            "infer_time_s": train_time,
        },
        "val": {
            "num_samples": len(val_set),
            "infer_speed_img_s": val_speed,
            "infer_time_s": val_time,
        },
        "test": {
            "num_samples": len(test_set),
            "metrics": test_metrics,
        },
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Train Infer Speed: {train_speed:.2f} img/s | Val Infer Speed: {val_speed:.2f} img/s",
        flush=True,
    )
    print(
        f"Test Precision: {test_metrics['precision']:.6f}"
        f" | Recall: {test_metrics['recall']:.6f}"
        f" | F1: {test_metrics['f1']:.6f}"
        f" | OA: {test_metrics['oa']:.6f}"
        f" | IoU: {test_metrics['iou']:.6f}"
        f" | Test Infer Speed: {test_metrics['infer_speed_img_s']:.2f} img/s",
        flush=True,
    )
    print(f"Saved: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

