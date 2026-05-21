import argparse
import json
import os
import time
from datetime import datetime

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader

from lightmnet3_train_ablation_common import LEVIRCDDataset, load_model


@torch.no_grad()
def compute_metrics_full(model, loader, device):
    model.eval()
    amp_enabled = (device.type == "cuda")

    tp = fp = fn = tn = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_start = time.time()

    for img_a, img_b, label in loader:
        img_a = img_a.to(device, non_blocking=True, memory_format=torch.channels_last)
        img_b = img_b.to(device, non_blocking=True, memory_format=torch.channels_last)
        label = label.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            pred = model(img_a, img_b)

        pred_map = torch.argmax(pred, dim=1, keepdim=True).float()
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


@torch.no_grad()
def infer_single(model, transform, device, img_a, img_b):
    height, width = img_a.shape[:2]
    augmented = transform(image=img_a, image_b=img_b, mask=np.zeros((height, width), dtype=np.uint8))
    a_tensor = augmented["image"].unsqueeze(0).to(device, memory_format=torch.channels_last)
    b_tensor = augmented["image_b"].unsqueeze(0).to(device, memory_format=torch.channels_last)
    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        logits = model(a_tensor, b_tensor)
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred * 255


def save_binary_masks(model, data_root, split, save_dir, num_images, device):
    model.eval()
    transform = A.Compose([A.Normalize(), ToTensorV2()], additional_targets={"image_b": "image"})
    os.makedirs(save_dir, exist_ok=True)

    root = os.path.join(data_root, split)
    a_dir = os.path.join(root, "A")
    b_dir = os.path.join(root, "B")

    all_names = sorted(os.listdir(a_dir))
    selected = all_names[:num_images]

    for name in selected:
        a_path = os.path.join(a_dir, name)
        b_path = os.path.join(b_dir, name)
        if not (os.path.exists(a_path) and os.path.exists(b_path)):
            continue

        img_a = np.array(Image.open(a_path).convert("RGB"))
        img_b = np.array(Image.open(b_path).convert("RGB"))
        mask = infer_single(model, transform, device, img_a, img_b)

        stem = os.path.splitext(name)[0]
        out_path = os.path.join(save_dir, f"{stem}_pred.png")
        Image.fromarray(mask, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default=r"E:\pyCharmProjects\lightMnet\train2.pth")
    parser.add_argument("--data_root", type=str, default=r"E:\pyCharmProjects\LEVIR-CD")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_masks", type=int, default=16)
    parser.add_argument("--mask_save_dir", type=str, default=r"E:\pyCharmProjects\lightMnet\train2_masks_levircd_test")
    parser.add_argument("--output_path", type=str, default=r"E:\pyCharmProjects\lightMnet\ablation_runs\test_eval\train2_levircd_metrics.json")
    args = parser.parse_args()

    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight not found: {args.weight_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Weight: {args.weight_path}", flush=True)
    print(f"Data root: {args.data_root}, split: {args.split}", flush=True)

    transform = A.Compose([A.Normalize(), ToTensorV2()], additional_targets={"image_b": "image"})
    dataset = LEVIRCDDataset(args.data_root, split=args.split, transform=transform)
    print(f"Test samples: {len(dataset)}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    model = load_model("lightmnet3", pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)
    state_dict = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully", flush=True)

    metrics = compute_metrics_full(model, loader, device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "data_root": args.data_root,
        "split": args.split,
        "num_samples": len(dataset),
        "weight_path": os.path.abspath(args.weight_path),
        "metrics": metrics,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n========== Test Metrics ==========", flush=True)
    print(f"  OA        : {metrics['oa']:.6f}", flush=True)
    print(f"  Precision : {metrics['precision']:.6f}", flush=True)
    print(f"  Recall    : {metrics['recall']:.6f}", flush=True)
    print(f"  F1        : {metrics['f1']:.6f}", flush=True)
    print(f"  IoU       : {metrics['iou']:.6f}", flush=True)
    print(f"  Infer Speed: {metrics['infer_speed_img_s']:.2f} img/s", flush=True)
    print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}", flush=True)
    print(f"==================================", flush=True)

    print(f"\nSaving binary masks (first {args.num_masks} images) to: {args.mask_save_dir}", flush=True)
    save_binary_masks(model, args.data_root, args.split, args.mask_save_dir, args.num_masks, device)

    print(f"\nMetrics saved: {args.output_path}", flush=True)
    print(f"Masks saved to: {args.mask_save_dir}", flush=True)


if __name__ == "__main__":
    main()
