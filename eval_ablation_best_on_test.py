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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def build_loader(data_root, split, batch_size, num_workers):
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ], additional_targets={"image_b": "image"})

    dataset = LEVIRCDDataset(
        data_root,
        split=split,
        transform=transform,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    return dataset, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdd_root", type=str, default=r"E:/pyCharmProjects/CDD/Real/subset")
    parser.add_argument("--levircd_root", type=str, default=r"E:/pyCharmProjects/LEVIR-CD")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"e:/pyCharmProjects/lightMnet/ablation_runs/test_eval/ablation_test_metrics.json",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    ensure_dir(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiments = [
        {
            "experiment_name": "CDD-NoParam",
            "data_root": args.cdd_root,
            "model_module": "lightmnet3_ablation_no_param",
            "weight_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/weights/cdd_no_param_best.pth",
        },
        {
            "experiment_name": "CDD-NoSemantic",
            "data_root": args.cdd_root,
            "model_module": "lightmnet3_ablation_no_semantic",
            "weight_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/weights/cdd_no_semantic_best.pth",
        },
        {
            "experiment_name": "CDD-NoAttention",
            "data_root": args.cdd_root,
            "model_module": "lightmnet3_ablation_no_attention",
            "weight_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/weights/cdd_no_attention_best.pth",
        },
        {
            "experiment_name": "LEVIRCD-NoParam",
            "data_root": args.levircd_root,
            "model_module": "lightmnet3_ablation_no_param",
            "weight_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/weights/levircd_no_param_best.pth",
        },
        {
            "experiment_name": "LEVIRCD-NoSemantic",
            "data_root": args.levircd_root,
            "model_module": "lightmnet3_ablation_no_semantic",
            "weight_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/weights/levircd_no_semantic_best.pth",
        },
        {
            "experiment_name": "LEVIRCD-NoAttention",
            "data_root": args.levircd_root,
            "model_module": "lightmnet3_ablation_no_attention",
            "weight_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/weights/levircd_no_attention_best.pth",
        },
    ]

    payload = {
        "split": args.split,
        "device": str(device),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiments": [],
    }

    for exp in experiments:
        dataset, loader = build_loader(
            exp["data_root"],
            args.split,
            args.batch_size,
            args.num_workers,
        )

        model = load_model(exp["model_module"], pretrained=True).to(device)
        model = model.to(memory_format=torch.channels_last)

        state_dict = torch.load(exp["weight_path"], map_location=device)
        model.load_state_dict(state_dict)

        metrics = compute_metrics_full(model, loader, device)

        item = {
            "experiment_name": exp["experiment_name"],
            "data_root": exp["data_root"],
            "split": args.split,
            "num_samples": len(dataset),
            "model_module": exp["model_module"],
            "weight_path": os.path.abspath(exp["weight_path"]),
            "metrics": metrics,
        }

        payload["experiments"].append(item)

        print(
            f"{exp['experiment_name']}"
            f" | Precision: {metrics['precision']:.6f}"
            f" | Recall: {metrics['recall']:.6f}"
            f" | F1: {metrics['f1']:.6f}"
            f" | OA: {metrics['oa']:.6f}"
            f" | IoU: {metrics['iou']:.6f}"
            f" | Infer Speed: {metrics['infer_speed_img_s']:.2f} img/s",
            flush=True,
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
