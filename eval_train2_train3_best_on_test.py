import argparse
import json
import os
import re
import time
from datetime import datetime

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from lightmnet3_train_ablation_common import LEVIRCDDataset, load_model


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_train2_defaults(train2_path):
    text = read_text(train2_path)

    data_root_match = re.search(r"data_root\s*=\s*r'([^']+)'", text)
    save_path_match = re.search(r"save_path\s*=\s*'([^']+)'", text)

    if not data_root_match or not save_path_match:
        raise ValueError("Failed to parse data_root/save_path from train2 script")

    return {
        "data_root": data_root_match.group(1),
        "save_path": save_path_match.group(1),
    }


def parse_train3_defaults(train3_path):
    text = read_text(train3_path)

    data_root_match = re.search(r"default\s*=\s*r'([^']+)'", text)
    save_path_match = re.search(r"--save_path'?,[\s\S]*?default='([^']+)'", text)
    resume_path_match = re.search(r"--resume_path'?,[\s\S]*?default='([^']*)'", text)

    if not data_root_match or not save_path_match:
        raise ValueError("Failed to parse data_root/save_path from train3 script")

    return {
        "data_root": data_root_match.group(1),
        "save_path": save_path_match.group(1),
        "resume_path": resume_path_match.group(1) if resume_path_match else "",
    }


def resolve_weight_path(project_root, candidate):
    if not candidate:
        return ""

    candidate = candidate.replace("\\", "/")
    if os.path.isabs(candidate):
        return os.path.abspath(candidate)

    return os.path.abspath(os.path.join(project_root, candidate))


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
    parser.add_argument(
        "--train2_path",
        type=str,
        default=r"e:/pyCharmProjects/lightMnet/lightmnet3-train2.py",
    )
    parser.add_argument(
        "--train3_path",
        type=str,
        default=r"e:/pyCharmProjects/lightMnet/lightmnet3-train3.py",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"e:/pyCharmProjects/lightMnet/ablation_runs/test_eval/train2_train3_test_metrics.json",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.dirname(args.train3_path))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    train2_defaults = parse_train2_defaults(args.train2_path)
    train3_defaults = parse_train3_defaults(args.train3_path)

    train2_weight = resolve_weight_path(project_root, train2_defaults["save_path"])
    train3_weight = resolve_weight_path(
        project_root,
        train3_defaults["resume_path"] or train3_defaults["save_path"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiments = [
        {
            "name": "train2-best",
            "data_root": train2_defaults["data_root"],
            "model_module": "lightmnet3",
            "weight_path": train2_weight,
        },
        {
            "name": "train3-best",
            "data_root": train3_defaults["data_root"],
            "model_module": "lightmnet3",
            "weight_path": train3_weight,
        },
    ]

    payload = {
        "split": args.split,
        "device": str(device),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "train2_path": os.path.abspath(args.train2_path),
            "train3_path": os.path.abspath(args.train3_path),
        },
        "experiments": [],
    }

    for exp in experiments:
        if not os.path.exists(exp["weight_path"]):
            raise FileNotFoundError(f"Weight not found: {exp['weight_path']}")

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

        payload["experiments"].append({
            "name": exp["name"],
            "data_root": exp["data_root"],
            "split": args.split,
            "num_samples": len(dataset),
            "model_module": exp["model_module"],
            "weight_path": exp["weight_path"],
            "metrics": metrics,
        })

        print(
            f"{exp['name']}"
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
