"""
LightMNet3 参数实验结果汇总评估脚本

功能:
    1. 汇总所有参数实验的验证集结果 (从 result JSON 中读取)
    2. 对每个实验的最佳权重在测试集上做最终评估
    3. 生成汇总报告 (JSON + 控制台表格)

使用方法:
    python eval_param_experiments.py                          # 汇总所有实验结果
    python eval_param_experiments.py --test                   # 在测试集上评估最佳权重
    python eval_param_experiments.py --output report.json     # 指定输出文件
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from lightmnet3_train_ablation_common import LEVIRCDDataset, load_model

PROJECT_ROOT = r"E:\pyCharmProjects\lightMnet"
PARAM_RUNS_DIR = os.path.join(PROJECT_ROOT, "param_runs")
RESULTS_DIR = os.path.join(PARAM_RUNS_DIR, "results")
WEIGHTS_DIR = os.path.join(PARAM_RUNS_DIR, "weights")

CDD_DATA_ROOT = r"E:/pyCharmProjects/CDD/Real/subset"
LEVIRCD_DATA_ROOT = r"E:/pyCharmProjects/LEVIR-CD"


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


def collect_val_results():
    """从 result JSON 文件中收集所有实验的验证集结果"""
    results = []

    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}", flush=True)
        return results

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith("_result.json"):
            continue

        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        results.append({
            "experiment_name": data["experiment_name"],
            "hyperparams": data["hyperparams"],
            "best_epoch": data["best_epoch"],
            "best_f1": data["best_f1"],
            "best_metrics": data["best_metrics"],
            "stop_reason": data["stop_reason"],
            "weight_path": data["save_path"],
        })

    return results


def print_summary_table(results, title="Parameter Experiment Summary"):
    """打印汇总表格"""
    print(f"\n{'='*120}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*120}", flush=True)

    header = (
        f"  {'Experiment':<30s}"
        f"  {'lr':<10s}"
        f"  {'wd':<10s}"
        f"  {'pos_w':<7s}"
        f"  {'freeze':<18s}"
        f"  {'opt':<10s}"
        f"  {'Best F1':<8s}"
        f"  {'Prec':<8s}"
        f"  {'Rec':<8s}"
    )
    print(header, flush=True)
    print(f"  {'-'*118}", flush=True)

    for r in sorted(results, key=lambda x: x["best_f1"], reverse=True):
        hp = r["hyperparams"]
        bm = r["best_metrics"]
        print(
            f"  {r['experiment_name']:<30s}"
            f"  {hp['lr']:<10.0e}"
            f"  {hp['weight_decay']:<10.0e}"
            f"  {hp['pos_weight']:<7.1f}"
            f"  {hp['freeze_mode']:<18s}"
            f"  {hp['optimizer']:<10s}"
            f"  {r['best_f1']:<8.4f}"
            f"  {bm.get('precision', 0):<8.4f}"
            f"  {bm.get('recall', 0):<8.4f}",
            flush=True,
        )

    print(f"{'='*120}\n", flush=True)


def print_grouped_summary(results):
    """按参数维度分组打印汇总"""
    groups = {
        "lr": [],
        "pos_weight": [],
        "weight_decay": [],
        "freeze_mode": [],
        "optimizer": [],
    }

    for r in results:
        hp = r["hyperparams"]
        for key in groups:
            val = hp.get(key)
            if val is not None:
                groups[key].append((val, r["best_f1"], r["experiment_name"]))

    print(f"\n{'='*80}", flush=True)
    print(f"  Grouped Analysis by Parameter Dimension", flush=True)
    print(f"{'='*80}", flush=True)

    for param_name, items in groups.items():
        if not items:
            continue

        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        print(f"\n  --- {param_name} ---", flush=True)
        for val, f1, name in items_sorted:
            print(f"    {val!s:<12s}  F1={f1:.4f}  ({name})", flush=True)

    print(f"{'='*80}\n", flush=True)


def evaluate_on_test(results, batch_size=8, num_workers=2):
    """对每个实验的最佳权重在测试集上做最终评估"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}", flush=True)
    print(f"  Test Set Evaluation", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"{'='*80}", flush=True)

    test_results = []

    for r in results:
        weight_path = r["weight_path"]
        if not os.path.exists(weight_path):
            print(f"  Weight not found, skipping: {weight_path}", flush=True)
            continue

        hp = r["hyperparams"]
        data_root = r.get("data_root", "")
        if not data_root:
            if "cdd" in r["experiment_name"]:
                data_root = CDD_DATA_ROOT
            else:
                data_root = LEVIRCD_DATA_ROOT

        dataset, loader = build_loader(data_root, "test", batch_size, num_workers)

        model = load_model("lightmnet3", pretrained=True).to(device)
        model = model.to(memory_format=torch.channels_last)
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)

        metrics = compute_metrics_full(model, loader, device)

        test_results.append({
            "experiment_name": r["experiment_name"],
            "hyperparams": hp,
            "data_root": data_root,
            "split": "test",
            "num_samples": len(dataset),
            "weight_path": weight_path,
            "metrics": metrics,
        })

        print(
            f"  {r['experiment_name']:<30s}"
            f"  F1={metrics['f1']:.4f}"
            f"  Prec={metrics['precision']:.4f}"
            f"  Rec={metrics['recall']:.4f}"
            f"  OA={metrics['oa']:.4f}"
            f"  IoU={metrics['iou']:.4f}",
            flush=True,
        )

    return test_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Evaluate best weights on test set")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path for the summary report")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    results = collect_val_results()

    if not results:
        print("No experiment results found. Run experiments first.", flush=True)
        return

    print(f"\nFound {len(results)} experiment results.", flush=True)

    print_summary_table(results)
    print_grouped_summary(results)

    if args.test:
        test_results = evaluate_on_test(results, args.batch_size, args.num_workers)

        print_summary_table(
            [{
                "experiment_name": tr["experiment_name"],
                "hyperparams": tr["hyperparams"],
                "best_epoch": 0,
                "best_f1": tr["metrics"]["f1"],
                "best_metrics": tr["metrics"],
                "stop_reason": "test_eval",
                "weight_path": tr["weight_path"],
            } for tr in test_results],
            title="Test Set Evaluation Summary",
        )

        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                PARAM_RUNS_DIR, "test_eval", "param_experiment_test_metrics.json"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {
            "split": "test",
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiments": test_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nTest results saved: {output_path}", flush=True)
    else:
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                PARAM_RUNS_DIR, "param_experiment_summary.json"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "num_experiments": len(results),
            "experiments": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
