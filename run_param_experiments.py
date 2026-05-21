"""
LightMNet3 参数实验队列调度器

按顺序依次运行参数实验，记录日志和状态。

使用方法:
    python run_param_experiments.py                          # 运行所有实验
    python run_param_experiments.py --dataset cdd            # 只运行 CDD 实验
    python run_param_experiments.py --dataset levircd        # 只运行 LEVIR-CD 实验
    python run_param_experiments.py --resume                 # 从上次中断处恢复
    python run_param_experiments.py --dry-run                # 只打印实验列表，不运行

实验阶段:
    第一阶段: lr 扫描 (CDD)
    第二阶段: pos_weight 扫描 (CDD)
    第三阶段: weight_decay 扫描 (CDD)
    第四阶段: 冻结策略扫描 (CDD)
    第五阶段: 优化器扫描 (CDD)
    第六阶段: 最优参数迁移到 LEVIR-CD
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = r"E:\pyCharmProjects\lightMnet"
PARAM_RUNS_DIR = os.path.join(PROJECT_ROOT, "param_runs")
LOGS_DIR = os.path.join(PARAM_RUNS_DIR, "logs")
WEIGHTS_DIR = os.path.join(PARAM_RUNS_DIR, "weights")
RESULTS_DIR = os.path.join(PARAM_RUNS_DIR, "results")
QUEUE_STATUS_PATH = os.path.join(PARAM_RUNS_DIR, "param_queue_status.json")

CDD_DATA_ROOT = r"E:/pyCharmProjects/CDD/Real/subset"
LEVIRCD_DATA_ROOT = r"E:/pyCharmProjects/LEVIR-CD"

TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "param_experiment_train.py")

EPOCHS = 100
NUM_WORKERS = 2
TARGET_F1 = 1.0


def make_experiments():
    experiments = []

    # =========================================================
    # 第一阶段: lr 扫描 (CDD)
    # =========================================================
    for lr in [1e-5, 5e-5, 1e-4, 5e-4]:
        name = f"cdd_lr_{lr:.0e}"
        experiments.append({
            "experiment_name": name,
            "data_root": CDD_DATA_ROOT,
            "save_path": os.path.join(WEIGHTS_DIR, f"{name}_best.pth"),
            "result_path": os.path.join(RESULTS_DIR, f"{name}_result.json"),
            "lr": lr,
            "weight_decay": 5e-3,
            "pos_weight": 3.0,
            "batch_size": 4,
            "freeze_mode": "freeze_layer0_2",
            "optimizer": "AdamW",
        })

    # =========================================================
    # 第二阶段: pos_weight 扫描 (CDD)
    # =========================================================
    for pos_w in [1.0, 2.0, 5.0, 7.0]:
        name = f"cdd_posw_{pos_w}"
        experiments.append({
            "experiment_name": name,
            "data_root": CDD_DATA_ROOT,
            "save_path": os.path.join(WEIGHTS_DIR, f"{name}_best.pth"),
            "result_path": os.path.join(RESULTS_DIR, f"{name}_result.json"),
            "lr": 5e-5,
            "weight_decay": 5e-3,
            "pos_weight": pos_w,
            "batch_size": 4,
            "freeze_mode": "freeze_layer0_2",
            "optimizer": "AdamW",
        })

    # =========================================================
    # 第三阶段: weight_decay 扫描 (CDD)
    # =========================================================
    for wd in [1e-4, 1e-3, 1e-2]:
        name = f"cdd_wd_{wd:.0e}"
        experiments.append({
            "experiment_name": name,
            "data_root": CDD_DATA_ROOT,
            "save_path": os.path.join(WEIGHTS_DIR, f"{name}_best.pth"),
            "result_path": os.path.join(RESULTS_DIR, f"{name}_result.json"),
            "lr": 5e-5,
            "weight_decay": wd,
            "pos_weight": 3.0,
            "batch_size": 4,
            "freeze_mode": "freeze_layer0_2",
            "optimizer": "AdamW",
        })

    # =========================================================
    # 第四阶段: 冻结策略扫描 (CDD)
    # =========================================================
    for freeze_mode in ["freeze_layer0_1", "freeze_layer0_3"]:
        name = f"cdd_{freeze_mode}"
        experiments.append({
            "experiment_name": name,
            "data_root": CDD_DATA_ROOT,
            "save_path": os.path.join(WEIGHTS_DIR, f"{name}_best.pth"),
            "result_path": os.path.join(RESULTS_DIR, f"{name}_result.json"),
            "lr": 5e-5,
            "weight_decay": 5e-3,
            "pos_weight": 3.0,
            "batch_size": 4,
            "freeze_mode": freeze_mode,
            "optimizer": "AdamW",
        })

    # =========================================================
    # 第五阶段: 优化器扫描 (CDD)
    # =========================================================
    for opt in ["Adam", "SGD"]:
        name = f"cdd_opt_{opt}"
        experiments.append({
            "experiment_name": name,
            "data_root": CDD_DATA_ROOT,
            "save_path": os.path.join(WEIGHTS_DIR, f"{name}_best.pth"),
            "result_path": os.path.join(RESULTS_DIR, f"{name}_result.json"),
            "lr": 5e-5,
            "weight_decay": 5e-3,
            "pos_weight": 3.0,
            "batch_size": 4,
            "freeze_mode": "freeze_layer0_2",
            "optimizer": opt,
        })

    # =========================================================
    # 第六阶段: 最优参数迁移到 LEVIR-CD
    # (使用 CDD 上找到的最优参数组合)
    # =========================================================
    for lr in [1e-4, 5e-5]:
        for pos_w in [2.0, 4.0]:
            name = f"levircd_lr_{lr:.0e}_posw_{pos_w}"
            experiments.append({
                "experiment_name": name,
                "data_root": LEVIRCD_DATA_ROOT,
                "save_path": os.path.join(WEIGHTS_DIR, f"{name}_best.pth"),
                "result_path": os.path.join(RESULTS_DIR, f"{name}_result.json"),
                "lr": lr,
                "weight_decay": 1e-2,
                "pos_weight": pos_w,
                "batch_size": 4,
                "freeze_mode": "freeze_layer0_2",
                "optimizer": "AdamW",
            })

    return experiments


def load_queue_status():
    if os.path.exists(QUEUE_STATUS_PATH):
        with open(QUEUE_STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "current_index": 0, "started_at": None}


def save_queue_status(status):
    os.makedirs(os.path.dirname(QUEUE_STATUS_PATH), exist_ok=True)
    with open(QUEUE_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def run_experiment(exp, log_file):
    cmd = [
        sys.executable, "-u", TRAIN_SCRIPT,
        "--experiment_name", exp["experiment_name"],
        "--data_root", exp["data_root"],
        "--save_path", exp["save_path"],
        "--result_path", exp["result_path"],
        "--lr", str(exp["lr"]),
        "--weight_decay", str(exp["weight_decay"]),
        "--pos_weight", str(exp["pos_weight"]),
        "--batch_size", str(exp["batch_size"]),
        "--epochs", str(EPOCHS),
        "--num_workers", str(NUM_WORKERS),
        "--target_f1", str(TARGET_F1),
        "--freeze_mode", exp["freeze_mode"],
        "--optimizer", exp["optimizer"],
    ]

    print(f"\n{'='*80}", flush=True)
    print(f"Starting experiment: {exp['experiment_name']}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"Log file: {log_file}", flush=True)
    print(f"{'='*80}\n", flush=True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp['experiment_name']}\n")
        f.write(f"Started at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'='*80}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()

        process.wait()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Finished at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Return code: {process.returncode}\n")

    return process.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "cdd", "levircd"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last interrupted experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print experiment list, do not run")
    args = parser.parse_args()

    all_experiments = make_experiments()

    if args.dataset == "cdd":
        experiments = [e for e in all_experiments if e["data_root"] == CDD_DATA_ROOT]
    elif args.dataset == "levircd":
        experiments = [e for e in all_experiments if e["data_root"] == LEVIRCD_DATA_ROOT]
    else:
        experiments = all_experiments

    print(f"\nTotal experiments: {len(experiments)}", flush=True)
    print(f"{'='*80}", flush=True)
    for i, exp in enumerate(experiments):
        print(f"  [{i+1:02d}/{len(experiments):02d}] {exp['experiment_name']}: "
              f"lr={exp['lr']}, wd={exp['weight_decay']}, "
              f"pos_w={exp['pos_weight']}, freeze={exp['freeze_mode']}, "
              f"opt={exp['optimizer']}", flush=True)
    print(f"{'='*80}\n", flush=True)

    if args.dry_run:
        print("Dry run mode. No experiments will be executed.", flush=True)
        return

    if args.resume:
        status = load_queue_status()
        start_index = status.get("current_index", 0)
        completed = set(status.get("completed", []))
        print(f"Resume mode. Starting from index {start_index}. "
              f"Already completed: {len(completed)}", flush=True)
    else:
        start_index = 0
        completed = set()
        status = {
            "completed": [],
            "current_index": 0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_queue_status(status)

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for i, exp in enumerate(experiments):
        if i < start_index:
            continue

        if exp["experiment_name"] in completed:
            print(f"Skipping already completed: {exp['experiment_name']}", flush=True)
            continue

        log_file = os.path.join(LOGS_DIR, f"{exp['experiment_name']}.log")

        success = run_experiment(exp, log_file)

        status["completed"].append(exp["experiment_name"])
        status["current_index"] = i + 1
        save_queue_status(status)

        if not success:
            print(f"\nExperiment {exp['experiment_name']} FAILED! "
                  f"Check log: {log_file}", flush=True)
            print("Queue paused. Fix the issue and resume with --resume.", flush=True)
            sys.exit(1)

        print(f"\nExperiment {exp['experiment_name']} completed successfully.", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("All experiments completed!", flush=True)
    print(f"Results saved in: {RESULTS_DIR}", flush=True)
    print(f"Weights saved in: {WEIGHTS_DIR}", flush=True)
    print(f"Logs saved in: {LOGS_DIR}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
