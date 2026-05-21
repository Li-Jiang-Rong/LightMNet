import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RUN_ROOT = os.path.join(PROJECT_ROOT, "ablation_runs")
LOG_DIR = os.path.join(RUN_ROOT, "logs")
STATUS_PATH = os.path.join(RUN_ROOT, "ablation_queue_status.json")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

EXPERIMENTS = [
    {
        "name": "CDD-NoParam",
        "script": "train_cdd_ablation_no_param.py",
        "log_path": os.path.join(LOG_DIR, "cdd_no_param.log"),
    },
    {
        "name": "CDD-NoSemantic",
        "script": "train_cdd_ablation_no_semantic.py",
        "log_path": os.path.join(LOG_DIR, "cdd_no_semantic.log"),
    },
    {
        "name": "CDD-NoAttention",
        "script": "train_cdd_ablation_no_attention.py",
        "log_path": os.path.join(LOG_DIR, "cdd_no_attention.log"),
    },
    {
        "name": "LEVIRCD-NoParam",
        "script": "train_levircd_ablation_no_param.py",
        "log_path": os.path.join(LOG_DIR, "levircd_no_param.log"),
    },
    {
        "name": "LEVIRCD-NoSemantic",
        "script": "train_levircd_ablation_no_semantic.py",
        "log_path": os.path.join(LOG_DIR, "levircd_no_semantic.log"),
    },
    {
        "name": "LEVIRCD-NoAttention",
        "script": "train_levircd_ablation_no_attention.py",
        "log_path": os.path.join(LOG_DIR, "levircd_no_attention.log"),
    },
]


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUN_ROOT, "weights"), exist_ok=True)
    os.makedirs(os.path.join(RUN_ROOT, "results"), exist_ok=True)


def write_status(payload):
    ensure_dirs()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_initial_status():
    return {
        "queue_status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "current_index": 0,
        "total_experiments": len(EXPERIMENTS),
        "current_experiment": None,
        "completed_experiments": [],
        "pending_experiments": [item["name"] for item in EXPERIMENTS],
    }


def run_single_experiment(python_exe, experiment, index, status):
    status["current_index"] = index + 1
    status["current_experiment"] = {
        "name": experiment["name"],
        "script": experiment["script"],
        "log_path": experiment["log_path"],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }
    status["pending_experiments"] = [
        item["name"] for item in EXPERIMENTS[index + 1:]
    ]
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_status(status)

    print(f"===== Running {experiment['name']} =====", flush=True)

    with open(experiment["log_path"], "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [python_exe, "-u", os.path.join(PROJECT_ROOT, experiment["script"])],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            log_file.flush()

        process.wait()

    current = status["current_experiment"]
    current["finished_at"] = datetime.now().isoformat(timespec="seconds")
    current["exit_code"] = process.returncode
    current["status"] = "completed" if process.returncode == 0 else "failed"

    status["completed_experiments"].append(current.copy())
    status["current_experiment"] = None
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if process.returncode != 0:
        status["queue_status"] = "failed"
        write_status(status)
        raise RuntimeError(f"{experiment['name']} failed with code {process.returncode}")

    write_status(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
    )
    args = parser.parse_args()

    ensure_dirs()
    status = build_initial_status()
    write_status(status)

    for index, experiment in enumerate(EXPERIMENTS):
        run_single_experiment(args.python_exe, experiment, index, status)

    status["queue_status"] = "completed"
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_status(status)
    print("===== All ablation experiments completed =====", flush=True)


if __name__ == "__main__":
    main()
