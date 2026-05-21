import json
import os
import re


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
STATUS_PATH = os.path.join(
    PROJECT_ROOT,
    "ablation_runs",
    "ablation_queue_status.json",
)


def load_status():
    if not os.path.exists(STATUS_PATH):
        raise FileNotFoundError(f"Status file not found: {STATUS_PATH}")

    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_log_path(status):
    current = status.get("current_experiment")
    if current and current.get("log_path"):
        return current["log_path"], current["name"]

    completed = status.get("completed_experiments", [])
    if completed:
        last_item = completed[-1]
        return last_item.get("log_path"), last_item.get("name")

    return None, None


def read_recent_epoch_lines(log_path, max_lines=10):
    if not log_path or not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip() for line in f]

    pattern = re.compile(r"^Epoch\s+\d+")
    epoch_lines = [line for line in lines if pattern.match(line)]
    return epoch_lines[-max_lines:]


def main():
    status = load_status()
    log_path, experiment_name = resolve_log_path(status)
    recent_lines = read_recent_epoch_lines(log_path, max_lines=10)

    print(f"Queue Status: {status.get('queue_status')}")
    print(
        f"Progress: "
        f"{len(status.get('completed_experiments', []))}"
        f"/"
        f"{status.get('total_experiments', 0)}"
    )

    if experiment_name:
        print(f"Current Or Latest Experiment: {experiment_name}")
    if log_path:
        print(f"Log Path: {log_path}")

    print("Recent 10 Epoch Logs:")
    if recent_lines:
        for line in recent_lines:
            print(line)
    else:
        print("No epoch logs found yet.")


if __name__ == "__main__":
    main()
