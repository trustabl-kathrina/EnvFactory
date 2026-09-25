"""Independent, bounded heartbeat for First-Divergence v1.

The app wakes the task every 90 minutes; this process writes a local liveness
record every five minutes and stops at the experiment deadline or final status.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1"
DEADLINE = dt.datetime.fromisoformat("2026-09-25T11:50:00+08:00")
TERMINAL = {"DONE", "STOPPED", "ERROR"}

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def read_control() -> dict:
    path = LOGS / "control.json"
    return json.loads(path.read_text()) if path.exists() else {}

def gpu_status() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], check=True, capture_output=True,
            text=True, timeout=15,
        )
        return result.stdout.strip().splitlines()
    except Exception as exc:
        return [f"unknown:{type(exc).__name__}:{exc}"]

def beat() -> str:
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    control = read_control()
    status = control.get("status", "RUNNING")
    error = control.get("error")
    if now >= DEADLINE and status not in TERMINAL:
        status = "STOPPED"
        error = "ten_hour_deadline_reached; experiment controller must stop owned jobs"
    gpus = gpu_status()
    payload = {
        "timestamp": now.isoformat(), "status": status,
        "cycle": control.get("cycle", 0), "phase": control.get("phase", "AUDIT"),
        "last_completed_action": control.get("last_completed_action", "heartbeat_started"),
        "current_job": control.get("current_job", "none"),
        "gpu0_status": gpus[0] if gpus else "unknown",
        "gpu1_status": gpus[1] if len(gpus) > 1 else "unknown",
        "latest_checkpoint": control.get("latest_checkpoint", "none"),
        "latest_report": control.get("latest_report", "none"),
        "error": error, "pid": os.getpid(),
        "deadline": DEADLINE.isoformat(),
    }
    write_json(LOGS / "heartbeat.json", payload)
    with (LOGS / "heartbeat.log").open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return status

def monitor(interval: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "heartbeat.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("heartbeat already running")
        while True:
            status = beat()
            if status in TERMINAL:
                break
            time.sleep(interval)

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("monitor")
    start.add_argument("--interval-seconds", type=int, default=300)
    update = sub.add_parser("set")
    update.add_argument("--status", default="RUNNING")
    update.add_argument("--cycle", type=int, required=True)
    update.add_argument("--phase", required=True)
    update.add_argument("--last-completed-action", default="none")
    update.add_argument("--current-job", default="none")
    update.add_argument("--latest-checkpoint", default="none")
    update.add_argument("--latest-report", default="none")
    update.add_argument("--error")
    args = parser.parse_args()
    if args.command == "monitor":
        if not 30 <= args.interval_seconds <= 300:
            raise ValueError("heartbeat interval must be within 30..300 seconds")
        monitor(args.interval_seconds)
    else:
        write_json(LOGS / "control.json", {
            "status": args.status, "cycle": args.cycle, "phase": args.phase,
            "last_completed_action": args.last_completed_action,
            "current_job": args.current_job,
            "latest_checkpoint": args.latest_checkpoint,
            "latest_report": args.latest_report, "error": args.error,
        })

if __name__ == "__main__":
    main()
