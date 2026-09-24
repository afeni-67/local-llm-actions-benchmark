#!/usr/bin/env python3
"""
Local CPU LLM benchmark on synthetic JSON classification tasks.
Runs entirely on-device via llama-cpp-python (no external inference API).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import psutil

from model_runner import classify, download_model, load_llm, model_path
from validate_output import extract_json_object, validate_record


def print_host_info() -> dict:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(mem.total / 1e9, 2),
        "ram_available_gb": round(mem.available / 1e9, 2),
        "disk_total_gb": round(disk.total / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
    }
    print("=== Host ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    try:
        uname = subprocess.check_output(["uname", "-a"], text=True).strip()
        print(f"  uname: {uname}")
        info["uname"] = uname
    except Exception:
        pass
    return info


def peak_rss_mb() -> float:
    # Linux: ru_maxrss is kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic_tickets.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="Max records (default: all)")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    host = print_host_info()
    if host["ram_available_gb"] < 3.5:
        print("ERROR: Less than ~3.5 GB RAM available; Qwen3-4B Q4_K_M may not load.", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Model download ===")
    try:
        if args.skip_download and model_path().exists():
            path = model_path()
            print(f"Using existing {path}")
        else:
            path = download_model()
        size_gb = path.stat().st_size / 1e9
        print(f"Verified size: {size_gb:.2f} GB")
        if size_gb < 1.5:
            print("ERROR: Model file unexpectedly small.", file=sys.stderr)
            return 3
    except Exception as e:
        print(f"ERROR: model download/verify failed: {e}", file=sys.stderr)
        return 3

    print("=== Load model (CPU only) ===")
    try:
        llm = load_llm(path)
    except Exception as e:
        print(f"ERROR: model load failed: {e}", file=sys.stderr)
        return 4

    if not args.data.exists():
        print(f"ERROR: dataset missing: {args.data}", file=sys.stderr)
        return 5

    records = load_dataset(args.data, args.limit)
    n = len(records)
    print(f"=== Benchmark: {n} records ===")

    results = []
    ok = 0
    fail = 0
    t_infer_sum = 0.0
    wall0 = time.perf_counter()

    for i, rec in enumerate(records, 1):
        try:
            out = classify(llm, rec["text"])
            parsed = extract_json_object(out["raw"])
            valid, reason = validate_record(parsed) if parsed is not None else (False, "no JSON")
            t_infer_sum += out["elapsed_s"]
            if valid:
                ok += 1
            else:
                fail += 1
            results.append(
                {
                    "id": rec["id"],
                    "success": valid,
                    "reason": reason,
                    "elapsed_s": round(out["elapsed_s"], 4),
                    "category": (parsed or {}).get("category"),
                    "sentiment": (parsed or {}).get("sentiment"),
                    "confidence": (parsed or {}).get("confidence"),
                    "raw_preview": (out["raw"] or "")[:200],
                }
            )
        except Exception as e:
            fail += 1
            results.append(
                {
                    "id": rec["id"],
                    "success": False,
                    "reason": f"exception: {e}",
                    "elapsed_s": None,
                    "category": None,
                    "sentiment": None,
                    "confidence": None,
                    "raw_preview": "",
                }
            )
        if i % 25 == 0 or i == n:
            elapsed = time.perf_counter() - wall0
            rpm = (i / elapsed) * 60 if elapsed > 0 else 0
            print(f"  progress {i}/{n} ok={ok} fail={fail} rpm={rpm:.2f}")

    wall = time.perf_counter() - wall0
    rpm = (n / wall) * 60 if wall > 0 else 0
    avg = (t_infer_sum / n) if n else 0
    # 6h = 21600 seconds of wall time
    est_6h = int(rpm * 360) if rpm > 0 else 0  # records in 6 hours of continuous processing
    peak_mb = peak_rss_mb()
    proc = psutil.Process()
    rss_now = proc.memory_info().rss / (1024 * 1024)

    summary = {
        "model_repo": os.environ.get("MODEL_REPO", "Qwen/Qwen3-4B-GGUF"),
        "model_file": os.environ.get("MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf"),
        "quantization": "Q4_K_M (GGUF)",
        "runtime": "llama-cpp-python (CPU, n_gpu_layers=0)",
        "host": host,
        "total_records": n,
        "successful_records": ok,
        "failed_records": fail,
        "success_rate": round(ok / n, 4) if n else 0,
        "records_per_minute": round(rpm, 3),
        "average_inference_seconds": round(avg, 4),
        "total_inference_seconds": round(t_infer_sum, 2),
        "wall_clock_seconds": round(wall, 2),
        "peak_rss_mb_approx": round(peak_mb, 1),
        "rss_mb_end": round(rss_now, 1),
        "estimated_records_in_6h_job": est_6h,
        "notes": (
            "GitHub-hosted job hard limit is 6 hours (360 minutes). "
            "estimated_records_in_6h_job extrapolates current records/minute to a full 6h wall clock "
            "and is not a guarantee."
        ),
    }

    json_path = args.out_dir / "benchmark_summary.json"
    csv_path = args.out_dir / "benchmark_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "success",
                "reason",
                "elapsed_s",
                "category",
                "sentiment",
                "confidence",
                "raw_preview",
            ],
        )
        w.writeheader()
        w.writerows(results)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if k == "host":
            continue
        print(f"  {k}: {v}")
    print(f"  wrote {json_path}")
    print(f"  wrote {csv_path}")

    if ok == 0:
        print("ERROR: zero successful JSON classifications", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
