#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("element vertex"):
                parts = line.split()
                return int(parts[2]) if len(parts) >= 3 else None
            if line.strip() == "end_header":
                return None
    return None


def disk_usage(path: Path) -> str:
    try:
        result = subprocess.run(
            ["du", "-sh", str(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.split()[0]
    except Exception:
        return "unknown"


def collect_pruning(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status_path in sorted((run_root / "posttrain_gpop").glob("*/*/*/*/status.json")):
        status = read_json(status_path)
        source_gaussians = status.get("source_gaussians")
        actual_pruned = status.get("actual_total_pruned")
        final_iteration = status.get("final_iteration")
        model_dir = Path(status.get("model_dir", ""))
        final_ply = model_dir / "point_cloud" / f"iteration_{final_iteration}" / "point_cloud.ply"
        final_gaussians = read_ply_vertex_count(final_ply)
        prune_percent = None
        if source_gaussians:
            prune_percent = 100.0 * float(actual_pruned or 0) / float(source_gaussians)
        rows.append(
            {
                "step": status.get("step"),
                "scene": status.get("scene"),
                "status": status.get("status"),
                "cycles": status.get("c_cycles"),
                "source_gaussians": source_gaussians,
                "actual_total_pruned": actual_pruned,
                "prune_percent": f"{prune_percent:.3f}" if prune_percent is not None else "",
                "final_gaussians": final_gaussians,
                "final_iteration": final_iteration,
                "fine_tune": status.get("fine_tune"),
                "model_dir": str(model_dir),
            }
        )
    return rows


def collect_metrics(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status_path in sorted((run_root / "evaluate").glob("*/*/*/*/status.json")):
        status = read_json(status_path)
        metrics = status.get("metrics", {})
        rows.append(
            {
                "step": status.get("step"),
                "source_step": status.get("source_step"),
                "scene": status.get("scene"),
                "iteration": status.get("iteration"),
                "render_rc": status.get("render_rc"),
                "metric_rc": status.get("metric_rc"),
                "PSNR": metrics.get("PSNR"),
                "SSIM": metrics.get("SSIM"),
                "LPIPS": metrics.get("LPIPS"),
            }
        )
    return rows


def collect_comparisons(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status_path in sorted((run_root / "compare").glob("*/*/*/*/status.json")):
        status = read_json(status_path)
        summary = status.get("summary", {})
        worst = (summary.get("worst_views") or [{}])[0]
        rows.append(
            {
                "step": status.get("step"),
                "candidate_step": status.get("candidate_step"),
                "reference_step": status.get("reference_step"),
                "scene": status.get("scene"),
                "candidate_iteration": status.get("candidate_iteration"),
                "reference_iteration": status.get("reference_iteration"),
                "charts_dir": summary.get("charts_dir"),
                "worst_count": summary.get("worst_count"),
                "top_worst_image": worst.get("image"),
                "top_worst5pct_mae": worst.get("candidate_worst5pct_mae"),
            }
        )
    return rows


def write_markdown(run_root: Path, out_dir: Path, pruning: list[dict[str, Any]], metrics: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Summary",
        "",
        f"Run folder: `{run_root}`",
        f"Disk used: `{disk_usage(run_root)}`",
        "",
        "## Metrics",
        "",
        "| step | PSNR | SSIM | LPIPS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row.get('source_step') or row.get('step')} | {row.get('PSNR')} | {row.get('SSIM')} | {row.get('LPIPS')} |"
        )

    lines.extend(["", "## Pruning", "", "| step | pruned | final gaussians | final iter |", "| --- | ---: | ---: | ---: |"])
    for row in pruning:
        lines.append(
            f"| {row.get('step')} | {row.get('prune_percent')}% | {row.get('final_gaussians')} | {row.get('final_iteration')} |"
        )

    lines.extend(["", "## Comparison Charts", ""])
    for row in comparisons:
        lines.append(
            f"- `{row.get('candidate_step')}`: `{row.get('charts_dir')}`; top worst view `{row.get('top_worst_image')}`"
        )

    (out_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a run_experiment output folder.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    run_root = args.run_root
    out_dir = run_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    pruning = collect_pruning(run_root)
    metrics = collect_metrics(run_root)
    comparisons = collect_comparisons(run_root)

    write_csv(out_dir / "pruning_summary.csv", pruning)
    write_csv(out_dir / "metrics_summary.csv", metrics)
    write_csv(out_dir / "comparison_summary.csv", comparisons)
    write_markdown(run_root, out_dir, pruning, metrics, comparisons)
    print(f"Wrote summary to {out_dir}")


if __name__ == "__main__":
    main()
