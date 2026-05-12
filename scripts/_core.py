#!/usr/bin/env python3
"""Core utilities for unified experiment runner (KISS/DRY)."""

from __future__ import annotations

import json
import os
import re
import selectors
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _strip_jsonc(text: str) -> str:
    out = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i = text.find("\n", i)
            if i == -1:
                break
            out.append("\n")
            i += 1
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            i = len(text) if end == -1 else end + 2
            continue

        out.append(ch)
        i += 1

    cleaned = "".join(out)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def read_json(path: str | Path):
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(_strip_jsonc(text)) if file_path.suffix.lower() == ".jsonc" else json.loads(text)



def write_json(path: str | Path, data) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_scene_jobs(config: Dict) -> List[Tuple[str, str, str]]:
    sel = config["selection"]
    mode = sel["mode"]

    if mode == "single_scene":
        scene_path = sel["scene_path"]
        name = Path(scene_path).name
        return [("single", name, scene_path)]

    if mode == "scene_list":
        dataset_root = sel["dataset_root"]
        names = sel["scenes"]
        return [("list", n, str(Path(dataset_root) / n)) for n in names]

    if mode == "split_group":
        split_path = sel.get("split_file") or sel.get("split_config")
        split = read_json(split_path)
        dataset_root = sel.get("dataset_root", split.get("dataset_root", "/home/mvv/projects/datasets/DTU"))
        groups = sel["groups"]
        jobs = []
        for g in groups:
            for n in split[g]:
                jobs.append((g, n, str(Path(dataset_root) / n)))
        return jobs

    raise ValueError(f"Unknown selection mode: {mode}")


def build_train_command(scene_path: str, model_path: str, phase: Dict, defaults: Dict) -> List[str]:
    iterations = int(phase.get("iterations", defaults.get("iterations", 30000)))
    resolution = int(phase.get("resolution", defaults.get("resolution", 2)))
    save_point_cloud = bool(phase.get("save_point_cloud", defaults.get("save_point_cloud", True)))
    run_tests = bool(phase.get("run_tests", defaults.get("run_tests", True)))
    # Default to final-only saves: use a huge save_every so only the last iteration is saved.
    save_every = int(phase.get("save_every", defaults.get("save_every", 999_999_999)))

    save_iters = [i for i in range(save_every, iterations + 1, save_every)]
    if iterations not in save_iters:
        save_iters.append(iterations)

    checkpoint_every = int(phase.get("checkpoint_every", defaults.get("checkpoint_every", 999_999_999)))
    default_ckpt_iters = [i for i in range(checkpoint_every, iterations + 1, checkpoint_every)]
    if iterations not in default_ckpt_iters:
        default_ckpt_iters.append(iterations)
    ckpt_iters = [int(x) for x in phase.get("checkpoint_iterations", default_ckpt_iters)]
    test_iters = [int(x) for x in phase.get("test_iterations", save_iters)]

    cmd = [
        sys.executable,
        "train.py",
        "-s",
        scene_path,
        "-m",
        model_path,
        "-r",
        str(resolution),
        "--iterations",
        str(iterations),
        "--eval",
        "--save_iterations",
        *[str(x) for x in save_iters],
        "--checkpoint_iterations",
        *[str(x) for x in ckpt_iters],
        "--test_iterations",
        *[str(x) for x in test_iters],
    ]
    if not save_point_cloud:
        cmd.append("--skip_save_iterations")
    if not run_tests:
        cmd.append("--skip_test_iterations")

    if phase.get("use_decoupled_appearance", defaults.get("use_decoupled_appearance", True)):
        cmd.append("--use_decoupled_appearance")

    # Optional GPOP schedule in training loop.
    gpop = phase.get("gpop", {})
    if gpop.get("enabled") or gpop.get("enable"):
        cmd.append("--gaussianpop_enable")

        # Handle schedule_iterations (list) or prune_iterations (CSV string)
        schedule_iters = gpop.get("schedule_iterations") or gpop.get("prune_iterations")
        if schedule_iters:
            if isinstance(schedule_iters, list):
                schedule_iters = ",".join(str(x) for x in schedule_iters)
            cmd += ["--gaussianpop_prune_iterations", schedule_iters]

        # Handle prune_ratios (list or CSV string)
        prune_ratios = gpop.get("prune_ratios")
        if prune_ratios:
            if isinstance(prune_ratios, list):
                prune_ratios = ",".join(str(x) for x in prune_ratios)
            cmd += ["--gaussianpop_prune_ratios", prune_ratios]

        # Views per quantization
        if "views_per_quant" in gpop:
            cmd += ["--gaussianpop_views_per_quant", str(gpop["views_per_quant"])]

        # Error aggregation method and parameters
        if "error_agg_method" in gpop:
            cmd += ["--gaussianpop_error_agg_method", str(gpop["error_agg_method"])]
        if "error_topk_percent" in gpop:
            cmd += ["--gaussianpop_error_topk_percent", str(gpop["error_topk_percent"])]
        if "error_lp_p" in gpop:
            cmd += ["--gaussianpop_error_lp_p", str(gpop["error_lp_p"])]
        if "error_visibility_mode" in gpop:
            cmd += ["--gaussianpop_visibility_mode", str(gpop["error_visibility_mode"])]
        if "error_min_visible_views" in gpop:
            cmd += ["--gaussianpop_error_min_visible_views", str(gpop["error_min_visible_views"])]

    # Common train params passthrough.
    reserved = {
        "iterations",
        "resolution",
        "save_every",
        "save_point_cloud",
        "checkpoint_every",
        "checkpoint_iterations",
        "test_iterations",
        "run_tests",
        "use_decoupled_appearance",
        "gpop",
        "gaussianpop_enable",
        "gaussianpop_prune_iterations",
        "gaussianpop_prune_ratios",
        "gaussianpop_views_per_quant",
        "gaussianpop_error_agg_method",
        "gaussianpop_error_topk_percent",
        "gaussianpop_error_lp_p",
        "gaussianpop_visibility_mode",
        "gaussianpop_error_min_visible_views",
        "start_checkpoint",
        "name",
        "op",
    }

    for k, v in defaults.items():
        if k in reserved:
            continue
        if k in phase.get("train_args", {}):
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    for k, v in phase.get("train_args", {}).items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    if phase.get("start_checkpoint"):
        cmd += ["--start_checkpoint", phase["start_checkpoint"]]

    return cmd


def run_logged_command(cmd: List[str], job_dir: Path, label: str, global_log_path: Path | None = None) -> int:
    ensure_dir(job_dir)
    out_path = job_dir / f"{label}.stdout.log"
    err_path = job_dir / f"{label}.stderr.log"
    cmdlog_path = job_dir / "commands.log"

    with open(cmdlog_path, "a", encoding="utf-8") as clog:
        clog.write("$ " + " ".join(cmd) + "\n")

    if global_log_path:
        with open(global_log_path, "a", encoding="utf-8") as glog:
            glog.write(f"[{datetime.now().isoformat()}] START {label}\n")
            glog.write("$ " + " ".join(cmd) + "\n")

    with open(out_path, "a", encoding="utf-8") as out, open(err_path, "a", encoding="utf-8") as err:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        sel = selectors.DefaultSelector()
        if proc.stdout:
            sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
        if proc.stderr:
            sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

        while sel.get_map():
            for key, _ in sel.select():
                stream = key.fileobj
                line = stream.readline()
                if line == "":
                    sel.unregister(stream)
                    continue

                if key.data == "stdout":
                    out.write(line)
                    out.flush()
                else:
                    err.write(line)
                    err.flush()

                if global_log_path:
                    with open(global_log_path, "a", encoding="utf-8") as glog:
                        glog.write(f"[{datetime.now().isoformat()}] [{label}] [{key.data}] {line}")

        rc = proc.wait()

    if global_log_path:
        with open(global_log_path, "a", encoding="utf-8") as glog:
            glog.write(f"[{datetime.now().isoformat()}] END {label} rc={rc}\n")

    return rc


def run_render_metric(
    scene_path: str,
    model_path: str,
    iteration: int,
    job_dir: Path,
    global_log_path: Path | None = None,
) -> Dict:
    render_cmd = [
        sys.executable,
        "render.py",
        "-m",
        model_path,
        "-s",
        scene_path,
        "--iteration",
        str(iteration),
        "--skip_train",
    ]
    rc_render = run_logged_command(render_cmd, job_dir, "render", global_log_path=global_log_path)

    rc_metric = -1
    if rc_render == 0:
        metric_cmd = [sys.executable, "metric.py", "-m", model_path]
        rc_metric = run_logged_command(metric_cmd, job_dir, "metric", global_log_path=global_log_path)

    metrics = {}
    results_path = Path(model_path) / "results.json"
    if results_path.exists():
        try:
            data = read_json(results_path)
            if isinstance(data, dict) and data:
                metrics = next(iter(data.values()), {})
        except Exception:
            metrics = {}

    return {
        "render_rc": rc_render,
        "metric_rc": rc_metric,
        "metrics": metrics,
    }


def visualize_high_error(model_path: str, out_dir: Path, iteration: int, top_percent: float = 5.0) -> Dict:
    import numpy as np
    from PIL import Image

    test_dir = Path(model_path) / "test"
    if iteration >= 0:
        method_dir = test_dir / f"ours_{iteration}"
    else:
        candidates = sorted([p for p in test_dir.glob("ours_*") if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"No ours_* in {test_dir}")
        method_dir = candidates[-1]

    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    overlays = out_dir / "overlays"
    ensure_dir(overlays)

    names = sorted([p.name for p in renders_dir.glob("*.png") if (gt_dir / p.name).exists()])
    per_image = []

    for name in names:
        r = np.asarray(Image.open(renders_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        err = np.abs(r - g).mean(axis=2)
        thr = float(np.quantile(err, 1.0 - top_percent / 100.0))
        mask = err >= thr

        # Build a standalone heatmap rather than blending on top of the
        # rendered image, which makes spatial error patterns easier to inspect.
        scale = max(float(np.quantile(err, 0.995)), 1e-6)
        err_norm = np.clip(err / scale, 0.0, 1.0)

        heat = np.zeros_like(r)
        heat[..., 0] = np.clip(err_norm * 2.0, 0.0, 1.0)
        heat[..., 1] = np.clip((err_norm - 0.5) * 2.0, 0.0, 1.0)

        panel = np.concatenate([g, r, heat], axis=1)
        Image.fromarray((panel * 255.0).astype(np.uint8)).save(overlays / name)

        per_image.append(
            {
                "image": name,
                "mae": float(err.mean()),
                "high_error_threshold": thr,
                "high_error_ratio": float(mask.mean()),
                "high_error_mean": float(err[mask].mean()) if np.any(mask) else 0.0,
            }
        )

    summary = {
        "model_path": model_path,
        "method_dir": str(method_dir),
        "top_percent": top_percent,
        "images": len(per_image),
        "mean_mae": float(np.mean([x["mae"] for x in per_image])) if per_image else None,
        "per_image": per_image,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def write_run_manifest(run_root: Path, config: Dict) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "config": config,
    }
    write_json(run_root / "run_manifest.json", manifest)
