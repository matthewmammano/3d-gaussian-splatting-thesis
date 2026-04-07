#!/usr/bin/env python3
"""
GaussianPOP 30k: baseline (tuned RaDe-GS) vs paper-exact GaussianPOP.
Run from repo root:  python scripts/gpop_ablation.py [output_root]

Example nohup run:
  cd /home/mvv/THESIS/RaDe-GS && \
  LOG="../RaDe-GS_RUNS/gpop_30k_$(date +%Y%m%d_%H%M%S).log" && \
  nohup python -u scripts/gpop_ablation.py > "$LOG" 2>&1 < /dev/null & \
  echo "$LOG" && tail -f "$LOG"
"""

import csv, json, os, subprocess, sys
from datetime import datetime

EXPERIMENT_NAME = "gpop_30k"

SCENES = [
    "../RaDe-GS_DATA/dtu_preproc/2DGS_data/DTU/scan105",
]

# ── Baseline: best RaDe-GS params from 2026-03-23 sweep, now at 30k ──
BASELINE = {
    "iterations": 30_000,
    "use_decoupled_appearance": True,
    "min_opacity": 0.14,
    "max_screen_size": 16,
    "densify_grad_threshold": 0.00035,
    "big_points_ws": 0.075,
}

# ── GaussianPOP base: vanilla 3DGS defaults + paper settings ──
# Paper Sec 4.1: prune at 15k,20k for 30k run; P={0.65,0.70} for DTU; all views
GPOP_BASE = {
    "iterations": 30_000,
    "use_decoupled_appearance": True,
    # vanilla 3DGS defaults for densification (paper comparison basis)
    "min_opacity": 0.05,
    "max_screen_size": 20,
    "densify_grad_threshold": 0.0002,
    "big_points_ws": 0.1,
    # GaussianPOP params
    "gaussianpop_enable": True,
    "gaussianpop_prune_iterations": "15000,20000",
    "gaussianpop_prune_ratios": "0.65,0.70",   # paper DTU recommendation
    "gaussianpop_views_per_quant": 0,           # 0 = all training views
}

CONFIGS = [
    # ── Baseline: tuned RaDe-GS, no GaussianPOP ──
    {"name": "baseline_30k", "description": "Tuned RaDe-GS 30k, no pruning",
     "train_args": {**BASELINE}},

    # ── GaussianPOP: paper-exact on vanilla densification ──
    {"name": "gpop_paper", "description": "Paper DTU: P=0.65,0.70 at 15k,20k, all views",
     "train_args": {**GPOP_BASE}},
]

DEFAULT_ITERS = 30_000

# ── HELPERS ───────────────────────────────────────────────────────────────

def scene_tag(path):
    return os.path.basename(path.rstrip("/"))

def build_cmd(dataset, output, args):
    cmd = ["python", "train.py", "-s", dataset, "-m", output, "--eval"]
    for k, v in args.items():
        if isinstance(v, bool):
            if v: cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    return cmd

def run(cmd, label=""):
    if label: print(f"    {label}")
    return subprocess.run(cmd).returncode == 0

def count_gaussians(out, iters):
    ply = os.path.join(out, f"point_cloud/iteration_{iters}/point_cloud.ply")
    if not os.path.exists(ply): return None
    try:
        with open(ply, "rb") as f:
            for line in f:
                s = line.decode("utf-8", errors="ignore").strip()
                if s.startswith("element vertex"): return int(s.split()[-1])
                if s == "end_header": break
    except Exception: pass
    return None

def read_metrics(out):
    rj = os.path.join(out, "results.json")
    if not os.path.exists(rj): return {}
    try:
        with open(rj) as f:
            data = json.load(f)
        return next(iter(data.values()), {})
    except Exception: return {}

def mean(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None

# ── PIPELINE ──────────────────────────────────────────────────────────────

def run_pipeline(config, dataset, out_path):
    iters = config["train_args"].get("iterations", DEFAULT_ITERS)
    ply = os.path.join(out_path, f"point_cloud/iteration_{iters}/point_cloud.ply")

    if os.path.exists(ply):
        print(f"    ✓ trained")
    else:
        cmd = build_cmd(dataset, out_path, config["train_args"])
        print(f"    TRAIN → {out_path}")
        if not run(cmd):
            return {"status": "FAIL"}

    if os.path.isdir(os.path.join(out_path, "test")):
        print(f"    ✓ rendered")
    else:
        run(["python", "render.py", "-m", out_path, "-s", dataset], "RENDER")

    if os.path.exists(os.path.join(out_path, "results.json")):
        print(f"    ✓ metrics")
    else:
        run(["python", "metric.py", "-m", out_path], "METRIC")

    m = read_metrics(out_path)
    return {
        "status": "OK",
        "gaussians": count_gaussians(out_path, iters),
        "psnr": m.get("PSNR"), "ssim": m.get("SSIM"), "lpips": m.get("LPIPS"),
    }

# ── OUTPUT ────────────────────────────────────────────────────────────────

FMT = "{:<24} {:<10} {:>10} {:>8} {:>8} {:>8} {:>6}"

def fmt(val, prec=4):
    if val is None: return "N/A"
    if isinstance(val, int): return f"{val:,}"
    return f"{val:.{prec}f}"

def print_summary(all_results):
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}\n")
    print(FMT.format("Config", "Scene", "Gaussians", "PSNR", "SSIM", "LPIPS", "Status"))
    print("-" * 80)
    for cfg_name, rows in all_results.items():
        for r in rows:
            print(FMT.format(
                r["config"], r["scene"],
                fmt(r.get("gaussians")), fmt(r.get("psnr"), 2),
                fmt(r.get("ssim")), fmt(r.get("lpips")), r["status"],
            ))
        if len(rows) > 1:
            ok = [r for r in rows if r["status"] == "OK"]
            print(FMT.format(
                cfg_name, "** MEAN **",
                fmt(mean([r.get("gaussians") for r in ok])),
                fmt(mean([r.get("psnr") for r in ok]), 2),
                fmt(mean([r.get("ssim") for r in ok])),
                fmt(mean([r.get("lpips") for r in ok])),
                f"{len(ok)}/{len(rows)}",
            ))
        print()

def save_csv(all_results, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config", "Scene", "Gaussians", "PSNR", "SSIM", "LPIPS", "Status"])
        for cfg_name, rows in all_results.items():
            for r in rows:
                w.writerow([
                    r["config"], r["scene"], r.get("gaussians") or "N/A",
                    fmt(r.get("psnr")), fmt(r.get("ssim")), fmt(r.get("lpips")), r["status"],
                ])
            ok = [r for r in rows if r["status"] == "OK"]
            if len(rows) > 1:
                w.writerow([
                    cfg_name, "MEAN",
                    fmt(mean([r.get("gaussians") for r in ok])),
                    fmt(mean([r.get("psnr") for r in ok])),
                    fmt(mean([r.get("ssim") for r in ok])),
                    fmt(mean([r.get("lpips") for r in ok])),
                    f"{len(ok)}/{len(rows)}",
                ])

# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(f"Usage: python scripts/{os.path.basename(__file__)} [output_root]")
        print(f"  {len(CONFIGS)} configs × {len(SCENES)} scenes = {len(CONFIGS)*len(SCENES)} runs")
        sys.exit(0)

    output_root = sys.argv[1] if len(sys.argv) > 1 else "../RaDe-GS_RUNS"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{datetime.now().strftime('%Y-%m-%d')}-{EXPERIMENT_NAME}")
    os.makedirs(run_dir, exist_ok=True)

    tags = [scene_tag(s) for s in SCENES]
    total = len(CONFIGS) * len(SCENES)

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"{'=' * 80}")
    print(f"  Scenes  : {', '.join(tags)}")
    print(f"  Configs : {len(CONFIGS)}")
    print(f"  Total   : {total} runs")
    print(f"  Run dir : {run_dir}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    all_results = {}
    step = 0

    for config in CONFIGS:
        name = config["name"]
        all_results[name] = []
        for scene_path, tag in zip(SCENES, tags):
            step += 1
            out = os.path.join(run_dir, name, tag)
            os.makedirs(out, exist_ok=True)
            print(f"\n[{step}/{total}] {name} / {tag}")
            r = run_pipeline(config, scene_path, out)
            r.update({"config": name, "scene": tag, "description": config["description"]})
            all_results[name].append(r)

    print_summary(all_results)
    csv_path = os.path.join(run_dir, f"{EXPERIMENT_NAME}_{ts}.csv")
    save_csv(all_results, csv_path)
    print(f"✓ CSV: {csv_path}")
    print(f"✓ Run: {run_dir}")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()
