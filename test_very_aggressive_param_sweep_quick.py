#!/usr/bin/env python3
"""
Quick-mode parameter sensitivity sweep around the very aggressive setup.
Runs 8 configs (2 perturbations x 4 parameters), changing one parameter at a time.
"""

import os
import sys
import subprocess
import csv
from datetime import datetime

# Very aggressive reference values
# min_opacity=0.13, max_screen_size=16, densify_grad_threshold=0.00035, big_points_ws=0.075
BASE = {
    "min_opacity": 0.13,
    "max_screen_size": 16,
    "densify_grad_threshold": 0.00035,
    "big_points_ws": 0.075,
}

# 8 total configs: each parameter nudged in both directions, one-at-a-time
CONFIGS = [
    {
        "name": "min_opacity_down",
        "description": "min_opacity 0.13 -> 0.12",
        "min_opacity": 0.12,
        "max_screen_size": BASE["max_screen_size"],
        "densify_grad_threshold": BASE["densify_grad_threshold"],
        "big_points_ws": BASE["big_points_ws"],
    },
    {
        "name": "min_opacity_up",
        "description": "min_opacity 0.13 -> 0.14",
        "min_opacity": 0.14,
        "max_screen_size": BASE["max_screen_size"],
        "densify_grad_threshold": BASE["densify_grad_threshold"],
        "big_points_ws": BASE["big_points_ws"],
    },
    {
        "name": "max_screen_size_down",
        "description": "max_screen_size 16 -> 15",
        "min_opacity": BASE["min_opacity"],
        "max_screen_size": 15,
        "densify_grad_threshold": BASE["densify_grad_threshold"],
        "big_points_ws": BASE["big_points_ws"],
    },
    {
        "name": "max_screen_size_up",
        "description": "max_screen_size 16 -> 17",
        "min_opacity": BASE["min_opacity"],
        "max_screen_size": 17,
        "densify_grad_threshold": BASE["densify_grad_threshold"],
        "big_points_ws": BASE["big_points_ws"],
    },
    {
        "name": "grad_threshold_down",
        "description": "densify_grad_threshold 0.00035 -> 0.000325",
        "min_opacity": BASE["min_opacity"],
        "max_screen_size": BASE["max_screen_size"],
        "densify_grad_threshold": 0.000325,
        "big_points_ws": BASE["big_points_ws"],
    },
    {
        "name": "grad_threshold_up",
        "description": "densify_grad_threshold 0.00035 -> 0.000375",
        "min_opacity": BASE["min_opacity"],
        "max_screen_size": BASE["max_screen_size"],
        "densify_grad_threshold": 0.000375,
        "big_points_ws": BASE["big_points_ws"],
    },
    {
        "name": "big_points_ws_down",
        "description": "big_points_ws 0.075 -> 0.070",
        "min_opacity": BASE["min_opacity"],
        "max_screen_size": BASE["max_screen_size"],
        "densify_grad_threshold": BASE["densify_grad_threshold"],
        "big_points_ws": 0.070,
    },
    {
        "name": "big_points_ws_up",
        "description": "big_points_ws 0.075 -> 0.080",
        "min_opacity": BASE["min_opacity"],
        "max_screen_size": BASE["max_screen_size"],
        "densify_grad_threshold": BASE["densify_grad_threshold"],
        "big_points_ws": 0.080,
    },
]


def apply_config(config):
    """Apply configuration by modifying source files."""
    print(f"\nApplying config: {config['name']}")
    print(f"  min_opacity: {config['min_opacity']}")
    print(f"  max_screen_size: {config['max_screen_size']}")
    print(f"  densify_grad_threshold: {config['densify_grad_threshold']}")
    print(f"  big_points_ws: {config['big_points_ws']}")

    # 1) train.py
    with open("train.py", "r") as f:
        content = f.read()

    content = content.replace(
        "gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)",
        f"gaussians.densify_and_prune(opt.densify_grad_threshold, {config['min_opacity']}, scene.cameras_extent, size_threshold)",
    )
    content = content.replace(
        "size_threshold = 20 if iteration > opt.opacity_reset_interval else None",
        f"size_threshold = {config['max_screen_size']} if iteration > opt.opacity_reset_interval else None",
    )

    with open("train.py", "w") as f:
        f.write(content)

    # 2) scene/gaussian_model.py
    with open("scene/gaussian_model.py", "r") as f:
        content = f.read()

    content = content.replace(
        "big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent",
        f"big_points_ws = self.get_scaling.max(dim=1).values > {config['big_points_ws']} * extent",
    )

    with open("scene/gaussian_model.py", "w") as f:
        f.write(content)

    # 3) arguments/__init__.py
    with open("arguments/__init__.py", "r") as f:
        content = f.read()

    content = content.replace(
        "self.densify_grad_threshold = 0.0002",
        f"self.densify_grad_threshold = {config['densify_grad_threshold']}",
    )

    with open("arguments/__init__.py", "w") as f:
        f.write(content)


def backup_files():
    files = ["train.py", "scene/gaussian_model.py", "arguments/__init__.py"]
    for file_path in files:
        subprocess.run(["cp", file_path, f"{file_path}.backup"], check=True)
    print("✓ Original files backed up")


def restore_files():
    files = ["train.py", "scene/gaussian_model.py", "arguments/__init__.py"]
    for file_path in files:
        backup_path = f"{file_path}.backup"
        if os.path.exists(backup_path):
            subprocess.run(["cp", backup_path, file_path], check=True)
    print("✓ Original files restored")


def run_training(config, dataset_path, output_base, iterations=16000):
    """Run one training job in QUICK mode."""
    output_path = os.path.join(output_base, config["name"])
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "python",
        "train.py",
        "-s",
        dataset_path,
        "-m",
        output_path,
        "--eval",
        "--iterations",
        str(iterations),
    ]

    print(f"\n{'=' * 72}")
    print(f"RUNNING: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Output: {output_path}")
    print("Mode: QUICK (16k iterations)")
    print(f"{'=' * 72}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def count_gaussians(output_path, iterations=16000):
    """Count final Gaussians from final PLY header."""
    ply_file = os.path.join(output_path, f"point_cloud/iteration_{iterations}/point_cloud.ply")
    if os.path.exists(ply_file):
        try:
            with open(ply_file, "rb") as f:
                for line in f:
                    try:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("element vertex"):
                            return int(line_str.split()[-1])
                    except UnicodeDecodeError:
                        break
        except Exception as exc:
            print(f"Warning: Could not read PLY file: {exc}")
    return None


def save_results_csv(results, output_base):
    csv_path = os.path.join(output_base, f"very_aggressive_sweep_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Config Name",
                "Description",
                "Gaussian Count",
                "Min Opacity",
                "Max Screen Size",
                "Grad Threshold",
                "Big Points WS",
                "Status",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["name"],
                    result["description"],
                    result["count"] if result["count"] is not None else "N/A",
                    result["config"]["min_opacity"],
                    result["config"]["max_screen_size"],
                    result["config"]["densify_grad_threshold"],
                    result["config"]["big_points_ws"],
                    "SUCCESS" if result["count"] is not None else "FAILED",
                ]
            )

    return csv_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_very_aggressive_param_sweep_quick.py <dataset_path> [output_base]")
        print("\nExample:")
        print(
            "  python test_very_aggressive_param_sweep_quick.py "
            "../RaDe-GS_DATA/dtu_preproc/2DGS_data/DTU/scan105 "
            "../RaDe-GS_RUNS/very_aggressive_sweep_quick"
        )
        print(f"\nThis will run {len(CONFIGS)} configs in QUICK mode (16k iter each)")
        print("Estimated time: ~8-10 hours total")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_base = sys.argv[2] if len(sys.argv) > 2 else "../RaDe-GS_RUNS/very_aggressive_sweep_quick"
    iterations = 16000  # forced quick mode

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("VERY AGGRESSIVE PARAM SWEEP (QUICK MODE)")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_base}")
    print("Mode: QUICK (16k iterations per config)")
    print(f"Configs: {len(CONFIGS)} (single-parameter +/- perturbations)")
    print("Reference: min_opacity=0.13, max_screen_size=16, grad=0.00035, big_points_ws=0.075")
    print(f"{'=' * 80}\n")

    backup_files()
    results = []

    try:
        for idx, config in enumerate(CONFIGS, start=1):
            print(f"\n[{idx}/{len(CONFIGS)}] Testing: {config['name']}")

            output_path = os.path.join(output_base, config["name"])
            ply_file = os.path.join(output_path, f"point_cloud/iteration_{iterations}/point_cloud.ply")

            if os.path.exists(ply_file):
                print(f"✓ {config['name']} already completed, skipping...")
                gaussian_count = count_gaussians(output_path, iterations)
                results.append(
                    {
                        "name": config["name"],
                        "description": config["description"],
                        "count": gaussian_count,
                        "config": config,
                    }
                )
                continue

            restore_files()
            apply_config(config)

            success = run_training(config, dataset_path, output_base, iterations)

            if success:
                gaussian_count = count_gaussians(output_path, iterations)
                results.append(
                    {
                        "name": config["name"],
                        "description": config["description"],
                        "count": gaussian_count,
                        "config": config,
                    }
                )
            else:
                print(f"✗ Training failed for {config['name']}")
                results.append(
                    {
                        "name": config["name"],
                        "description": config["description"],
                        "count": None,
                        "config": config,
                    }
                )

    finally:
        restore_files()
        for backup in [
            "train.py.backup",
            "scene/gaussian_model.py.backup",
            "arguments/__init__.py.backup",
        ]:
            if os.path.exists(backup):
                os.remove(backup)

    print(f"\n\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Config':<24} {'Gaussians':>12} {'Opacity':>10} {'Screen':>8} {'Grad':>10} {'WS':>8}")
    print(f"{'-' * 24} {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 8}")

    for result in results:
        if result["count"] is None:
            print(f"{result['name']:<24} {'FAILED':>12}")
            continue

        cfg = result["config"]
        print(
            f"{result['name']:<24} "
            f"{result['count']:>12,} "
            f"{cfg['min_opacity']:>10.3f} "
            f"{cfg['max_screen_size']:>8} "
            f"{cfg['densify_grad_threshold']:>10.6f} "
            f"{cfg['big_points_ws']:>8.3f}"
        )

    csv_path = save_results_csv(results, output_base)
    print()
    print(f"✓ Results saved to CSV: {csv_path}")
    print(f"✓ All outputs saved to: {output_base}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
