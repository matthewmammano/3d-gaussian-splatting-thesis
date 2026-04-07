#!/usr/bin/env python3
"""
Test 2 configurations targeting ~30% Gaussian reduction
Regular mode: 30k iterations (~2 hours each = ~4 hours total)
These are MILDER than aggressive, positioned between baseline and moderate to hit 30% target
"""
import os
import sys
import subprocess
import csv
from pathlib import Path
from datetime import datetime

# Define 2 configurations targeting 30% reduction (mild tuning)
# Baseline: 280,746 Gaussians
# Target 30% reduction: ~196,522 Gaussians
CONFIGS = [
    {
        "name": "light_pruning_30pct",
        "description": "Light pruning targeting 30% reduction (slightly stricter than baseline)",
        "min_opacity": 0.07,
        "max_screen_size": 19,
        "densify_grad_threshold": 0.00025,
        "big_points_ws": 0.09,
        "densification_interval": 100,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "opacity_reset_interval": 3000,
        "percent_dense": 0.01
    },
    {
        "name": "mild_pruning_30pct",
        "description": "Mild pruning targeting 30% reduction (balanced approach)",
        "min_opacity": 0.08,
        "max_screen_size": 18,
        "densify_grad_threshold": 0.000275,
        "big_points_ws": 0.085,
        "densification_interval": 100,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "opacity_reset_interval": 3000,
        "percent_dense": 0.01
    },
]


def apply_config(config):
    """Apply configuration by modifying source files"""
    print(f"\nApplying config: {config['name']}")
    print(f"  min_opacity: {config['min_opacity']}")
    print(f"  max_screen_size: {config['max_screen_size']}")
    print(f"  densify_grad_threshold: {config['densify_grad_threshold']}")
    print(f"  big_points_ws: {config['big_points_ws']}")
    
    # 1. Modify train.py
    with open('train.py', 'r') as f:
        content = f.read()
    
    # Replace hardcoded values
    content = content.replace(
        'gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)',
        f'gaussians.densify_and_prune(opt.densify_grad_threshold, {config["min_opacity"]}, scene.cameras_extent, size_threshold)'
    )
    content = content.replace(
        'size_threshold = 20 if iteration > opt.opacity_reset_interval else None',
        f'size_threshold = {config["max_screen_size"]} if iteration > opt.opacity_reset_interval else None'
    )
    
    with open('train.py', 'w') as f:
        f.write(content)
    
    # 2. Modify scene/gaussian_model.py
    with open('scene/gaussian_model.py', 'r') as f:
        content = f.read()
    
    content = content.replace(
        'big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent',
        f'big_points_ws = self.get_scaling.max(dim=1).values > {config["big_points_ws"]} * extent'
    )
    
    with open('scene/gaussian_model.py', 'w') as f:
        f.write(content)
    
    # 3. Modify arguments/__init__.py
    with open('arguments/__init__.py', 'r') as f:
        content = f.read()
    
    content = content.replace(
        'self.densify_grad_threshold = 0.0002',
        f'self.densify_grad_threshold = {config["densify_grad_threshold"]}'
    )
    
    with open('arguments/__init__.py', 'w') as f:
        f.write(content)


def backup_files():
    """Backup original files"""
    files = ['train.py', 'scene/gaussian_model.py', 'arguments/__init__.py']
    for f in files:
        subprocess.run(['cp', f, f'{f}.backup'], check=True)
    print("✓ Original files backed up")


def restore_files():
    """Restore original files"""
    files = ['train.py', 'scene/gaussian_model.py', 'arguments/__init__.py']
    for f in files:
        if os.path.exists(f'{f}.backup'):
            subprocess.run(['cp', f'{f}.backup', f], check=True)
    print("✓ Original files restored")


def run_training(config, dataset_path, output_base, iterations=30000):
    """Run training with given config"""
    output_path = os.path.join(output_base, config['name'])
    os.makedirs(output_path, exist_ok=True)
    
    cmd = [
        'python', 'train.py',
        '-s', dataset_path,
        '-m', output_path,
        '--eval',
        '--iterations', str(iterations)
    ]
    
    print(f"\n{'='*70}")
    print(f"RUNNING: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Output: {output_path}")
    print(f"Mode: REGULAR (30k iterations)")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def count_gaussians(output_path, iterations=30000):
    """Count final Gaussians from PLY file"""
    ply_file = os.path.join(output_path, f'point_cloud/iteration_{iterations}/point_cloud.ply')
    if os.path.exists(ply_file):
        try:
            with open(ply_file, 'rb') as f:
                for line in f:
                    try:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('element vertex'):
                            return int(line_str.split()[-1])
                    except UnicodeDecodeError:
                        # Hit binary data, stop reading
                        break
        except Exception as e:
            print(f"Warning: Could not read PLY file: {e}")
    return None


def save_results_csv(results, output_base, baseline_count, target_count):
    """Save results to CSV file"""
    csv_path = os.path.join(output_base, f'30percent_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Config Name',
            'Description',
            'Gaussian Count',
            'Reduction %',
            'Distance from Target',
            'Min Opacity',
            'Max Screen Size',
            'Grad Threshold',
            'Big Points WS',
            'Status'
        ])
        
        # Data rows
        for result in results:
            if result['count'] is not None:
                if baseline_count:
                    pct = (baseline_count - result['count']) / baseline_count * 100
                    reduction_pct = f"{pct:.1f}"
                else:
                    reduction_pct = "N/A"
                
                if target_count:
                    dist = abs(result['count'] - target_count)
                    distance_str = f"{dist:,}"
                else:
                    distance_str = "N/A"
                
                writer.writerow([
                    result['name'],
                    result['description'],
                    result['count'],
                    reduction_pct,
                    distance_str,
                    result['config']['min_opacity'],
                    result['config']['max_screen_size'],
                    result['config']['densify_grad_threshold'],
                    result['config']['big_points_ws'],
                    'SUCCESS'
                ])
            else:
                writer.writerow([
                    result['name'],
                    result['description'],
                    'N/A',
                    'N/A',
                    'N/A',
                    result['config']['min_opacity'],
                    result['config']['max_screen_size'],
                    result['config']['densify_grad_threshold'],
                    result['config']['big_points_ws'],
                    'FAILED'
                ])
    
    return csv_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_30_percent_reduction.py <dataset_path> [output_base]")
        print("\nExample:")
        print("  python test_30_percent_reduction.py ../RaDe-GS_DATA/dtu_preproc/2DGS_data/DTU/scan105 ../RaDe-GS_RUNS/pruning_test_30pct")
        print(f"\nThis will run {len(CONFIGS)} configurations in REGULAR mode (30k iterations each)")
        print("  Estimated time: ~4 hours total (~2 hours per config)")
        print(f"  Target: ~196,522 Gaussians (30% reduction from 280,746 baseline)")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_base = sys.argv[2] if len(sys.argv) > 2 else '../RaDe-GS_RUNS/pruning_test_30pct'
    iterations = 30000  # ALWAYS regular mode
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("30% REDUCTION TARGET TEST (REGULAR MODE)")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_base}")
    print(f"Mode: REGULAR (30k iterations per config)")
    print(f"Configurations to test: {len(CONFIGS)}")
    print(f"Estimated time: ~4 hours")
    print(f"Target reduction: 30% (280,746 → ~196,522 Gaussians)")
    print(f"{'='*80}\n")
    
    # Backup files
    backup_files()
    
    results = []
    
    try:
        for i, config in enumerate(CONFIGS, 1):
            print(f"\n[{i}/{len(CONFIGS)}] Testing: {config['name']}")
            
            # Check if config already completed
            output_path = os.path.join(output_base, config['name'])
            ply_file = os.path.join(output_path, f'point_cloud/iteration_{iterations}/point_cloud.ply')
            
            if os.path.exists(ply_file):
                print(f"✓ {config['name']} already completed, skipping...")
                gaussian_count = count_gaussians(output_path, iterations)
                results.append({
                    'name': config['name'],
                    'description': config['description'],
                    'count': gaussian_count,
                    'config': config
                })
                continue
            
            # Restore then apply new config
            restore_files()
            apply_config(config)
            
            # Run training
            success = run_training(config, dataset_path, output_base, iterations)
            
            if success:
                output_path = os.path.join(output_base, config['name'])
                gaussian_count = count_gaussians(output_path, iterations)
                results.append({
                    'name': config['name'],
                    'description': config['description'],
                    'count': gaussian_count,
                    'config': config
                })
            else:
                print(f"✗ Training failed for {config['name']}")
                results.append({
                    'name': config['name'],
                    'description': config['description'],
                    'count': None,
                    'config': config
                })
    
    finally:
        # Always restore
        restore_files()
        # Clean up backups
        for f in ['train.py.backup', 'scene/gaussian_model.py.backup', 'arguments/__init__.py.backup']:
            if os.path.exists(f):
                os.remove(f)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY - 30% REDUCTION TARGET")
    print(f"{'='*80}\n")
    
    baseline_count = 280746  # From previous run
    target_count = int(baseline_count * 0.7)
    
    # Print table
    print(f"{'Config':<25} {'Gaussians':>12} {'Reduction':>12} {'vs Target':>15}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*15}")
    
    for result in results:
        if result['count'] is not None:
            pct = (baseline_count - result['count']) / baseline_count * 100
            reduction = f"{pct:+.1f}%"
            
            delta = result['count'] - target_count
            if delta > 0:
                vs_target = f"+{delta:,}"
            else:
                vs_target = f"{delta:,}"
            
            print(f"{result['name']:<25} {result['count']:>12,} {reduction:>12} {vs_target:>15}")
        else:
            print(f"{result['name']:<25} {'FAILED':>12}")
    
    print()
    print(f"Baseline: {baseline_count:,} Gaussians")
    print(f"Target for 30% reduction: {target_count:,} Gaussians\n")
    
    # Save to CSV
    csv_path = save_results_csv(results, output_base, baseline_count, target_count)
    print(f"✓ Results saved to CSV: {csv_path}")
    print(f"✓ All outputs saved to: {output_base}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
