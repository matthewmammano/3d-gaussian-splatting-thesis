#!/usr/bin/env python3
"""
Render and visualize comparison of baseline vs moderate vs aggressive pruning
"""
import os
import sys
import subprocess
from pathlib import Path

# Configuration paths
configs = [
    ("baseline", "../RaDe-GS_RUNS/pruning_test/baseline"),
    ("moderate_pruning", "../RaDe-GS_RUNS/pruning_test/moderate_pruning"),
    ("aggressive_pruning", "../RaDe-GS_RUNS/pruning_test/aggressive_pruning"),
]

dataset_path = "../RaDe-GS_DATA/dtu_preproc/2DGS_data/DTU/scan105"

print("=" * 80)
print("RENDERING TEST SETS FOR COMPARISON")
print("=" * 80)

# Render test set for each config
for name, model_path in configs:
    print(f"\n[Rendering {name}]")
    
    cmd = [
        "python", "render.py",
        "-m", model_path,
        "-s", dataset_path,
        "--iteration", "16000",
        "--skip_train"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✓ {name} rendered successfully")
    else:
        print(f"✗ {name} rendering failed")

print("\n" + "=" * 80)
print("CREATING COMPARISON VISUALIZATION")
print("=" * 80)

# Create comparison grid
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_test_images(model_path, iteration=16000):
    """Load all test set renders"""
    render_dir = Path(model_path) / "test" / f"ours_{iteration}" / "renders"
    if not render_dir.exists():
        return []
    
    images = sorted(render_dir.glob("*.png"))
    return [Image.open(img) for img in images[:4]]  # First 4 test images

# Load images for each config
all_images = {}
for name, model_path in configs:
    imgs = load_test_images(model_path)
    if imgs:
        all_images[name] = imgs
        print(f"✓ Loaded {len(imgs)} images for {name}")
    else:
        print(f"✗ No images found for {name}")

if not all_images:
    print("ERROR: No rendered images found!")
    sys.exit(1)

# Create comparison grid
num_views = len(next(iter(all_images.values())))
num_configs = len(all_images)

# Get image size
sample_img = next(iter(all_images.values()))[0]
img_width, img_height = sample_img.size

# Add padding and labels
padding = 10
label_height = 40
text_height = 50

# Calculate grid size
grid_width = num_configs * img_width + (num_configs + 1) * padding
grid_height = num_views * (img_height + text_height) + (num_views + 1) * padding + label_height

# Create canvas
canvas = Image.new('RGB', (grid_width, grid_height), 'white')
draw = ImageDraw.Draw(canvas)

# Try to use a nice font
try:
    font_title = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 24)
    font_label = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
except:
    font_title = ImageFont.load_default()
    font_label = ImageFont.load_default()

# Add column headers
config_names = [
    "Baseline - 280,746 Gaussians", 
    "Moderate - 114,294 Gaussians (59.3% reduction)", 
    "Aggressive - 92,664 Gaussians (67.0% reduction)"
]
for col_idx, (config_key, config_path) in enumerate(configs):
    x = padding + col_idx * (img_width + padding)
    # Draw text without anchor to avoid multiline issues
    bbox = draw.textbbox((0, 0), config_names[col_idx], font=font_label)
    text_width = bbox[2] - bbox[0]
    draw.text((x + (img_width - text_width) // 2, 10), config_names[col_idx], fill='black', font=font_label)

# Paste images in grid
for row_idx in range(num_views):
    for col_idx, (config_key, _) in enumerate(configs):
        if config_key in all_images and row_idx < len(all_images[config_key]):
            img = all_images[config_key][row_idx]
            
            x = padding + col_idx * (img_width + padding)
            y = label_height + padding + row_idx * (img_height + text_height + padding)
            
            canvas.paste(img, (x, y))
            
            # Add view label on leftmost column
            if col_idx == 0:
                draw.text((5, y + img_height // 2), f"View {row_idx + 1}", fill='black', font=font_label, anchor='lm')

# Save
output_path = "../RaDe-GS_RUNS/pruning_test/comparison_visualization.png"
canvas.save(output_path)
print(f"\n✓ Comparison saved to: {output_path}")
print(f"  Grid size: {grid_width}x{grid_height}")
print(f"  Views: {num_views}")
print("=" * 80)
