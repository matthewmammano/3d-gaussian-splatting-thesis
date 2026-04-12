# RaDe-GS Thesis Lab

**Stevens Institute of Technology, 2026 Master's Thesis**

Goal: evaluate and improve 3D Gaussian Splatting pipelines — pruning/densification scheduling, Gaussian count reduction, and quality-efficiency tradeoffs.

## Credit
Derived from [RaDe-GS](https://github.com/HKUST-SAIL/RaDe-GS) (Zhang et al.), with components from 3DGS (GraphDECO), Mip-Splatting, 2DGS, GOF, and PGSR. Pruning via GaussianPOP.

---

## Setup

### 1. Clone
```bash
git clone --recursive https://github.com/matthewmammano/3d-gaussian-splatting-thesis.git
cd 3d-gaussian-splatting-thesis
```

### 2. Choose your GPU environment

<details>
<summary><b>SM52 — GTX Titan X / Maxwell (CUDA 11.7, PyTorch 1.13.1) [TESTED]</b></summary>

> **Why these versions?** CUDA 13.0+ dropped SM52 compilation support. PyTorch cu126+ dropped SM52 kernels. MKL 2024.1+ is ABI-incompatible with PyTorch 1.13.1 (`iJIT_NotifyEvent` crash). These constraints force the specific pinned versions captured in `environment_sm52.yml`.

**Step 1: Create the conda env from the lockfile**
```bash
conda env create -f environment_sm52.yml
conda activate 3dgs-thesis
```

**Step 2: Verify torch**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 1.13.1 / True
```

**Step 3: Fix libcudart symlink (conda packaging bug)**
```bash
# Check if the symlink target exists:
ls -la $CONDA_PREFIX/lib/libcudart.so
# If it points to a nonexistent file, fix it:
ln -sf libcudart.so.11.7.99 $CONDA_PREFIX/lib/libcudart.so
```

**Step 4: Build CUDA submodules**
```bash
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
TORCH_CUDA_ARCH_LIST="5.2" pip install submodules/diff-gaussian-rasterization --no-build-isolation
TORCH_CUDA_ARCH_LIST="5.2" pip install submodules/simple-knn --no-build-isolation
```

**Step 5: Verify everything**
```bash
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print('All OK')"
```

</details>

<details>
<summary><b>Modern GPU — RTX 20 series+ (CUDA 13.x, PyTorch latest) [UNTESTED]</b></summary>

> This path follows the upstream RaDe-GS instructions and has not been verified in this repo.

```bash
conda create -n radegs python=3.12 -y
conda activate radegs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/modern.txt
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
```

</details>

### 3. Get DTU dataset

Download the [preprocessed DTU dataset from 2DGS](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) (~3.5GB):

```bash
pip install gdown  # if not already installed
mkdir -p ~/projects/datasets
gdown --folder https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9 -O ~/projects/datasets/DTU
cd ~/projects/datasets/DTU
tar -xzf dtu.tar.gz
# If extracted into DTU/DTU/, flatten it:
mv DTU/* . && rmdir DTU
```

Verify structure:
```bash
ls ~/projects/datasets/DTU/scan105/
# Expected: cameras.npz  database.db  depths  images  mask  points.ply  sparse
```

For geometry evaluation, also download the [official DTU point clouds](https://roboimagedata.compute.dtu.dk/?page_id=36) and place under `dtu_eval/Offical_DTU_Dataset`.

---

## Train
```bash
DATA=~/projects/datasets/DTU/scan105
OUT=output/scan105_baseline

# Baseline RaDe-GS (30K iterations, ~2.5hrs on Titan X)
python train.py -s "$DATA" -m "$OUT" -r 2 --use_decoupled_appearance --eval

# Quick test (1K iterations, ~2min)
python train.py -s "$DATA" -m "$OUT" -r 2 --use_decoupled_appearance --eval --iterations 1000

# + GaussianPOP pruning
python train.py -s "$DATA" -m "$OUT" -r 2 --use_decoupled_appearance --eval \
  --gaussianpop_enable \
  --gaussianpop_prune_iterations 15000,20000 \
  --gaussianpop_prune_ratios 0.35,0.35 \
  --gaussianpop_views_per_quant 8
```

### Train flags explained
| Flag | Purpose |
|------|---------|
| `-s` | Source dataset path (scene folder with images/sparse) |
| `-m` | Model output path (checkpoints, results saved here) |
| `-r 2` | Resolution downscale factor (2x, standard for DTU) |
| `--use_decoupled_appearance` | Per-image appearance modeling (handles lighting variation) |
| `--eval` | Hold out test views for PSNR/SSIM/LPIPS evaluation |
| `--iterations N` | Override iteration count (default 30000) |

## Evaluate
```bash
python render.py -m "$OUT" -s "$DATA"
python metric.py -m "$OUT"
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `iJIT_NotifyEvent` on `import torch` | MKL ≥2024.1 ABI-incompatible with PyTorch 1.13.1 | Nuke env, recreate from `environment_sm52.yml` (has MKL pinned) |
| `cannot find -lcudart` during build | Linker can't find CUDA runtime | `export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH` and fix symlink |
| `No module named 'pkg_resources'` | setuptools ≥70 removed it | `pip install setuptools==69.5.1` (already in yml) |
| `CUDA version mismatch` (13.x vs 11.7) | System nvcc found instead of conda's | Verify `which nvcc` → should be `$CONDA_PREFIX/bin/nvcc` |
| `forward_with_aux` binding error | Stale rasterizer build | `TORCH_CUDA_ARCH_LIST="5.2" pip install submodules/diff-gaussian-rasterization --no-build-isolation` |

---

## Project structure
```
├── train.py                  # Training entry point
├── render.py                 # Render trained model
├── metric.py                 # Compute PSNR/SSIM/LPIPS
├── gaussian_renderer/        # Differentiable renderer
├── scene/                    # Scene + Gaussian model
├── submodules/               # CUDA extensions (diff-gaussian-rasterization, simple-knn)
├── environment_sm52.yml      # Conda env lockfile (SM52 / Titan X)
├── requirements/modern.txt   # Pip deps for modern GPUs (untested)
├── docs/                     # Parameter reference, pruning results
├── scripts/                  # Training/eval scripts
└── dtu_eval/                 # DTU mesh evaluation
```

## Thesis focus
- Pruning/densification behavior and scheduling
- GaussianPOP integration and ablation
- Gaussian count reduction (~30%+ target)
- Quality impact (PSNR / SSIM / LPIPS)
- Efficiency-quality tradeoffs
