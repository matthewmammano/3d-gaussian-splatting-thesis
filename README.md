# RaDe-GS Thesis Lab

This repository is used for **Stevens Institute of Technology, 2026 Master’s Thesis** experimentation.

Goal: evaluate and improve current 3D Gaussian Splatting pipelines (especially pruning/densification, quality vs Gaussian count, and runtime-quality tradeoffs).

## Credit to original work
This codebase is derived from and built upon:

- RaDe-GS: Rasterizing Depth in Gaussian Splatting (Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, Ping Tan)
- Gaussian Splatting (GraphDECO)
- Mip-Splatting
- 2D Gaussian Splatting
- Gaussian Opacity Fields (GOF)
- DTU and Tanks and Temples evaluation toolchains


# Setup
1) Clone:
```
git clone --recursive https://github.com/matthewmammano/3d-gaussian-splatting-thesis.git
cd 3d-gaussian-splatting-thesis
```

2) Choose environment:

SM52 / GTX TITAN X:
```
conda create -n radegs_sm52 python=3.10 -y
conda activate radegs_sm52
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements_SM52.txt
conda install -c nvidia cuda-toolkit=11.7
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
```

Modern GPU:
```
conda create -n radegs python=3.12 -y
conda activate radegs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements_MODERN.txt
```

3) Build extension:
```
pip install submodules/diff-gaussian-rasterization --no-build-isolation
```

# Quick run
```
python train.py -s <dataset_path> -m <output_path> -r 2 --use_decoupled_appearance
python render.py -m <output_path> -s <dataset_path>
python metric.py -m <output_path>
```

# Thesis focus
- Pruning/densification behavior and scheduling
- Gaussian count reduction targets (e.g., ~30%)
- Quality impact (PSNR / SSIM / LPIPS)
- Efficiency-quality tradeoffs for improved 3DGS pipelines
