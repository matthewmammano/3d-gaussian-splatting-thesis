# RaDe-GS Thesis Lab

Minimal thesis repo for `RaDe-GS` training and pruning tests.

## Initial setup
```bash
git clone --recursive <repo-url>
cd RaDe-GS

# choose one env
source ~/miniconda3/bin/activate radegs
# or
source ~/miniconda3/bin/activate radegs_sm52

pip install submodules/diff-gaussian-rasterization --no-build-isolation
```

- Run all commands from the repo root (`RaDe-GS/`).
- Set `DATA` to the dataset path and `OUT` to the output/run directory.

## Train
```bash
DATA=/path/to/dataset
OUT=/path/to/output

# baseline
python train.py -s "$DATA" -m "$OUT" -r 2

# + decoupled appearance
python train.py -s "$DATA" -m "$OUT" -r 2 --use_decoupled_appearance

# + coord-map regularization
python train.py -s "$DATA" -m "$OUT" -r 2 --use_decoupled_appearance --use_coord_map

# + scheduled GaussianPOP pruning
python train.py -s "$DATA" -m "$OUT" -r 2 --use_decoupled_appearance \
  --gaussianpop_enable \
  --gaussianpop_prune_iterations 15000,20000 \
  --gaussianpop_prune_ratios 0.35,0.35 \
  --gaussianpop_views_per_quant 8
```

## Evaluate
```bash
python render.py -m "$OUT" -s "$DATA"
python metric.py -m "$OUT"
```

## Troubleshooting
- If CUDA is not detected inside Flatpak VS Code on Fedora, ensure `LD_LIBRARY_PATH=/run/host/usr/lib64` is available in the environment.
- If you see `forward_with_aux` or related rasterizer binding errors after pulling renderer/CUDA changes, reinstall:
  ```bash
  pip install submodules/diff-gaussian-rasterization --no-build-isolation
  ```

See `docs/PARAMETER_REFERENCE.md` for the short parameter notes.
