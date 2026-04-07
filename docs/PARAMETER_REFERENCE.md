# Complete Guide: Pruning & Densification Parameters

**RaDe-GS Pipeline - All Adjustable Parameters**

---

## Parameter Categories

### 1. Timing Parameters (When operations occur)
### 2. Threshold Parameters (What gets pruned/densified)
### 3. Interval Parameters (How often operations run)
### 4. Behavior Parameters (How operations work)
### 5. Learning Rates (Indirect impact on densification)
### 6. Gradient Accumulation (Internal tracking)
### 7. Dataset/Runtime Gates (Conditional behavior)

---

## 1. TIMING PARAMETERS

### `densify_from_iter`
- **Default:** 500
- **Location:** `arguments/__init__.py`
- **What it controls:** First iteration when densification (clone/split) begins
- **Effect:** Delays densification to allow initial optimization. Lower = earlier densification starts.
- **Typical range:** 300-1000

### `densify_until_iter`
- **Default:** 15,000
- **Location:** `arguments/__init__.py`
- **What it controls:** Last iteration when densification occurs. After this, only optimization (no add/remove).
- **Effect:** Determines how long Gaussians can be added/removed. Gaussian count is final after this.
- **Typical range:** 10,000-20,000

---

## GaussianPOP pruning flags

### `gaussianpop_enable`
- **Default:** `False`
- Enables scheduled GaussianPOP error quantification + pruning during training.

### `gaussianpop_prune_iterations`
- **Default:** `"15000,20000"`
- CSV list of prune iterations.

### `gaussianpop_prune_ratios`
- **Default:** `"0.5,0.5"`
- CSV list of prune fractions used at those iterations.

### `gaussianpop_views_per_quant`
- **Default:** `0`
- Number of train views used for scoring before pruning. `0` means all views.

---

## 2. THRESHOLD PARAMETERS (Pruning Criteria)

### `min_opacity` (Pruning)
- **Default:** 0.05
- **Location:** `train.py` line 192 (hardcoded)
- **What it controls:** Minimum opacity threshold for keeping a Gaussian
- **Effect:** Gaussians with `opacity < min_opacity` are pruned. Higher value = more aggressive pruning.
- **Formula:** `prune_mask = (opacity < min_opacity)`
- **Typical range:** 0.01-0.20
- **Impact:** Primary opacity-based pruning control

### `max_screen_size` (Pruning)
- **Default:** 20 pixels (after iter 3000), None (before)
- **Location:** `train.py` line 191 (hardcoded)
- **What it controls:** Maximum screen-space radius allowed
- **Effect:** Gaussians with `max_radii2D > max_screen_size` are pruned. Removes Gaussians that appear too large on screen.
- **Formula:** `big_points_vs = max_radii2D > max_screen_size`
- **Activation gate:** Disabled before `opacity_reset_interval` because `size_threshold=None`
- **Typical range:** 10-30 pixels
- **Impact:** Prevents oversized screen-space splats

### `big_points_ws` (Pruning)
- **Default:** 0.1 * scene_extent
- **Location:** `scene/gaussian_model.py` line 735 (hardcoded)
- **What it controls:** Maximum world-space scale allowed
- **Effect:** Gaussians with `max_scaling > 0.1 * extent` are pruned. Removes physically large Gaussians.
- **Formula:** `big_points_ws = get_scaling.max(dim=1) > 0.1 * extent`
- **Typical range:** 0.05-0.15 * extent
- **Impact:** Prevents oversized world-space Gaussians

### `densify_grad_threshold` (Densification)
- **Default:** 0.0002
- **Location:** `arguments/__init__.py`
- **What it controls:** Minimum position gradient to trigger densification
- **Effect:** Higher = fewer Gaussians added. Determines which areas need more detail.
- **Formula:** `selected = gradient_norm >= threshold`
- **Typical range:** 0.0001-0.0010
- **Impact:** Primary densification control - higher = fewer new Gaussians

### `max_grad` (Alias used inside densification)
- **Default:** Not independent (comes from `densify_grad_threshold`)
- **Location:** `scene/gaussian_model.py` (`densify_and_prune(max_grad, ...)`)
- **What it controls:** High-gradient ratio used for adaptive quantile thresholding
- **Formula:** `ratio = mean(norm(grads) >= max_grad)`
- **Impact:** Indirectly controls adaptive `Q` and therefore clone/split selection

### `percent_dense` (Densification)
- **Default:** 0.01 (1% of scene extent)
- **Location:** `arguments/__init__.py`
- **What it controls:** Size threshold for clone vs split decision
- **Effect:** 
  - If `scaling > percent_dense * extent`: SPLIT (large Gaussians subdivide)
  - If `scaling <= percent_dense * extent`: CLONE (small Gaussians duplicate)
- **Typical range:** 0.005-0.02
- **Impact:** Determines clone/split behavior based on Gaussian size

---

## 3. INTERVAL PARAMETERS

### `densification_interval`
- **Default:** 100 iterations
- **Location:** `arguments/__init__.py`
- **What it controls:** How often densify_and_prune() runs
- **Effect:** Lower = more frequent densification/pruning operations
- **Typical range:** 50-200
- **Impact:** Frequency of Gaussian count adjustments

### `opacity_reset_interval`
- **Default:** 3,000 iterations
- **Location:** `arguments/__init__.py`
- **What it controls:** How often all Gaussian opacities are reset
- **Effect:** Resets opacities to min(current_opacity, 0.01) with 3D filter adjustment. Gives "second chances" to low-opacity Gaussians.
- **Formula:** `if iteration % opacity_reset_interval == 0: reset_opacity()`
- **Reset behavior:** `new_opacity = min(opacity_with_filter, 0.01) / filter_coefficient`
- **Typical range:** 2,000-5,000
- **Impact:** Prevents premature pruning of temporarily low-opacity Gaussians

---

## 4. BEHAVIOR PARAMETERS (Hardcoded)

### Split Factor: `0.8 * N`
- **Default:** 0.8 * 2 = 1.6
- **Location:** `scene/gaussian_model.py` line 684
- **What it controls:** Scale reduction when splitting large Gaussians
- **Effect:** New Gaussians after split are `1/1.6 ≈ 62.5%` of original size
- **Formula:** `new_scaling = old_scaling / (0.8 * N)` where N=2
- **Impact:** Controls overlap after splitting

### Split Count: `N`
- **Default:** 2
- **Location:** `scene/gaussian_model.py` line 662 (function signature)
- **What it controls:** How many new Gaussians created when splitting
- **Effect:** Each large Gaussian becomes N=2 smaller ones
- **Impact:** Doubles Gaussian count in high-gradient areas

### Quantile Calculation: `Q`
- **Location:** `scene/gaussian_model.py` line 723
- **What it controls:** Adaptive threshold based on gradient distribution
- **Formula:** `Q = quantile(grads_abs, 1 - ratio)`
- **Effect:** Automatically adjusts to maintain consistent densification behavior
- **Impact:** Balances clone/split decisions dynamically

---

## 5. LEARNING RATES (Indirect Impact on Densification)

These parameters don't directly control pruning/densification but affect optimization speed, which influences gradient accumulation and densification triggers.

### `position_lr_init` / `position_lr_final`
- **Default:** 0.00016 → 0.0000016
- **Location:** `arguments/__init__.py`
- **What it controls:** Learning rate for Gaussian positions
- **Indirect effect:** Faster position updates = higher gradients = more densification triggers

### `opacity_lr`
- **Default:** 0.05
- **Location:** `arguments/__init__.py`
- **What it controls:** Learning rate for opacity optimization
- **Indirect effect:** Faster opacity learning = faster convergence to pruning thresholds

### `scaling_lr`
- **Default:** 0.005
- **Location:** `arguments/__init__.py`
- **What it controls:** Learning rate for Gaussian scale
- **Indirect effect:** Affects how quickly Gaussians grow/shrink to trigger size-based pruning

---

## 6. GRADIENT ACCUMULATION (Internal)

### `xyz_gradient_accum`
- **What it tracks:** Position gradient accumulation (XY components)
- **Used for:** Determining which Gaussians have high positional error
- **Formula:** `grads = xyz_gradient_accum / denom`

### `xyz_gradient_accum_abs`
- **What it tracks:** Absolute gradient accumulation (Z component)
- **Used for:** Adaptive quantile threshold calculation
- **Formula:** `Q = quantile(grads_abs, 1 - ratio)`

---

## 7. DATASET/RUNTIME GATES (Conditional behavior)

### `dataset.disable_filter3D`
- **Location:** `train.py`
- **What it controls:** Whether 3D filter is reset or recomputed after densify/prune
- **Effect:** Changes filtered opacity/scale behavior and can shift pruning outcomes

### `dataset.white_background`
- **Location:** `train.py`
- **What it controls:** Adds an extra opacity reset at `iteration == densify_from_iter`
- **Effect:** Alters early-stage opacity/pruning dynamics

### Hard gate: `iteration < densify_until_iter`
- **Location:** `train.py`
- **What it controls:** Whether add/remove operations are allowed
- **Effect:** After this gate closes, Gaussian count is frozen

### Scale normalizer: `scene.cameras_extent`
- **Location:** Passed from `train.py` into `densify_and_prune(...)`
- **What it controls:** Normalizes `percent_dense*extent` and world-size pruning threshold
- **Effect:** Same threshold values behave differently across scenes

---

## PARAMETER INTERACTION SUMMARY

### To REDUCE Gaussian count (aggressive pruning):
1. ↑ **Increase** `min_opacity` (0.05 → 0.10-0.15)
2. ↓ **Decrease** `max_screen_size` (20 → 15-17)
3. ↓ **Decrease** `big_points_ws` (0.1 → 0.07-0.08)
4. ↑ **Increase** `densify_grad_threshold` (0.0002 → 0.0003-0.0005)
5. ↑ **Increase** `densification_interval` (100 → 150-200)
6. ↓ **Decrease** `densify_until_iter` (15000 → 12000-14000)

### To INCREASE Gaussian count (more detail):
1. ↓ **Decrease** `min_opacity` (0.05 → 0.02-0.03)
2. ↑ **Increase** `max_screen_size` (20 → 25-30)
3. ↑ **Increase** `big_points_ws` (0.1 → 0.12-0.15)
4. ↓ **Decrease** `densify_grad_threshold` (0.0002 → 0.0001-0.00015)
5. ↓ **Decrease** `densification_interval` (100 → 50-75)
6. ↑ **Increase** `densify_until_iter` (15000 → 18000-20000)

---

## CRITICAL LOCATIONS IN CODE

### Editable Parameters:
- [`arguments/__init__.py`](arguments/__init__.py#L76-L97) - All timing and gradient thresholds
- [`train.py`](train.py#L191-L192) - `min_opacity`, `max_screen_size` (hardcoded)
- [`scene/gaussian_model.py`](scene/gaussian_model.py#L735) - `big_points_ws` factor (hardcoded as 0.1)

### Core Logic:
- [`scene/gaussian_model.py`](scene/gaussian_model.py#L717-L740) - `densify_and_prune()` main function
- [`scene/gaussian_model.py`](scene/gaussian_model.py#L662-L689) - `densify_and_split()` 
- [`scene/gaussian_model.py`](scene/gaussian_model.py#L690-L715) - `densify_and_clone()`
- [`scene/gaussian_model.py`](scene/gaussian_model.py#L743-L747) - `add_densification_stats()`

---

## TESTING RESULTS (DTU Scan 105)

| Config | min_opacity | max_screen | grad_thresh | big_points_ws | Result |
|--------|-------------|------------|-------------|---------------|--------|
| Baseline | 0.05 | 20 | 0.0002 | 0.1 | 280,746 Gaussians |
| Moderate | 0.10 | 17 | 0.0003 | 0.08 | 114,294 (-59.3%) |
| Aggressive | 0.13 | 16 | 0.00035 | 0.075 | 92,664 (-67.0%) |

**Target for 30% reduction: ~196,522 Gaussians**

---

**Last Updated:** March 9, 2026  
**Based on:** RaDe-GS codebase testing
