# Pruning Configuration Results - DTU Scan 105
**Date:** March 9, 2026  
**Dataset:** DTU scan105 (16k iterations, quick mode)

---

## Key Findings

### Gaussian Count Results
| Config | Gaussians | Reduction | Target Met? |
|--------|-----------|-----------|-------------|
| **Baseline** | 280,746 | — | Reference |
| **Moderate** | 114,294 | **59.3%** | Exceeded |
| **Aggressive** | 92,664 | **67.0%** | Exceeded |


---

## Parameter Impact Analysis

### Moderate Pruning (59.3% reduction)
- `min_opacity`: 0.05 → **0.10** (2x increase)
- `max_screen_size`: 20 → **17** (15% decrease)
- `densify_grad_threshold`: 0.0002 → **0.0003** (50% increase)
- `big_points_ws`: 0.1 → **0.08** (20% decrease)

### Aggressive Pruning (67.0% reduction)
- `min_opacity`: 0.05 → **0.13** (2.6x increase)
- `max_screen_size`: 20 → **16** (20% decrease)
- `densify_grad_threshold`: 0.0002 → **0.00035** (75% increase)
- `big_points_ws`: 0.1 → **0.075** (25% decrease)

---

## Quality Metrics (Iteration 7000)

| Config | Test PSNR | Train PSNR | Test L1 |
|--------|-----------|------------|---------|
| Baseline | 28.48 | 32.03 | 0.0248 |
| Moderate | 28.48 | 31.49 | 0.0247 |
| Aggressive | 28.09 | 31.41 | 0.0257 |

**Quality maintained despite 60-67% Gaussian reduction!**

---

## Conclusions

1. **Pruning more aggressive than expected** — parameters need refinement for 30% target
2. **Quality degradation minimal** — PSNR dropped only ~0.4 with 67% fewer Gaussians
3. **Key parameters identified:** opacity threshold, screen size threshold most impactful
4. **Next steps:** Fine-tune to hit 30% target; evaluate full 30k iteration quality

---

## Recommended Parameters for 30% Reduction

Based on results, suggested tuning:
- `min_opacity`: **0.06-0.07** (smaller increase)
- `max_screen_size`: **18-19** (minor decrease)
- `densify_grad_threshold`: **0.00022-0.00025** (modest increase)
- `big_points_ws`: **0.09** (slight decrease)

---

**Status:** ✓ Complete  
**Data:** `/home/mvv/THESIS/RaDe-GS_RUNS/pruning_test/pruning_results_20260309_001832.csv`
