from pathlib import Path
import numpy as np
from PIL import Image

sum_r = Path("output/20260505_171218-posttrain_sum_ref_truck/posttrain_gpop/posttrain_p80_sum_ref/list/Truck/default/model/test/ours_30007/renders")
k_r = Path("output/20260505_112916-topk_fullsweep_min5_p80_truck/posttrain_gpop/posttrain_p80_topk10_min5/list/Truck/default/model/test/ours_35009/renders")
gt_r = Path("output/20260505_112916-topk_fullsweep_min5_p80_truck/posttrain_gpop/posttrain_p80_topk10_min5/list/Truck/default/model/test/ours_35009/gt")

files = sorted(
    {p.name for p in sum_r.glob("*.png")}
    & {p.name for p in k_r.glob("*.png")}
    & {p.name for p in gt_r.glob("*.png")}
)

diff_maxes = []
rows = []
for n in files:
    gt = np.asarray(Image.open(gt_r / n).convert("RGB"), dtype=np.float32) / 255.0
    s = np.asarray(Image.open(sum_r / n).convert("RGB"), dtype=np.float32) / 255.0
    k = np.asarray(Image.open(k_r / n).convert("RGB"), dtype=np.float32) / 255.0

    se = np.abs(s - gt).mean(axis=2)
    ke = np.abs(k - gt).mean(axis=2)
    d = ke - se

    diff_maxes.append(np.abs(d).max())
    rows.append((n, se.mean(), ke.mean(), d.mean(), np.abs(d).mean(), np.abs(d).max(), se.max(), ke.max()))

scale = max(float(np.percentile(np.array(diff_maxes), 95)), 1e-6)
print(f"global_diff_scale_95pct={scale:.6f}")
print("name se_mean ke_mean d_mean absd_mean absd_max se_max ke_max absd_max/scale")
for r in rows[:8]:
    ratio = r[5] / scale
    print(r[0], *[f"{x:.6f}" for x in r[1:]], f"{ratio:.3f}")

print("\nTop 8 by SUM max error:")
rows2 = sorted(rows, key=lambda x: x[6], reverse=True)[:8]
for r in rows2:
    ratio = r[5] / scale
    print(r[0], f"se_max={r[6]:.4f}", f"ke_max={r[7]:.4f}", f"absd_max={r[5]:.4f}", f"ratio={ratio:.3f}")
