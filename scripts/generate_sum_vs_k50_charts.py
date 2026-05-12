#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rb_diverging(x: np.ndarray, scale: float) -> np.ndarray:
    z = np.clip(x / scale, -1.0, 1.0)
    out = np.zeros(z.shape + (3,), dtype=np.float32)
    pos = z >= 0
    out[..., 0] = np.where(pos, z, 0.0)
    out[..., 2] = np.where(~pos, -z, 0.0)
    out[..., 1] = 0.05
    return np.clip(out, 0.0, 1.0)


def err_cmap(x: np.ndarray, scale: float) -> np.ndarray:
    z = np.clip(x / scale, 0.0, 1.0)
    out = np.zeros(z.shape + (3,), dtype=np.float32)
    out[..., 0] = np.clip(1.8 * z, 0, 1)
    out[..., 1] = np.clip((z - 0.35) * 2.0, 0, 1)
    out[..., 2] = np.clip((z - 0.75) * 4.0, 0, 1)
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    sum_base = Path(
        "output/20260505_171218-posttrain_sum_ref_truck/posttrain_gpop/posttrain_p80_sum_ref/list/Truck/default/model/test/ours_30007"
    )
    k50_base = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/posttrain_gpop/posttrain_p80_topk50_min5/list/Truck/default/model/test/ours_35009"
    )
    out_dir = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/compare_sumref30007_vs_topk50/charts"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_r = sum_base / "renders"
    k50_r = k50_base / "renders"
    gt_dir = k50_base / "gt"

    sum_files = {p.name for p in sum_r.glob("*.png")}
    k50_files = {p.name for p in k50_r.glob("*.png")}
    gt_files = {p.name for p in gt_dir.glob("*.png")}
    files = sorted(sum_files & k50_files & gt_files)
    if not files:
        raise RuntimeError("No aligned PNG files found among sum/k50/gt")

    kdiff_vals = []
    abs_sum_vals = []
    abs_k50_vals = []
    for n in files:
        gt = np.asarray(Image.open(gt_dir / n).convert("RGB"), dtype=np.float32) / 255.0
        s = np.asarray(Image.open(sum_r / n).convert("RGB"), dtype=np.float32) / 255.0
        k = np.asarray(Image.open(k50_r / n).convert("RGB"), dtype=np.float32) / 255.0
        dk = (k - s).mean(axis=2)
        kdiff_vals.append(np.abs(dk).max())
        abs_sum_vals.append(np.abs(s - gt).mean(axis=2).max())
        abs_k50_vals.append(np.abs(k - gt).mean(axis=2).max())

    kdiff_scale = max(float(np.percentile(np.array(kdiff_vals), 95)), 1e-6)
    err_scale = max(float(np.percentile(np.array(abs_sum_vals + abs_k50_vals), 95)), 1e-6)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    labels = [
        "GROUND TRUTH",
        "PRED SUM GPOP (POST-TRAIN)",
        "HEATMAP: K50 - SUM (RED=K50 HIGHER, BLUE=SUM HIGHER)",
        "PRED K=50 MEAN GPOP",
        "HEATMAP: |SUM - GT|",
        "HEATMAP: |K50 - GT|",
    ]

    for n in files:
        gt = np.asarray(Image.open(gt_dir / n).convert("RGB"), dtype=np.float32) / 255.0
        s = np.asarray(Image.open(sum_r / n).convert("RGB"), dtype=np.float32) / 255.0
        k = np.asarray(Image.open(k50_r / n).convert("RGB"), dtype=np.float32) / 255.0
        dk = (k - s).mean(axis=2)
        e_sum = np.abs(s - gt).mean(axis=2)
        e_k50 = np.abs(k - gt).mean(axis=2)

        p3 = rb_diverging(dk, kdiff_scale)
        p5 = err_cmap(e_sum, err_scale)
        p6 = err_cmap(e_k50, err_scale)

        panels = [gt, s, p3, k, p5, p6]
        pil_panels = [Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)) for p in panels]

        w, h = pil_panels[0].size
        header_h = 56
        footer_h = 32
        canvas = Image.new("RGB", (w * 6, h + header_h + footer_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)

        for j, p in enumerate(pil_panels):
            x = j * w
            canvas.paste(p, (x, header_h))
            draw.rectangle([x, 0, x + w, header_h], fill=(30, 30, 30))
            draw.text((x + 8, 16), labels[j], fill=(245, 245, 245), font=font)

        draw.text(
            (10, h + header_h + 4),
            f"view={n} | sum_iter=30007 | k50_iter=35009",
            fill=(220, 220, 220),
            font=font,
        )
        canvas.save(out_dir / n)

    print(f"WROTE {len(files)} charts to {out_dir}")


if __name__ == "__main__":
    main()