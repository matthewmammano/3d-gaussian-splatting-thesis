#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rb_diverging(x: np.ndarray, scale: float, boost: float = 4.0, gamma: float = 0.55) -> np.ndarray:
    # Aggressive contrast boost so subtle differences are much more visible.
    z = np.clip((x / scale) * boost, -1.0, 1.0)
    mag = np.power(np.abs(z), gamma)
    out = np.zeros(z.shape + (3,), dtype=np.float32)
    pos = z >= 0
    out[..., 0] = np.where(pos, mag, 0.0)
    out[..., 2] = np.where(~pos, mag, 0.0)
    out[..., 1] = 0.05
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    sum_base = Path(
        "output/20260505_171218-posttrain_sum_ref_truck/posttrain_gpop/posttrain_p80_sum_ref/list/Truck/default/model/test/ours_30007"
    )
    k10_base = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/posttrain_gpop/posttrain_p80_topk10_min5/list/Truck/default/model/test/ours_35009"
    )
    out_dir = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/compare_sumref30007_vs_topk10_boosted/charts"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_r = sum_base / "renders"
    k10_r = k10_base / "renders"
    gt_dir = k10_base / "gt"

    sum_files = {p.name for p in sum_r.glob("*.png")}
    k10_files = {p.name for p in k10_r.glob("*.png")}
    gt_files = {p.name for p in gt_dir.glob("*.png")}
    files = sorted(sum_files & k10_files & gt_files)
    if not files:
        raise RuntimeError("No aligned PNG files found among sum/k10/gt")

    diff_scales = []
    for name in files:
        gt = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_img = np.asarray(Image.open(sum_r / name).convert("RGB"), dtype=np.float32) / 255.0
        k10_img = np.asarray(Image.open(k10_r / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_err = np.abs(sum_img - gt).mean(axis=2)
        k10_err = np.abs(k10_img - gt).mean(axis=2)
        diff_scales.append(np.abs(k10_err - sum_err).max())

    # Lower percentile + boost creates a much stronger color response.
    diff_scale = max(float(np.percentile(np.array(diff_scales), 70)), 1e-6)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    labels = [
        "GROUND TRUTH",
        "PRED SUM GPOP (POST-TRAIN)",
        "PRED K=10 MEAN GPOP",
        "ERROR DIFF TO GT (BOOSTED) (RED=K10 WORSE, BLUE=SUM WORSE)",
    ]

    for name in files:
        gt = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_img = np.asarray(Image.open(sum_r / name).convert("RGB"), dtype=np.float32) / 255.0
        k10_img = np.asarray(Image.open(k10_r / name).convert("RGB"), dtype=np.float32) / 255.0

        sum_err = np.abs(sum_img - gt).mean(axis=2)
        k10_err = np.abs(k10_img - gt).mean(axis=2)
        err_diff = k10_err - sum_err
        diff_panel = rb_diverging(err_diff, diff_scale, boost=4.0, gamma=0.55)

        panels = [gt, sum_img, k10_img, diff_panel]
        pil_panels = [Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)) for panel in panels]

        width, height = pil_panels[0].size
        header_h = 56
        footer_h = 32
        canvas = Image.new("RGB", (width * 4, height + header_h + footer_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)

        for idx, panel in enumerate(pil_panels):
            x = idx * width
            canvas.paste(panel, (x, header_h))
            draw.rectangle([x, 0, x + width, header_h], fill=(30, 30, 30))
            draw.text((x + 8, 16), labels[idx], fill=(245, 245, 245), font=font)

        draw.text(
            (10, height + header_h + 4),
            f"view={name} | sum_iter=30007 | k10_iter=35009",
            fill=(220, 220, 220),
            font=font,
        )
        canvas.save(out_dir / name)

    print(f"WROTE {len(files)} charts to {out_dir}")


if __name__ == "__main__":
    main()