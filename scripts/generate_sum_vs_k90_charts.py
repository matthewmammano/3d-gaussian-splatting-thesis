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


def main() -> None:
    sum_base = Path(
        "output/20260505_171218-posttrain_sum_ref_truck/posttrain_gpop/posttrain_p80_sum_ref/list/Truck/default/model/test/ours_30007"
    )
    k90_base = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/posttrain_gpop/posttrain_p80_topk90_min5/list/Truck/default/model/test/ours_35009"
    )
    out_dir = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/compare_sumref30007_vs_topk90/charts"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_r = sum_base / "renders"
    k90_r = k90_base / "renders"
    gt_dir = k90_base / "gt"

    sum_files = {p.name for p in sum_r.glob("*.png")}
    k90_files = {p.name for p in k90_r.glob("*.png")}
    gt_files = {p.name for p in gt_dir.glob("*.png")}
    files = sorted(sum_files & k90_files & gt_files)
    if not files:
        raise RuntimeError("No aligned PNG files found among sum/k90/gt")

    diff_scales = []
    for name in files:
        gt = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_img = np.asarray(Image.open(sum_r / name).convert("RGB"), dtype=np.float32) / 255.0
        k90_img = np.asarray(Image.open(k90_r / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_err = np.abs(sum_img - gt).mean(axis=2)
        k90_err = np.abs(k90_img - gt).mean(axis=2)
        diff_scales.append(np.abs(k90_err - sum_err).max())

    diff_scale = max(float(np.percentile(np.array(diff_scales), 95)), 1e-6)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    labels = [
        "GROUND TRUTH",
        "PRED SUM GPOP (POST-TRAIN)",
        "PRED K=90 MEAN GPOP",
        "ERROR DIFF TO GT (RED=K90 WORSE, BLUE=SUM WORSE)",
    ]

    for name in files:
        gt = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_img = np.asarray(Image.open(sum_r / name).convert("RGB"), dtype=np.float32) / 255.0
        k90_img = np.asarray(Image.open(k90_r / name).convert("RGB"), dtype=np.float32) / 255.0

        sum_err = np.abs(sum_img - gt).mean(axis=2)
        k90_err = np.abs(k90_img - gt).mean(axis=2)
        err_diff = k90_err - sum_err
        diff_panel = rb_diverging(err_diff, diff_scale)

        panels = [gt, sum_img, k90_img, diff_panel]
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
            f"view={name} | sum_iter=30007 | k90_iter=35009",
            fill=(220, 220, 220),
            font=font,
        )
        canvas.save(out_dir / name)

    print(f"WROTE {len(files)} charts to {out_dir}")


if __name__ == "__main__":
    main()