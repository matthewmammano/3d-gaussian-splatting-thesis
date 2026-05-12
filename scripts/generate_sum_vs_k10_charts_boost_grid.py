#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rb_diverging(x: np.ndarray, scale: float, boost: float) -> np.ndarray:
    z = np.clip((x / scale) * boost, -1.0, 1.0)
    out = np.zeros(z.shape + (3,), dtype=np.float32)
    pos = z >= 0
    out[..., 0] = np.where(pos, np.abs(z), 0.0)
    out[..., 2] = np.where(~pos, np.abs(z), 0.0)
    out[..., 1] = 0.05
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    boosts = [1.1, 1.25, 1.4, 1.6, 1.8, 2.0]

    sum_base = Path(
        "output/20260505_171218-posttrain_sum_ref_truck/posttrain_gpop/posttrain_p80_sum_ref/list/Truck/default/model/test/ours_30007"
    )
    k10_base = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/posttrain_gpop/posttrain_p80_topk10_min5/list/Truck/default/model/test/ours_35009"
    )
    root_out = Path(
        "output/20260505_112916-topk_fullsweep_min5_p80_truck/compare_sumref30007_vs_topk10_boost_grid"
    )
    root_out.mkdir(parents=True, exist_ok=True)

    sum_r = sum_base / "renders"
    k10_r = k10_base / "renders"
    gt_dir = k10_base / "gt"

    sum_files = {p.name for p in sum_r.glob("*.png")}
    k10_files = {p.name for p in k10_r.glob("*.png")}
    gt_files = {p.name for p in gt_dir.glob("*.png")}
    files = sorted(sum_files & k10_files & gt_files)
    if not files:
        raise RuntimeError("No aligned PNG files found among sum/k10/gt")

    diff_maxes = []
    cache = {}
    for name in files:
        gt = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_img = np.asarray(Image.open(sum_r / name).convert("RGB"), dtype=np.float32) / 255.0
        k10_img = np.asarray(Image.open(k10_r / name).convert("RGB"), dtype=np.float32) / 255.0
        sum_err = np.abs(sum_img - gt).mean(axis=2)
        k10_err = np.abs(k10_img - gt).mean(axis=2)
        err_diff = k10_err - sum_err
        diff_maxes.append(np.abs(err_diff).max())
        cache[name] = (gt, sum_img, k10_img, err_diff)

    # Keep scale fixed across all boost variants for fair visual comparison.
    diff_scale = max(float(np.percentile(np.array(diff_maxes), 95)), 1e-6)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    for boost in boosts:
        boost_tag = str(boost).replace(".", "p")
        out_dir = root_out / f"boost_{boost_tag}" / "charts"
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = [
            "GROUND TRUTH",
            "PRED SUM GPOP (POST-TRAIN)",
            "PRED K=10 MEAN GPOP",
            f"ERROR DIFF TO GT (BOOST={boost}x) (RED=K10 WORSE, BLUE=SUM WORSE)",
        ]

        for name in files:
            gt, sum_img, k10_img, err_diff = cache[name]
            diff_panel = rb_diverging(err_diff, diff_scale, boost=boost)

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
                f"view={name} | sum_iter=30007 | k10_iter=35009 | scale=95pct",
                fill=(220, 220, 220),
                font=font,
            )
            canvas.save(out_dir / name)

        print(f"WROTE {len(files)} charts to {out_dir}")


if __name__ == "__main__":
    main()
