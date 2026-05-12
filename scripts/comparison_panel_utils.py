#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OKABE_ITO_ORANGE = np.array([0.90, 0.47, 0.00], dtype=np.float32)
OKABE_ITO_BLUE = np.array([0.00, 0.45, 0.70], dtype=np.float32)
LOW_INK = np.array([0.08, 0.08, 0.08], dtype=np.float32)


@dataclass
class ComparisonRecord:
    name: str
    gt: np.ndarray
    reference: np.ndarray
    candidate: np.ndarray
    reference_error: np.ndarray
    candidate_error: np.ndarray
    error_delta: np.ndarray
    score: float


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def aligned_png_names(reference_renders: Path, candidate_renders: Path, gt_dir: Path) -> list[str]:
    reference_files = {p.name for p in reference_renders.glob("*.png")}
    candidate_files = {p.name for p in candidate_renders.glob("*.png")}
    gt_files = {p.name for p in gt_dir.glob("*.png")}
    names = sorted(reference_files & candidate_files & gt_files)
    if not names:
        raise RuntimeError(f"No aligned PNG files found in {reference_renders}, {candidate_renders}, {gt_dir}")
    return names


def top_fraction_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    keep = max(1, int(flat.size * fraction))
    return float(np.partition(flat, flat.size - keep)[-keep:].mean())


def gaussian_blur(values: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return values
    half_width = max(1, int(radius * 3.0))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / (2.0 * radius * radius))
    kernel = kernel / kernel.sum()

    src = values.astype(np.float32, copy=False)
    padded_x = np.pad(src, ((0, 0), (half_width, half_width)), mode="edge")
    tmp = np.empty_like(src, dtype=np.float32)
    for row in range(src.shape[0]):
        tmp[row, :] = np.convolve(padded_x[row, :], kernel, mode="valid")

    padded_y = np.pad(tmp, ((half_width, half_width), (0, 0)), mode="edge")
    out = np.empty_like(src, dtype=np.float32)
    for col in range(src.shape[1]):
        out[:, col] = np.convolve(padded_y[:, col], kernel, mode="valid")
    return out


def clean_diverging_panel(
    signed_values: np.ndarray,
    scale: float,
    threshold_percentile: float = 92.0,
    blur_radius: float = 1.6,
    gamma: float = 0.72,
) -> np.ndarray:
    smoothed = gaussian_blur(signed_values, blur_radius)
    magnitude = np.abs(smoothed)
    threshold = float(np.percentile(magnitude, threshold_percentile))
    threshold = max(threshold, scale * 0.08)
    denom = max(scale - threshold, 1e-6)
    soft_mask = np.clip((magnitude - threshold) / denom, 0.0, 1.0)
    soft_mask = gaussian_blur(soft_mask, max(blur_radius * 0.35, 0.1))
    intensity = np.power(np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0), gamma) * np.clip(soft_mask, 0.0, 1.0)

    colors = np.where(smoothed[..., None] >= 0.0, OKABE_ITO_ORANGE, OKABE_ITO_BLUE)
    return np.clip(LOW_INK * (1.0 - intensity[..., None]) + colors * intensity[..., None], 0.0, 1.0)


def build_records(reference_renders: Path, candidate_renders: Path, gt_dir: Path) -> list[ComparisonRecord]:
    records: list[ComparisonRecord] = []
    for name in aligned_png_names(reference_renders, candidate_renders, gt_dir):
        gt = load_rgb(gt_dir / name)
        reference = load_rgb(reference_renders / name)
        candidate = load_rgb(candidate_renders / name)
        reference_error = np.abs(reference - gt).mean(axis=2)
        candidate_error = np.abs(candidate - gt).mean(axis=2)
        error_delta = candidate_error - reference_error
        score = top_fraction_mean(candidate_error, 0.05)
        records.append(
            ComparisonRecord(
                name=name,
                gt=gt,
                reference=reference,
                candidate=candidate,
                reference_error=reference_error,
                candidate_error=candidate_error,
                error_delta=error_delta,
                score=score,
            )
        )
    return sorted(records, key=lambda item: item.score, reverse=True)


def load_font(size: int = 22):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_panel(
    panels: Sequence[np.ndarray],
    labels: Sequence[str],
    footer: str,
    out_path: Path,
) -> None:
    pil_panels = [Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)) for panel in panels]
    width, height = pil_panels[0].size
    header_h = 58
    footer_h = 34
    font = load_font(21)
    canvas = Image.new("RGB", (width * len(pil_panels), height + header_h + footer_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for idx, panel in enumerate(pil_panels):
        x = idx * width
        canvas.paste(panel, (x, header_h))
        draw.rectangle([x, 0, x + width, header_h], fill=(28, 28, 28))
        draw.text((x + 8, 17), labels[idx], fill=(245, 245, 245), font=font)

    draw.text((10, height + header_h + 5), footer, fill=(220, 220, 220), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def generate_prediction_comparison(
    reference_base: Path,
    candidate_base: Path,
    out_dir: Path,
    reference_label: str,
    candidate_label: str,
    reference_iter: int | str,
    candidate_iter: int | str,
    worst_count: int = 12,
    threshold_percentile: float = 92.0,
    blur_radius: float = 1.6,
) -> dict:
    reference_renders = reference_base / "renders"
    candidate_renders = candidate_base / "renders"
    gt_dir = candidate_base / "gt"
    records = build_records(reference_renders, candidate_renders, gt_dir)
    selected = records[: max(1, worst_count)]

    scale = max(float(np.percentile([np.abs(r.error_delta).max() for r in records], 95)), 1e-6)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        "GROUND TRUTH",
        reference_label,
        candidate_label,
        "CLEAN ERROR DELTA: ORANGE=CANDIDATE WORSE, BLUE=REFERENCE WORSE",
    ]
    summary_rows = []
    for rank, record in enumerate(selected, start=1):
        delta_panel = clean_diverging_panel(
            record.error_delta,
            scale=scale,
            threshold_percentile=threshold_percentile,
            blur_radius=blur_radius,
        )
        save_panel(
            [record.gt, record.reference, record.candidate, delta_panel],
            labels,
            (
                f"rank={rank} | view={record.name} | ref_iter={reference_iter} | "
                f"candidate_iter={candidate_iter} | worst5pct_mae={record.score:.6f}"
            ),
            charts_dir / f"{rank:02d}_{record.name}",
        )
        summary_rows.append(
            {
                "rank": rank,
                "image": record.name,
                "candidate_worst5pct_mae": record.score,
                "candidate_mean_mae": float(record.candidate_error.mean()),
                "reference_mean_mae": float(record.reference_error.mean()),
                "mean_error_delta": float(record.error_delta.mean()),
                "abs_delta_95pct": float(np.percentile(np.abs(record.error_delta), 95)),
            }
        )

    with (out_dir / "worst_views.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "charts_dir": str(charts_dir),
        "worst_count": len(selected),
        "total_aligned_images": len(records),
        "threshold_percentile": threshold_percentile,
        "blur_radius": blur_radius,
        "scale": scale,
        "worst_views": summary_rows,
    }
