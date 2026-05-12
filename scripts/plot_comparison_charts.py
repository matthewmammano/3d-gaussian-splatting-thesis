#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path('comparison_charts')


def load_rows(csv_name: str):
    rows = []
    with (OUT / csv_name).open() as f:
        r = csv.DictReader(f)
        for row in r:
            row['order_key'] = int(float(row['order_key']))
            for k in ['train_psnr', 'test_psnr', 'test_ssim', 'test_lpips']:
                row[k] = float(row[k]) if row[k] not in ('', None) else None
            rows.append(row)
    return sorted(rows, key=lambda x: x['order_key'])


def save_psnr_plot(rows, title: str, xlabel: str, out_name: str):
    x = [r['method'] for r in rows]
    train = [r['train_psnr'] for r in rows]
    test = [r['test_psnr'] for r in rows]

    plt.figure(figsize=(14, 5))
    plt.plot(x, train, marker='o', label='Train PSNR')
    plt.plot(x, test, marker='o', label='Test PSNR')
    plt.xticks(rotation=35, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('PSNR (dB)')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=180)
    plt.close()


def save_ssim_lpips_plot(rows, title: str, xlabel: str, out_name: str):
    x = [r['method'] for r in rows]
    ssim = [r['test_ssim'] for r in rows]
    lpips = [r['test_lpips'] for r in rows]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(x, ssim, marker='o', color='#2a9d8f')
    ax1.set_ylabel('Test SSIM', color='#2a9d8f')
    ax1.tick_params(axis='y', labelcolor='#2a9d8f')
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels(x, rotation=35, ha='right')
    ax1.set_xlabel(xlabel)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, lpips, marker='s', color='#e76f51')
    ax2.set_ylabel('Test LPIPS (lower better)', color='#e76f51')
    ax2.tick_params(axis='y', labelcolor='#e76f51')

    plt.title(title)
    fig.tight_layout()
    plt.savefig(OUT / out_name, dpi=180)
    plt.close(fig)


def main():
    topk = load_rows('topk_sweep_metrics.csv')
    agg = load_rows('agg_sweep_metrics.csv')

    save_psnr_plot(topk, 'TopK Sweep: Train vs Test PSNR', 'TopK method (min_visible=5)', 'topk_psnr.png')
    save_ssim_lpips_plot(topk, 'TopK Sweep: Test SSIM and LPIPS', 'TopK method (min_visible=5)', 'topk_ssim_lpips.png')

    save_psnr_plot(agg, 'GPOP Aggregation Sweep: Train vs Test PSNR', 'Aggregation method', 'agg_psnr.png')
    save_ssim_lpips_plot(agg, 'GPOP Aggregation Sweep: Test SSIM and LPIPS', 'Aggregation method', 'agg_ssim_lpips.png')

    print('WROTE plots to comparison_charts/')


if __name__ == '__main__':
    main()
