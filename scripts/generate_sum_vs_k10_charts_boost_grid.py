#!/usr/bin/env python3
from pathlib import Path

from comparison_panel_utils import generate_prediction_comparison


def main() -> None:
    root = Path("output/20260505_112916-topk_fullsweep_min5_p80_truck/compare_sumref30007_vs_topk10_threshold_grid")
    for threshold in (88.0, 92.0, 96.0):
        summary = generate_prediction_comparison(
            reference_base=Path(
                "output/20260505_171218-posttrain_sum_ref_truck/posttrain_gpop/posttrain_p80_sum_ref/list/Truck/default/model/test/ours_30007"
            ),
            candidate_base=Path(
                "output/20260505_112916-topk_fullsweep_min5_p80_truck/posttrain_gpop/posttrain_p80_topk10_min5/list/Truck/default/model/test/ours_35009"
            ),
            out_dir=root / f"threshold_{int(threshold)}",
            reference_label="SUM GPOP",
            candidate_label="TOPK10 GPOP",
            reference_iter=30007,
            candidate_iter=35009,
            worst_count=12,
            threshold_percentile=threshold,
            blur_radius=1.6,
        )
        print(f"WROTE {summary['worst_count']} charts to {summary['charts_dir']}")


if __name__ == "__main__":
    main()
