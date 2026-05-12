#!/usr/bin/env python3
"""Unified experiment runner: optimize / evaluate / analyze / pipeline."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Allow direct execution: python scripts/run_experiment.py --config ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._core import (
    build_train_command,
    ensure_dir,
    generate_model_comparison,
    now_tag,
    read_json,
    resolve_scene_jobs,
    run_logged_command,
    run_render_metric,
    visualize_high_error,
    write_json,
    write_run_manifest,
)


def _artifact_key(step_id: str, variant: str, group: str, scene: str) -> str:
    return f"{step_id}|{variant}|{group}|{scene}"


def _read_ply_vertex_count(ply_path: Path) -> int | None:
    if not ply_path.exists():
        return None
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("element vertex"):
                parts = line.split()
                if len(parts) >= 3:
                    return int(parts[2])
                return None
            if line.strip() == "end_header":
                break
    return None


def _read_last_pruned_count(stdout_log_path: Path) -> int | None:
    if not stdout_log_path.exists():
        return None
    last = None
    pat = re.compile(r"GaussianPOP prune ratio\s+[0-9.]+\s+-> pruned\s+(\d+)")
    with open(stdout_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                last = int(m.group(1))
    return last


def _remove_transient_training_artifacts(
    model_dir: Path, iteration: int, global_log_path: Path | None = None
) -> None:
    """Remove per-cycle artifacts after a later checkpoint has superseded them."""
    paths = [
        model_dir / f"chkpnt{iteration}.pth",
        model_dir / "point_cloud" / f"iteration_{iteration}",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            if global_log_path is not None:
                with open(global_log_path, "a", encoding="utf-8") as glog:
                    glog.write(
                        f"[{datetime.now().isoformat()}] WARN failed to remove transient artifact {path}: {exc}\n"
                    )


def _resolve_checkpoint(
    step_cfg: Dict, scene_key: Tuple[str, str], artifacts: Dict
) -> str | None:
    resume = step_cfg.get("resume_from")
    if not resume:
        return None

    group, scene = scene_key
    src_step = resume["step_id"]
    src_variant = resume.get("variant", "default")
    it = int(resume["checkpoint_iteration"])
    k = _artifact_key(src_step, src_variant, group, scene)
    if k not in artifacts:
        raise KeyError(f"resume_from artifact not found: {k}")
    model_dir = Path(artifacts[k]["model_dir"])
    ckpt = model_dir / f"chkpnt{it}.pth"
    return str(ckpt)


def _resolve_external_source_model_dir(
    step_cfg: Dict, group: str, scene: str
) -> Path | None:
    source = step_cfg.get("source_model_dir") or step_cfg.get("source_model_dirs")
    if not source:
        return None
    if isinstance(source, dict):
        source = (
            source.get(f"{group}/{scene}")
            or source.get(scene)
            or source.get(group)
            or source.get("default")
        )
    return Path(source) if source else None


def _symlink_existing(src: Path, dst: Path) -> None:
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src:
            return
        raise FileExistsError(f"Refusing to replace existing path: {dst}")
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _posttrain_gpop_overrides(step_cfg: Dict) -> Dict:
    """Collect optional GaussianPOP aggregation overrides for post-train cycles.

    Post-train steps historically defined these in steps[].phases[0].gpop,
    but the cycle runner synthesizes phases internally. This helper forwards
    those fields into generated cycle commands.
    """
    supported = (
        "error_agg_method",
        "error_topk_percent",
        "error_lp_p",
        "error_visibility_mode",
        "error_min_visible_views",
    )
    overrides = {}

    step_gpop = step_cfg.get("gpop", {})
    if isinstance(step_gpop, dict):
        for key in supported:
            if key in step_gpop:
                overrides[key] = step_gpop[key]

    for phase in step_cfg.get("phases", []):
        phase_gpop = phase.get("gpop", {})
        if isinstance(phase_gpop, dict):
            for key in supported:
                if key in phase_gpop:
                    overrides[key] = phase_gpop[key]
            break

    return overrides


def run_optimize_step(
    step_cfg: Dict,
    run_root: Path,
    jobs: List[Tuple[str, str, str]],
    defaults: Dict,
    execution: Dict,
    artifacts: Dict,
    global_log_path: Path,
):
    step_id = step_cfg["id"]
    variants = step_cfg.get(
        "variants", [{"name": "default", "phases": step_cfg.get("phases", [])}]
    )

    for group, scene, scene_path in jobs:
        for variant in variants:
            variant_name = variant["name"]
            job_dir = run_root / "optimize" / step_id / group / scene / variant_name
            model_dir = job_dir / "model"
            ensure_dir(model_dir)

            if (
                execution.get("skip_existing", False)
                and (model_dir / "point_cloud").exists()
            ):
                status = {
                    "step": step_id,
                    "variant": variant_name,
                    "group": group,
                    "scene": scene,
                    "status": "skipped_existing",
                }
                write_json(job_dir / "status.json", status)
                artifacts[_artifact_key(step_id, variant_name, group, scene)] = {
                    "model_dir": str(model_dir)
                }
                continue

            phase_results = []
            for i, phase in enumerate(variant.get("phases", []), start=1):
                if phase.get("op", "train") != "train":
                    phase_results.append(
                        {
                            "phase_index": i,
                            "op": phase.get("op"),
                            "rc": 0,
                            "note": "ignored",
                        }
                    )
                    continue

                phase_local = dict(phase)
                ckpt = _resolve_checkpoint(step_cfg, (group, scene), artifacts)
                if ckpt and not phase_local.get("start_checkpoint"):
                    phase_local["start_checkpoint"] = ckpt

                cmd = build_train_command(
                    scene_path=scene_path,
                    model_path=str(model_dir),
                    phase=phase_local,
                    defaults=defaults,
                )
                rc = run_logged_command(
                    cmd, job_dir, f"phase_{i}_train", global_log_path=global_log_path
                )
                phase_results.append(
                    {
                        "phase_index": i,
                        "op": "train",
                        "rc": rc,
                        "checkpoint": phase_local.get("start_checkpoint"),
                    }
                )
                if rc != 0:
                    break

            ok = all(
                x.get("rc", 1) == 0 for x in phase_results if x.get("op") == "train"
            )
            status = {
                "step": step_id,
                "variant": variant_name,
                "group": group,
                "scene": scene,
                "scene_path": scene_path,
                "model_dir": str(model_dir),
                "status": "ok" if ok else "failed",
                "phases": phase_results,
            }
            write_json(job_dir / "status.json", status)

            if ok:
                artifacts[_artifact_key(step_id, variant_name, group, scene)] = {
                    "model_dir": str(model_dir)
                }


def run_evaluate_step(
    step_cfg: Dict,
    run_root: Path,
    jobs: List[Tuple[str, str, str]],
    artifacts: Dict,
    global_log_path: Path,
):
    step_id = step_cfg["id"]
    src_step = step_cfg["from_step_id"]
    src_variant = step_cfg.get("from_variant", "default")
    iteration = int(step_cfg.get("iteration", 30000))

    for group, scene, scene_path in jobs:
        k = _artifact_key(src_step, src_variant, group, scene)
        if k not in artifacts:
            with open(global_log_path, "a", encoding="utf-8") as glog:
                glog.write(
                    f"[{datetime.now().isoformat()}] WARN evaluate skipping {step_id}/{group}/{scene}: source artifact missing ({k})\n"
                )
            continue
        artifact = artifacts[k]
        model_dir = artifact["model_dir"]
        # Use final_iteration from artifact if config didn't specify one explicitly
        if "iteration" not in step_cfg:
            iteration = int(artifact.get("final_iteration", iteration))

        job_dir = run_root / "evaluate" / step_id / group / scene / src_variant
        ensure_dir(job_dir)
        result = run_render_metric(
            scene_path=scene_path,
            model_path=model_dir,
            iteration=iteration,
            job_dir=job_dir,
            global_log_path=global_log_path,
        )
        status = {
            "step": step_id,
            "group": group,
            "scene": scene,
            "source_step": src_step,
            "source_variant": src_variant,
            "iteration": iteration,
            **result,
        }
        write_json(job_dir / "status.json", status)


def run_external_model_step(
    step_cfg: Dict,
    run_root: Path,
    jobs: List[Tuple[str, str, str]],
    artifacts: Dict,
    global_log_path: Path,
):
    step_id = step_cfg["id"]
    variant = step_cfg.get("variant_name", "default")
    iteration = int(step_cfg.get("iteration", step_cfg.get("source_iteration", 30000)))

    for group, scene, _scene_path in jobs:
        model_dir = _resolve_external_source_model_dir(step_cfg, group, scene)
        if model_dir is None:
            with open(global_log_path, "a", encoding="utf-8") as glog:
                glog.write(
                    f"[{datetime.now().isoformat()}] WARN external_model skipping {step_id}/{group}/{scene}: missing source_model_dir\n"
                )
            continue
        source_model_dir = model_dir
        ckpt = source_model_dir / f"chkpnt{iteration}.pth"
        ply = (
            source_model_dir
            / "point_cloud"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
        cfg_args = source_model_dir / "cfg_args"
        if not ckpt.exists():
            with open(global_log_path, "a", encoding="utf-8") as glog:
                glog.write(
                    f"[{datetime.now().isoformat()}] WARN external_model skipping {step_id}/{group}/{scene}: missing checkpoint {ckpt}\n"
                )
            continue
        if not ply.exists() or not cfg_args.exists():
            with open(global_log_path, "a", encoding="utf-8") as glog:
                glog.write(
                    f"[{datetime.now().isoformat()}] WARN external_model skipping {step_id}/{group}/{scene}: missing required baseline files ply={ply.exists()} cfg_args={cfg_args.exists()}\n"
                )
            continue

        if bool(step_cfg.get("link_into_run", True)):
            job_dir = run_root / "external_model" / step_id / group / scene / variant
            linked_model_dir = job_dir / "model"
            _symlink_existing(ckpt, linked_model_dir / f"chkpnt{iteration}.pth")
            _symlink_existing(cfg_args, linked_model_dir / "cfg_args")
            _symlink_existing(
                ply,
                linked_model_dir
                / "point_cloud"
                / f"iteration_{iteration}"
                / "point_cloud.ply",
            )
            for name in ("cameras.json", "input.ply", f"chkpnt{iteration}.txt"):
                src = source_model_dir / name
                if src.exists():
                    _symlink_existing(src, linked_model_dir / name)
            model_dir = linked_model_dir
            write_json(
                job_dir / "status.json",
                {
                    "step": step_id,
                    "variant": variant,
                    "group": group,
                    "scene": scene,
                    "status": "ok",
                    "source_model_dir": str(source_model_dir),
                    "model_dir": str(model_dir),
                    "final_iteration": iteration,
                    "link_into_run": True,
                },
            )

        artifacts[_artifact_key(step_id, variant, group, scene)] = {
            "model_dir": str(model_dir),
            "source_model_dir": str(source_model_dir),
            "final_iteration": iteration,
        }


def run_analyze_step(
    step_cfg: Dict,
    run_root: Path,
    jobs: List[Tuple[str, str, str]],
    artifacts: Dict,
    global_log_path: Path,
):
    step_id = step_cfg["id"]
    src_step = step_cfg["from_step_id"]
    src_variant = step_cfg.get("from_variant", "default")
    iteration = int(step_cfg.get("iteration", 30000))
    top_percent = float(step_cfg.get("top_percent", 5.0))
    worst_count = int(step_cfg.get("worst_count", 12))
    threshold_percentile = float(step_cfg.get("threshold_percentile", 92.0))
    blur_radius = float(step_cfg.get("blur_radius", 1.6))

    for group, scene, _scene_path in jobs:
        with open(global_log_path, "a", encoding="utf-8") as glog:
            glog.write(
                f"[{datetime.now().isoformat()}] ANALYZE_START step={step_id} group={group} scene={scene}\n"
            )
        k = _artifact_key(src_step, src_variant, group, scene)
        if k not in artifacts:
            with open(global_log_path, "a", encoding="utf-8") as glog:
                glog.write(
                    f"[{datetime.now().isoformat()}] WARN analyze skipping {step_id}/{group}/{scene}: source artifact missing ({k})\n"
                )
            continue
        model_dir = artifacts[k]["model_dir"]

        job_dir = run_root / "analyze" / step_id / group / scene / src_variant
        ensure_dir(job_dir)
        summary = visualize_high_error(
            model_path=model_dir,
            out_dir=job_dir,
            iteration=iteration,
            top_percent=top_percent,
            worst_count=worst_count,
            threshold_percentile=threshold_percentile,
            blur_radius=blur_radius,
        )
        write_json(
            job_dir / "status.json",
            {"step": step_id, "group": group, "scene": scene, "summary": summary},
        )
        with open(global_log_path, "a", encoding="utf-8") as glog:
            glog.write(
                f"[{datetime.now().isoformat()}] ANALYZE_END step={step_id} group={group} scene={scene}\n"
            )


def run_compare_step(
    step_cfg: Dict,
    run_root: Path,
    jobs: List[Tuple[str, str, str]],
    artifacts: Dict,
    global_log_path: Path,
):
    step_id = step_cfg["id"]
    candidate_step = step_cfg["from_step_id"]
    candidate_variant = step_cfg.get("from_variant", "default")
    reference_step = step_cfg["reference_step_id"]
    reference_variant = step_cfg.get("reference_variant", "default")
    worst_count = int(step_cfg.get("worst_count", 12))
    threshold_percentile = float(step_cfg.get("threshold_percentile", 92.0))
    blur_radius = float(step_cfg.get("blur_radius", 1.6))

    for group, scene, _scene_path in jobs:
        candidate_key = _artifact_key(candidate_step, candidate_variant, group, scene)
        reference_key = _artifact_key(reference_step, reference_variant, group, scene)
        if candidate_key not in artifacts or reference_key not in artifacts:
            with open(global_log_path, "a", encoding="utf-8") as glog:
                glog.write(
                    f"[{datetime.now().isoformat()}] WARN compare skipping {step_id}/{group}/{scene}: missing artifact candidate={candidate_key in artifacts} reference={reference_key in artifacts}\n"
                )
            continue

        candidate = artifacts[candidate_key]
        reference = artifacts[reference_key]
        candidate_iteration = int(
            step_cfg.get("iteration", candidate.get("final_iteration", 30000))
        )
        reference_iteration = int(
            step_cfg.get("reference_iteration", reference.get("final_iteration", 30000))
        )
        job_dir = run_root / "compare" / step_id / group / scene / candidate_variant
        ensure_dir(job_dir)
        summary = generate_model_comparison(
            reference_model_path=str(reference["model_dir"]),
            reference_iteration=reference_iteration,
            candidate_model_path=str(candidate["model_dir"]),
            candidate_iteration=candidate_iteration,
            out_dir=job_dir,
            reference_label=str(step_cfg.get("reference_label", reference_step)),
            candidate_label=str(step_cfg.get("candidate_label", candidate_step)),
            worst_count=worst_count,
            threshold_percentile=threshold_percentile,
            blur_radius=blur_radius,
        )
        write_json(
            job_dir / "status.json",
            {
                "step": step_id,
                "group": group,
                "scene": scene,
                "candidate_step": candidate_step,
                "reference_step": reference_step,
                "candidate_iteration": candidate_iteration,
                "reference_iteration": reference_iteration,
                "summary": summary,
            },
        )


def run_posttrain_gpop_step(
    step_cfg: Dict,
    run_root: Path,
    jobs: List[Tuple[str, str, str]],
    defaults: Dict,
    execution: Dict,
    artifacts: Dict,
    global_log_path: Path,
):
    step_id = step_cfg["id"]
    src_step = step_cfg.get("from_step_id")
    src_variant = step_cfg.get("from_variant", "default")
    src_iteration = int(
        step_cfg.get("from_iteration", step_cfg.get("source_iteration", 30000))
    )
    c_cycles = int(step_cfg.get("c_cycles", 8))
    prune_ratio_mode = step_cfg.get("prune_ratio_mode", "total_across_cycles")
    cycle_prune_ratio = float(step_cfg.get("cycle_prune_ratio", 0.9))
    cycle_prune_ratios = step_cfg.get("cycle_prune_ratios", [])
    views_per_quant = int(step_cfg.get("views_per_quant", 0))
    fine_tune_iterations = int(step_cfg.get("fine_tune_iterations", 5000))
    keep_intermediate_checkpoints = bool(
        step_cfg.get(
            "keep_intermediate_checkpoints",
            defaults.get("keep_intermediate_checkpoints", False),
        )
    )
    output_variant = step_cfg.get("variant_name", "default")
    gpop_overrides = _posttrain_gpop_overrides(step_cfg)

    if prune_ratio_mode not in {"total_across_cycles", "per_cycle"}:
        raise ValueError(
            f"Unknown prune_ratio_mode: {prune_ratio_mode}. Use 'total_across_cycles' or 'per_cycle'."
        )

    for group, scene, scene_path in jobs:
        if src_step:
            src_key = _artifact_key(src_step, src_variant, group, scene)
            if src_key not in artifacts:
                with open(global_log_path, "a", encoding="utf-8") as glog:
                    glog.write(
                        f"[{datetime.now().isoformat()}] WARN posttrain skipping {step_id}/{group}/{scene}: source artifact missing ({src_key})\n"
                    )
                continue
            src_model_dir = Path(artifacts[src_key]["model_dir"])
            source_status = {
                "step_id": src_step,
                "variant": src_variant,
                "checkpoint_iteration": src_iteration,
                "model_dir": str(src_model_dir),
            }
        else:
            src_model_dir = _resolve_external_source_model_dir(step_cfg, group, scene)
            if src_model_dir is None:
                with open(global_log_path, "a", encoding="utf-8") as glog:
                    glog.write(
                        f"[{datetime.now().isoformat()}] WARN posttrain skipping {step_id}/{group}/{scene}: missing from_step_id or source_model_dir\n"
                    )
                continue
            source_status = {
                "checkpoint_iteration": src_iteration,
                "model_dir": str(src_model_dir),
            }

        src_ckpt = src_model_dir / f"chkpnt{src_iteration}.pth"
        if not src_ckpt.exists():
            with open(global_log_path, "a", encoding="utf-8") as glog:
                glog.write(
                    f"[{datetime.now().isoformat()}] WARN posttrain skipping {step_id}/{group}/{scene}: source checkpoint missing ({src_ckpt})\n"
                )
            continue

        job_dir = run_root / "posttrain_gpop" / step_id / group / scene / output_variant
        model_dir = job_dir / "model"
        ensure_dir(model_dir)

        if (
            execution.get("skip_existing", False)
            and (model_dir / "point_cloud").exists()
        ):
            existing_status = {}
            status_path = job_dir / "status.json"
            if status_path.exists():
                try:
                    existing_status = read_json(status_path)
                except Exception:
                    pass
            # Only write a skipped marker if no real status exists yet
            if existing_status.get("status") != "ok":
                write_json(
                    job_dir / "status.json",
                    {
                        "step": step_id,
                        "group": group,
                        "scene": scene,
                        "status": "skipped_existing",
                        "source": source_status,
                    },
                )
            artifact_entry: Dict[str, object] = {"model_dir": str(model_dir)}
            if "final_iteration" in existing_status:
                artifact_entry["final_iteration"] = existing_status["final_iteration"]
            else:
                # Infer final_iteration from the highest checkpoint in the model dir
                import glob as _glob

                ckpts = sorted(_glob.glob(str(model_dir / "chkpnt*.pth")))
                if ckpts:
                    import re as _re

                    m = _re.search(r"chkpnt(\d+)\.pth$", ckpts[-1])
                    if m:
                        artifact_entry["final_iteration"] = int(m.group(1))
            artifacts[_artifact_key(step_id, output_variant, group, scene)] = (
                artifact_entry
            )
            continue

        current_iter = src_iteration
        current_ckpt = str(src_ckpt)
        cycle_results: List[Dict[str, object]] = []
        source_ply = (
            src_model_dir
            / "point_cloud"
            / f"iteration_{src_iteration}"
            / "point_cloud.ply"
        )
        source_gaussians = _read_ply_vertex_count(source_ply)
        target_total_pruned = (
            int(source_gaussians * cycle_prune_ratio)
            if (
                source_gaussians is not None
                and not cycle_prune_ratios
                and prune_ratio_mode == "total_across_cycles"
            )
            else None
        )
        actual_total_pruned = 0

        for cycle_idx in range(1, c_cycles + 1):
            if cycle_prune_ratios and cycle_idx - 1 < len(cycle_prune_ratios):
                ratio = float(cycle_prune_ratios[cycle_idx - 1])
            elif prune_ratio_mode == "per_cycle":
                ratio = cycle_prune_ratio
            else:
                # Paper-aligned iterative behavior: treat cycle_prune_ratio as
                # total target P across all cycles and split pruning budget.
                # Track counts from stdout so transient cycles do not need PLYs.
                current_gaussians = (
                    max(source_gaussians - actual_total_pruned, 0)
                    if source_gaussians is not None
                    else None
                )

                if (
                    target_total_pruned is None
                    or current_gaussians is None
                    or current_gaussians <= 0
                ):
                    ratio = cycle_prune_ratio
                else:
                    remaining_target = max(target_total_pruned - actual_total_pruned, 0)
                    remaining_cycles = c_cycles - cycle_idx + 1
                    step_target_pruned = (
                        remaining_target + remaining_cycles - 1
                    ) // remaining_cycles
                    ratio = min(max(step_target_pruned / current_gaussians, 0.0), 1.0)

            target_iter = current_iter + 1
            keep_cycle_outputs = keep_intermediate_checkpoints or (
                fine_tune_iterations <= 0 and cycle_idx == c_cycles
            )
            phase = {
                "name": f"posttrain_cycle_{cycle_idx}",
                "iterations": target_iter,
                "start_checkpoint": current_ckpt,
                "save_point_cloud": keep_cycle_outputs,
                "run_tests": bool(step_cfg.get("run_cycle_tests", False)),
                "checkpoint_iterations": [target_iter],
                "gpop": {
                    "enabled": True,
                    "schedule_iterations": [target_iter],
                    "prune_ratios": [ratio],
                    "views_per_quant": views_per_quant,
                    **gpop_overrides,
                },
            }
            cmd = build_train_command(
                scene_path=scene_path,
                model_path=str(model_dir),
                phase=phase,
                defaults=defaults,
            )
            rc = run_logged_command(
                cmd,
                job_dir,
                f"cycle_{cycle_idx}_prune",
                global_log_path=global_log_path,
            )
            cycle_stdout_log = job_dir / f"cycle_{cycle_idx}_prune.stdout.log"
            pruned_count = _read_last_pruned_count(cycle_stdout_log)
            if pruned_count is not None:
                actual_total_pruned += pruned_count
            cycle_ckpt = model_dir / f"chkpnt{target_iter}.pth"
            cycle_results.append(
                {
                    "cycle": cycle_idx,
                    "prune_ratio": ratio,
                    "pruned_count": pruned_count,
                    "target_iteration": target_iter,
                    "rc": rc,
                    "checkpoint": str(cycle_ckpt),
                }
            )
            if rc != 0:
                break
            if not cycle_ckpt.exists():
                cycle_results[-1]["rc"] = 2
                cycle_results[-1]["error"] = "expected cycle checkpoint missing"
                break
            if (
                not keep_intermediate_checkpoints
                and Path(current_ckpt).parent == model_dir
            ):
                _remove_transient_training_artifacts(
                    model_dir, current_iter, global_log_path
                )
            current_iter = target_iter
            current_ckpt = str(cycle_ckpt)

        cycles_ok = (
            all(x.get("rc", 1) == 0 for x in cycle_results)
            and len(cycle_results) == c_cycles
        )

        fine_tune_result = None
        if cycles_ok and fine_tune_iterations > 0:
            ft_target = current_iter + fine_tune_iterations
            ft_phase = {
                "name": "posttrain_finetune",
                "iterations": ft_target,
                "start_checkpoint": current_ckpt,
                "train_args": step_cfg.get("fine_tune_train_args", {}),
                "save_every": int(
                    step_cfg.get(
                        "fine_tune_save_every", defaults.get("save_every", 999_999_999)
                    )
                ),
            }
            ft_cmd = build_train_command(
                scene_path=scene_path,
                model_path=str(model_dir),
                phase=ft_phase,
                defaults=defaults,
            )
            ft_rc = run_logged_command(
                ft_cmd, job_dir, "fine_tune", global_log_path=global_log_path
            )
            ft_ckpt = model_dir / f"chkpnt{ft_target}.pth"
            fine_tune_result = {
                "rc": ft_rc,
                "iterations": fine_tune_iterations,
                "target_iteration": ft_target,
                "checkpoint": str(ft_ckpt),
            }
            if ft_rc == 0 and ft_ckpt.exists():
                if (
                    not keep_intermediate_checkpoints
                    and Path(current_ckpt).parent == model_dir
                ):
                    _remove_transient_training_artifacts(
                        model_dir, current_iter, global_log_path
                    )
                current_iter = ft_target
                current_ckpt = str(ft_ckpt)

        ok = cycles_ok and (
            fine_tune_result is None or fine_tune_result.get("rc", 1) == 0
        )
        status = {
            "step": step_id,
            "group": group,
            "scene": scene,
            "status": "ok" if ok else "failed",
            "source": source_status,
            "prune_ratio_mode": prune_ratio_mode,
            "source_gaussians": source_gaussians,
            "target_total_pruned": target_total_pruned,
            "actual_total_pruned": actual_total_pruned,
            "c_cycles": c_cycles,
            "cycles": cycle_results,
            "post_prune_iteration": cycle_results[-1]["target_iteration"]
            if cycle_results
            else src_iteration,
            "fine_tune": fine_tune_result,
            "final_checkpoint": current_ckpt,
            "final_iteration": current_iter,
            "model_dir": str(model_dir),
        }
        write_json(job_dir / "status.json", status)

        # Always register the artifact so downstream evaluate/analyze steps can
        # still run (and report failure gracefully) even when this step failed.
        artifacts[_artifact_key(step_id, output_variant, group, scene)] = {
            "model_dir": str(model_dir),
            "final_iteration": current_iter,
        }


def execute_steps(config: Dict, run_root: Path):
    defaults = config.get("defaults", {})
    execution = config.get("execution", {})
    jobs = resolve_scene_jobs(config)
    artifacts = {}
    global_log_path = run_root / "run.log"

    for step in config["steps"]:
        mode = step["mode"]
        with open(global_log_path, "a", encoding="utf-8") as glog:
            glog.write(
                f"[{datetime.now().isoformat()}] STEP_START id={step['id']} mode={mode}\n"
            )
        if mode == "optimize":
            run_optimize_step(
                step, run_root, jobs, defaults, execution, artifacts, global_log_path
            )
        elif mode == "external_model":
            run_external_model_step(step, run_root, jobs, artifacts, global_log_path)
        elif mode == "posttrain_gpop":
            run_posttrain_gpop_step(
                step, run_root, jobs, defaults, execution, artifacts, global_log_path
            )
        elif mode == "evaluate":
            run_evaluate_step(step, run_root, jobs, artifacts, global_log_path)
        elif mode == "analyze":
            run_analyze_step(step, run_root, jobs, artifacts, global_log_path)
        elif mode == "compare":
            run_compare_step(step, run_root, jobs, artifacts, global_log_path)
        elif mode == "pipeline":
            continue
        else:
            raise ValueError(f"Unknown step mode: {mode}")
        with open(global_log_path, "a", encoding="utf-8") as glog:
            glog.write(
                f"[{datetime.now().isoformat()}] STEP_END id={step['id']} mode={mode}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument(
        "--config", required=True, type=str, help="Path to experiment JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config output_dir for this run",
    )
    args = parser.parse_args()

    config = read_json(args.config)
    exp_name = config["experiment_name"]
    output_root = Path(config.get("output_root", "output"))
    if args.output_dir:
        run_root = Path(args.output_dir)
        config["output_dir"] = str(run_root)
    elif "output_dir" in config:
        run_root = Path(config["output_dir"])
    else:
        run_root = output_root / f"{now_tag()}-{exp_name}"
    ensure_dir(run_root)

    write_json(run_root / "resolved_config.json", config)
    write_run_manifest(run_root, config)

    run_log_path = run_root / "run.log"
    with open(run_log_path, "a", encoding="utf-8") as glog:
        glog.write(f"[{datetime.now().isoformat()}] RUN_START {run_root}\n")
        glog.write(f"config={args.config}\n")

    # Copy the original experiment config to the run folder for reference
    config_src = Path(args.config).resolve()
    config_dst = run_root / config_src.name
    shutil.copy2(config_src, config_dst)

    execute_steps(config, run_root)

    with open(run_log_path, "a", encoding="utf-8") as glog:
        glog.write(f"[{datetime.now().isoformat()}] RUN_END {run_root}\n")

    print(f"Run complete: {run_root}")
    print(f"Config copied to: {config_dst}")


if __name__ == "__main__":
    main()
