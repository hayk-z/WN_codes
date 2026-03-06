#!/usr/bin/env python3
"""HSE DOS/PDOS/Band workflow with sequential steps from start-step to end."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ase.io import read

# Allow running as script from repository root.
try:
    from dftkit.workflows.dos_pdos_band_workflow import (
        coerce_scalar_for_vasp,
        copy_restart_files,
        load_records,
        load_workflow_yaml,
        log_message,
        monitor_submitted_jobs,
        normalize_kpts,
        parse_ids,
        parse_shell_conf,
        prepare_step_input,
        read_first_valid_structure,
        run_step4_symmetry_analysis,
        sanitize_name,
        select_records,
        submit_step1,
        write_band_kpoints_from_2d_path,
        write_myrun,
    )
except ModuleNotFoundError:
    import sys

    THIS_FILE = Path(__file__).resolve()
    SRC_ROOT = THIS_FILE.parents[2]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from dftkit.workflows.dos_pdos_band_workflow import (
        coerce_scalar_for_vasp,
        copy_restart_files,
        load_records,
        load_workflow_yaml,
        log_message,
        monitor_submitted_jobs,
        normalize_kpts,
        parse_ids,
        parse_shell_conf,
        prepare_step_input,
        read_first_valid_structure,
        run_step4_symmetry_analysis,
        sanitize_name,
        select_records,
        submit_step1,
        write_band_kpoints_from_2d_path,
        write_myrun,
    )


DEFAULT_CONFIG = Path("configs/dos_calc_hse.yaml")
DEFAULT_SLURM_CONFIG = Path("configs/slurm_ysu2.conf")
DEFAULT_OUTPUT_ROOT = Path("data/calculations")
DEFAULT_CALC_NAME = "DOS_HSE_calc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSE DOS/PDOS/Band workflow")
    parser.add_argument("--input-db", type=Path, required=True, help="Input database (.db or .json)")
    parser.add_argument("--calc-name", type=str, default=DEFAULT_CALC_NAME, help="Calculation name")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional IDs (space/comma separated)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Workflow YAML config")
    parser.add_argument("--slurm-config", type=Path, default=DEFAULT_SLURM_CONFIG, help="SLURM config")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-step", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-wait-hours", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_step4_defaults(workflow_cfg: dict[str, Any]) -> dict[str, Any]:
    step4_cfg = workflow_cfg.get("step4_band", {})
    if not isinstance(step4_cfg, dict):
        step4_cfg = {}
    if "vasp_tags" not in step4_cfg or not isinstance(step4_cfg["vasp_tags"], dict):
        step4_cfg["vasp_tags"] = {}

    tags = step4_cfg["vasp_tags"]
    tags.setdefault("kpts", [18, 18, 1])
    tags.setdefault("nsw", 0)
    tags.setdefault("ibrion", -1)
    tags.setdefault("icharge", 11)
    tags.setdefault("lorbit", 11)
    tags.setdefault("lwave", False)
    tags.setdefault("lcharg", False)
    tags.pop("nedos", None)
    workflow_cfg["step4_band"] = step4_cfg
    return step4_cfg


def copy_restart_files_maybe(
    src_dir: Path,
    dst_dir: Path,
    names: list[str],
    dry_run: bool,
    log_file: Path,
) -> None:
    missing = [name for name in names if not (src_dir / name).exists()]
    if missing:
        if dry_run:
            log_message(
                log_file,
                f"[DRY-RUN] Missing restart files in {src_dir.name}: {', '.join(missing)}; skipping copy.",
            )
            return
        raise FileNotFoundError(f"Required restart file(s) missing in {src_dir}: {', '.join(missing)}")
    copy_restart_files(src_dir, dst_dir, names)


def main() -> None:
    args = parse_args()
    if not args.input_db.exists():
        raise FileNotFoundError(f"Input database not found: {args.input_db}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.slurm_config.exists():
        raise FileNotFoundError(f"SLURM config not found: {args.slurm_config}")

    workflow_cfg = load_workflow_yaml(args.config)
    slurm_cfg = parse_shell_conf(args.slurm_config)
    ids_filter = parse_ids(args.ids)

    records = select_records(load_records(args.input_db), ids_filter)
    if not records:
        raise ValueError("No records selected. Check --ids or database content.")

    calc_root = (args.output_root / f"{sanitize_name(args.input_db.stem)}_{sanitize_name(args.calc_name)}").resolve()
    calc_root.mkdir(parents=True, exist_ok=True)
    log_file = calc_root / "workflow.log"

    log_message(log_file, f"Starting HSE workflow: calc_name={args.calc_name}")
    log_message(log_file, f"Selected records: {len(records)}")
    log_message(log_file, f"SLURM config: {args.slurm_config}")
    log_message(log_file, f"Start step: {args.start_step}")

    job_prefix = str(workflow_cfg.get("job_name_prefix", "dos_hse"))

    for current_step in range(args.start_step, 5):
        submitted_jobs: list[dict[str, Any]] = []
        log_message(log_file, f"=== Starting step {current_step} for {len(records)} IDs ===")

        for rec in records:
            rid = rec["id"]
            material = sanitize_name(str(rec.get("material", f"id_{rid}")))
            mat_dir = calc_root / f"id_{rid}_{material}"
            step1_dir = mat_dir / "01_relax"
            step2_dir = mat_dir / "02_scf"
            step3_dir = mat_dir / "03_dos"
            step4_dir = mat_dir / "04_band"
            for d in (step1_dir, step2_dir, step3_dir, step4_dir):
                d.mkdir(parents=True, exist_ok=True)

            if current_step == 1:
                log_message(log_file, f"Preparing step1 relaxation for ID={rid}, material={material}")
                prepare_step_input(step1_dir, rec["atoms"], workflow_cfg, step_key="step1_relax")
                job_name = f"{job_prefix}_id{rid}_r1"
                run_dir = step1_dir
                step_label = "step1_relax"

            elif current_step == 2:
                relaxed_atoms, source_structure = read_first_valid_structure([step1_dir / "CONTCAR", step1_dir / "POSCAR"])
                log_message(log_file, f"Preparing step2 HSE-SCF for ID={rid}, material={material} from {source_structure.name}")
                base_kpts = normalize_kpts(workflow_cfg.get("step1_relax", {}).get("vasp_tags", {}).get("kpts"))
                prepare_step_input(step2_dir, relaxed_atoms, workflow_cfg, step_key="step2_scf", base_kpts=base_kpts)
                copy_restart_files_maybe(step1_dir, step2_dir, ["CHGCAR", "WAVECAR"], args.dry_run, log_file)
                job_name = f"{job_prefix}_id{rid}_s2"
                run_dir = step2_dir
                step_label = "step2_scf"

            elif current_step == 3:
                scf_atoms, source_structure = read_first_valid_structure([step2_dir / "CONTCAR", step2_dir / "POSCAR"])
                log_message(log_file, f"Preparing step3 HSE-DOS for ID={rid}, material={material} from {source_structure.name}")
                base_kpts = normalize_kpts(workflow_cfg.get("step2_scf", {}).get("vasp_tags", {}).get("kpts"))
                prepare_step_input(step3_dir, scf_atoms, workflow_cfg, step_key="step3_dos", base_kpts=base_kpts)
                copy_restart_files_maybe(step2_dir, step3_dir, ["CHGCAR", "WAVECAR"], args.dry_run, log_file)
                job_name = f"{job_prefix}_id{rid}_s3"
                run_dir = step3_dir
                step_label = "step3_dos"

            else:
                atoms, source_structure = read_first_valid_structure([
                    step3_dir / "CONTCAR",
                    step3_dir / "POSCAR",
                    step2_dir / "CONTCAR",
                    step2_dir / "POSCAR",
                    step1_dir / "CONTCAR",
                    step1_dir / "POSCAR",
                ])
                log_message(log_file, f"Preparing step4 HSE-band for ID={rid}, material={material} from {source_structure.name}")
                layer_result, _ = run_step4_symmetry_analysis(step4_dir, atoms, workflow_cfg, rec)
                step4_cfg = apply_step4_defaults(workflow_cfg)
                prepare_step_input(step4_dir, atoms, workflow_cfg, step_key="step4_band")
                copy_restart_files_maybe(step2_dir, step4_dir, ["CHGCAR", "WAVECAR"], args.dry_run, log_file)

                kpath_string_2d = str(layer_result.get("kpath_string_2d", "")).strip()
                kpoints_2d = layer_result.get("kpath_kpoints_2d", {})
                aperiodic_dir = int(layer_result.get("aperiodic_dir", 2))
                points_per_segment = int(coerce_scalar_for_vasp(step4_cfg.get("band_points_per_segment", 40)))
                if not isinstance(kpoints_2d, dict) or not kpoints_2d:
                    raise ValueError("Step 4 failed to produce canonical 2D k-point coordinates")
                write_band_kpoints_from_2d_path(
                    step4_dir / "KPOINTS",
                    kpath_string_2d=kpath_string_2d,
                    kpoints_2d={str(k): v for k, v in kpoints_2d.items()},
                    aperiodic_dir=aperiodic_dir,
                    points_per_segment=points_per_segment,
                )
                job_name = f"{job_prefix}_id{rid}_s4"
                run_dir = step4_dir
                step_label = "step4_band"

            write_myrun(run_dir / "myrun.sh", slurm_cfg, job_name=job_name, workdir=run_dir)
            job_info = submit_step1(step_dir=run_dir, job_name=job_name, log_file=log_file, dry_run=args.dry_run)
            job_info["rid"] = rid
            job_info["step_label"] = step_label
            submitted_jobs.append(job_info)

        if args.dry_run:
            for job in submitted_jobs:
                log_message(log_file, f"[DRY-RUN] Would wait for completion: {job['job_name']}")
                log_message(log_file, f"ID={job.get('rid', '?')}: {job.get('step_label', 'step')} completed")
        elif submitted_jobs:
            log_message(log_file, f"Submitted {len(submitted_jobs)} jobs for step {current_step}. Start monitoring...")
            monitor_submitted_jobs(
                jobs=submitted_jobs,
                log_file=log_file,
                poll_seconds=args.poll_seconds,
                max_wait_hours=args.max_wait_hours,
            )

        log_message(log_file, f"=== Step {current_step} finished ===")

    log_message(log_file, "Workflow finished")


if __name__ == "__main__":
    main()
