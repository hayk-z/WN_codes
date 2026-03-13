#!/usr/bin/env python3
"""Workflow to compute ZPE corrections and update Gibbs free energy for H adsorption DB."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.constraints import FixAtoms
from ase.db import connect

# Allow running as script from repository root.
try:
    from dftkit.utils.calc_funcs_vasp import set_vasp
    from dftkit.workflows.dos_pdos_band_workflow import (
        coerce_scalar_for_vasp,
        load_records,
        load_workflow_yaml,
        log_message,
        monitor_submitted_jobs,
        parse_ids,
        parse_shell_conf,
        resolve_potpaw_pbe_root,
        sanitize_name,
        select_records,
        submit_step1,
        write_myrun,
    )
except ModuleNotFoundError:
    THIS_FILE = Path(__file__).resolve()
    SRC_ROOT = THIS_FILE.parents[2]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from dftkit.utils.calc_funcs_vasp import set_vasp
    from dftkit.workflows.dos_pdos_band_workflow import (
        coerce_scalar_for_vasp,
        load_records,
        load_workflow_yaml,
        log_message,
        monitor_submitted_jobs,
        parse_ids,
        parse_shell_conf,
        resolve_potpaw_pbe_root,
        sanitize_name,
        select_records,
        submit_step1,
        write_myrun,
    )


DEFAULT_CONFIG = Path("configs/zpe_calc.yaml")
DEFAULT_SLURM_CONFIG = Path("configs/slurm_ysu2.conf")
DEFAULT_OUTPUT_ROOT = Path("data/calculations")
DEFAULT_CALC_NAME = "ZPE_Gibbs_calc"
DEFAULT_POTCAR_DIR = Path("data/potcars")
CM1_TO_EV = 1.2398419843320026e-4


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def normalize_kpts(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return int(value[0]), int(value[1]), int(value[2])
    if isinstance(value, str):
        parts = [p for p in re.split(r"[xX, ]+", value.strip()) if p]
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"Invalid k-point mesh: {value!r}")


def build_h2_structure(bond_length: float, box_length: float) -> Atoms:
    center = box_length / 2.0
    half_bond = bond_length / 2.0
    return Atoms(
        symbols="H2",
        positions=[
            [center, center, center - half_bond],
            [center, center, center + half_bond],
        ],
        cell=[box_length, box_length, box_length],
        pbc=[True, True, True],
    )


def build_hstar_constrained_structure(atoms: Atoms) -> Atoms:
    """Fix non-H slab atoms (F F F) and keep only H atoms movable (T T T)."""
    constrained = atoms.copy()
    symbols = constrained.get_chemical_symbols()
    fixed_mask = [sym != "H" for sym in symbols]
    if all(fixed_mask):
        raise ValueError("No H atom found in structure; cannot prepare H* vibrational input")
    constrained.set_constraint(FixAtoms(mask=fixed_mask))
    return constrained


def prepare_vasp_input(
    work_dir: Path,
    atoms: Atoms,
    vasp_tags: dict[str, Any],
    potcar_root: Path,
) -> None:
    resolved_potcar = resolve_potpaw_pbe_root(potcar_root)
    os.environ["VASP_PP_PATH"] = str(resolved_potcar.parent)

    kwargs = {key: coerce_scalar_for_vasp(value) for key, value in vasp_tags.items()}
    if "encut_ev" in kwargs and "encut" not in kwargs:
        kwargs["encut"] = kwargs.pop("encut_ev")
    if "kpoints" in kwargs and "kpts" not in kwargs:
        kwargs["kpts"] = kwargs.pop("kpoints")
    if "kpts" in kwargs:
        kwargs["kpts"] = normalize_kpts(kwargs["kpts"])

    calc = Vasp(**kwargs)
    with pushd(work_dir):
        set_vasp(atoms, calc)


def prepare_vasp_input_with_dryrun_fallback(
    work_dir: Path,
    atoms: Atoms,
    vasp_tags: dict[str, Any],
    potcar_root: Path,
    dry_run: bool,
    log_file: Path,
) -> None:
    try:
        prepare_vasp_input(work_dir, atoms, vasp_tags, potcar_root=potcar_root)
    except Exception as exc:
        if not dry_run:
            raise
        log_message(
            log_file,
            f"[DRY-RUN] Input generation fallback in {work_dir}: {type(exc).__name__}: {exc}",
        )
        (work_dir / "DRY_RUN_INPUT_PLACEHOLDER.txt").write_text(
            "Dry-run fallback: VASP input generation skipped because runtime resources "
            f"(e.g. POTCAR) are unavailable.\nError: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )


def parse_outcar_frequencies_cm1(outcar_path: Path) -> list[dict[str, Any]]:
    if not outcar_path.exists():
        raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")

    modes: list[dict[str, Any]] = []
    for raw in outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if "cm-1" not in line:
            continue
        if " f  =" not in line and " f/i=" not in line:
            continue

        m = re.search(r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*cm-1", line)
        if not m:
            continue

        freq = float(m.group(1))
        imaginary = " f/i=" in line
        if imaginary:
            freq = -abs(freq)

        mode_id_match = re.match(r"^\s*(\d+)\s", raw)
        mode_id = int(mode_id_match.group(1)) if mode_id_match else None

        modes.append({"mode": mode_id, "frequency_cm1": freq, "imaginary": imaginary, "raw": line})

    if not modes:
        raise ValueError(f"No vibrational frequencies found in {outcar_path}")
    return modes


def zpe_from_modes(modes: list[dict[str, Any]]) -> dict[str, Any]:
    real_pos = [float(m["frequency_cm1"]) for m in modes if (not m["imaginary"] and float(m["frequency_cm1"]) > 0.0)]
    imag = [abs(float(m["frequency_cm1"])) for m in modes if m["imaginary"]]

    if not real_pos:
        raise ValueError("No positive real frequencies were found for ZPE calculation")

    zpe_ev = 0.5 * sum(real_pos) * CM1_TO_EV
    return {
        "real_frequencies_cm1": real_pos,
        "imaginary_frequencies_cm1": imag,
        "zpe_ev": zpe_ev,
    }


def zpe_hstar_from_three_modes(modes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute ZPE(H*) using only the 3 vibrational modes of adsorbed H.

    Since slab atoms are fixed in POSCAR selective dynamics, H should be the
    only movable atom and these 3 modes are the H* vibrational contribution.
    """
    real_pos = sorted(
        [float(m["frequency_cm1"]) for m in modes if (not m["imaginary"] and float(m["frequency_cm1"]) > 0.0)],
        reverse=True,
    )
    imag = [abs(float(m["frequency_cm1"])) for m in modes if m["imaginary"]]

    if len(real_pos) < 3:
        raise ValueError(f"Expected at least 3 positive real H* modes, found {len(real_pos)}")

    selected = real_pos[:3]
    zpe_ev = 0.5 * sum(selected) * CM1_TO_EV
    return {
        "selected_hstar_modes_cm1": selected,
        "real_frequencies_cm1": real_pos,
        "imaginary_frequencies_cm1": imag,
        "zpe_ev": zpe_ev,
    }


def zpe_h2_from_stretching_mode(modes: list[dict[str, Any]]) -> dict[str, Any]:
    real_pos = [float(m["frequency_cm1"]) for m in modes if (not m["imaginary"] and float(m["frequency_cm1"]) > 0.0)]
    imag = [abs(float(m["frequency_cm1"])) for m in modes if m["imaginary"]]

    if not real_pos:
        raise ValueError("No positive real H2 frequencies were found")

    stretching_mode = max(real_pos)
    zpe_h2_ev = 0.5 * stretching_mode * CM1_TO_EV
    return {
        "stretching_mode_cm1": stretching_mode,
        "real_frequencies_cm1": real_pos,
        "imaginary_frequencies_cm1": imag,
        "zpe_h2_ev": zpe_h2_ev,
    }


def load_general_inputs(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in general inputs file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"General inputs must be a JSON object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def try_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "name",
        "composition",
        "adsorption_site_type",
        "delta_zpe_ev",
        "zpe_h_star_ev",
        "half_zpe_h2_ev",
        "delta_g_h_ev",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZPE/Gibbs workflow for H adsorption DB")
    parser.add_argument("--input-db", type=Path, required=True, help="Input database (.db or .json)")
    parser.add_argument("--calc-name", type=str, default=DEFAULT_CALC_NAME, help="Calculation name")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional IDs (space/comma separated)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Workflow YAML config")
    parser.add_argument("--slurm-config", type=Path, default=DEFAULT_SLURM_CONFIG, help="SLURM config")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--potcar",
        "--potcar-root",
        dest="potcar",
        type=Path,
        default=None,
        help=(
            "Path to POTCAR directory. Accepts either .../potpaw_PBE or its parent "
            f"(default: {DEFAULT_POTCAR_DIR})"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-wait-hours", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true", help="Create files but do not submit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_db.exists():
        raise FileNotFoundError(f"Input database not found: {args.input_db}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.slurm_config.exists():
        raise FileNotFoundError(f"SLURM config not found: {args.slurm_config}")

    workflow_cfg = load_workflow_yaml(args.config)
    selected_potcar = args.potcar or Path(str(workflow_cfg.get("potcar_root", DEFAULT_POTCAR_DIR)))
    slurm_cfg = parse_shell_conf(args.slurm_config)

    ids_filter = parse_ids(args.ids)
    records = select_records(load_records(args.input_db), ids_filter)
    if not records:
        raise ValueError("No records selected. Check --ids or database content.")

    calc_root = (args.output_root / f"{sanitize_name(args.input_db.stem)}_{sanitize_name(args.calc_name)}").resolve()
    calc_root.mkdir(parents=True, exist_ok=True)
    log_file = calc_root / "workflow.log"
    general_inputs_file = calc_root / "general_inputs.json"

    log_message(log_file, f"Starting ZPE/Gibbs workflow: calc_name={args.calc_name}")
    log_message(log_file, f"Selected records: {len(records)}")
    log_message(log_file, f"SLURM config: {args.slurm_config}")
    log_message(log_file, f"POTCAR path: {selected_potcar}")

    cfg_general = workflow_cfg.get("general_inputs", {})
    if not isinstance(cfg_general, dict):
        raise ValueError("Config key 'general_inputs' must be a mapping")

    t_delta_s_default = float(coerce_scalar_for_vasp(cfg_general.get("t_delta_s_ev", -0.2)))
    general_inputs = load_general_inputs(general_inputs_file)
    if "t_delta_s_ev" not in general_inputs:
        general_inputs["t_delta_s_ev"] = t_delta_s_default

    h2_cfg = workflow_cfg.get("h2_reference", {})
    if not isinstance(h2_cfg, dict):
        raise ValueError("Config key 'h2_reference' must be a mapping")
    h2_tags = h2_cfg.get("vasp_tags", {})
    if not isinstance(h2_tags, dict):
        raise ValueError("Config key 'h2_reference.vasp_tags' must be a mapping")

    job_prefix = str(workflow_cfg.get("job_name_prefix", "zpe"))

    # Step 1: Ensure H2 ZPE is available in general inputs (calculate once if missing).
    zpe_h2_existing = general_inputs.get("zpe_h2_ev")
    need_h2 = zpe_h2_existing is None
    if need_h2:
        h2_dir = calc_root / "h2_reference" / "01_vib"
        h2_dir.mkdir(parents=True, exist_ok=True)

        bond = float(coerce_scalar_for_vasp(h2_cfg.get("bond_length_angstrom", 0.74)))
        box = float(coerce_scalar_for_vasp(h2_cfg.get("box_length_angstrom", 15.0)))
        h2_atoms = build_h2_structure(bond_length=bond, box_length=box)

        log_message(log_file, "ZPE(H2) missing in general inputs; preparing H2 vibration job")
        prepare_vasp_input_with_dryrun_fallback(
            h2_dir,
            h2_atoms,
            h2_tags,
            potcar_root=selected_potcar,
            dry_run=args.dry_run,
            log_file=log_file,
        )
        write_myrun(h2_dir / "myrun.sh", slurm_cfg, f"{job_prefix}_h2_vib", h2_dir)

        job = submit_step1(step_dir=h2_dir, job_name=f"{job_prefix}_h2_vib", log_file=log_file, dry_run=args.dry_run)
        if not args.dry_run:
            monitor_submitted_jobs(
                jobs=[{"job_id": job["job_id"], "job_name": job["job_name"], "step_dir": h2_dir, "step_label": "h2_vib"}],
                log_file=log_file,
                poll_seconds=args.poll_seconds,
                max_wait_hours=args.max_wait_hours,
            )
            h2_modes = parse_outcar_frequencies_cm1(h2_dir / "OUTCAR")
            h2_data = zpe_h2_from_stretching_mode(h2_modes)
            general_inputs["zpe_h2_ev"] = h2_data["zpe_h2_ev"]
            general_inputs["h2_reference"] = {
                "path": str(h2_dir),
                "stretching_mode_cm1": h2_data["stretching_mode_cm1"],
                "real_frequencies_cm1": h2_data["real_frequencies_cm1"],
                "imaginary_frequencies_cm1": h2_data["imaginary_frequencies_cm1"],
            }
            if h2_data["imaginary_frequencies_cm1"]:
                log_message(
                    log_file,
                    f"WARNING: H2 has imaginary frequencies (cm^-1): {h2_data['imaginary_frequencies_cm1']}",
                )
            write_json(general_inputs_file, general_inputs)
            log_message(log_file, f"Computed ZPE(H2) = {h2_data['zpe_h2_ev']:.6f} eV")
        else:
            log_message(log_file, "[DRY-RUN] Skipping H2 OUTCAR parsing and ZPE(H2) calculation")
    else:
        log_message(log_file, f"Using existing ZPE(H2) from general inputs: {zpe_h2_existing} eV")

    h_star_cfg = workflow_cfg.get("h_star_vib", {})
    if not isinstance(h_star_cfg, dict):
        raise ValueError("Config key 'h_star_vib' must be a mapping")
    h_star_tags = h_star_cfg.get("vasp_tags", {})
    if not isinstance(h_star_tags, dict):
        raise ValueError("Config key 'h_star_vib.vasp_tags' must be a mapping")

    # Step 2: Prepare and submit H* vibration calculations for selected records.
    submitted_jobs: list[dict[str, Any]] = []
    for rec in records:
        rid = rec["id"]
        material = sanitize_name(str(rec.get("material", f"id_{rid}")))
        mat_dir = calc_root / f"id_{rid}_{material}" / "01_hstar_vib"
        mat_dir.mkdir(parents=True, exist_ok=True)

        log_message(log_file, f"Preparing H* vibration step for ID={rid}, material={material}")
        constrained_atoms = build_hstar_constrained_structure(rec["atoms"])
        prepare_vasp_input_with_dryrun_fallback(
            mat_dir,
            constrained_atoms,
            h_star_tags,
            potcar_root=selected_potcar,
            dry_run=args.dry_run,
            log_file=log_file,
        )

        job_name = f"{job_prefix}_id{rid}_vib"
        write_myrun(mat_dir / "myrun.sh", slurm_cfg, job_name, mat_dir)
        submitted = submit_step1(step_dir=mat_dir, job_name=job_name, log_file=log_file, dry_run=args.dry_run)
        submitted["rid"] = rid
        submitted["material"] = material
        submitted["step_label"] = "hstar_vib"
        submitted_jobs.append(submitted)

    if submitted_jobs and not args.dry_run:
        monitor_submitted_jobs(
            jobs=submitted_jobs,
            log_file=log_file,
            poll_seconds=args.poll_seconds,
            max_wait_hours=args.max_wait_hours,
        )

    results: list[dict[str, Any]] = []
    if args.dry_run:
        for rec in records:
            rid = rec["id"]
            material = sanitize_name(str(rec.get("material", f"id_{rid}")))
            results.append(
                {
                    "id": rid,
                    "material": material,
                    "status": "dry_run_prepared",
                    "hstar_directory": str(calc_root / f"id_{rid}_{material}" / "01_hstar_vib"),
                }
            )
        write_json(calc_root / "zpe_results.json", {"dry_run": True, "results": results})
        log_message(log_file, "Dry run complete")
        return

    zpe_h2_ev = general_inputs.get("zpe_h2_ev")
    if zpe_h2_ev is None:
        raise RuntimeError("ZPE(H2) is still missing after workflow execution")
    zpe_h2_ev = float(zpe_h2_ev)
    t_delta_s_ev = float(general_inputs.get("t_delta_s_ev", t_delta_s_default))

    db_updatable = args.input_db.suffix.lower() == ".db"
    report_rows: list[dict[str, Any]] = []

    db_context = connect(args.input_db) if db_updatable else nullcontext(None)
    with db_context as db:
        for rec in records:
            rid = rec["id"]
            material = sanitize_name(str(rec.get("material", f"id_{rid}")))
            vib_dir = calc_root / f"id_{rid}_{material}" / "01_hstar_vib"
            outcar = vib_dir / "OUTCAR"

            modes = parse_outcar_frequencies_cm1(outcar)
            zpe_data = zpe_hstar_from_three_modes(modes)
            zpe_h_star_ev = float(zpe_data["zpe_ev"])
            half_zpe_h2_ev = 0.5 * zpe_h2_ev
            delta_zpe_ev = zpe_h_star_ev - half_zpe_h2_ev

            adsorption_energy_ev: float | None = None
            gibbs_corrected_ev: float | None = None
            composition: Any = None
            adsorption_site_type: str | None = None
            if db is not None:
                row = db.get(id=rid)
                adsorption_raw = getattr(row, "adsorption_energy_ev", None)
                composition = try_parse_json(getattr(row, "composition", None))
                adsorption_site_type = getattr(row, "h_adsorption_type", None)
                if adsorption_raw is not None:
                    adsorption_energy_ev = float(adsorption_raw)
                    gibbs_corrected_ev = adsorption_energy_ev + delta_zpe_ev - t_delta_s_ev

            result = {
                "id": rid,
                "material": material,
                "hstar_directory": str(vib_dir),
                "frequencies_cm1": [float(m["frequency_cm1"]) for m in modes],
                "selected_hstar_modes_cm1": zpe_data["selected_hstar_modes_cm1"],
                "imaginary_frequencies_cm1": zpe_data["imaginary_frequencies_cm1"],
                "zpe_h_star_ev": zpe_h_star_ev,
                "zpe_h2_ev": zpe_h2_ev,
                "half_zpe_h2_ev": half_zpe_h2_ev,
                "delta_zpe_ev": delta_zpe_ev,
                "t_delta_s_ev": t_delta_s_ev,
                "adsorption_energy_ev": adsorption_energy_ev,
                "gibbs_free_energy_ev": gibbs_corrected_ev,
                "composition": composition,
                "adsorption_site_type": adsorption_site_type,
            }
            results.append(result)
            report_rows.append(
                {
                    "id": rid,
                    "name": material,
                    "composition": json.dumps(composition, ensure_ascii=True) if isinstance(composition, (dict, list)) else composition,
                    "adsorption_site_type": adsorption_site_type,
                    "delta_zpe_ev": delta_zpe_ev,
                    "zpe_h_star_ev": zpe_h_star_ev,
                    "half_zpe_h2_ev": half_zpe_h2_ev,
                    "delta_g_h_ev": gibbs_corrected_ev,
                }
            )

            if zpe_data["imaginary_frequencies_cm1"]:
                log_message(
                    log_file,
                    f"WARNING: ID={rid} has imaginary frequencies (cm^-1): {zpe_data['imaginary_frequencies_cm1']}",
                )

            if db is not None:
                update_payload = {
                    "zpe_h_star_ev": zpe_h_star_ev,
                    "zpe_h2_ev": zpe_h2_ev,
                    "half_zpe_h2_ev": half_zpe_h2_ev,
                    "delta_zpe_ev": delta_zpe_ev,
                    "t_delta_s_ev": t_delta_s_ev,
                    "zpe_modes_cm1": json.dumps(result["frequencies_cm1"]),
                    "zpe_hstar_modes_used_cm1": json.dumps(result["selected_hstar_modes_cm1"]),
                    "zpe_imaginary_modes_cm1": json.dumps(result["imaginary_frequencies_cm1"]),
                }
                if adsorption_energy_ev is not None and gibbs_corrected_ev is not None:
                    update_payload["gibbs_free_energy_ev"] = gibbs_corrected_ev
                    update_payload["delta_g_h_ev"] = gibbs_corrected_ev
                db.update(rid, **update_payload)

    if not db_updatable:
        log_message(log_file, f"Input '{args.input_db}' is not an ASE DB; skipping DB updates")

    write_json(general_inputs_file, general_inputs)
    write_json(
        calc_root / "zpe_results.json",
        {
            "dry_run": False,
            "input_db": str(args.input_db),
            "calc_root": str(calc_root),
            "zpe_h2_ev": zpe_h2_ev,
            "t_delta_s_ev": t_delta_s_ev,
            "results": results,
        },
    )
    write_report_csv(calc_root / "zpe_gibbs_report.csv", report_rows)
    log_message(log_file, f"Wrote report: {calc_root / 'zpe_gibbs_report.csv'}")
    log_message(log_file, "Workflow finished")


if __name__ == "__main__":
    main()
