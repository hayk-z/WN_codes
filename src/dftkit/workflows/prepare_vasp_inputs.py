#!/usr/bin/env python3
"""Prepare VASP input folders from ASE DB / JSON / YAML exports."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.db import connect

# Allow direct script execution from repo root.
try:
    from dftkit.utils.calc_funcs_vasp import set_vasp
except ModuleNotFoundError:
    THIS_FILE = Path(__file__).resolve()
    SRC_ROOT = THIS_FILE.parents[2]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from dftkit.utils.calc_funcs_vasp import set_vasp


DEFAULT_DFT_CONFIG = Path("configs/dft_defaults.yaml")
DEFAULT_SLURM_CONFIG = Path("configs/slurm_ysu2.conf")
DEFAULT_OUTPUT_ROOT = Path("data/calculations")
DEFAULT_POTCAR_ROOT = Path("/mnt/dftevn/opt/vasp/pseudo/potpaw_PBE")


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse a flat key:value YAML file (used for dft_defaults.yaml)."""
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            data[key] = value[1:-1]
            continue
        if value.startswith("'") and value.endswith("'"):
            data[key] = value[1:-1]
            continue
        if value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
            continue
        if value.lower() in {"none", "null"}:
            data[key] = None
            continue
        try:
            if "." in value or "e" in value.lower():
                data[key] = float(value)
            else:
                data[key] = int(value)
            continue
        except ValueError:
            data[key] = value
    return data


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_cluster_conf(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    modules: list[str] = []
    in_modules = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if in_modules:
            if line == ")":
                in_modules = False
                config["MODULES"] = modules
                continue
            item = line.rstrip(",").strip()
            if len(item) >= 2 and item[0] == item[-1] and item[0] in {"'", '"'}:
                item = item[1:-1]
            modules.append(item)
            continue

        if line.startswith("MODULES=") and line.endswith("("):
            in_modules = True
            modules = []
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            config[key] = value

    if "MODULES" not in config:
        config["MODULES"] = []
    return config


def atoms_from_structure_dict(structure: dict[str, Any]) -> Atoms:
    return Atoms(
        symbols=structure["symbols"],
        positions=structure["positions"],
        cell=structure["cell"],
        pbc=structure["pbc"],
    )


def load_records(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    records: list[dict[str, Any]] = []

    if suffix == ".db":
        with connect(input_path) as db:
            for row in db.select():
                records.append(
                    {
                        "id": row.id,
                        "material": getattr(row, "material", f"id_{row.id}"),
                        "atoms": row.toatoms(),
                    }
                )
        return records

    if suffix == ".json":
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "records" in raw:
            raw = raw["records"]
        if not isinstance(raw, list):
            raise ValueError("JSON input must be a list of records")
        for rec in raw:
            if not isinstance(rec, dict) or "structure" not in rec:
                raise ValueError("JSON record missing 'structure'")
            records.append(
                {
                    "id": rec.get("id"),
                    "material": rec.get("material", f"id_{rec.get('id', 'unknown')}"),
                    "atoms": atoms_from_structure_dict(rec["structure"]),
                }
            )
        return records

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YAML input requires PyYAML. Install it in your conda env: conda install pyyaml"
            ) from exc
        raw = yaml.safe_load(input_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "records" in raw:
            raw = raw["records"]
        if not isinstance(raw, list):
            raise ValueError("YAML input must be a list of records")
        for rec in raw:
            if not isinstance(rec, dict) or "structure" not in rec:
                raise ValueError("YAML record missing 'structure'")
            records.append(
                {
                    "id": rec.get("id"),
                    "material": rec.get("material", f"id_{rec.get('id', 'unknown')}"),
                    "atoms": atoms_from_structure_dict(rec["structure"]),
                }
            )
        return records

    raise ValueError(f"Unsupported input type: {input_path.suffix}. Use .db, .json, .yml, .yaml")


def parse_ids(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for value in values:
        for part in value.split(","):
            token = part.strip()
            if token:
                out.add(token)
    return out


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "material"


def parse_kpoint_mesh(kpoints_value: Any) -> tuple[int, int, int]:
    if isinstance(kpoints_value, str):
        tokens = re.split(r"[xX, ]+", kpoints_value.strip())
        nums = [int(t) for t in tokens if t]
    elif isinstance(kpoints_value, (list, tuple)):
        nums = [int(v) for v in kpoints_value]
    else:
        raise ValueError(f"Unsupported kpoints format: {kpoints_value!r}")

    if len(nums) != 3:
        raise ValueError(f"kpoints must define 3 integers, got: {kpoints_value!r}")
    return nums[0], nums[1], nums[2]


def dft_config_to_vasp_kwargs(dft_cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate dft_defaults.yaml keys to ASE Vasp calculator kwargs."""
    kwargs: dict[str, Any] = {}

    if "kpoints" in dft_cfg:
        kwargs["kpts"] = parse_kpoint_mesh(dft_cfg["kpoints"])
    else:
        raise ValueError("dft_defaults.yaml missing required key: kpoints")

    key_map = {
        "encut_ev": "encut",
    }
    skip = {"calculator", "kpoints"}
    for key, value in dft_cfg.items():
        if key in skip or value is None:
            continue
        mapped_key = key_map.get(key, key)
        kwargs[mapped_key] = value

    # Ensure this pipeline writes files only (no runtime command needed here).
    kwargs.setdefault("xc", str(dft_cfg.get("xc", "PBE")))
    return kwargs


def write_vasp_inputs_with_ase(
    job_dir: Path,
    atoms: Atoms,
    dft_cfg: dict[str, Any],
    potcar_root: Path,
) -> None:
    """Write INCAR/KPOINTS/POTCAR/POSCAR via ASE Vasp + calc_funcs_vasp.set_vasp."""
    potcar_parent = potcar_root.parent
    if not potcar_parent.exists():
        raise FileNotFoundError(f"POTCAR parent directory not found: {potcar_parent}")

    # ASE resolves POTCARs using VASP_PP_PATH/<family>/<element>/POTCAR.
    os.environ["VASP_PP_PATH"] = str(potcar_parent)
    vasp_kwargs = dft_config_to_vasp_kwargs(dft_cfg)
    calc = Vasp(**vasp_kwargs)

    with pushd(job_dir):
        set_vasp(atoms, calc)


def render_run_command(slurm_cfg: dict[str, object]) -> str:
    ntasks = str(slurm_cfg.get("NTASKS", "1"))
    run_command = str(slurm_cfg.get("RUN_COMMAND", "")).strip()
    if not run_command:
        run_command = "vasp_std"
    if "{NTASKS}" in run_command:
        return run_command.replace("{NTASKS}", ntasks)
    return f"mpirun -np {ntasks} {run_command}"


def write_myrun(path: Path, slurm_cfg: dict[str, object], workdir: Path) -> None:
    required = ["JOB_NAME", "PARTITION", "WALLTIME", "NODES", "NTASKS", "MEMORY"]
    missing = [k for k in required if not str(slurm_cfg.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Missing required SLURM config keys: {', '.join(missing)}")

    modules = slurm_cfg.get("MODULES", [])
    if not isinstance(modules, list):
        modules = []

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {slurm_cfg['JOB_NAME']}",
        f"#SBATCH -p {slurm_cfg['PARTITION']}",
        f"#SBATCH -t {slurm_cfg['WALLTIME']}",
        f"#SBATCH -N {slurm_cfg['NODES']}",
        f"#SBATCH -n {slurm_cfg['NTASKS']}",
        f"#SBATCH --mem {slurm_cfg['MEMORY']}",
    ]
    if str(slurm_cfg.get("OUTPUT_FILE", "")).strip():
        lines.append(f"#SBATCH -o {slurm_cfg['OUTPUT_FILE']}")

    abs_workdir = workdir.resolve()

    lines.extend(
        [
            "",
            f"export OMP_NUM_THREADS={slurm_cfg.get('OMP_NUM_THREADS', '1')}",
            f"cd {abs_workdir}",
        ]
    )

    if modules:
        lines.append("module purge")
        for module in modules:
            lines.append(f"module load {module}")

    lines.append(render_run_command(slurm_cfg))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(path, 0o755)


def select_records(records: list[dict[str, Any]], ids_filter: set[str]) -> list[dict[str, Any]]:
    if not ids_filter:
        return records
    selected: list[dict[str, Any]] = []
    for rec in records:
        rid = rec.get("id")
        if rid is None:
            continue
        if str(rid) in ids_filter:
            selected.append(rec)
    return selected


def prepare(
    input_db: Path,
    calc_name: str,
    dft_config: Path,
    slurm_config: Path,
    output_root: Path,
    potcar_root: Path,
    ids_filter: set[str],
) -> tuple[Path, int]:
    # 1) Load structures from DB/JSON/YAML and apply optional ID filter.
    records = load_records(input_db)
    records = select_records(records, ids_filter)
    if not records:
        raise ValueError("No records selected. Check --ids or input dataset.")

    # 2) Load calculation defaults and cluster submission settings.
    dft_cfg = parse_simple_yaml(dft_config)
    slurm_cfg = parse_cluster_conf(slurm_config)

    target_root = output_root / f"{sanitize_name(input_db.stem)}_{sanitize_name(calc_name)}"
    target_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for rec in records:
        rid = rec.get("id", "unknown")
        atoms = rec["atoms"]
        material = sanitize_name(str(rec.get("material", f"id_{rid}")))
        job_dir = target_root / f"id_{rid}_{material}"
        job_dir.mkdir(parents=True, exist_ok=True)

        # 3) Build VASP inputs with ASE calculator + shared calc_funcs utility.
        write_vasp_inputs_with_ase(job_dir=job_dir, atoms=atoms, dft_cfg=dft_cfg, potcar_root=potcar_root)
        # 4) Add cluster-specific run script beside the VASP inputs.
        write_myrun(job_dir / "myrun.sh", slurm_cfg, job_dir)
        count += 1

    return target_root, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VASP input folders from DB/JSON/YAML")
    parser.add_argument("--input-db", type=Path, required=True, help="Input .db, .json, .yml, or .yaml")
    parser.add_argument("--calc-name", type=str, required=True, help="Calculation name for output folder")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional record IDs (space or comma separated)")
    parser.add_argument("--dft-config", type=Path, default=DEFAULT_DFT_CONFIG)
    parser.add_argument("--slurm-config", type=Path, default=DEFAULT_SLURM_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--potcar-root", type=Path, default=DEFAULT_POTCAR_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids_filter = parse_ids(args.ids)
    target_root, count = prepare(
        input_db=args.input_db,
        calc_name=args.calc_name,
        dft_config=args.dft_config,
        slurm_config=args.slurm_config,
        output_root=args.output_root,
        potcar_root=args.potcar_root,
        ids_filter=ids_filter,
    )
    print(f"Prepared {count} VASP job directories")
    print(f"Output root: {target_root}")


if __name__ == "__main__":
    main()
