#!/usr/bin/env python3
"""Prepare and run DOS/PDOS/BAND workflow (Step 1: relaxation implemented)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.db import connect
from ase.io import read
import spglib
import seekpath

# Allow running as a script from repository root.
try:
    from dftkit.utils.bsub_funcs_vasp import bsub_run, bsub_stat
    from dftkit.utils.calc_funcs_vasp import set_vasp
except ModuleNotFoundError:
    THIS_FILE = Path(__file__).resolve()
    SRC_ROOT = THIS_FILE.parents[2]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from dftkit.utils.bsub_funcs_vasp import bsub_run, bsub_stat
    from dftkit.utils.calc_funcs_vasp import set_vasp


DEFAULT_CONFIG = Path("configs/dos_calc_pbe.yaml")
DEFAULT_SLURM_CONFIG = Path("configs/slurm_ysu2.conf")
DEFAULT_OUTPUT_ROOT = Path("data/calculations")
DEFAULT_CALC_NAME = "DOS_calc"
DEFAULT_POTCAR_DIR = Path("data/potcars")


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_message(log_file: Path, text: str) -> None:
    line = f"[{ts()}] {text}"
    print(line)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_shell_conf(path: Path) -> dict[str, object]:
    cfg: dict[str, object] = {}
    modules: list[str] = []
    in_modules = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if in_modules:
            if line == ")":
                in_modules = False
                cfg["MODULES"] = modules
                continue
            item = line.rstrip(",").strip()
            if len(item) >= 2 and item[0] == item[-1] and item[0] in {"'", '"'}:
                item = item[1:-1]
            if item:
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
            cfg[key] = value

    if "MODULES" not in cfg:
        cfg["MODULES"] = []
    return cfg


def load_workflow_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        data = parse_yaml_fallback(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Workflow config must be a YAML mapping: {path}")
    return data


def resolve_potpaw_pbe_root(potcar: Path) -> Path:
    """Resolve a user path to the potpaw_PBE directory expected by ASE."""
    base = potcar.expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"POTCAR path not found: {base}")
    if base.is_file():
        raise ValueError(f"POTCAR path must be a directory, got file: {base}")

    if base.name == "potpaw_PBE":
        return base

    candidate = base / "potpaw_PBE"
    if candidate.exists() and candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"POTCAR directory must be 'potpaw_PBE' or contain it as a subfolder: {base}"
    )


def parse_yaml_value(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if inner == "":
            return []
        return [parse_yaml_value(item) for item in inner.split(",")]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def coerce_scalar_for_vasp(value: Any) -> Any:
    """Convert numeric-looking strings from YAML into int/float for ASE VASP tags."""
    if not isinstance(value, str):
        return value
    raw = value.strip()
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if re.fullmatch(r"[+-]?\d+", raw):
            return int(raw)
        if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", raw) or re.fullmatch(
            r"[+-]?\d+[eE][+-]?\d+", raw
        ):
            return float(raw)
    except ValueError:
        return value
    return value


def parse_yaml_fallback(text: str) -> dict[str, Any]:
    """Minimal YAML parser for nested mappings + inline lists."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            node: dict[str, Any] = {}
            current[key] = node
            stack.append((indent + 2, node))
        else:
            current[key] = parse_yaml_value(value)

    return root


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


def parse_kpoint_mesh(value: str) -> tuple[int, int, int]:
    tokens = [t for t in re.split(r"[xX, ]+", value.strip()) if t]
    if len(tokens) != 3:
        raise ValueError(f"KPOINTS must have 3 integers, got: {value!r}")
    return int(tokens[0]), int(tokens[1]), int(tokens[2])


def atoms_from_structure_dict(structure: dict[str, Any]) -> Atoms:
    return Atoms(
        symbols=structure["symbols"],
        positions=structure["positions"],
        cell=structure["cell"],
        pbc=structure["pbc"],
    )


def load_records(input_db: Path) -> list[dict[str, Any]]:
    suffix = input_db.suffix.lower()
    rows: list[dict[str, Any]] = []

    if suffix == ".db":
        with connect(input_db) as db:
            for row in db.select():
                rows.append(
                    {
                        "id": row.id,
                        "material": getattr(row, "material", f"id_{row.id}"),
                        "atoms": row.toatoms(),
                        "db_high_symmetry_path": getattr(row, "high_symmetry_path", None),
                        "db_space_group_number": getattr(row, "space_group_number", None),
                        "db_crystal_system": getattr(row, "crystal_system", None),
                        "db_kpts": json.loads(getattr(row, "kpts", "{}") or "{}"),
                    }
                )
        return rows

    if suffix == ".json":
        data = json.loads(input_db.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records")
        for rec in data:
            if not isinstance(rec, dict) or "structure" not in rec:
                raise ValueError("Each JSON record must include 'structure'")
            rows.append(
                {
                    "id": rec.get("id"),
                    "material": rec.get("material", f"id_{rec.get('id', 'unknown')}"),
                    "atoms": atoms_from_structure_dict(rec["structure"]),
                    "db_high_symmetry_path": rec.get("high_symmetry_path"),
                    "db_space_group_number": rec.get("space_group_number"),
                    "db_crystal_system": rec.get("crystal_system"),
                    "db_kpts": rec.get("kpts", {}),
                }
            )
        return rows

    raise ValueError(f"Unsupported input file: {input_db}. Use .db or .json")


def select_records(records: list[dict[str, Any]], ids_filter: set[str]) -> list[dict[str, Any]]:
    if not ids_filter:
        return records
    selected: list[dict[str, Any]] = []
    for rec in records:
        rid = rec.get("id")
        if rid is not None and str(rid) in ids_filter:
            selected.append(rec)
    return selected


def normalize_kpts(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return parse_kpoint_mesh(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return int(value[0]), int(value[1]), int(value[2])
    return None


def build_step_vasp_kwargs(
    workflow_cfg: dict[str, Any],
    step_key: str,
    base_kpts: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    step_cfg = workflow_cfg.get(step_key, {})
    if not isinstance(step_cfg, dict):
        raise ValueError(f"workflow config key '{step_key}' must be a mapping")

    step_tags = step_cfg.get("vasp_tags", {})
    if not isinstance(step_tags, dict):
        raise ValueError(f"workflow config key '{step_key}.vasp_tags' must be a mapping")

    base_tags: dict[str, Any] = {}
    if step_key != "step1_relax":
        inherited = workflow_cfg.get("step1_relax", {}).get("vasp_tags", {})
        if isinstance(inherited, dict):
            base_tags = dict(inherited)

    merged = dict(base_tags)
    merged.update(step_tags)

    # If k-point scaling is requested and step-specific kpts is not set, use scaled base kpts.
    if "kpts_scale" in step_cfg and "kpts" not in step_tags:
        merged.pop("kpts", None)
        merged.pop("kpoints", None)

    kwargs = {key: coerce_scalar_for_vasp(value) for key, value in merged.items()}
    alias_map = {"encut_ev": "encut", "kpoints": "kpts"}
    for old_key, new_key in alias_map.items():
        if old_key in kwargs and new_key not in kwargs:
            kwargs[new_key] = kwargs.pop(old_key)

    kpts = normalize_kpts(kwargs.get("kpts"))
    if kpts is None and base_kpts is not None:
        scale = int(coerce_scalar_for_vasp(step_cfg.get("kpts_scale", 1)) or 1)
        if scale < 1:
            scale = 1
        scale_z = bool(coerce_scalar_for_vasp(step_cfg.get("kpts_scale_z", False)))
        if scale_z:
            kpts = tuple(max(1, int(scale * v)) for v in base_kpts)
        else:
            kpts = (
                max(1, int(scale * base_kpts[0])),
                max(1, int(scale * base_kpts[1])),
                int(base_kpts[2]),
            )
    if kpts is not None:
        kwargs["kpts"] = kpts

    return kwargs


def render_run_command(cfg: dict[str, object]) -> str:
    ntasks = str(cfg.get("NTASKS", "1"))
    run_command = str(cfg.get("RUN_COMMAND", "vasp_std")).strip() or "vasp_std"
    if "{NTASKS}" in run_command:
        run_command = run_command.replace("{NTASKS}", ntasks)
    if re.fullmatch(r"srun\s+vasp_std", run_command):
        return run_command
    if re.search(r"\bmpirun\b", run_command):
        return run_command
    return f"mpirun -np {ntasks} {run_command}"


def write_myrun(path: Path, cfg: dict[str, object], job_name: str, workdir: Path) -> None:
    required = ["PARTITION", "WALLTIME", "NODES", "NTASKS", "MEMORY"]
    missing = [k for k in required if not str(cfg.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Missing cluster config keys: {', '.join(missing)}")

    modules = cfg.get("MODULES", [])
    if not isinstance(modules, list):
        modules = []

    lines = [
        "#!/bin/bash -l",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {cfg['PARTITION']}",
        f"#SBATCH -t {cfg['WALLTIME']}",
        f"#SBATCH -N {cfg['NODES']}",
        f"#SBATCH -n {cfg['NTASKS']}",
        f"#SBATCH --mem {cfg['MEMORY']}",
    ]
    out_file = str(cfg.get("OUTPUT_FILE", "")).strip()
    if out_file:
        lines.append(f"#SBATCH -o {out_file}")

    lines.extend(
        [
            "",
            f"export OMP_NUM_THREADS={cfg.get('OMP_NUM_THREADS', '1')}",
            f"cd {workdir.resolve()}",
        ]
    )

    if modules:
        lines.append("module purge")
        for module in modules:
            lines.append(f"module load {module}")

    lines.append(render_run_command(cfg))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(path, 0o755)


def prepare_step_input(
    step_dir: Path,
    atoms: Atoms,
    workflow_cfg: dict[str, Any],
    step_key: str,
    base_kpts: tuple[int, int, int] | None = None,
) -> None:
    potcar_root = resolve_potpaw_pbe_root(Path(str(workflow_cfg.get("potcar_root", DEFAULT_POTCAR_DIR))))

    # ASE VASP expects VASP_PP_PATH to point to parent of potpaw_PBE.
    os.environ["VASP_PP_PATH"] = str(potcar_root.parent)

    vasp_kwargs = build_step_vasp_kwargs(workflow_cfg, step_key=step_key, base_kpts=base_kpts)
    calc = Vasp(**vasp_kwargs)
    with pushd(step_dir):
        set_vasp(atoms, calc)


def copy_restart_files(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    for name in names:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Required restart file not found: {src}")
        shutil.copy2(src, dst_dir / name)


def read_first_valid_structure(candidates: list[Path]) -> tuple[Atoms, Path]:
    errors: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            return read(path), path
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError(
        "Could not read any candidate structure file.\n" + ("\n".join(errors) if errors else "No files found.")
    )


def select_structure_file_for_step4(step1_dir: Path, step2_dir: Path, step3_dir: Path) -> Path:
    candidates = [
        step3_dir / "CONTCAR",
        step3_dir / "POSCAR",
        step2_dir / "CONTCAR",
        step2_dir / "POSCAR",
        step1_dir / "CONTCAR",
        step1_dir / "POSCAR",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Step 4 requested but no CONTCAR/POSCAR found in 03_dos, 02_scf, or 01_relax."
    )


def run_step4_symmetry_analysis(
    step4_dir: Path,
    atoms: Atoms,
    workflow_cfg: dict[str, Any],
    rec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    def to_json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_json_safe(v) for v in obj]
        if hasattr(obj, "tolist"):
            try:
                return to_json_safe(obj.tolist())
            except Exception:
                pass
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    step4_cfg = workflow_cfg.get("step4_band", {})
    if not isinstance(step4_cfg, dict):
        step4_cfg = {}
    sym_tol = float(coerce_scalar_for_vasp(step4_cfg.get("symmetry_tolerance", 1e-2)))
    angle_tol = float(coerce_scalar_for_vasp(step4_cfg.get("angle_tolerance", 5.0)))

    cell = atoms.cell.array.tolist()
    scaled_positions = atoms.get_scaled_positions().tolist()
    numbers = atoms.get_atomic_numbers().tolist()
    spgcell = (cell, scaled_positions, numbers)

    # By default we treat the largest lattice direction as non-periodic (2D slab).
    aperiodic_dir_cfg = step4_cfg.get("aperiodic_dir", "auto")
    if str(aperiodic_dir_cfg).lower() == "auto":
        lengths = atoms.cell.lengths()
        aperiodic_dir = int(max(range(3), key=lambda i: lengths[i]))
    else:
        aperiodic_dir = int(coerce_scalar_for_vasp(aperiodic_dir_cfg))
        if aperiodic_dir not in (0, 1, 2):
            raise ValueError("step4_band.aperiodic_dir must be 0, 1, 2, or 'auto'")

    layer_result: dict[str, Any] = {
        "symmetry_tolerance": sym_tol,
        "angle_tolerance": angle_tol,
        "aperiodic_dir": aperiodic_dir,
    }
    seekpath_result: dict[str, Any] = {
        "symmetry_tolerance": sym_tol,
        "angle_tolerance": angle_tol,
        "aperiodic_dir": aperiodic_dir,
    }

    def two_d_lattice_type(atoms_obj: Atoms, non_periodic_dir: int) -> str:
        idx = [i for i in range(3) if i != non_periodic_dir]
        v1 = atoms_obj.cell.array[idx[0]]
        v2 = atoms_obj.cell.array[idx[1]]
        l1 = float((v1 @ v1) ** 0.5)
        l2 = float((v2 @ v2) ** 0.5)
        cosang = max(-1.0, min(1.0, float((v1 @ v2) / (l1 * l2))))
        angle = float(math.degrees(math.acos(cosang)))
        rel = abs(l1 - l2) / max(l1, l2, 1e-12)
        len_tol = float(coerce_scalar_for_vasp(step4_cfg.get("lattice_length_rel_tol", 0.03)))
        ang_tol = float(coerce_scalar_for_vasp(step4_cfg.get("lattice_angle_tol_deg", 3.0)))
        if abs(angle - 90.0) <= ang_tol and rel <= len_tol:
            return "square"
        if abs(angle - 90.0) <= ang_tol:
            return "rectangular"
        if rel <= len_tol and (abs(angle - 60.0) <= ang_tol or abs(angle - 120.0) <= ang_tol):
            return "hexagonal"
        return "oblique"

    def format_kpath_string(path_segments: list[Any]) -> str:
        if not path_segments:
            return ""
        label_map = {"GAMMA": "G", "\\Gamma": "G"}
        chain: list[str] = []
        for idx_seg, seg in enumerate(path_segments):
            if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                continue
            s = label_map.get(str(seg[0]), str(seg[0]))
            e = label_map.get(str(seg[1]), str(seg[1]))
            if idx_seg == 0:
                chain.extend([s, e])
                continue
            if chain and chain[-1] == s:
                chain.append(e)
            else:
                chain.extend(["|", s, e])
        return " ".join(chain).strip()

    def canonical_2d_kpath(lattice_type: str) -> str:
        lut = {
            "square": "G X M G",
            "rectangular": "G X S Y G",
            "hexagonal": "G K M G",
            "oblique": "G X Y G",
        }
        return lut.get(lattice_type, "G X Y G")

    def canonical_2d_kpoints(lattice_type: str) -> dict[str, list[float]]:
        lut = {
            "square": {"G": [0.0, 0.0], "X": [0.5, 0.0], "M": [0.5, 0.5]},
            "rectangular": {
                "G": [0.0, 0.0],
                "X": [0.5, 0.0],
                "S": [0.5, 0.5],
                "Y": [0.0, 0.5],
            },
            "hexagonal": {"G": [0.0, 0.0], "K": [1.0 / 3.0, 1.0 / 3.0], "M": [0.5, 0.0]},
            "oblique": {"G": [0.0, 0.0], "X": [0.5, 0.0], "Y": [0.0, 0.5]},
        }
        return lut.get(lattice_type, lut["oblique"])

    try:
        layer_ds = spglib.get_symmetry_layerdataset(
            spgcell,
            aperiodic_dir=aperiodic_dir,
            symprec=sym_tol,
        )
        if layer_ds is None:
            raise RuntimeError("spglib returned no layer dataset")

        layer_group = spglib.get_layergroup(
            spgcell,
            aperiodic_dir=aperiodic_dir,
            symprec=sym_tol,
        )
        if layer_group is None:
            raise RuntimeError("spglib returned no layer group")

        standardized_cell = (
            layer_ds.std_lattice,
            layer_ds.std_positions,
            layer_ds.std_types,
        )
        kpath_raw = seekpath.get_path(
            standardized_cell,
            symprec=sym_tol,
            angle_tolerance=angle_tol,
        )

        point_coords_3d = dict(kpath_raw.get("point_coords", {}))
        path_3d = list(kpath_raw.get("path", []))

        # 2D projection: remove the aperiodic component from k-points.
        point_coords_2d: dict[str, list[float]] = {}
        for label, coord in point_coords_3d.items():
            if not isinstance(coord, (list, tuple)) or len(coord) != 3:
                continue
            in_plane = [float(coord[i]) for i in range(3) if i != aperiodic_dir]
            point_coords_2d[str(label)] = in_plane

        path_2d = [[str(p[0]), str(p[1])] for p in path_3d if isinstance(p, (list, tuple)) and len(p) == 2]
        lattice2d = two_d_lattice_type(atoms, aperiodic_dir)
        kpath_string_canonical = canonical_2d_kpath(lattice2d)
        kpoints_canonical = canonical_2d_kpoints(lattice2d)

        layer_result.update(
            {
                "layer_group_number": int(layer_group.number),
                "layer_group_symbol": str(layer_group.international),
                "layer_hall": str(layer_group.hall),
                "pointgroup": str(layer_group.pointgroup),
                "lattice_type_2d": lattice2d,
                "kpath_string_2d": kpath_string_canonical,
                "kpath_kpoints_2d": kpoints_canonical,
            }
        )
        seekpath_result.update(
            {
                "bravais_lattice": str(kpath_raw.get("bravais_lattice", "")),
                "bravais_lattice_extended": str(kpath_raw.get("bravais_lattice_extended", "")),
                "kpath_path_2d": path_2d,
                "kpath_kpoints_2d": point_coords_2d,
                "kpath_string_2d_canonical": kpath_string_canonical,
                "kpath_kpoints_2d_canonical": kpoints_canonical,
                "lattice_type_2d": lattice2d,
                "source": "seekpath",
            }
        )
    except Exception as exc:
        layer_result["error"] = f"{type(exc).__name__}: {exc}"
        seekpath_result["error"] = f"{type(exc).__name__}: {exc}"

    db_result = {
        "db_space_group_number": rec.get("db_space_group_number"),
        "db_crystal_system": rec.get("db_crystal_system"),
        "db_high_symmetry_path": rec.get("db_high_symmetry_path"),
        "db_kpts": rec.get("db_kpts", {}),
    }

    layer_result = to_json_safe(layer_result)
    seekpath_result = to_json_safe(seekpath_result)
    db_result = to_json_safe(db_result)

    # Explicitly remove obsolete ASE file from previous runs.
    ase_old = step4_dir / "ase_output.json"
    if ase_old.exists():
        ase_old.unlink()

    (step4_dir / "pymatgen_output.json").write_text(
        json.dumps(layer_result, indent=2), encoding="utf-8"
    )
    (step4_dir / "seekpath_output.json").write_text(
        json.dumps(seekpath_result, indent=2), encoding="utf-8"
    )
    (step4_dir / "db_kpath_summary.json").write_text(json.dumps(db_result, indent=2), encoding="utf-8")

    summary_lines = [
        "Step 4 2D Band/Symmetry Summary",
        f"symmetry_tolerance: {sym_tol}",
        f"angle_tolerance: {angle_tol}",
        f"aperiodic_dir: {aperiodic_dir}",
        f"lattice_type_2d: {layer_result.get('lattice_type_2d')}",
        f"kpath_string_2d: {layer_result.get('kpath_string_2d')}",
        "",
        "[spglib + seekpath]",
        f"layer_group_number: {layer_result.get('layer_group_number')}",
        f"layer_group_symbol: {layer_result.get('layer_group_symbol')}",
        f"pointgroup: {layer_result.get('pointgroup')}",
        f"kpath_segments: {len(seekpath_result.get('kpath_path_2d', []))}",
        f"error: {layer_result.get('error', '')}",
        "",
        "[2D path comparison]",
        f"spglib_path_2d: {layer_result.get('kpath_string_2d')}",
        f"seekpath_path_2d_canonical: {seekpath_result.get('kpath_string_2d_canonical')}",
        f"paths_match: {layer_result.get('kpath_string_2d') == seekpath_result.get('kpath_string_2d_canonical')}",
        "",
        "[DB]",
        f"db_space_group_number: {db_result.get('db_space_group_number')}",
        f"db_crystal_system: {db_result.get('db_crystal_system')}",
    ]

    # Add canonical 2D k-point coordinates used by kpath_string_2d.
    kpoint_coords = layer_result.get("kpath_kpoints_2d", {})
    if isinstance(kpoint_coords, dict) and kpoint_coords:
        summary_lines.extend(["", "[2D k-point coordinates]"])
        labels_in_path = str(layer_result.get("kpath_string_2d", "")).split()
        seen_labels: set[str] = set()
        for label in labels_in_path:
            if label == "|":
                continue
            if label in seen_labels:
                continue
            seen_labels.add(label)
            coord = kpoint_coords.get(label)
            if isinstance(coord, list) and len(coord) == 2:
                summary_lines.append(f"{label}: [{coord[0]}, {coord[1]}]")

    (step4_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return layer_result, seekpath_result


def write_band_kpoints_from_2d_path(
    kpoints_file: Path,
    kpath_string_2d: str,
    kpoints_2d: dict[str, list[float]],
    aperiodic_dir: int,
    points_per_segment: int,
) -> None:
    labels = [tok for tok in kpath_string_2d.split() if tok != "|"]
    if len(labels) < 2:
        raise ValueError(f"Invalid 2D k-path string: {kpath_string_2d!r}")

    def to_3d(coord2d: list[float]) -> list[float]:
        if len(coord2d) != 2:
            raise ValueError(f"Invalid 2D k-point coordinate: {coord2d}")
        out = [0.0, 0.0, 0.0]
        in_plane_idx = [i for i in range(3) if i != aperiodic_dir]
        out[in_plane_idx[0]] = float(coord2d[0])
        out[in_plane_idx[1]] = float(coord2d[1])
        return out

    lines = [
        "KPOINTS",
        str(max(2, int(points_per_segment))),
        "Line-mode",
        "Reciprocal",
    ]
    for i in range(len(labels) - 1):
        a = labels[i]
        b = labels[i + 1]
        if a not in kpoints_2d or b not in kpoints_2d:
            raise KeyError(f"K-point label missing coordinates: {a if a not in kpoints_2d else b}")
        ka = to_3d(kpoints_2d[a])
        kb = to_3d(kpoints_2d[b])
        lines.append(f"{ka[0]:.8f} {ka[1]:.8f} {ka[2]:.8f} ! {a}")
        lines.append(f"{kb[0]:.8f} {kb[1]:.8f} {kb[2]:.8f} ! {b}")
        lines.append("")

    kpoints_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def submit_step1(
    step_dir: Path,
    job_name: str,
    log_file: Path,
    dry_run: bool,
) -> dict[str, Any]:
    abs_step_dir = step_dir.resolve()

    if dry_run:
        log_message(log_file, f"[DRY-RUN] Would submit: {abs_step_dir / 'myrun.sh'}")
        return {"job_id": "DRYRUN", "step_dir": abs_step_dir, "job_name": job_name}

    with pushd(abs_step_dir):
        bsub_run(job_name, "myrun.sh")

        run_id_file = abs_step_dir / "run_id.txt"
        job_id = None
        for _ in range(10):
            if run_id_file.exists():
                text = run_id_file.read_text(encoding="utf-8").strip()
                match = re.search(r"\b(\d+)\b", text)
                if match:
                    job_id = match.group(1)
                    break
            time.sleep(0.5)
        if job_id is None:
            raise RuntimeError(f"Submission failed or run_id.txt has no job id in {abs_step_dir}")

        log_message(log_file, f"Submitted job: {job_name} (job_id={job_id})")
        return {"job_id": job_id, "step_dir": abs_step_dir, "job_name": job_name}


def monitor_submitted_jobs(
    jobs: list[dict[str, Any]],
    log_file: Path,
    poll_seconds: int,
    max_wait_hours: float,
) -> None:
    start = time.time()
    finished: set[str] = set()
    last_status: dict[str, str] = {}

    while len(finished) < len(jobs):
        for job in jobs:
            job_name = str(job["job_name"])
            if job_name in finished:
                continue

            step_dir = Path(str(job["step_dir"]))
            with pushd(step_dir):
                status = bsub_stat(job_name)

            if status != last_status.get(job_name):
                log_message(log_file, f"{job_name} status: {status}")
                last_status[job_name] = status

            if status == "Finished":
                rid = job.get("rid", "?")
                step_label = str(job.get("step_label", "step"))
                log_message(log_file, f"ID={rid}: {step_label} completed")
                finished.add(job_name)

        elapsed_h = (time.time() - start) / 3600.0
        if elapsed_h > max_wait_hours:
            pending = [str(job["job_name"]) for job in jobs if str(job["job_name"]) not in finished]
            raise TimeoutError(f"Timeout while waiting. Pending jobs: {pending}")

        if len(finished) < len(jobs):
            time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOS/PDOS/Band workflow (Step1 relaxation implemented)"
    )
    parser.add_argument("--input-db", type=Path, required=True, help="Input database (.db or .json)")
    parser.add_argument("--calc-name", type=str, default=DEFAULT_CALC_NAME, help="Calculation name")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional IDs (space/comma separated)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Workflow YAML config")
    parser.add_argument(
        "--slurm-config",
        type=Path,
        default=DEFAULT_SLURM_CONFIG,
        help="Cluster SLURM config file",
    )
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
    parser.add_argument(
        "--start-step",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Start step. Workflow runs sequentially from this step to 4.",
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
    # CLI --potcar overrides YAML; otherwise use YAML value or project default.
    selected_potcar = args.potcar or Path(str(workflow_cfg.get("potcar_root", DEFAULT_POTCAR_DIR)))
    workflow_cfg["potcar_root"] = str(selected_potcar)
    slurm_cfg = parse_shell_conf(args.slurm_config)
    ids_filter = parse_ids(args.ids)

    records = load_records(args.input_db)
    records = select_records(records, ids_filter)
    if not records:
        raise ValueError("No records selected. Check --ids or database content.")

    calc_root = (args.output_root / f"{sanitize_name(args.input_db.stem)}_{sanitize_name(args.calc_name)}").resolve()
    calc_root.mkdir(parents=True, exist_ok=True)
    log_file = calc_root / "workflow.log"

    log_message(log_file, f"Starting workflow: calc_name={args.calc_name}")
    log_message(log_file, f"Selected records: {len(records)}")
    log_message(log_file, f"SLURM config: {args.slurm_config}")
    log_message(log_file, f"POTCAR path: {selected_potcar}")
    log_message(log_file, f"Start step: {args.start_step}")

    job_prefix = str(workflow_cfg.get("job_name_prefix", "dos"))

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
                relaxed_atoms, source_structure = read_first_valid_structure(
                    [step1_dir / "CONTCAR", step1_dir / "POSCAR"]
                )
                log_message(
                    log_file,
                    f"Preparing step2 SCF for ID={rid}, material={material} from {source_structure.name}",
                )
                base_kpts = normalize_kpts(
                    workflow_cfg.get("step1_relax", {}).get("vasp_tags", {}).get("kpts")
                )
                prepare_step_input(
                    step2_dir,
                    relaxed_atoms,
                    workflow_cfg,
                    step_key="step2_scf",
                    base_kpts=base_kpts,
                )
                job_name = f"{job_prefix}_id{rid}_s2"
                run_dir = step2_dir
                step_label = "step2_scf"

            elif current_step == 3:
                scf_atoms, source_structure = read_first_valid_structure(
                    [step2_dir / "CONTCAR", step2_dir / "POSCAR"]
                )
                log_message(
                    log_file,
                    f"Preparing step3 DOS for ID={rid}, material={material} from {source_structure.name}",
                )
                base_kpts = normalize_kpts(
                    workflow_cfg.get("step2_scf", {}).get("vasp_tags", {}).get("kpts")
                ) or normalize_kpts(workflow_cfg.get("step1_relax", {}).get("vasp_tags", {}).get("kpts"))
                prepare_step_input(
                    step3_dir,
                    scf_atoms,
                    workflow_cfg,
                    step_key="step3_dos",
                    base_kpts=base_kpts,
                )
                copy_restart_files(step2_dir, step3_dir, ["CHGCAR", "WAVECAR"])
                job_name = f"{job_prefix}_id{rid}_s3"
                run_dir = step3_dir
                step_label = "step3_dos"

            else:
                atoms, source_structure = read_first_valid_structure(
                    [
                        step3_dir / "CONTCAR",
                        step3_dir / "POSCAR",
                        step2_dir / "CONTCAR",
                        step2_dir / "POSCAR",
                        step1_dir / "CONTCAR",
                        step1_dir / "POSCAR",
                    ]
                )
                log_message(
                    log_file,
                    f"Preparing step4 2D band for ID={rid}, material={material} from {source_structure.name}",
                )
                layer_result, _seekpath_result = run_step4_symmetry_analysis(step4_dir, atoms, workflow_cfg, rec)

                step4_cfg = workflow_cfg.get("step4_band", {})
                if not isinstance(step4_cfg, dict):
                    step4_cfg = {}
                if "vasp_tags" not in step4_cfg:
                    step4_cfg["vasp_tags"] = {}
                if isinstance(step4_cfg.get("vasp_tags"), dict):
                    step4_cfg["vasp_tags"].setdefault("kpts", [18, 18, 1])
                    step4_cfg["vasp_tags"].setdefault("icharge", 11)
                    step4_cfg["vasp_tags"].setdefault("lorbit", 11)
                    step4_cfg["vasp_tags"].setdefault("nsw", 0)
                    step4_cfg["vasp_tags"].setdefault("ibrion", -1)
                    step4_cfg["vasp_tags"].setdefault("lwave", False)
                    step4_cfg["vasp_tags"].setdefault("lcharg", False)
                    step4_cfg["vasp_tags"].pop("nedos", None)
                workflow_cfg["step4_band"] = step4_cfg

                prepare_step_input(step4_dir, atoms, workflow_cfg, step_key="step4_band")
                copy_restart_files(step2_dir, step4_dir, ["CHGCAR", "WAVECAR"])

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
            job_info = submit_step1(
                step_dir=run_dir,
                job_name=job_name,
                log_file=log_file,
                dry_run=args.dry_run,
            )
            job_info["rid"] = rid
            job_info["step_label"] = step_label
            submitted_jobs.append(job_info)

        if args.dry_run:
            for job in submitted_jobs:
                log_message(log_file, f"[DRY-RUN] Would wait for completion: {job['job_name']}")
                log_message(
                    log_file,
                    f"ID={job.get('rid', '?')}: {job.get('step_label', 'step')} completed",
                )
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
