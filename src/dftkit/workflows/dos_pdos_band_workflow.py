#!/usr/bin/env python3
"""Prepare and run DOS/PDOS/BAND workflow (Step 1: relaxation implemented)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.db import connect
from ase.io import read

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
        return run_command.replace("{NTASKS}", ntasks)
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
    potcar_root = Path(str(workflow_cfg.get("potcar_root", ""))).expanduser()
    if not potcar_root.exists():
        raise FileNotFoundError(f"POTCAR_ROOT does not exist: {potcar_root}")

    # ASE VASP expects VASP_PP_PATH to point to parent of potpaw_PBE.
    os.environ["VASP_PP_PATH"] = str(potcar_root.parent)

    vasp_kwargs = build_step_vasp_kwargs(workflow_cfg, step_key=step_key, base_kpts=base_kpts)
    calc = Vasp(**vasp_kwargs)
    with pushd(step_dir):
        set_vasp(atoms, calc)


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

        log_message(log_file, f"Submitted step1 relaxation: {job_name} (job_id={job_id})")
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
        "--start-step",
        type=int,
        choices=[1, 2],
        default=1,
        help="Step to execute: 1=relax, 2=scf (from 01_relax/CONTCAR)",
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
    log_message(log_file, f"Start step: {args.start_step}")

    job_prefix = str(workflow_cfg.get("job_name_prefix", "dos"))
    submitted_jobs: list[dict[str, Any]] = []

    for rec in records:
        rid = rec["id"]
        material = sanitize_name(str(rec.get("material", f"id_{rid}")))
        mat_dir = calc_root / f"id_{rid}_{material}"
        step1_dir = mat_dir / "01_relax"
        step2_dir = mat_dir / "02_scf"
        step3_dir = mat_dir / "03_pdos"
        step4_dir = mat_dir / "04_band"

        # Create full step tree now; only step1 is populated/submitted in this stage.
        for d in (step1_dir, step2_dir, step3_dir, step4_dir):
            d.mkdir(parents=True, exist_ok=True)

        if args.start_step == 1:
            log_message(log_file, f"Preparing step1 relaxation for ID={rid}, material={material}")
            prepare_step_input(step1_dir, rec["atoms"], workflow_cfg, step_key="step1_relax")
            job_name = f"{job_prefix}_id{rid}_r1"
        else:
            contcar = step1_dir / "CONTCAR"
            if not contcar.exists():
                raise FileNotFoundError(
                    f"Step 2 requested but CONTCAR not found for ID={rid}: {contcar}"
                )
            log_message(log_file, f"Preparing step2 SCF for ID={rid}, material={material} from CONTCAR")
            relaxed_atoms = read(contcar)
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

        run_dir = step1_dir if args.start_step == 1 else step2_dir
        write_myrun(run_dir / "myrun.sh", slurm_cfg, job_name=job_name, workdir=run_dir)

        job_info = submit_step1(
            step_dir=run_dir,
            job_name=job_name,
            log_file=log_file,
            dry_run=args.dry_run,
        )
        job_info["rid"] = rid
        job_info["step_label"] = "step1_relax" if args.start_step == 1 else "step2_scf"
        submitted_jobs.append(job_info)

    if args.dry_run:
        for job in submitted_jobs:
            log_message(log_file, f"[DRY-RUN] Would wait for completion: {job['job_name']}")
            log_message(
                log_file,
                f"ID={job.get('rid', '?')}: {job.get('step_label', 'step')} completed",
            )
    else:
        log_message(log_file, f"Submitted {len(submitted_jobs)} jobs. Start monitoring...")
        monitor_submitted_jobs(
            jobs=submitted_jobs,
            log_file=log_file,
            poll_seconds=args.poll_seconds,
            max_wait_hours=args.max_wait_hours,
        )

    log_message(log_file, "Workflow finished")


if __name__ == "__main__":
    main()
