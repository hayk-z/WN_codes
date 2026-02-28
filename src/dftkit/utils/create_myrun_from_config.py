#!/usr/bin/env python3
"""Generate a SLURM myrun.sh script from a cluster config file."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


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
            module = line.rstrip(",").strip()
            if module:
                modules.append(_strip_quotes(module))
            continue

        if line.startswith("MODULES=") and line.endswith("("):
            in_modules = True
            modules = []
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = _strip_quotes(value.strip())

    if "MODULES" not in config:
        config["MODULES"] = []
    return config


def build_script(config: dict[str, object], workdir: Path) -> str:
    required = ["JOB_NAME", "PARTITION", "WALLTIME", "NODES", "NTASKS", "MEMORY"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    ntasks = str(config["NTASKS"])
    if not re.fullmatch(r"\d+", ntasks):
        raise ValueError(f"NTASKS must be an integer, got: {ntasks!r}")

    modules = config.get("MODULES", [])
    if not isinstance(modules, list):
        modules = []

    output_file = str(config.get("OUTPUT_FILE", "slurm-%j.out"))
    omp_threads = str(config.get("OMP_NUM_THREADS", "1"))
    run_command = str(config.get("RUN_COMMAND", "")).strip()
    if not run_command:
        raise ValueError("Missing required config key: RUN_COMMAND")

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {config['JOB_NAME']}",
        f"#SBATCH -p {config['PARTITION']}",
        f"#SBATCH -t {config['WALLTIME']}",
        f"#SBATCH -N {config['NODES']}",
        f"#SBATCH -n {ntasks}",
        f"#SBATCH --mem {config['MEMORY']}",
        f"#SBATCH -o {output_file}",
        "",
        f"export OMP_NUM_THREADS={omp_threads}",
        f"cd {workdir}",
    ]

    if modules:
        lines.append("module purge")
        for module in modules:
            lines.append(f"module load {module}")

    # RUN_COMMAND is sourced from config.
    # Recommended format in config: RUN_COMMAND="mpirun -np {NTASKS} vasp_std"
    # If {NTASKS} is missing, append command after mpirun -np <NTASKS>.
    if "{NTASKS}" in run_command:
        final_command = run_command.replace("{NTASKS}", ntasks)
    else:
        final_command = f"mpirun -np {ntasks} {run_command}"

    lines.append(final_command)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create myrun.sh from a cluster .conf file")
    parser.add_argument(
        "--config",
        default="configs/slurm_ysu2.conf",
        help="Path to cluster config file (default: configs/slurm_ysu2.conf)",
    )
    parser.add_argument(
        "--output",
        default="myrun.sh",
        help="Output script path (default: myrun.sh)",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for the generated script (default: current directory)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    out_path = Path(args.output).resolve()
    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()

    config = parse_cluster_conf(config_path)
    script = build_script(config, workdir)

    out_path.write_text(script, encoding="utf-8")
    os.chmod(out_path, 0o755)
    print(f"Created: {out_path}")


if __name__ == "__main__":
    main()
