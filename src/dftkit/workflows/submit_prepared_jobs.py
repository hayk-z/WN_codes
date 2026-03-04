#!/usr/bin/env python3
"""Submit prepared VASP job folders that contain myrun.sh."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


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


def extract_id_from_dirname(dirname: str) -> str | None:
    # Expected folder format: id_<id>_<material_name>
    match = re.match(r"^id_(\d+)_", dirname)
    if not match:
        return None
    return match.group(1)


def collect_job_dirs(prepared_root: Path, ids_filter: set[str], script_name: str) -> list[Path]:
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root not found: {prepared_root}")
    if not prepared_root.is_dir():
        raise ValueError(f"Prepared root is not a directory: {prepared_root}")

    jobs: list[tuple[int, Path]] = []
    for child in prepared_root.iterdir():
        if not child.is_dir():
            continue
        rid = extract_id_from_dirname(child.name)
        if rid is None:
            continue
        if ids_filter and rid not in ids_filter:
            continue
        runfile = child / script_name
        if runfile.exists():
            jobs.append((int(rid), child))

    jobs.sort(key=lambda x: x[0])
    return [job_dir for _, job_dir in jobs]


def submit_job(job_dir: Path, script_name: str, submit_cmd: str, dry_run: bool) -> tuple[bool, str]:
    cmd = [submit_cmd, script_name]
    if dry_run:
        return True, f"[DRY-RUN] {' '.join(cmd)} (cwd={job_dir})"

    proc = subprocess.run(cmd, cwd=job_dir, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        error_text = proc.stderr.strip() or proc.stdout.strip() or "Unknown submission error"
        return False, error_text

    output = proc.stdout.strip() or "Submitted"
    return True, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit prepared VASP jobs via myrun.sh")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Directory containing id_<id>_<material>/ subfolders",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional DB record IDs (space or comma separated). If omitted, submit all.",
    )
    parser.add_argument(
        "--script-name",
        type=str,
        default="myrun.sh",
        help="Run script file name in each prepared job folder (default: myrun.sh)",
    )
    parser.add_argument(
        "--submit-cmd",
        type=str,
        default="sbatch",
        help="Scheduler submit command (default: sbatch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without calling scheduler",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids_filter = parse_ids(args.ids)
    job_dirs = collect_job_dirs(args.prepared_root, ids_filter, args.script_name)
    if not job_dirs:
        raise ValueError("No matching prepared job directories found.")

    submitted = 0
    failed = 0

    for job_dir in job_dirs:
        ok, message = submit_job(
            job_dir=job_dir,
            script_name=args.script_name,
            submit_cmd=args.submit_cmd,
            dry_run=args.dry_run,
        )
        label = job_dir.name
        if ok:
            submitted += 1
            print(f"[OK] {label}: {message}")
        else:
            failed += 1
            print(f"[FAIL] {label}: {message}")

    print(f"Summary: ok={submitted}, failed={failed}, total={len(job_dirs)}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
