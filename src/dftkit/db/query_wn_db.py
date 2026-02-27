#!/usr/bin/env python3
"""Query/export helper for wn_materials ASE database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ase.db import connect

DEFAULT_DB = Path("data/processed/wn_materials.db")
DEFAULT_JSON = Path("data/processed/wn_materials_export.json")
DEFAULT_YAML = Path("data/processed/wn_materials_export.yaml")


def atoms_to_dict(atoms) -> dict[str, Any]:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.cell.tolist(),
        "pbc": atoms.pbc.tolist(),
        "positions": atoms.positions.tolist(),
    }


def row_to_record(row) -> dict[str, Any]:
    atoms = row.toatoms()
    record = {
        "id": row.id,
        "material": getattr(row, "material", None),
        "material_uid": getattr(row, "material_uid", None),
        "crystal_system": getattr(row, "crystal_system", None),
        "high_symmetry_path": getattr(row, "high_symmetry_path", None),
        "kpts": json.loads(getattr(row, "kpts", "{}") or "{}"),
        "lattice_type": getattr(row, "lattice_type", None),
        "space_group_number": getattr(row, "space_group_number", None),
        "structure_path": getattr(row, "structure_path", None),
        "structure": atoms_to_dict(atoms),
    }
    for key, value in row.key_value_pairs.items():
        if key not in record:
            record[key] = value
    return record


def record_matches(record: dict[str, Any], material: str | None, crystal_system: str | None) -> bool:
    if material and record.get("material") != material:
        return False
    if crystal_system and str(record.get("crystal_system", "")).lower() != crystal_system.lower():
        return False
    return True


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def to_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines: list[str] = []

    if isinstance(value, dict):
        if not value:
            return [pad + "{}"]
        for key, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(to_yaml_lines(v, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(v)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [pad + "[]"]
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(pad + "-")
                lines.extend(to_yaml_lines(item, indent + 2))
            else:
                lines.append(pad + f"- {yaml_scalar(item)}")
        return lines

    return [pad + yaml_scalar(value)]


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(to_yaml_lines(data)) + "\n"
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query/export wn_materials ASE db")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--material", type=str, default=None)
    parser.add_argument("--crystal-system", type=str, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--yaml-out", type=Path, default=None)
    parser.add_argument("--export-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")

    records: list[dict[str, Any]] = []
    with connect(args.db) as db:
        for row in db.select():
            rec = row_to_record(row)
            if record_matches(rec, args.material, args.crystal_system):
                records.append(rec)

    if not args.export_all:
        records = records[: args.limit]

    print(f"Matched records: {len(records)}")
    for rec in records:
        print(
            f"- {rec['material']} | {rec['material_uid']} | "
            f"{rec['crystal_system']} | SG {rec['space_group_number']}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.json_out}")

    if args.yaml_out:
        write_yaml(args.yaml_out, records)
        print(f"Wrote YAML: {args.yaml_out}")


if __name__ == "__main__":
    main()
