#!/usr/bin/env python3
"""Build ASE database for W-N materials from raw adsorption export files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from ase.db import connect
from ase.io import read

DEFAULT_CSV = Path("data/raw/adsorption_materials_export/Adsorption_gibbs_with_i0.csv")
DEFAULT_MATERIALS_ROOT = Path("data/raw/adsorption_materials_export/materials")
DEFAULT_DB = Path("data/processed/wn_materials.db")


def normalize_csv_key(column_name: str, used: set[str]) -> str:
    key = column_name.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    if not key:
        key = "column"
    if key[0].isdigit():
        key = f"c_{key}"
    if key == "material":
        key = "csv_material"

    base = key
    idx = 1
    while key in used:
        idx += 1
        key = f"{base}_{idx}"
    used.add(key)
    return key


def parse_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def read_csv_rows(csv_path: Path) -> tuple[list[str], dict[str, dict[str, str]], dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {csv_path}")

        raw_headers = list(reader.fieldnames)
        clean_headers = [h.strip() for h in raw_headers]
        if "Material" not in clean_headers:
            raise ValueError(f"Missing 'Material' column in {csv_path}")

        # Stable header key map for all rows.
        used_keys: set[str] = set()
        csv_key_map: dict[str, str] = {}
        for header in clean_headers:
            csv_key_map[header] = normalize_csv_key(header, used_keys)

        material_to_row: dict[str, dict[str, str]] = {}
        material_ids: list[str] = []
        for row in reader:
            clean_row = {
                raw_headers[i].strip(): (row[raw_headers[i]] or "").strip()
                for i in range(len(raw_headers))
            }
            material = clean_row.get("Material", "")
            if not material:
                continue
            if material not in material_to_row:
                material_ids.append(material)
            material_to_row[material] = clean_row

    return material_ids, material_to_row, csv_key_map


def load_material_entry(material_dir: Path) -> dict[str, Any]:
    json_path = material_dir / "material_database_entry.json"
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_db(csv_path: Path, materials_root: Path, db_path: Path, expected_count: int = 9) -> int:
    material_ids, csv_rows, csv_key_map = read_csv_rows(csv_path)
    if len(material_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} materials from CSV, found {len(material_ids)} in {csv_path}"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    row_count = 0
    with connect(db_path) as db:
        for material in material_ids:
            material_dir = materials_root / material
            if not material_dir.exists():
                raise FileNotFoundError(f"Material folder not found: {material_dir}")

            structure_path = material_dir / "FINAL_STRUCTURE.vasp"
            atoms = read(structure_path)
            entry = load_material_entry(material_dir)

            material_record = entry.get("material_record", {})
            symmetry = material_record.get("symmetry", {})
            high_symmetry = symmetry.get("high_symmetry_path", {})
            csv_row = csv_rows.get(material)
            if csv_row is None:
                raise ValueError(f"Material '{material}' missing in CSV rows")

            unique_id = (
                material_record.get("struct_id")
                or material_record.get("full_name")
                or material
            )

            row_kwargs: dict[str, Any] = {
                "material": material,
                "material_uid": str(unique_id),
                "crystal_system": str(symmetry.get("crystal_system", "")),
                "high_symmetry_path": str(high_symmetry.get("path", "")),
                "kpts": json.dumps(high_symmetry.get("kpts", {}), sort_keys=True),
                "lattice_type": str(symmetry.get("lattice_type", "")),
                "space_group_number": int(symmetry.get("space_group_number", -1)),
                "structure_path": str(structure_path),
                "composition": str(csv_row.get("Composition", "")),
                "energy_above_hull_ev_atom": float(csv_row.get("Energy Above Hull (eV/atom)", "nan")),
            }

            # Add all CSV columns as ASE row fields with safe key names.
            for column_name, raw_value in csv_row.items():
                key = csv_key_map[column_name]
                value = parse_csv_value(raw_value)
                if value is None:
                    continue
                row_kwargs[key] = value

            db.write(
                atoms,
                data={
                    "metadata": entry,
                    "composition_by_element": material_record.get("composition", {}),
                },
                **row_kwargs,
            )
            row_count += 1

    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ASE database wn_materials.db from raw exports.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--materials-root", type=Path, default=DEFAULT_MATERIALS_ROOT)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--expected-count", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_count = build_db(
        csv_path=args.csv,
        materials_root=args.materials_root,
        db_path=args.output_db,
        expected_count=args.expected_count,
    )
    print(f"Created ASE database: {args.output_db}")
    print(f"Rows written: {row_count}")


if __name__ == "__main__":
    main()
