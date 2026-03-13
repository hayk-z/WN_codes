#!/usr/bin/env python3
"""Build processed ASE/JSON/YAML database for H adsorption structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ase.db import connect
from ase.io import read

DEFAULT_INPUT_ROOT = Path("data/raw/h_adsorption_structures_export")
DEFAULT_DB_OUT = Path("data/processed/h_adsorption_materials.db")
DEFAULT_JSON_OUT = Path("data/processed/h_adsorption_materials_export.json")
DEFAULT_YAML_OUT = Path("data/processed/h_adsorption_materials_export.yaml")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    as_float = to_float(value)
    if as_float is None:
        return None
    return int(as_float)


def atoms_to_dict(atoms) -> dict[str, Any]:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.cell.tolist(),
        "pbc": atoms.pbc.tolist(),
        "positions": atoms.positions.tolist(),
    }


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
    path.write_text("\n".join(to_yaml_lines(data)) + "\n", encoding="utf-8")


def resolve_structure(material_dir: Path, complex_record: dict[str, Any]) -> tuple[Path, str]:
    copied = complex_record.get("copied_files", {})
    if isinstance(copied, dict):
        rel = copied.get("relaxed_struct")
        if rel:
            candidate = material_dir / "relaxed_structures" / Path(str(rel)).name
            if candidate.exists():
                return candidate, "relaxed"

    full_name = str(complex_record.get("full_name", "")).strip()
    if full_name:
        candidate = material_dir / "relaxed_structures" / f"{full_name}.vasp"
        if candidate.exists():
            return candidate, "relaxed"

    if isinstance(copied, dict):
        init = copied.get("initial_struct")
        if init:
            candidate = material_dir / "initial_structures" / Path(str(init)).name
            if candidate.exists():
                return candidate, "initial"

    if full_name:
        candidate = material_dir / "initial_structures" / f"{full_name}.vasp"
        if candidate.exists():
            return candidate, "initial"

    raise FileNotFoundError(f"No structure file found for material dir: {material_dir}")


def build_records(input_root: Path) -> list[dict[str, Any]]:
    materials_dir = input_root / "materials"
    if not materials_dir.exists():
        raise FileNotFoundError(f"Materials directory not found: {materials_dir}")

    records: list[dict[str, Any]] = []
    for material_json in sorted(materials_dir.glob("*/h_adsorption_data.json")):
        material_dir = material_json.parent
        data = json.loads(material_json.read_text(encoding="utf-8"))

        parent_material = str(data.get("material", material_dir.name))
        csv_row = data.get("csv_row", {}) if isinstance(data.get("csv_row"), dict) else {}
        material_record = data.get("material_record", {}) if isinstance(data.get("material_record"), dict) else {}
        symmetry = material_record.get("symmetry", {}) if isinstance(material_record.get("symmetry"), dict) else {}
        composition = material_record.get("composition", {}) if isinstance(material_record.get("composition"), dict) else {}

        e_above_hull = to_float(csv_row.get("Energy Above Hull (eV/atom)"))
        if e_above_hull is None:
            e_above_hull = to_float(material_record.get("e_above_hull"))

        for complex_record in data.get("adsorption_complexes", []):
            if not isinstance(complex_record, dict):
                continue

            try:
                structure_path, structure_source = resolve_structure(material_dir, complex_record)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}; skipping complex {complex_record.get('full_name')}")
                continue
            atoms = read(structure_path)
            site = complex_record.get("site", {}) if isinstance(complex_record.get("site"), dict) else {}

            x = to_float(site.get("x"))
            y = to_float(site.get("y"))
            z = to_float(site.get("z"))
            adsorption_coord = [x, y, z]

            row = {
                "material": str(complex_record.get("full_name", "")),
                "parent_material": parent_material,
                "h_adsorption_type": site.get("name"),
                "h_adsorption_coordinate": adsorption_coord,
                "composition": composition,
                "surface_area_a2": to_float(csv_row.get("Surface Area (A^2)")),
                "surface_area_cm2": to_float(csv_row.get("Surface Area (cm^2)")),
                "i0": to_float(csv_row.get("i0")),
                "gibbs_free_energy_ev": to_float(csv_row.get("Gibbs free energy (eV)")),
                "e_above_hull": e_above_hull,
                "crystal_system": symmetry.get("crystal_system"),
                "space_group_symbol": symmetry.get("space_group_symbol"),
                "space_group_number": to_int(symmetry.get("space_group_number")),
                "structure_path": str(structure_path),
                "structure_source": structure_source,
                "structure_parameters": {
                    "natoms": len(atoms),
                    "cell": atoms.cell.tolist(),
                    "pbc": atoms.pbc.tolist(),
                },
                "structure": atoms_to_dict(atoms),
                "_atoms_obj": atoms,
            }
            records.append(row)
    return records


def write_ase_db(records: list[dict[str, Any]], db_out: Path) -> list[dict[str, Any]]:
    db_out.parent.mkdir(parents=True, exist_ok=True)
    if db_out.exists():
        db_out.unlink()

    exported: list[dict[str, Any]] = []
    with connect(db_out) as db:
        for record in records:
            atoms = record["_atoms_obj"]
            kv = {
                "material": record["material"],
                "parent_material": record["parent_material"],
                "h_adsorption_type": record["h_adsorption_type"],
                "h_adsorption_coordinate": json.dumps(record["h_adsorption_coordinate"]),
                "composition": json.dumps(record["composition"], sort_keys=True),
                "surface_area_a2": record["surface_area_a2"],
                "surface_area_cm2": record["surface_area_cm2"],
                "i0": record["i0"],
                "gibbs_free_energy_ev": record["gibbs_free_energy_ev"],
                "e_above_hull": record["e_above_hull"],
                "crystal_system": record["crystal_system"],
                "space_group_symbol": record["space_group_symbol"],
                "space_group_number": record["space_group_number"],
                "structure_path": record["structure_path"],
                "structure_source": record["structure_source"],
                "structure_parameters": json.dumps(record["structure_parameters"]),
            }
            row_id = db.write(atoms, **kv)
            out_rec = {k: v for k, v in record.items() if k != "_atoms_obj"}
            out_rec["id"] = row_id
            exported.append(out_rec)
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ASE/JSON/YAML database for H adsorption structures")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--db-out", type=Path, default=DEFAULT_DB_OUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--yaml-out", type=Path, default=DEFAULT_YAML_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(args.input_root)
    exported = write_ase_db(records, args.db_out)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    write_yaml(args.yaml_out, exported)

    print(f"Wrote ASE DB: {args.db_out}")
    print(f"Wrote JSON: {args.json_out}")
    print(f"Wrote YAML: {args.yaml_out}")
    print(f"Total records: {len(exported)}")


if __name__ == "__main__":
    main()
