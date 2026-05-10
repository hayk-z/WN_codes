#!/usr/bin/env python3
"""Plot DOS, PDOS, and band structure from prepared workflow directories."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from pymatgen.electronic_structure.plotter import BSPlotter, BSPlotterProjected, DosPlotter
from pymatgen.io.vasp.outputs import Vasprun

ORBITAL_ORDER = ("s", "p", "d", "f")
ONE_COLUMN_WIDTH_IN = 90.0 / 25.4  # Elsevier single-column width
SINGLE_PANEL_HEIGHT_IN = 60.0 / 25.4  # 60 mm height
COMBINED_PANEL_HEIGHT_IN = 60.0 / 25.4  # 60 mm height
EXPORT_DPI = 1000
AXIS_FONT_SIZE = 7
TITLE_FONT_SIZE = 8
DOS_LINEWIDTH_SCALE = 0.65
LEGEND_ANCHOR_X = 1.16
LEGEND_ANCHOR_Y = 1.0
LEGEND_RIGHT_MARGIN = 0.76

# Prefer Times New Roman for all figure text.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "material"


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


def load_records_from_db(input_db: Path) -> list[dict[str, str]]:
    suffix = input_db.suffix.lower()
    if suffix == ".db":
        from ase.db import connect

        with connect(input_db) as db:
            out: list[dict[str, str]] = []
            for row in db.select():
                kv = getattr(row, "key_value_pairs", {}) or {}
                material = str(kv.get("material", "")).strip()
                out.append({"id": str(row.id), "material": material})
            return out

    if suffix == ".json":
        data = json.loads(input_db.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records")
        out: list[dict[str, str]] = []
        for rec in data:
            if isinstance(rec, dict) and rec.get("id") is not None:
                material = str(rec.get("material", "")).strip()
                out.append({"id": str(rec["id"]), "material": material})
        return out

    raise ValueError(f"Unsupported input-db format: {input_db} (use .db or .json)")


def find_material_dir(calc_root: Path, rid: str, material_hint: str | None = None) -> Path | None:
    matches = sorted(calc_root.glob(f"id_{rid}_*"))
    material_token = sanitize_name(material_hint).lower() if material_hint else ""
    if material_token:
        for match in matches:
            if material_token in match.name.lower():
                return match
        fallback_matches = sorted(calc_root.glob("id_*"))
        for match in fallback_matches:
            if material_token in match.name.lower():
                return match
    if matches:
        return matches[0]
    return None


def _style_axes(fig: plt.Figure) -> None:
    for ax in fig.get_axes():
        ax.tick_params(axis="both", which="major", labelsize=AXIS_FONT_SIZE)
        ax.xaxis.label.set_size(AXIS_FONT_SIZE + 0.5)
        ax.yaxis.label.set_size(AXIS_FONT_SIZE + 0.5)
        ax.title.set_size(TITLE_FONT_SIZE)


def _save_figure(fig: plt.Figure, out_file: Path, height_in: float) -> None:
    fig.set_size_inches(ONE_COLUMN_WIDTH_IN, height_in, forward=True)
    _style_axes(fig)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout(pad=0.3)
    fig.savefig(out_file, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_file.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_file.with_suffix(".jpg"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_plot_object(plot_obj, out_file: Path) -> None:
    if hasattr(plot_obj, "figure"):
        fig = plot_obj.figure
    elif hasattr(plot_obj, "gcf"):
        fig = plot_obj.gcf()
    else:
        fig = plt.gcf()
    _save_figure(fig, out_file, SINGLE_PANEL_HEIGHT_IN)


def _style_dos_lines(ax: plt.Axes) -> None:
    for line in ax.get_lines():
        line.set_linewidth(max(0.5, line.get_linewidth() * DOS_LINEWIDTH_SCALE))


def _place_legend_outside_top_right(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if not filtered:
        return
    handles, labels = zip(*filtered)
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(LEGEND_ANCHOR_X, LEGEND_ANCHOR_Y),
        borderaxespad=0.0,
        frameon=False,
        fontsize=AXIS_FONT_SIZE,
    )
def _orbital_name(orbital_key) -> str:
    name = getattr(orbital_key, "name", str(orbital_key))
    return name.split(".")[-1].lower()


def _iter_spd_dos(spd_map: dict):
    keyed = {_orbital_name(k): dos for k, dos in spd_map.items()}
    for orb in ORBITAL_ORDER:
        if orb in keyed:
            yield orb, keyed[orb]


def _find_h_s_peak_energy(complete_dos, e_min: float | None, e_max: float | None) -> float | None:
    element_dos = complete_dos.get_element_dos()
    h_element = next((el for el in element_dos if str(el) == "H"), None)
    if h_element is None:
        return None

    h_spd = complete_dos.get_element_spd_dos(h_element)
    h_s_dos = next((dos for orb, dos in _iter_spd_dos(h_spd) if orb == "s"), None)
    if h_s_dos is None:
        return None

    energies = np.array(h_s_dos.energies) - float(h_s_dos.efermi)
    densities = _sum_spin_density(h_s_dos.densities)
    mask = np.ones_like(energies, dtype=bool)
    if e_min is not None:
        mask &= energies >= e_min
    if e_max is not None:
        mask &= energies <= e_max
    if not np.any(mask):
        return None

    masked_densities = densities[mask]
    if masked_densities.size == 0 or float(np.max(masked_densities)) <= 0.0:
        return None

    masked_energies = energies[mask]
    peak_idx = int(np.argmax(masked_densities))
    return float(masked_energies[peak_idx])


def _add_hs_marker_vertical(ax, energy: float | None) -> None:
    if energy is None:
        return
    ax.axvline(energy, color="tab:green", ls="--", lw=1.1, alpha=0.95, label=f"H-s peak ({energy:.2f} eV)")


def _add_hs_marker_horizontal(ax, energy: float | None) -> None:
    if energy is None:
        return
    ax.axhline(energy, color="tab:green", ls="--", lw=1.1, alpha=0.95, label=f"H-s peak ({energy:.2f} eV)")


def _finalize_dos_plot(plot: plt.Axes, show_labels: bool, show_hs_orbital_marker: bool) -> None:
    _style_dos_lines(plot)
    if show_labels or show_hs_orbital_marker:
        _place_legend_outside_top_right(plot)
        plot.figure.subplots_adjust(right=LEGEND_RIGHT_MARGIN)
    else:
        legend = plot.get_legend()
        if legend is not None:
            legend.remove()


def save_total_dos_plot(
    dos_vr: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
    show_labels: bool,
    show_hs_orbital_marker: bool = False,
) -> None:
    vr = Vasprun(str(dos_vr), parse_dos=True, parse_eigen=False)
    plotter = DosPlotter(sigma=0.05)
    plotter.add_dos("Total DOS", vr.complete_dos)
    xlim = (e_min, e_max) if (e_min is not None and e_max is not None) else None
    plot = plotter.get_plot(xlim=xlim)
    if show_hs_orbital_marker:
        hs_peak = _find_h_s_peak_energy(vr.complete_dos, e_min, e_max)
        _add_hs_marker_vertical(plot, hs_peak)
    _finalize_dos_plot(plot, show_labels, show_hs_orbital_marker)
    save_plot_object(plot, out_file)


def save_element_pdos_plot(
    dos_vr: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
    show_labels: bool,
    show_hs_orbital_marker: bool = False,
) -> None:
    vr = Vasprun(str(dos_vr), parse_dos=True, parse_eigen=False)
    element_dos = vr.complete_dos.get_element_dos()
    if not element_dos:
        raise ValueError("No element-projected DOS found in vasprun.xml")

    plotter = DosPlotter(sigma=0.05)
    for element, dos in element_dos.items():
        plotter.add_dos(str(element), dos)

    xlim = (e_min, e_max) if (e_min is not None and e_max is not None) else None
    plot = plotter.get_plot(xlim=xlim)
    if show_hs_orbital_marker:
        hs_peak = _find_h_s_peak_energy(vr.complete_dos, e_min, e_max)
        _add_hs_marker_vertical(plot, hs_peak)
    _finalize_dos_plot(plot, show_labels, show_hs_orbital_marker)
    save_plot_object(plot, out_file)


def save_total_orbital_dos_plot(
    dos_vr: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
    show_labels: bool,
    show_hs_orbital_marker: bool = False,
) -> None:
    vr = Vasprun(str(dos_vr), parse_dos=True, parse_eigen=False)
    complete_dos = vr.complete_dos
    spd_dos = complete_dos.get_spd_dos()
    if not spd_dos:
        raise ValueError("No orbital-projected DOS found in vasprun.xml")

    plotter = DosPlotter(sigma=0.05)
    plotter.add_dos("Total DOS", complete_dos)
    added = False
    for orb, dos in _iter_spd_dos(spd_dos):
        plotter.add_dos(f"{orb.upper()}", dos)
        added = True
    if not added:
        raise ValueError("No s/p/d/f orbital DOS channels found")

    xlim = (e_min, e_max) if (e_min is not None and e_max is not None) else None
    plot = plotter.get_plot(xlim=xlim)
    if show_hs_orbital_marker:
        hs_peak = _find_h_s_peak_energy(complete_dos, e_min, e_max)
        _add_hs_marker_vertical(plot, hs_peak)
    _finalize_dos_plot(plot, show_labels, show_hs_orbital_marker)
    save_plot_object(plot, out_file)


def save_element_orbital_pdos_plot(
    dos_vr: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
    show_labels: bool,
    show_hs_orbital_marker: bool = False,
) -> None:
    vr = Vasprun(str(dos_vr), parse_dos=True, parse_eigen=False)
    complete_dos = vr.complete_dos
    element_dos = complete_dos.get_element_dos()
    if not element_dos:
        raise ValueError("No element-projected DOS found in vasprun.xml")

    plotter = DosPlotter(sigma=0.05)
    added = False
    for element in element_dos:
        spd_dos = complete_dos.get_element_spd_dos(element)
        for orb, dos in _iter_spd_dos(spd_dos):
            plotter.add_dos(f"{element}-{orb.upper()}", dos)
            added = True
    if not added:
        raise ValueError("No element s/p/d/f orbital DOS channels found")

    xlim = (e_min, e_max) if (e_min is not None and e_max is not None) else None
    plot = plotter.get_plot(xlim=xlim)
    if show_hs_orbital_marker:
        hs_peak = _find_h_s_peak_energy(complete_dos, e_min, e_max)
        _add_hs_marker_vertical(plot, hs_peak)
    _finalize_dos_plot(plot, show_labels, show_hs_orbital_marker)
    save_plot_object(plot, out_file)


def save_n_w_orbital_pdos_plot(
    dos_vr: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
    show_labels: bool,
    show_hs_orbital_marker: bool = False,
) -> None:
    vr = Vasprun(str(dos_vr), parse_dos=True, parse_eigen=False)
    complete_dos = vr.complete_dos
    element_dos = complete_dos.get_element_dos()
    if not element_dos:
        raise ValueError("No element-projected DOS found in vasprun.xml")

    requested = {"N": {"s", "p"}, "W": {"s", "p", "d", "f"}}
    plotter = DosPlotter(sigma=0.05)
    added = False
    for element in element_dos:
        element_name = str(element)
        if element_name not in requested:
            continue
        spd_dos = complete_dos.get_element_spd_dos(element)
        for orb, dos in _iter_spd_dos(spd_dos):
            if orb not in requested[element_name]:
                continue
            plotter.add_dos(f"{element_name}-{orb}", dos)
            added = True
    if not added:
        raise ValueError("Requested N/W orbital PDOS channels were not found")

    xlim = (e_min, e_max) if (e_min is not None and e_max is not None) else None
    plot = plotter.get_plot(xlim=xlim)
    if show_hs_orbital_marker:
        hs_peak = _find_h_s_peak_energy(complete_dos, e_min, e_max)
        _add_hs_marker_vertical(plot, hs_peak)
    _finalize_dos_plot(plot, show_labels, show_hs_orbital_marker)
    save_plot_object(plot, out_file)


def save_band_plot(
    band_vr: Path,
    kpoints_file: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
) -> None:
    vr = Vasprun(str(band_vr), parse_projected_eigen=False)
    bs = vr.get_band_structure(kpoints_filename=str(kpoints_file), line_mode=True)
    plotter = BSPlotter(bs)
    ylim = (e_min, e_max) if (e_min is not None and e_max is not None) else None
    plot = plotter.get_plot(ylim=ylim)
    save_plot_object(plot, out_file)


def save_projected_band_plot(
    band_vr: Path,
    kpoints_file: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
) -> None:
    vr = Vasprun(str(band_vr), parse_projected_eigen=True)
    bs = vr.get_band_structure(kpoints_filename=str(kpoints_file), line_mode=True)
    plotter = BSPlotterProjected(bs)
    plot = plotter.get_elt_projected_plots_color()
    if e_min is not None and e_max is not None:
        axes = plot if isinstance(plot, (list, tuple, np.ndarray)) else [plot]
        for ax in axes:
            ax.set_ylim(e_min, e_max)
    save_plot_object(plot, out_file)


def _sum_spin_density(densities: dict) -> np.ndarray:
    total = None
    for arr in densities.values():
        arr_np = np.array(arr)
        total = arr_np if total is None else total + arr_np
    if total is None:
        raise ValueError("No DOS densities found")
    return total


def save_band_dos_combined_plot(
    band_vr: Path,
    dos_vr: Path,
    kpoints_file: Path,
    out_file: Path,
    e_min: float | None,
    e_max: float | None,
    show_labels: bool,
    show_hs_orbital_marker: bool = False,
) -> None:
    vr_band = Vasprun(str(band_vr), parse_projected_eigen=False)
    bs = vr_band.get_band_structure(kpoints_filename=str(kpoints_file), line_mode=True)
    bs_data = BSPlotter(bs).bs_plot_data()

    vr_dos = Vasprun(str(dos_vr), parse_dos=True, parse_eigen=False)
    cdos = vr_dos.complete_dos
    energies = np.array(cdos.energies) - float(cdos.efermi)
    total_dos = _sum_spin_density(cdos.densities)
    elem_dos = cdos.get_element_dos()

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH_IN, COMBINED_PANEL_HEIGHT_IN))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.3, 1.2], wspace=0.1)
    ax_band = fig.add_subplot(gs[0, 0])
    ax_dos = fig.add_subplot(gs[0, 1], sharey=ax_band)

    for spin_branches in bs_data["energy"].values():
        for ib, branch_bands in enumerate(spin_branches):
            dist = bs_data["distances"][ib]
            for band in branch_bands:
                ax_band.plot(dist, band, color="black", lw=1.0, alpha=0.9)

    ticks = bs_data.get("ticks", {})
    if "distance" in ticks and "label" in ticks:
        ax_band.set_xticks(ticks["distance"])
        labels = [("" if lbl is None else str(lbl).replace("$\\Gamma$", "G")) for lbl in ticks["label"]]
        ax_band.set_xticklabels(labels)
        for x in ticks["distance"]:
            ax_band.axvline(x, color="0.85", lw=0.8, zorder=0)

    ax_band.axhline(0.0, color="tab:red", ls="--", lw=1.0)
    ax_band.set_ylabel("E - Ef (eV)")
    ax_band.set_xlabel("k-path")
    ax_band.set_title("Band Structure")
    ax_band.grid(axis="y", alpha=0.2)

    ax_dos.plot(total_dos, energies, color="black", lw=0.8, label="Total DOS")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    for i, (el, dos_obj) in enumerate(elem_dos.items()):
        el_dos = _sum_spin_density(dos_obj.densities)
        ax_dos.plot(el_dos, energies, lw=0.6, color=colors[i % len(colors)], label=f"{el}")

    ax_dos.axhline(0.0, color="tab:red", ls="--", lw=1.0)
    if show_hs_orbital_marker:
        hs_peak = _find_h_s_peak_energy(cdos, e_min, e_max)
        _add_hs_marker_horizontal(ax_dos, hs_peak)
    ax_dos.set_xlabel("DOS")
    ax_dos.set_title("Total DOS + PDOS")
    ax_dos.grid(axis="y", alpha=0.2)
    if e_min is not None and e_max is not None:
        ax_band.set_ylim(e_min, e_max)

    mask = (energies >= (e_min if e_min is not None else energies.min())) & (
        energies <= (e_max if e_max is not None else energies.max())
    )
    x_max = float(np.max(total_dos[mask])) if np.any(mask) else float(np.max(total_dos))
    if x_max > 0:
        ax_dos.set_xlim(0, 1.15 * x_max)
    if show_labels:
        _place_legend_outside_top_right(ax_dos)
        fig.subplots_adjust(right=LEGEND_RIGHT_MARGIN)
    else:
        legend = ax_dos.get_legend()
        if legend is not None:
            legend.remove()

    plt.setp(ax_dos.get_yticklabels(), visible=False)
    _save_figure(fig, out_file, COMBINED_PANEL_HEIGHT_IN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DOS/PDOS/Band from workflow output folders")
    parser.add_argument("--input-db", type=Path, required=True, help="Input DB (.db or .json)")
    parser.add_argument("--calc-name", type=str, default="DOS_calc", help="Calculation name")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional IDs (space/comma separated)")
    parser.add_argument("--output-root", type=Path, default=Path("data/calculations"))
    parser.add_argument("--dos-step", type=str, default="03_dos", help="Folder name for DOS step")
    parser.add_argument("--band-step", type=str, default="04_band", help="Folder name for band step")
    parser.add_argument("--plots-subdir", type=str, default="plots", help="Output subfolder inside each id dir")
    parser.add_argument("--emin", type=float, default=-6.0, help="Lower energy bound (eV)")
    parser.add_argument("--emax", type=float, default=6.0, help="Upper energy bound (eV)")
    parser.add_argument(
        "--show-labels",
        dest="show_labels",
        action="store_true",
        default=True,
        help="Show plot legends/labels (default: enabled)",
    )
    parser.add_argument(
        "--hide-labels",
        dest="show_labels",
        action="store_false",
        help="Hide plot legends/labels",
    )
    parser.add_argument(
        "--show-hs-orbital-marker",
        action="store_true",
        help="Draw H-s orbital peak marker line in DOS plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_db.exists():
        raise FileNotFoundError(f"Input DB not found: {args.input_db}")

    calc_root = (args.output_root / f"{sanitize_name(args.input_db.stem)}_{sanitize_name(args.calc_name)}").resolve()
    if not calc_root.exists():
        raise FileNotFoundError(f"Calculation root not found: {calc_root}")

    ids = parse_ids(args.ids)
    db_records = load_records_from_db(args.input_db)
    selected_records = db_records if not ids else [rec for rec in db_records if rec["id"] in ids]
    if not selected_records:
        raise ValueError("No IDs selected.")

    print(f"Calculation root: {calc_root}")
    print(f"Selected IDs: {', '.join(rec['id'] for rec in selected_records)}")

    for rec in selected_records:
        rid = rec["id"]
        material_hint = rec.get("material", "")
        material_dir = find_material_dir(calc_root, rid, material_hint=material_hint)
        if material_dir is None:
            print(f"[SKIP] ID={rid}: directory not found under {calc_root}")
            continue

        dos_dir = material_dir / args.dos_step
        band_dir = material_dir / args.band_step
        out_dir = material_dir / args.plots_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ID={rid}] material_dir={material_dir.name}")

        dos_vr = dos_dir / "vasprun.xml"
        if dos_vr.exists():
            try:
                save_total_dos_plot(
                    dos_vr,
                    out_dir / "dos_total.png",
                    args.emin,
                    args.emax,
                    args.show_labels,
                    args.show_hs_orbital_marker,
                )
                print("  [OK] dos_total.png")
            except Exception as exc:
                print(f"  [ERR] DOS plot failed: {type(exc).__name__}: {exc}")

            try:
                save_element_pdos_plot(
                    dos_vr,
                    out_dir / "pdos_element.png",
                    args.emin,
                    args.emax,
                    args.show_labels,
                    args.show_hs_orbital_marker,
                )
                print("  [OK] pdos_element.png")
            except Exception as exc:
                print(f"  [ERR] PDOS plot failed: {type(exc).__name__}: {exc}")

            try:
                save_total_orbital_dos_plot(
                    dos_vr,
                    out_dir / "dos_total_spdf.png",
                    args.emin,
                    args.emax,
                    args.show_labels,
                    args.show_hs_orbital_marker,
                )
                print("  [OK] dos_total_spdf.png")
            except Exception as exc:
                print(f"  [ERR] Total orbital DOS plot failed: {type(exc).__name__}: {exc}")

            try:
                save_element_orbital_pdos_plot(
                    dos_vr,
                    out_dir / "pdos_element_spdf.png",
                    args.emin,
                    args.emax,
                    args.show_labels,
                    args.show_hs_orbital_marker,
                )
                print("  [OK] pdos_element_spdf.png")
            except Exception as exc:
                print(f"  [ERR] Element orbital PDOS plot failed: {type(exc).__name__}: {exc}")

            try:
                save_n_w_orbital_pdos_plot(
                    dos_vr,
                    out_dir / "pdos_n_sp_w_spdf.png",
                    args.emin,
                    args.emax,
                    args.show_labels,
                    args.show_hs_orbital_marker,
                )
                print("  [OK] pdos_n_sp_w_spdf.png")
            except Exception as exc:
                print(f"  [ERR] N/W orbital PDOS plot failed: {type(exc).__name__}: {exc}")
        else:
            print(f"  [SKIP] DOS vasprun missing: {dos_vr}")

        band_vr = band_dir / "vasprun.xml"
        band_kpoints = band_dir / "KPOINTS"
        if band_vr.exists() and band_kpoints.exists():
            try:
                save_band_plot(band_vr, band_kpoints, out_dir / "band_structure.png", args.emin, args.emax)
                print("  [OK] band_structure.png")
            except Exception as exc:
                print(f"  [ERR] Band plot failed: {type(exc).__name__}: {exc}")
            try:
                save_projected_band_plot(
                    band_vr,
                    band_kpoints,
                    out_dir / "band_structure_projected.png",
                    args.emin,
                    args.emax,
                )
                print("  [OK] band_structure_projected.png")
            except Exception as exc:
                print(f"  [ERR] Projected band plot failed: {type(exc).__name__}: {exc}")
            if dos_vr.exists():
                try:
                    save_band_dos_combined_plot(
                        band_vr,
                        dos_vr,
                        band_kpoints,
                        out_dir / "band_dos_combined.png",
                        args.emin,
                        args.emax,
                        args.show_labels,
                        args.show_hs_orbital_marker,
                    )
                    print("  [OK] band_dos_combined.png")
                except Exception as exc:
                    print(f"  [ERR] Combined band+DOS plot failed: {type(exc).__name__}: {exc}")
            else:
                print(f"  [SKIP] Combined band+DOS: DOS vasprun missing: {dos_vr}")
        else:
            print(f"  [SKIP] Band files missing: {band_vr} or {band_kpoints}")


if __name__ == "__main__":
    main()
