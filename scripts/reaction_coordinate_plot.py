import csv
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ONE_COLUMN_WIDTH_IN = 90.0 / 25.4  # Elsevier single-column width (90 mm)
ONE_COLUMN_HEIGHT_IN = 3.0
ONE_COLUMN_DPI = 1000
LABEL_FONT_SIZE = 5.0


def read_adsorption_csv(path: str = "Reports_gen/Adsorption_filtered.csv"):
    """Read adsorption CSV into (header, data) where data is a 2D NumPy object array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [[field.strip() for field in row] for row in reader]

    return header, np.array(rows, dtype=object)


def calculate_gibbs_free_energy(header, data, offset: float = 0.365):
    """Append Gibbs free energy column: G = Adsorption Energy + offset."""
    ads_idx = header.index("Adsorption Energy (eV)")
    nrows = data.shape[0]

    adsorption = np.full(nrows, np.nan, dtype=float)
    for i, value in enumerate(data[:, ads_idx]):
        text = str(value).strip()
        try:
            adsorption[i] = np.nan if (text == "" or text.upper() == "N/A") else float(text)
        except ValueError:
            adsorption[i] = np.nan

    gibbs = adsorption + offset
    out = np.empty((nrows, data.shape[1] + 1), dtype=object)
    out[:, :-1] = data
    out[:, -1] = gibbs
    return header + ["Gibbs free energy (eV)"], out


def plot_reaction_coordinate(
    header,
    data,
    plot_path: str = "Reports_gen/reaction_coordinate_plot.png",
    middle_points: int = 2,
    show_labels: bool = True,
    show_legend: bool = False,
):
    """
    Plot HER Gibbs free energy diagram with segmented linear connections.
    Total points per material = 2 + middle_points (default 5 points).
    Middle points are all assigned to H* energy to form a secant-like path.
    """
    gibbs_idx = header.index("Gibbs free energy (eV)")
    if "Material Composition" in header:
        label_idx = header.index("Material Composition")
    elif "Composition" in header:
        label_idx = header.index("Composition")
    else:
        label_idx = 0

    gibbs = data[:, gibbs_idx].astype(float)
    labels = data[:, label_idx]
    valid_mask = ~np.isnan(gibbs)
    gibbs = gibbs[valid_mask]
    labels = labels[valid_mask]

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH_IN, ONE_COLUMN_HEIGHT_IN))
    middle_points = max(2, int(middle_points))
    x = np.linspace(0.0, 2.0, middle_points + 2)
    mid_start = 1
    mid_end = mid_start + middle_points
    x_mid_center = float(np.mean(x[mid_start:mid_end]))
    x_labels = [r"$\mathrm{H^+ + e^-}$", r"$\mathrm{H^*}$", r"$\mathrm{\frac{1}{2}H_2}$"]

    if gibbs.size > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, gibbs.size))
        for idx, (c, dg, label) in enumerate(zip(colors, gibbs, labels)):
            y = np.zeros_like(x)
            y[mid_start:mid_end] = float(dg)
            line_label = str(label) if show_legend else None
            ax.plot(x, y, color=c, linewidth=1.8, alpha=0.9, label=line_label)
            ax.scatter([x_mid_center], [float(dg)], color=c, s=20, zorder=3)
            if show_labels:
                dy = 6 if idx % 2 == 0 else -7
                va = "bottom" if dy > 0 else "top"
                ax.annotate(
                    str(label),
                    xy=(x_mid_center, float(dg)),
                    xytext=(5, dy),
                    textcoords="offset points",
                    fontsize=LABEL_FONT_SIZE,
                    color="black",
                    va=va,
                    ha="left",
                )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks([x[0], x_mid_center, x[-1]])
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_xlabel("Reaction coordinate", fontsize=7.5)
    ax.set_ylabel(r"$\Delta G$ (eV)", fontsize=7.5)
    ax.set_title("HER Gibbs Free Energy Diagram", fontsize=8, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.grid(True, alpha=0.25)
    if show_legend and gibbs.size > 0:
        ax.legend(
            loc="upper right",
            fontsize=5.2,
            frameon=True,
            framealpha=0.85,
            borderpad=0.2,
            handlelength=1.0,
            labelspacing=0.2,
            labelcolor="black",
        )

    fig.tight_layout(pad=0.3)
    fig.savefig(plot_path, dpi=ONE_COLUMN_DPI)
    base, _ = os.path.splitext(plot_path)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HER reaction coordinate plot")
    parser.add_argument(
        "--show-labels",
        dest="show_labels",
        action="store_true",
        default=True,
        help="Show inline text labels on curves (default: on)",
    )
    parser.add_argument(
        "--hide-labels",
        dest="show_labels",
        action="store_false",
        help="Hide inline text labels on curves",
    )
    parser.add_argument(
        "--show-legend",
        dest="show_legend",
        action="store_true",
        default=False,
        help="Show legend box",
    )
    parser.add_argument(
        "--hide-legend",
        dest="show_legend",
        action="store_false",
        help="Hide legend box (default)",
    )
    parser.add_argument(
        "--middle-points",
        type=int,
        default=2,
        help="Number of middle H* points (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        header, data = read_adsorption_csv()
        header, data = calculate_gibbs_free_energy(header, data)
        plot_reaction_coordinate(
            header,
            data,
            "Reports_gen/reaction_coordinate_plot.png",
            middle_points=args.middle_points,
            show_labels=args.show_labels,
            show_legend=args.show_legend,
        )
        print("Saved reaction coordinate plot to: Reports_gen/reaction_coordinate_plot.png")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
