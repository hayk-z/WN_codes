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
PT_GIBBS = 0.08
PT_LOG_I0 = -2.63
PT_LABEL = "Pt"


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


def calculate_exchange_current(header, data):
    """
    Append exchange current i0 using row-wise surface area from file.
    k0 is computed per row as: k0 = 200 / surface_area_cm2
    """
    kb = 8.617333262145e-5  # eV/K
    t = 300.0               # K
    e = 1.602176634e-19     # C

    area_idx = header.index("Surface Area (cm^2)")
    gibbs_idx = header.index("Gibbs free energy (eV)")
    nrows = data.shape[0]

    i0_vals = np.full(nrows, np.nan, dtype=float)
    for i in range(nrows):
        area_text = str(data[i, area_idx]).strip()
        gibbs_text = str(data[i, gibbs_idx]).strip()
        try:
            area = np.nan if (area_text == "" or area_text.upper() == "N/A") else float(area_text)
            g = np.nan if (gibbs_text == "" or gibbs_text.upper() == "N/A") else float(gibbs_text)
        except ValueError:
            i0_vals[i] = np.nan
            continue

        if np.isnan(area) or np.isnan(g) or area <= 0:
            i0_vals[i] = np.nan
            continue

        k0 = 200.0 / area
        if g < 0:
            i0_vals[i] = np.log10(k0 * e / (1.0 + np.exp(-g / (kb * t))))
        else:
            i0_vals[i] = np.log10(k0 * e / (1.0 + np.exp(g / (kb * t))))

    out = np.empty((nrows, data.shape[1] + 1), dtype=object)
    out[:, :-1] = data
    out[:, -1] = i0_vals
    return header + ["i0"], out


def plot_volcano(
    header,
    data,
    plot_path: str = "Reports_gen/volcano_plot.png",
    show_labels: bool = True,
    show_legend: bool = False,
):
    """Plot i0 vs Gibbs free energy and save volcano plot."""
    gibbs_idx = header.index("Gibbs free energy (eV)")
    i0_idx = header.index("i0")
    if "Material Composition" in header:
        label_idx = header.index("Material Composition")
    elif "Composition" in header:
        label_idx = header.index("Composition")
    else:
        label_idx = 0
    gibbs = data[:, gibbs_idx].astype(float)
    i0 = data[:, i0_idx].astype(float)
    valid_mask = ~(np.isnan(gibbs) | np.isnan(i0))

    gibbs_clean = gibbs[valid_mask]
    i0_clean = i0[valid_mask]
    labels = data[valid_mask, label_idx]
    gibbs_clean = np.append(gibbs_clean, PT_GIBBS)
    i0_clean = np.append(i0_clean, PT_LOG_I0)
    labels = np.append(labels, PT_LABEL)

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH_IN, ONE_COLUMN_HEIGHT_IN))
    colors = plt.cm.tab20(np.linspace(0, 1, max(gibbs_clean.size, 1)))
    for idx, (c, x, y, label) in enumerate(zip(colors, gibbs_clean, i0_clean, labels)):
        scatter_label = str(label) if show_legend else None
        ax.scatter([x], [y], alpha=0.8, s=50, edgecolors="black", color=c, zorder=3, label=scatter_label)
        if show_labels:
            dy = 7 if idx % 2 == 0 else -8
            va = "bottom" if dy > 0 else "top"
            ax.annotate(
                str(label),
                xy=(x, y),
                xytext=(4, dy),
                textcoords="offset points",
                fontsize=LABEL_FONT_SIZE,
                color="black",
                va=va,
                ha="left",
            )

    # Symmetric volcano branches: left increases to x=0, right decreases from x=0
    if gibbs_clean.size > 1:
        left_mask = gibbs_clean < 0
        right_mask = gibbs_clean > 0

        slope_mag = None
        y0 = None

        if np.sum(left_mask) > 1:
            left_slope, left_intercept = np.polyfit(gibbs_clean[left_mask], i0_clean[left_mask], 1)
            slope_mag = abs(left_slope)
            y0 = left_intercept
        elif np.sum(right_mask) > 1:
            right_slope, right_intercept = np.polyfit(gibbs_clean[right_mask], i0_clean[right_mask], 1)
            slope_mag = abs(right_slope)
            y0 = right_intercept
        else:
            x_abs = np.abs(gibbs_clean)
            abs_slope, abs_intercept = np.polyfit(x_abs, i0_clean, 1)
            slope_mag = abs(abs_slope)
            y0 = abs_intercept

        if slope_mag is not None and y0 is not None:
            left_span = None
            if gibbs_clean.min() < 0:
                x_left = np.linspace(gibbs_clean.min(), 0.0, 100)
                y_left = y0 + slope_mag * x_left
                ax.plot(x_left, y_left, "k-", linewidth=2.5, zorder=2)
                left_span = abs(gibbs_clean.min())
            if gibbs_clean.max() > 0:
                right_span = left_span if left_span is not None else gibbs_clean.max()
                x_right = np.linspace(0.0, right_span, 100)
                y_right = y0 - slope_mag * x_right
                ax.plot(x_right, y_right, "k-", linewidth=2.5, zorder=2)

    ax.set_xlabel(r"$\Delta G_{H^*}$ (eV)", fontsize=7.5)
    ax.set_ylabel(r"$\log(i_0/(A\,cm^{-2}))$", fontsize=7.5)
    ax.set_title("Volcano Plot for 2D W-N materials", fontsize=8, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.set_ylim(-35, 0)
    ax.set_xlim(-2.2, 2.2)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    if show_legend and gibbs_clean.size > 0:
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


def save_csv(header, data, out_path: str = "Reports_gen/Adsorption_gibbs_with_i0.csv"):
    """Save header + 2D object array to CSV."""
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            out = []
            for value in row:
                if isinstance(value, float):
                    out.append("" if (np.isnan(value) or np.isinf(value)) else f"{value:.6g}")
                else:
                    out.append(value)
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate volcano plot for W-N materials")
    parser.add_argument(
        "--show-labels",
        dest="show_labels",
        action="store_true",
        default=True,
        help="Show inline text labels on points (default: on)",
    )
    parser.add_argument(
        "--hide-labels",
        dest="show_labels",
        action="store_false",
        help="Hide inline text labels on points",
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
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        header, data = read_adsorption_csv()
        header, data = calculate_gibbs_free_energy(header, data)
        header, data = calculate_exchange_current(header, data)
        save_csv(header, data, "Reports_gen/Adsorption_gibbs_with_i0.csv")
        plot_volcano(
            header,
            data,
            "Reports_gen/volcano_plot.png",
            show_labels=args.show_labels,
            show_legend=args.show_legend,
        )

        print("Saved augmented CSV to: Reports_gen/Adsorption_gibbs_with_i0.csv")
        print("Saved volcano plot to: Reports_gen/volcano_plot.png")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
