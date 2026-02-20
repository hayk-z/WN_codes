import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


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


def plot_volcano(header, data, plot_path: str = "Reports_gen/volcano_plot.png"):
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

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(gibbs_clean, i0_clean, alpha=0.6, s=50, edgecolors="black", zorder=3)

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

    for x, y, label in zip(gibbs_clean, i0_clean, labels):
        ax.annotate(str(label), (x, y), fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(r"$\Delta G_{H^*}$ (eV)", fontsize=12)
    ax.set_ylabel(r"$\log(i_0/(A\,cm^{-2}))$", fontsize=12)
    ax.set_title("Volcano Plot for 2D W-N materials", fontsize=14, fontweight="bold")
    ax.set_ylim(-35, 0)
    ax.set_xlim(-2.2, 2.2)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
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


if __name__ == "__main__":
    try:
        header, data = read_adsorption_csv()
        header, data = calculate_gibbs_free_energy(header, data)
        header, data = calculate_exchange_current(header, data)

        save_csv(header, data, "Reports_gen/Adsorption_gibbs_with_i0.csv")
        plot_volcano(header, data, "Reports_gen/volcano_plot.png")

        print("Saved augmented CSV to: Reports_gen/Adsorption_gibbs_with_i0.csv")
        print("Saved volcano plot to: Reports_gen/volcano_plot.png")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
