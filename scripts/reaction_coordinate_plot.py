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


def plot_reaction_coordinate(
    header,
    data,
    plot_path: str = "Reports_gen/reaction_coordinate_plot.png",
    middle_points: int = 2,
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

    fig, ax = plt.subplots(figsize=(11, 7))
    middle_points = max(2, int(middle_points))
    x = np.linspace(0.0, 2.0, middle_points + 2)
    mid_start = 1
    mid_end = mid_start + middle_points
    x_mid_center = float(np.mean(x[mid_start:mid_end]))
    x_labels = [r"$\mathrm{H^+ + e^-}$", r"$\mathrm{H^*}$", r"$\mathrm{\frac{1}{2}H_2}$"]

    if gibbs.size > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, gibbs.size))
        for c, dg, label in zip(colors, gibbs, labels):
            y = np.zeros_like(x)
            y[mid_start:mid_end] = float(dg)
            ax.plot(x, y, color=c, linewidth=1.8, alpha=0.9)
            ax.scatter([x_mid_center], [float(dg)], color=c, s=20, zorder=3)
            ax.annotate(str(label), (x_mid_center, float(dg)), fontsize=12, xytext=(5, 2), textcoords="offset points")

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks([x[0], x_mid_center, x[-1]])
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel("Reaction coordinate", fontsize=12)
    ax.set_ylabel(r"$\Delta G$ (eV)", fontsize=12)
    ax.set_title("HER Gibbs Free Energy Diagram", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    try:
        header, data = read_adsorption_csv()
        header, data = calculate_gibbs_free_energy(header, data)
        plot_reaction_coordinate(header, data, "Reports_gen/reaction_coordinate_plot.png")
        print("Saved reaction coordinate plot to: Reports_gen/reaction_coordinate_plot.png")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
