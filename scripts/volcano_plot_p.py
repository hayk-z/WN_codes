import pandas as pd


def load_adsorption(path: str = "Reports_gen/Adsorption_gibbs.csv") -> pd.DataFrame:
    """Load adsorption Gibbs CSV into a pandas DataFrame and return it as `fvariable`."""
    fvariable = pd.read_csv(path)

    # Compute a new column equal to 2 * (last column)
    if fvariable.shape[1] > 0:
        last_col = fvariable.columns[-1]
        new_col_name = f"two_{last_col}"
        fvariable[new_col_name] = 2 * pd.to_numeric(fvariable[last_col], errors="coerce")

    return fvariable


if __name__ == "__main__":
    fvariable = load_adsorption()
    print("Loaded DataFrame with shape:", fvariable.shape)
    print("Columns:", ", ".join(fvariable.columns))

print(fvariable)