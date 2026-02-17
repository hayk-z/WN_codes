import os
import sys
import numpy as np


def read_adsorption_csv(path: str = "Reports_gen/Adsorption_gibbs.csv"):
    """Read adsorption CSV into a 2D NumPy array.

    Returns (header_list, 2D_array) where 2D_array.shape = (nrows, ncols)
    Raises FileNotFoundError if the file is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Read header (first line)
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        header = [h.strip() for h in first.strip().split(",")]

    # Load data as 2D array, handling quoted values
    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                # Strip quotes from fields
                fields = [field.strip().strip('"') for field in line.strip().split(",")]
                rows.append(fields)
        
        # Convert to numpy array, then to float (will be mixed types but preserve structure)
        data = np.array(rows, dtype=object)
        
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV with NumPy: {e}")

    return header, data


if __name__ == "__main__":
    try:
        header, data = read_adsorption_csv()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Data shape (rows, columns):", data.shape)
    print("Headers:", ", ".join(header))
    print("First row:\n", data[0])


