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
        
        # Convert to numpy array with object dtype
        data = np.array(rows, dtype=object)
        
        # Convert last two columns to float, replacing N/A with nan
        for col_idx in [-2, -1]:
            float_col = []
            for val in data[:, col_idx]:
                try:
                    if val.upper() == 'N/A' or val == '':
                        float_col.append(np.nan)
                    else:
                        float_col.append(float(val))
                except (ValueError, AttributeError):
                    float_col.append(np.nan)
            data[:, col_idx] = np.array(float_col, dtype=float)
        
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


