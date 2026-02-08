import pandas as pd
import numpy as np
from pathlib import Path


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /app -> project root
RAW = PROJECT_ROOT / "data" / "raw" / "Sample_Superstore_dataset_for_Python_Project.csv"
OUT = PROJECT_ROOT / "data" / "processed" / "superstore_processed.csv"


# ----------------------------
# Robust CSV Reader
# ----------------------------
def read_superstore_csv(path: Path) -> pd.DataFrame:
    """
    Robust reader for Excel-exported CSVs:
    - Tries common Windows encodings
    - Tries common delimiters: ';', ',', tab
    - Uses python engine to tolerate messy quoting
    - Validates that expected columns exist
    """
    encodings = ["utf-8-sig", "cp1252", "latin1"]
    seps = [";", ",", "\t"]

    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",        # more tolerant than C engine
                    on_bad_lines="skip",    # skip broken lines instead of crashing
                )

                # Clean column names (strip spaces / NBSP)
                df.columns = (
                    df.columns.astype(str)
                    .str.replace("\xa0", " ", regex=False)
                    .str.strip()
                )

                # Validate: must contain these columns (case-insensitive match)
                cols_lower = {c.lower(): c for c in df.columns}
                required = ["sales", "profit", "order date"]
                if all(r in cols_lower for r in required):
                    return df

            except Exception as e:
                last_err = e

    raise RuntimeError(
        f"Could not parse CSV reliably. Last error: {last_err}\n"
        f"File: {path}"
    )


# ----------------------------
# Numeric cleaner
# ----------------------------
def to_num(series: pd.Series) -> pd.Series:
    """
    Convert messy numeric strings to numbers.
    Handles:
    - NBSP
    - thousands separators (commas)
    - parentheses negatives
    - stray currency/letters
    """
    s = series.astype(str).str.strip()

    # remove NBSP and regular spaces
    s = s.str.replace("\xa0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)

    # parentheses negatives: (123.45) -> -123.45
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    # remove thousands commas: 9,575.775 -> 9575.775
    s = s.str.replace(",", "", regex=False)

    # keep only digits, minus, dot
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)

    return pd.to_numeric(s, errors="coerce")


# ----------------------------
# Main rebuild
# ----------------------------
def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW}")

    df = read_superstore_csv(RAW)

    # Replace NBSP across entire df (text columns too)
    df = df.replace("\xa0", " ", regex=True)

    # Standardize key column names if needed (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}

    # Map expected names -> actual names in file
    def col(name: str) -> str:
        return cols_lower.get(name.lower(), name)

    sales_col = col("Sales")
    profit_col = col("Profit")
    disc_col = col("Discount")
    qty_col = col("Quantity")
    order_date_col = col("Order Date")
    ship_date_col = col("Ship Date")

    # Numeric conversions
    for c in [sales_col, profit_col, disc_col, qty_col]:
        if c in df.columns:
            df[c] = to_num(df[c])
                # ------------------------------------
    # Fix systematic Profit scaling issue
    # ------------------------------------
    PROFIT_SCALE = 4.9

    if profit_col in df.columns:
        df["Profit_raw"] = df[profit_col]  # keep original for audit
        df[profit_col] = df[profit_col] / PROFIT_SCALE



    # Date conversions
    if order_date_col in df.columns:
        df[order_date_col] = pd.to_datetime(df[order_date_col], errors="coerce")
    if ship_date_col in df.columns:
        df[ship_date_col] = pd.to_datetime(df[ship_date_col], errors="coerce")

    # Drop unusable rows
    must_have = [order_date_col, sales_col, profit_col]
    df = df.dropna(subset=[c for c in must_have if c in df.columns])

    # ----------------------------
    # Validation summary
    # ----------------------------
    total_sales = df[sales_col].sum()
    total_profit = df[profit_col].sum()
    margin = (total_profit / total_sales * 100) if total_sales else np.nan
    suspicious = int(((df[profit_col] > df[sales_col]) & (df[sales_col] > 0)).sum())

    print("\n=== REBUILD SUMMARY ===")
    print("Rows:", len(df))
    print("Total Sales:", float(total_sales))
    print("Total Profit:", float(total_profit))
    print("Profit Margin %:", float(margin))
    print("Rows where Profit > Sales (Sales>0):", suspicious)
    print("Detected columns:", list(df.columns))

    # Write output
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
