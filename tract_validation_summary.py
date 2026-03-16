from __future__ import annotations
import pandas as pd

DATA_PATH = "data/census_tract_ses_2023_with_311.csv"

def load_tract_table(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the merged tract table and add per-capita complaint metrics."""
    df = pd.read_csv(path).copy()
    df["complaints_per_1k_2015"] = df["complaint_count_2015"] / df["population"] * 1000
    df["complaints_per_1k_2025"] = df["complaint_count_2025"] / df["population"] * 1000
    df["complaint_change_per_1k"] = (
        df["complaints_per_1k_2025"] - df["complaints_per_1k_2015"]
    )
    return df

def pearson_corr(df: pd.DataFrame, x: str, y: str) -> float:
    """Return the Pearson correlation between two columns after dropping nulls."""
    subset = df[[x, y]].dropna()
    return subset[x].corr(subset[y])

def main() -> None:
    df = load_tract_table()

    print(f"Loaded {len(df)} census tracts from {DATA_PATH}")
    print()
    print("Pearson correlations")

    comparisons = [
        ("poverty_rate", "complaint_change_2025_minus_2015"),
        ("median_household_income", "complaint_change_2025_minus_2015"),
        ("poverty_rate", "complaint_change_per_1k"),
        ("median_household_income", "complaint_change_per_1k"),
        ("poverty_rate", "complaints_per_1k_2025"),
        ("median_household_income", "complaints_per_1k_2025"),
    ]

    for x, y in comparisons:
        corr = pearson_corr(df, x, y)
        print(f"  {x} vs {y}: {corr:.3f}")

if __name__ == "__main__":
    main()




# Results:
# Loaded 218 census tracts from data/census_tract_ses_2023_with_311.csv
# Pearson correlations
#   poverty_rate vs complaint_change_2025_minus_2015: 0.128
#   median_household_income vs complaint_change_2025_minus_2015: -0.043
#   poverty_rate vs complaint_change_per_1k: 0.098
#   median_household_income vs complaint_change_per_1k: -0.045
#   poverty_rate vs complaints_per_1k_2025: -0.083
#   median_household_income vs complaints_per_1k_2025: 0.290
