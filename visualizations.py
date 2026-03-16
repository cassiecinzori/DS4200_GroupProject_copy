"""
visualizations.py - Generate all visualizations for DS4200 project
Boston 311 Service Request Analysis: 2015 vs 2025
"""

from api311 import Year
from signatures import SignatureAnalyzer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List, Dict
from matplotlib.figure import Figure

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["font.family"] = "sans-serif"

# == Helpers ==


def save_figure(fig: Figure, filename: str):
    """Save figure to file"""
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig(os.path.join("figures", filename), dpi=300, bbox_inches="tight")
    print(f"Saved figure: {filename}")


def clean_request_type_name(name: str) -> str:
    """Make request type names more readable"""
    replacements = {
        "Missed Trash/Recycling/Yard Waste/Bulk Item": "Missed Trash/Recycling",
        "Request for Snow Plowing": "Snow Plowing",
        "Request for Pothole Repair": "Pothole Repair",
        "Street Light Outages": "Street Lights",
        "Pothole Repair (Internal)": "Pothole (Internal)",
        "Poor Conditions of Property": "Poor Property Conditions",
        "Improper Storage of Trash (Barrels)": "Improper Trash Storage",
        "Parks Lighting/Electrical Issues": "Parks Lighting",
    }
    return replacements.get(name, name)


# == Visualization Functions ==


def create_monthly_heatmap(
    year15: Year, year25: Year, top_n: int = 10, save: bool = True
) -> Figure:
    """
    Monthly heatmap visualization 1:
    Takes 2 Year objects (2015 and 2025) and generates monthly summaries by neighborhood and request type.
    The flow for each monthly series is:
    Break data into 2 groups:
    Cold Months (11, 12, 1, 2, 3, 4)
    Warm Months (5, 6, 7, 8, 9, 10)
        For each group, calculate the relative frequency of each request type (number of requests of that type
        in the month divided by total requests in that month).
        Order the data by the frequencies separately within each season.
        Get the top N most common request types across both years for that season.
        Create a heatmap with request types on the y-axis and months on the x-axis, where the color intensity
        represents the relative frequency of that request type in that month.
        Vertically stack Warm and Cold month heatmaps for each year.
        Display observation counts (n) in each subplot title.
    Place the 2015 and 2025 heatmap groups side by side for comparison.
    """
    print("Creating monthly heatmap...")

    COLD_MONTHS = [11, 12, 1, 2, 3, 4]
    WARM_MONTHS = [5, 6, 7, 8, 9, 10]
    MONTH_NAMES = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    # Get monthly summary data (type x month DataFrames)
    summary_15 = year15.summarize("neighborhood", "type")
    summary_25 = year25.summarize("neighborhood", "type")
    monthly_15 = summary_15["monthly"]
    monthly_25 = summary_25["monthly"]

    # Map column names to month numbers for slicing
    month_name_to_num = {v: k for k, v in MONTH_NAMES.items()}
    full_month_to_num = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    month_name_to_num.update(full_month_to_num)

    def compute_relative_freq(monthly_df):
        """Convert raw counts to relative frequency per month (column-wise)."""
        col_totals = monthly_df.sum(axis=0)
        col_totals = col_totals.replace(0, 1)
        return monthly_df.div(col_totals, axis=1)

    def split_by_season(monthly_df, month_lookup):
        """Split a type x month DataFrame into cold and warm DataFrames."""
        cold_cols = [
            c for c in monthly_df.columns if month_lookup.get(c) in COLD_MONTHS
        ]
        warm_cols = [
            c for c in monthly_df.columns if month_lookup.get(c) in WARM_MONTHS
        ]
        cold_order = [MONTH_NAMES[m] for m in COLD_MONTHS]
        warm_order = [MONTH_NAMES[m] for m in WARM_MONTHS]
        cold_cols_ordered = [
            c
            for co in cold_order
            for c in cold_cols
            if month_lookup.get(c) == month_name_to_num.get(co, co)
        ]
        warm_cols_ordered = [
            c
            for wo in warm_order
            for c in warm_cols
            if month_lookup.get(c) == month_name_to_num.get(wo, wo)
        ]
        return monthly_df[cold_cols_ordered], monthly_df[warm_cols_ordered]

    # Build column lookup from actual DataFrame columns
    col_lookup_15 = {c: month_name_to_num.get(c) for c in monthly_15.columns}
    col_lookup_25 = {c: month_name_to_num.get(c) for c in monthly_25.columns}

    # Split RAW counts by season first (before RF) for observation counts
    cold_15_raw, warm_15_raw = split_by_season(monthly_15, col_lookup_15)
    cold_25_raw, warm_25_raw = split_by_season(monthly_25, col_lookup_25)

    n_warm_15 = int(warm_15_raw.sum().sum())
    n_warm_25 = int(warm_25_raw.sum().sum())
    n_cold_15 = int(cold_15_raw.sum().sum())
    n_cold_25 = int(cold_25_raw.sum().sum())

    # Compute relative frequencies
    rel_15 = compute_relative_freq(monthly_15)
    rel_25 = compute_relative_freq(monthly_25)

    # Split RF by season
    cold_15, warm_15 = split_by_season(rel_15, col_lookup_15)
    cold_25, warm_25 = split_by_season(rel_25, col_lookup_25)

    # Rank within each season separately by mean RF across both years
    warm_mean = warm_15.mean(axis=1).add(warm_25.mean(axis=1), fill_value=0)
    warm_top = warm_mean.nlargest(top_n).index.tolist()

    cold_mean = cold_15.mean(axis=1).add(cold_25.mean(axis=1), fill_value=0)
    cold_top = cold_mean.nlargest(top_n).index.tolist()

    # Filter each season to its own top types
    warm_15 = warm_15.reindex(warm_top).fillna(0)
    warm_25 = warm_25.reindex(warm_top).fillna(0)
    cold_15 = cold_15.reindex(cold_top).fillna(0)
    cold_25 = cold_25.reindex(cold_top).fillna(0)

    # Clean labels
    for df in [cold_15, warm_15, cold_25, warm_25]:
        df.index = [clean_request_type_name(x) for x in df.index]

    # Create figure: 2 columns (2015, 2025) x 2 rows (warm on top, cold on bottom)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12),
        gridspec_kw={"hspace": 0.35, "wspace": 0.3},
    )

    vmax = max(
        warm_15.max().max(),
        cold_15.max().max(),
        warm_25.max().max(),
        cold_25.max().max(),
    )

    heatmap_kws = dict(
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Relative Frequency"},
    )

    # Row 0: Warm months
    sns.heatmap(warm_15, ax=axes[0, 0], **heatmap_kws)
    axes[0, 0].set_title(
        f"2015 — Warm Months (May–Oct)\nn = {n_warm_15:,}",
        fontsize=13,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Request Type")

    sns.heatmap(warm_25, ax=axes[0, 1], **heatmap_kws)
    axes[0, 1].set_title(
        f"2025 — Warm Months (May–Oct)\nn = {n_warm_25:,}",
        fontsize=13,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Month")
    axes[0, 1].set_ylabel("")

    # Row 1: Cold months
    sns.heatmap(cold_15, ax=axes[1, 0], **heatmap_kws)
    axes[1, 0].set_title(
        f"2015 — Cold Months (Nov–Apr)\nn = {n_cold_15:,}",
        fontsize=13,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Request Type")

    sns.heatmap(cold_25, ax=axes[1, 1], **heatmap_kws)
    axes[1, 1].set_title(
        f"2025 — Cold Months (Nov–Apr)\nn = {n_cold_25:,}",
        fontsize=13,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("")

    # Rotate tick labels
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)

    plt.suptitle(
        "Seasonal Patterns in Boston 311 Requests (Relative Frequency)",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save:
        save_figure(fig, "monthly_heatmap.png")
    plt.show()
    return fig


def create_composition_bars(
    year15: Year, year25: Year, top_n: int, save: bool = True
) -> Figure:
    """
    Neighborhood composition bar charts 2:


    """
    print("Creating neighborhood composition charts...")

    # Clean blank neighborhoods
    data15 = year15.data[
        year15.data["neighborhood"].notna()
        & (year15.data["neighborhood"].str.strip() != "")
    ]

    data25 = year25.data[
        year25.data["neighborhood"].notna()
        & (year25.data["neighborhood"].str.strip() != "")
    ]

    if top_n > len(data15["neighborhood"].value_counts()):
        top_n = len(data15["neighborhood"].value_counts())
        print(
            f"Warning: top_n exceeds number of neighborhoods. Using top_n={top_n} instead."
        )

    top_neighborhoods = data15["neighborhood"].value_counts().head(top_n).index.tolist()

    # Create 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    colors = ["#4C72B0", "#DD8452"]  # Blue for 2015, Orange for 2025
    bar_height = 0.35

    # Create horizontal bar chart for each neighborhood
    for idx, hood in enumerate(top_neighborhoods):

        hood_15 = data15[data15["neighborhood"] == hood]
        hood_25 = data25[data25["neighborhood"] == hood]

        # Get top 5 combined request types across both years
        combined_counts = (
            pd.concat([hood_15["type"], hood_25["type"]]).value_counts().head(5)
        )

        top_types = combined_counts.index.tolist()

        # Count occurrences for each year
        counts_15 = [(hood_15["type"] == t).sum() for t in top_types]
        counts_25 = [(hood_25["type"] == t).sum() for t in top_types]

        y_positions = range(len(top_types))

        ax = axes[idx]

        # Plot 2015 bars
        ax.barh(
            [y - bar_height / 2 for y in y_positions],
            counts_15,
            height=bar_height,
            color=colors[0],
            edgecolor="black",
            linewidth=0.6,
            label="2015" if idx == 0 else "",
        )

        # Plot 2025 bars
        ax.barh(
            [y + bar_height / 2 for y in y_positions],
            counts_25,
            height=bar_height,
            color=colors[1],
            edgecolor="black",
            linewidth=0.6,
            label="2025" if idx == 0 else "",
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([clean_request_type_name(t) for t in top_types], fontsize=9)

        ax.set_title(hood, fontsize=13, fontweight="bold", pad=8)

        ax.set_xlabel("Requests", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Remove unused subplots if any
    for i in range(len(top_neighborhoods), 6):
        fig.delaxes(axes[i])

    # Add shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()

    plt.suptitle(
        "How Neighborhood Request Patterns Changed (2015 → 2025)",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        fontsize=11,
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        save_figure(fig, "composition_bars.png")
    plt.show()

    return fig


"""
Nieghborhood signature drift Quanification analysis 3:


"""


def create_signature_drift(
    year15: Year, year25: Year, save: bool = True
) -> Tuple[Figure, pd.DataFrame]:
    """Signature drift analysis"""
    print("Creating signature drift analysis...")

    sa = SignatureAnalyzer(area_col="neighborhood", type_col="type")

    # Build signature vectors for each neighborhood
    sigs_15 = sa.build_signatures(year15.data, min_requests=30)
    sigs_25 = sa.build_signatures(year25.data, min_requests=30)

    drift = sa.compare_signatures(sigs_15, sigs_25)
    drift = drift.sort_values("distance", ascending=False).reset_index(drop=True)

    # Drop the last (lowest-drift) neighborhood
    drift = drift.iloc[:-1]

    # Get average request counts for context
    counts_15 = year15.data["neighborhood"].value_counts()
    counts_25 = year25.data["neighborhood"].value_counts()

    drift["avg_requests"] = drift["area"].map(
        lambda x: (counts_15.get(x, 0) + counts_25.get(x, 0)) / 2
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(22, 10))

    # Color bars using a single Blues colormap
    colors = plt.cm.get_cmap("Blues")(
        0.3 + 0.7 * (drift["distance"] / drift["distance"].max())
    )

    ax.bar(
        range(len(drift)),
        drift["distance"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    # Prevent last label from being clipped
    ax.set_xlim(-0.5, len(drift) - 0.3)

    # Add median reference line
    median = drift["distance"].median()
    ax.axhline(y=median, color="red", linestyle="--", linewidth=2.5, alpha=0.7)

    # Add median text annotation
    ax.text(
        len(drift) - 0.3,
        median + 0.02,
        f"Median: {median:.3f}",
        fontsize=11,
        ha="right",
        va="bottom",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="red",
            linewidth=2,
            alpha=0.9,
        ),
    )

    # Add neighborhood labels to all bars
    ax.set_xticks(range(len(drift)))
    ax.set_xticklabels(
        drift["area"],
        rotation=60,
        ha="right",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlabel("Neighborhoods (ranked by drift)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cosine Distance (2015 → 2025)", fontsize=13, fontweight="bold")
    ax.set_title(
        "311 Signature Drift by Neighborhood", fontsize=18, fontweight="bold", pad=30
    )

    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.subplots_adjust(bottom=0.30, top=0.92)
    plt.tight_layout()
    if save:
        save_figure(fig, "signature_drift.png")
    plt.show()

    print(drift.tail())

    return fig, drift


"""
Cluster distribution comparison 4:
"""


def create_cluster_comparison(
    year15: Year, year25: Year, save: bool = True
) -> Tuple[Figure, pd.Series, pd.Series]:
    """Cluster comparison"""
    print("Creating cluster analysis...")

    sa = SignatureAnalyzer(area_col="neighborhood", type_col="type")

    # Build signatures and apply k-means clustering
    sigs_15 = sa.build_signatures(year15.data, min_requests=30)
    sigs_25 = sa.build_signatures(year25.data, min_requests=30)

    labels_15, _ = sa.cluster(sigs_15, k=4)
    labels_25, _ = sa.cluster(sigs_25, k=4)

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    cluster_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    # 2015 cluster distribution
    cluster_counts_15 = labels_15.value_counts().sort_index()
    ax1.bar(
        cluster_counts_15.index,
        cluster_counts_15.values,
        color=cluster_colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax1.set_title("2015 Neighborhood Clusters", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Cluster Group", fontsize=12)
    ax1.set_ylabel("Number of Neighborhoods", fontsize=12)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"Cluster {i}" for i in range(4)])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # 2025 cluster distribution
    cluster_counts_25 = labels_25.value_counts().sort_index()
    ax2.bar(
        cluster_counts_25.index,
        cluster_counts_25.values,
        color=cluster_colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.set_title("2025 Neighborhood Clusters", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Cluster Group", fontsize=12)
    ax2.set_ylabel("Number of Neighborhoods", fontsize=12)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"Cluster {i}" for i in range(4)])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        "How Neighborhoods Group Together by Request Patterns",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        save_figure(fig, "cluster_comparison.png")
    plt.show()

    return fig, labels_15, labels_25


"""Visualization 5 in interactive_composition.py"""

# == Main Execution ==


def main():
    """Generate all static visualizations"""
    print("Loading data...")
    year15 = Year("data/cleaned2015.csv")
    year25 = Year("data/cleaned2025.csv")
    year15.make_points()
    year25.make_points()

    print(f"Loaded 2015: {len(year15.data):,} records")
    print(f"Loaded 2025: {len(year25.data):,} records")

    # Generate all visualizations
    fig = create_monthly_heatmap(year15, year25, save=True)
    fig = create_composition_bars(year15, year25, save=True)
    fig, drift = create_signature_drift(year15, year25, save=True)
    fig, labels15, labels25 = create_cluster_comparison(year15, year25, save=True)

    print("\nAll visualizations displayed!")


if __name__ == "__main__":
    main()
