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


def create_monthly_heatmap(year15: Year, year25: Year, save: bool = True) -> Figure:
    """Monthly request volume heatmap"""
    print("Creating monthly heatmap...")

    # Get monthly summary data
    summary_15 = year15.summarize("neighborhood", "type")
    summary_25 = year25.summarize("neighborhood", "type")

    monthly_15 = summary_15["monthly"]
    monthly_25 = summary_25["monthly"]

    # Get top 10 most common request types
    top_types = year15.data["type"].value_counts().head(10).index.tolist()

    # Filter to types that appear in both years
    monthly_15_filtered = monthly_15.reindex(top_types).fillna(0)
    monthly_25_filtered = monthly_25.reindex(top_types).fillna(0)

    # Clean up the labels
    monthly_15_filtered.index = [
        clean_request_type_name(x) for x in monthly_15_filtered.index
    ]
    monthly_25_filtered.index = [
        clean_request_type_name(x) for x in monthly_25_filtered.index
    ]

    # Create figure with side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 2015 heatmap
    sns.heatmap(
        monthly_15_filtered,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Requests"},
        ax=ax1,
        linewidths=0.5,
        vmin=0,
    )
    ax1.set_title("2015", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Request Type", fontsize=12)
    ax1.tick_params(axis="x", rotation=45, labelsize=10)
    ax1.tick_params(axis="y", rotation=0, labelsize=10)

    # 2025 heatmap
    sns.heatmap(
        monthly_25_filtered,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Requests"},
        ax=ax2,
        linewidths=0.5,
        vmin=0,
    )
    ax2.set_title("2025", fontsize=16, fontweight="bold", pad=15)
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Request Type", fontsize=12)
    ax2.tick_params(axis="x", rotation=45, labelsize=10)
    ax2.tick_params(axis="y", rotation=0, labelsize=10)

    plt.suptitle(
        "Seasonal Patterns in Boston 311 Requests",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    if save:
        save_figure(fig, "monthly_heatmap.png")
    plt.show()
    return fig


def create_composition_bars(year15: Year, year25: Year, save: bool = True) -> Figure:
    """Neighborhood composition comparison (2015 vs 2025) — Horizontal Bars"""
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

    # Get top 6 neighborhoods by request volume
    top_neighborhoods = data15["neighborhood"].value_counts().head(6).index.tolist()

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
