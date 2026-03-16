"""
visualizations.py - Generate all visualizations for DS4200 project and output to figures/
Boston 311 Service Request Analysis: 2015 vs 2025

This module contains functions to create the following visualizations:
1. Monthly heatmap comparison of request types by season (Matplotlib/Seaborn)
2. Neighborhood request composition over time (Altair)
3. Signature drift analysis (Matplotlib/Seaborn)
4. Cluster comparison analysis (Altair)
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
import altair as alt
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import AgglomerativeClustering

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

""" Visualization 1: Monthly Heatmap Comparison of Request Types by Season"""


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
    return fig


""" Visualization 2: Neighborhood Request Composition Analysis"""


def create_composition_bars(
    year15: Year,
    year25: Year,
    top_n_neighborhoods: int = 10,
    rf_cutoff: float = 0.02,
    save: bool = True,
) -> alt.HConcatChart:
    """
    Stacked bar chart showing request type composition per neighborhood.
    Uses relative frequency. Types below rf_cutoff are dropped entirely.
    2015 and 2025 placed side by side with a shared legend.

    Args:
        year15: Year object for 2015 data
        year25: Year object for 2025 data
        top_n_neighborhoods: number of top neighborhoods by total volume across both years
        rf_cutoff: minimum mean relative frequency to include a request type;
                   types below this threshold are excluded entirely
        save: whether to save the chart as PNG

    Returns:
        alt.HConcatChart
    """

    # Identify top neighborhoods by combined volume across both years
    combined = pd.concat(
        [
            year15.data[["neighborhood"]],
            year25.data[["neighborhood"]],
        ]
    ).dropna()
    top_neighborhoods = (
        combined["neighborhood"].value_counts().head(top_n_neighborhoods).index.tolist()
    )

    def build_rf_frame(year_data: pd.DataFrame, year_label: str) -> pd.DataFrame:
        """Compute relative frequency of each type within each neighborhood."""
        df = year_data[["neighborhood", "type"]].dropna()
        df = df[df["neighborhood"].isin(top_neighborhoods)]

        counts = df.groupby(["neighborhood", "type"]).size().reset_index(name="count")
        totals = counts.groupby("neighborhood")["count"].transform("sum")
        counts["rf"] = counts["count"] / totals
        counts["year"] = year_label
        return counts

    rf_15 = build_rf_frame(year15.data, "2015")
    rf_25 = build_rf_frame(year25.data, "2025")

    # Determine which types pass the RF cutoff (mean RF across all neighborhoods, both years)
    all_rf = pd.concat([rf_15, rf_25])
    mean_rf_by_type = all_rf.groupby("type")["rf"].mean()
    passing_types = mean_rf_by_type[mean_rf_by_type >= rf_cutoff].index.tolist()

    # Drop types below cutoff
    rf_15 = rf_15[rf_15["type"].isin(passing_types)].copy()
    rf_25 = rf_25[rf_25["type"].isin(passing_types)].copy()

    # Sort neighborhoods by total volume descending
    neighborhood_order = (
        pd.concat([rf_15, rf_25])
        .groupby("neighborhood")["count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Sort types by overall mean RF descending
    type_order = (
        pd.concat([rf_15, rf_25])
        .groupby("type")["rf"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Clean type names for display
    for df in [rf_15, rf_25]:
        df["type"] = df["type"].map(clean_request_type_name)
    type_order = [clean_request_type_name(t) for t in type_order]

    def make_chart(df: pd.DataFrame, title: str) -> alt.Chart:
        n_total = int(df["count"].sum())
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "neighborhood:N",
                    sort=neighborhood_order,
                    title="Neighborhood",
                    axis=alt.Axis(labelAngle=-45, labelLimit=140),
                ),
                y=alt.Y(
                    "rf:Q",
                    title="Relative Frequency",
                    stack="zero",
                    axis=alt.Axis(format=".0%"),
                ),
                color=alt.Color(
                    "type:N",
                    sort=type_order,
                    title="Request Type",
                    scale=alt.Scale(scheme="tableau20"),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("neighborhood:N", title="Neighborhood"),
                    alt.Tooltip("type:N", title="Request Type"),
                    alt.Tooltip("rf:Q", title="Relative Freq", format=".2%"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
            .properties(
                title=alt.Title(title, subtitle=f"n = {n_total:,}"),
                width=380,
                height=420,
            )
        )

    chart_15 = make_chart(rf_15, "2015")
    chart_25 = make_chart(rf_25, "2025")

    # Standalone shared legend (rendered as invisible chart with visible legend)
    legend = (
        alt.Chart(pd.concat([rf_15, rf_25]))
        .mark_point(opacity=0)
        .encode(
            color=alt.Color(
                "type:N",
                sort=type_order,
                title="Request Type",
                scale=alt.Scale(scheme="tableau20"),
                legend=alt.Legend(
                    orient="none",
                    legendX=0,
                    legendY=0,
                    direction="vertical",
                    titleFontSize=15,
                    titleFontWeight="bold",
                    labelFontSize=13,
                    symbolSize=250,
                    symbolStrokeWidth=0,
                    labelLimit=300,
                    titlePadding=12,
                    padding=20,
                    columns=1,
                    rowPadding=6,
                ),
            ),
        )
        .properties(width=0, height=0)
    )

    combined_chart = (
        alt.hconcat(chart_15, chart_25, legend, spacing=30)
        .resolve_scale(color="shared")
        .properties(
            title=alt.Title(
                "Request Type Composition by Neighborhood",
                subtitle=f"Types with mean relative frequency ≥ {rf_cutoff:.0%} · Neighborhoods ordered by total request volume",
                anchor="middle",
                fontSize=18,
                subtitleFontSize=12,
                subtitleColor="#666",
            )
        )
    )

    if save:
        combined_chart.save("figures/composition_bars.html")
        print("Saved to composition_bars.html")

    return combined_chart


""" Visualization 3: Signature Drift Analysis"""


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

    print(drift.tail())

    return fig, drift


""" Visualization 4: Cluster Comparison Analysis"""


def create_cluster_comparison(
    year15: Year,
    year25: Year,
    k: int = 3,
    top_n_types: int = 10,
    save: bool = True,
) -> Tuple[alt.VConcatChart, pd.DataFrame]:
    """
    Cluster profile + membership comparison visualization.

    1. Build aligned signatures for both years.
    2. Average each neighborhood's 2015 and 2025 vectors.
    3. Compute pairwise JSD distance matrix, cluster with agglomerative
       clustering (average linkage) to define k cluster identities.
    4. Derive centroids as mean signature per cluster.
    5. For each cluster, show top N request types by relative frequency.
    6. Independently assign each (neighborhood, year) pair to nearest
       centroid via JSD.
    7. Membership panel: stable neighborhoods shown once, shifted
       neighborhoods shown with year annotation in both clusters.

    Args:
        year15: Year object for 2015 data
        year25: Year object for 2025 data
        k: number of clusters
        top_n_types: number of top request types per cluster
        save: whether to save as HTML

    Returns:
        (chart, assignment_df)
    """
    print("Creating cluster analysis...")

    sa = SignatureAnalyzer(area_col="neighborhood", type_col="type")

    # Build and align signatures
    sigs_15 = sa.build_signatures(year15.data, min_requests=30)
    sigs_25 = sa.build_signatures(year25.data, min_requests=30)
    sigs_15_aligned, sigs_25_aligned = sa.align_signatures(sigs_15, sigs_25)

    # Common neighborhoods only — drop blanks/whitespace
    common_areas = sorted(sigs_15_aligned.index.intersection(sigs_25_aligned.index))
    common_areas = [a for a in common_areas if a and a.strip()]
    sigs_15_common = sigs_15_aligned.loc[common_areas]
    sigs_25_common = sigs_25_aligned.loc[common_areas]

    # Truncate long neighborhood names for display
    SHORT_NAMES = {
        "South Boston / South Boston Waterfront": "S. Boston / Waterfront",
        "Fenway / Kenmore / Audubon Circle / Longwood": "Fenway / Kenmore",
        "Downtown / Financial District": "Downtown / Fin. District",
    }

    def shorten_hood(name):
        return SHORT_NAMES.get(name, name)

    # ── Step 1: Average signatures and cluster with JSD + agglomerative ──
    sigs_avg = (sigs_15_common + sigs_25_common) / 2.0
    sigs_avg = sigs_avg.div(sigs_avg.sum(axis=1), axis=0)

    areas = sigs_avg.index.tolist()
    n = len(areas)

    # Pairwise JSD distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(sigs_avg.iloc[i].values, sigs_avg.iloc[j].values)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    agg = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    cluster_labels = agg.fit_predict(dist_matrix)
    labels = pd.Series(cluster_labels, index=areas, name="cluster")

    # Compute centroids as mean signature per cluster
    type_columns = sigs_avg.columns
    centroids = np.array(
        [sigs_avg.loc[labels[labels == cid].index].mean().values for cid in range(k)]
    )

    # ── Step 2: Build cluster profile from centroids ──
    profile_rows = []
    for cid in range(k):
        centroid = pd.Series(centroids[cid], index=type_columns)
        top = centroid.nlargest(top_n_types)
        for req_type, rf in top.items():
            profile_rows.append(
                {
                    "cluster": int(cid),
                    "type": clean_request_type_name(req_type),
                    "rf": rf,
                }
            )
    profile_df = pd.DataFrame(profile_rows)

    # ── Step 3: Assign each (neighborhood, year) via JSD to nearest centroid ──
    def assign_to_cluster(sig_vector, centroids_arr):
        v = np.clip(sig_vector, 0, None)
        if v.sum() > 0:
            v = v / v.sum()
        distances = [jensenshannon(v, c) for c in centroids_arr]
        return int(np.argmin(distances))

    assignments = []
    for hood in common_areas:
        c15 = assign_to_cluster(sigs_15_common.loc[hood].values, centroids)
        c25 = assign_to_cluster(sigs_25_common.loc[hood].values, centroids)
        assignments.append(
            {
                "neighborhood": hood,
                "cluster_2015": c15,
                "cluster_2025": c25,
                "shifted": c15 != c25,
            }
        )
    assignment_df = pd.DataFrame(assignments)

    # ── Step 4: Build membership display data ──
    mem_rows = []
    for _, row in assignment_df.iterrows():
        hood = row["neighborhood"]
        short = shorten_hood(hood)
        c15 = row["cluster_2015"]
        c25 = row["cluster_2025"]

        if not row["shifted"]:
            mem_rows.append(
                {
                    "cluster": f"Cluster {c15}",
                    "cluster_id": c15,
                    "label": short,
                    "sort_key": hood.lower(),
                    "shifted": False,
                }
            )
        else:
            mem_rows.append(
                {
                    "cluster": f"Cluster {c15}",
                    "cluster_id": c15,
                    "label": f"{short} (2015)",
                    "sort_key": hood.lower(),
                    "shifted": True,
                }
            )
            mem_rows.append(
                {
                    "cluster": f"Cluster {c25}",
                    "cluster_id": c25,
                    "label": f"{short} (2025)",
                    "sort_key": hood.lower(),
                    "shifted": True,
                }
            )
    mem_df = pd.DataFrame(mem_rows)
    mem_df = mem_df.sort_values(["cluster_id", "sort_key"]).reset_index(drop=True)
    label_order = mem_df["label"].tolist()

    # ── Step 5: Summary stats ──
    n_shifted = int(assignment_df["shifted"].sum())
    n_stable = len(assignment_df) - n_shifted

    def cluster_subtitle(cid):
        members = mem_df[mem_df["cluster_id"] == cid]
        stable = members[~members["shifted"]]["label"].tolist()
        shifted = members[members["shifted"]]["label"].tolist()
        parts = []
        if stable:
            preview = ", ".join(stable[:3])
            more = f" +{len(stable) - 3}" if len(stable) > 3 else ""
            parts.append(f"{len(stable)} stable: {preview}{more}")
        if shifted:
            parts.append(f"{len(shifted)} shifted")
        return " · ".join(parts) if parts else "No neighborhoods (centroid only)"

    # ── Step 6: Build Altair charts ──
    all_clusters = sorted(profile_df["cluster"].unique())
    CLUSTER_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]

    cluster_charts = []
    for cid in all_clusters:
        c_data = profile_df[profile_df["cluster"] == cid].copy()
        type_order = c_data.sort_values("rf", ascending=False)["type"].tolist()

        chart = (
            alt.Chart(c_data)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "type:N",
                    sort=type_order,
                    title=None,
                    axis=alt.Axis(labelAngle=-35, labelLimit=150, labelFontSize=11),
                ),
                y=alt.Y(
                    "rf:Q",
                    title="Relative Frequency",
                    axis=alt.Axis(format=".0%", labelFontSize=11),
                ),
                color=alt.value(CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]),
                tooltip=[
                    alt.Tooltip("type:N", title="Request Type"),
                    alt.Tooltip("rf:Q", title="Relative Freq", format=".2%"),
                ],
            )
            .properties(
                width=340,
                height=260,
                title=alt.Title(
                    f"Cluster {cid}",
                    subtitle=cluster_subtitle(cid),
                    subtitleFontSize=10,
                    subtitleColor="#666",
                    fontSize=14,
                    anchor="start",
                ),
            )
        )
        cluster_charts.append(chart)

    # Membership panel
    stable_points = (
        alt.Chart(mem_df[~mem_df["shifted"]])
        .mark_point(size=180, filled=True, opacity=0.9, shape="circle")
        .encode(
            y=alt.Y(
                "label:N",
                sort=label_order,
                title=None,
                axis=alt.Axis(labelFontSize=11, labelLimit=220),
            ),
            x=alt.X(
                "cluster:N",
                title=None,
                axis=alt.Axis(labelFontSize=12, orient="top", labelAngle=0),
            ),
            color=alt.Color(
                "cluster:N",
                title="Cluster",
                scale=alt.Scale(
                    domain=[f"Cluster {c}" for c in all_clusters],
                    range=[
                        CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in all_clusters
                    ],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Neighborhood"),
                alt.Tooltip("cluster:N", title="Cluster"),
            ],
        )
    )

    shifted_points = (
        alt.Chart(mem_df[mem_df["shifted"]])
        .mark_point(size=200, filled=True, opacity=0.9, shape="diamond")
        .encode(
            y=alt.Y("label:N", sort=label_order, title=None),
            x=alt.X("cluster:N", title=None),
            color=alt.Color(
                "cluster:N",
                scale=alt.Scale(
                    domain=[f"Cluster {c}" for c in all_clusters],
                    range=[
                        CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in all_clusters
                    ],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Neighborhood"),
                alt.Tooltip("cluster:N", title="Cluster"),
            ],
        )
    )

    membership_chart = alt.layer(stable_points, shifted_points).properties(
        width=180,
        height=max(260, len(label_order) * 18),
        title=alt.Title(
            "Cluster Membership",
            subtitle=[
                f"{n_stable} stable · {n_shifted} shifted between years",
                "● stable  ◆ shifted",
            ],
            subtitleFontSize=10,
            subtitleColor="#666",
            fontSize=14,
            anchor="start",
        ),
    )

    # ── Layout: adaptive grid ──
    # Make membership panel wider when displayed as its own row
    membership_chart = membership_chart.properties(width=720)

    if len(cluster_charts) >= 4:
        # 2x2 cluster grid + membership below as full-width row
        top_row = alt.hconcat(cluster_charts[0], cluster_charts[1], spacing=40)
        mid_row = alt.hconcat(cluster_charts[2], cluster_charts[3], spacing=40)
        combined = alt.vconcat(
            top_row, mid_row, membership_chart, spacing=40
        ).resolve_scale(color="independent")
    elif len(cluster_charts) == 3:
        top_row = alt.hconcat(cluster_charts[0], cluster_charts[1], spacing=40)
        mid_row = alt.hconcat(cluster_charts[2], spacing=40)
        combined = alt.vconcat(
            top_row, mid_row, membership_chart, spacing=40
        ).resolve_scale(color="independent")
    elif len(cluster_charts) == 2:
        top_row = alt.hconcat(cluster_charts[0], cluster_charts[1], spacing=40)
        combined = alt.vconcat(top_row, membership_chart, spacing=40).resolve_scale(
            color="independent"
        )
    else:
        combined = alt.vconcat(
            cluster_charts[0] if cluster_charts else membership_chart,
            membership_chart,
            spacing=40,
        ).resolve_scale(color="independent")

    combined = combined.properties(
        title=alt.Title(
            "How Neighborhoods Group by 311 Request Patterns",
            subtitle=[
                "Agglomerative clustering (JSD distance, average linkage) on averaged 2015+2025 signatures",
                "Neighborhood-year pairs assigned to nearest cluster centroid via Jensen-Shannon divergence",
            ],
            anchor="middle",
            fontSize=18,
            subtitleFontSize=12,
            subtitleColor="#666",
        ),
    )

    if save:
        combined.save("figures/cluster_comparison.html")
        print("Saved to figures/cluster_comparison.html")

    return combined, assignment_df


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
    print("\nGenerating visualizations...")
    print("1. Monthly heatmap comparison...")
    fig = create_monthly_heatmap(year15, year25, top_n=10, save=True)
    print("2. Neighborhood request composition...")
    fig = create_composition_bars(
        year15, year25, top_n_neighborhoods=10, rf_cutoff=0.02, save=True
    )
    print("3. Signature drift analysis...")
    fig, drift = create_signature_drift(year15, year25, save=True)
    print("4. Cluster comparison analysis...")
    fig, assignment_df = create_cluster_comparison(
        year15, year25, k=4, top_n_types=10, save=True
    )
    print("\nAll visualizations displayed!")

    print("Outputs in figures/ directory (PNG for Matplotlib, HTML for Altair).")

    print("Visualization 5 in `interactive_composition.py`")


if __name__ == "__main__":
    main()
