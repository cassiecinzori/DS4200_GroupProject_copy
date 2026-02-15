"""
visualizations.py - Generate all visualizations for DS4200 project
Boston 311 Service Request Analysis: 2015 vs 2025
"""

from api311 import Year
from signatures import SignatureAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'


def clean_request_type_name(name):
    """Make request type names more readable"""
    replacements = {
        'Missed Trash/Recycling/Yard Waste/Bulk Item': 'Missed Trash/Recycling',
        'Request for Snow Plowing': 'Snow Plowing',
        'Request for Pothole Repair': 'Pothole Repair',
        'Street Light Outages': 'Street Lights',
        'Pothole Repair (Internal)': 'Pothole (Internal)',
        'Poor Conditions of Property': 'Poor Property Conditions',
        'Improper Storage of Trash (Barrels)': 'Improper Trash Storage',
        'Parks Lighting/Electrical Issues': 'Parks Lighting',
    }
    return replacements.get(name, name)


def create_monthly_heatmap(year15, year25):
    """Monthly request volume heatmap"""
    print("Creating monthly heatmap...")

    summary_15 = year15.summarize("type")
    summary_25 = year25.summarize("type")

    monthly_15 = summary_15["monthly"]
    monthly_25 = summary_25["monthly"]

    top_types = year15.data["type"].value_counts().head(10).index.tolist()

    monthly_15_filtered = monthly_15.reindex(top_types).fillna(0)
    monthly_25_filtered = monthly_25.reindex(top_types).fillna(0)

    monthly_15_filtered.index = [clean_request_type_name(x) for x in monthly_15_filtered.index]
    monthly_25_filtered.index = [clean_request_type_name(x) for x in monthly_25_filtered.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(monthly_15_filtered, annot=True, fmt='.0f',
                cmap='YlOrRd', cbar_kws={'label': 'Number of Requests'},
                ax=ax1, linewidths=0.5, vmin=0)
    ax1.set_title('2015', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Request Type', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', rotation=0, labelsize=10)

    sns.heatmap(monthly_25_filtered, annot=True, fmt='.0f',
                cmap='YlOrRd', cbar_kws={'label': 'Number of Requests'},
                ax=ax2, linewidths=0.5, vmin=0)
    ax2.set_title('2025', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Request Type', fontsize=12)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', rotation=0, labelsize=10)

    plt.suptitle('Seasonal Patterns in Boston 311 Requests',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    return fig


def create_composition_bars(year15, year25):
    """Neighborhood composition bar charts"""
    print("Creating neighborhood composition charts...")

    top_neighborhoods = year15.data["neighborhood"].value_counts().head(6).index.tolist()
    data_list = []

    for hood in top_neighborhoods:
        hood_15 = year15.data[year15.data["neighborhood"] == hood]
        type_counts_15 = hood_15["type"].value_counts().head(5)

        for req_type, count in type_counts_15.items():
            data_list.append({
                "neighborhood": hood,
                "type": clean_request_type_name(req_type),
                "count": count,
                "year": "2015"
            })

        hood_25 = year25.data[year25.data["neighborhood"] == hood]
        type_counts_25 = hood_25["type"].value_counts().head(5)

        for req_type, count in type_counts_25.items():
            data_list.append({
                "neighborhood": hood,
                "type": clean_request_type_name(req_type),
                "count": count,
                "year": "2025"
            })

    df = pd.DataFrame(data_list)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ['#3498db', '#e74c3c']

    for idx, hood in enumerate(top_neighborhoods):
        hood_df = df[df["neighborhood"] == hood]
        pivot = hood_df.pivot_table(index='type', columns='year',
                                     values='count', fill_value=0)

        pivot.plot(kind='bar', ax=axes[idx], color=colors,
                   width=0.75, edgecolor='black', linewidth=0.7)

        axes[idx].set_title(f"{hood}", fontsize=13, fontweight='bold', pad=10)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Number of Requests', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=9)
        axes[idx].legend(title='Year', fontsize=10, loc='upper right')
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('How Neighborhood Request Patterns Changed: 2015 vs 2025',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fig


def create_signature_drift(year15, year25):
    """Signature drift analysis"""
    print("Creating signature drift analysis...")

    sa = SignatureAnalyzer(area_col="neighborhood", type_col="type")

    sigs_15 = sa.build_signatures(year15.data, min_requests=30)
    sigs_25 = sa.build_signatures(year25.data, min_requests=30)

    drift = sa.compare_signatures(sigs_15, sigs_25, metric="cosine")
    drift = drift.sort_values("distance", ascending=False).reset_index(drop=True)

    counts_15 = year15.data["neighborhood"].value_counts()
    counts_25 = year25.data["neighborhood"].value_counts()
    drift["avg_requests"] = drift["area"].map(
        lambda x: (counts_15.get(x, 0) + counts_25.get(x, 0)) / 2
    )

    fig, ax = plt.subplots(figsize=(15, 8))

    colors = plt.cm.RdYlGn_r(drift["distance"] / drift["distance"].max())
    bars = ax.bar(range(len(drift)), drift["distance"],
                   color=colors, edgecolor='black', linewidth=0.6, alpha=0.85)

    top_5_idx = drift.head(5).index
    for idx in top_5_idx:
        bars[idx].set_edgecolor('darkred')
        bars[idx].set_linewidth(2.5)

    for idx, row in drift.head(5).iterrows():
        ax.text(idx, row["distance"] + 0.015, row["area"],
                rotation=45, ha='left', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                         alpha=0.8, edgecolor='black', linewidth=1))

    median = drift["distance"].median()
    ax.axhline(y=median, color='navy', linestyle='--', linewidth=2.5,
               label=f'Median Change: {median:.3f}', alpha=0.7)

    ax.set_xlabel('Neighborhoods (sorted by amount of change)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Signature Distance (0 = no change, 1 = complete change)',
                  fontsize=13, fontweight='bold')
    ax.set_title('Which Boston Neighborhoods Changed Most? (2015-2025)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

    textstr = 'Largest Changes:\n' + '\n'.join(
        [f"{i+1}. {row['area']}: {row['distance']:.3f}"
         for i, row in drift.head(5).iterrows()]
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                 edgecolor='black', linewidth=1.5)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props,
            family='monospace')

    plt.tight_layout()
    plt.show()

    return fig, drift


def create_cluster_comparison(year15, year25):
    """Cluster comparison"""
    print("Creating cluster analysis...")

    sa = SignatureAnalyzer(area_col="neighborhood", type_col="type")

    sigs_15 = sa.build_signatures(year15.data, min_requests=30)
    sigs_25 = sa.build_signatures(year25.data, min_requests=30)

    labels_15, _ = sa.cluster(sigs_15, k=4)
    labels_25, _ = sa.cluster(sigs_25, k=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    cluster_counts_15 = labels_15.value_counts().sort_index()
    ax1.bar(cluster_counts_15.index, cluster_counts_15.values,
            color=cluster_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_title('2015 Neighborhood Clusters', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Cluster Group', fontsize=12)
    ax1.set_ylabel('Number of Neighborhoods', fontsize=12)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f'Cluster {i}' for i in range(4)])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    cluster_counts_25 = labels_25.value_counts().sort_index()
    ax2.bar(cluster_counts_25.index, cluster_counts_25.values,
            color=cluster_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_title('2025 Neighborhood Clusters', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Cluster Group', fontsize=12)
    ax2.set_ylabel('Number of Neighborhoods', fontsize=12)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f'Cluster {i}' for i in range(4)])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('How Neighborhoods Group Together by Request Patterns',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    return fig, labels_15, labels_25


def main():
    """Generate all static visualizations"""
    print("Loading data...")
    year15 = Year("data/cleaned2015.csv")
    year25 = Year("data/cleaned2025.csv")
    year15.make_points()
    year25.make_points()

    print(f"Loaded 2015: {len(year15.data):,} records")
    print(f"Loaded 2025: {len(year25.data):,} records")

    create_monthly_heatmap(year15, year25)
    create_composition_bars(year15, year25)
    create_signature_drift(year15, year25)
    create_cluster_comparison(year15, year25)

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()