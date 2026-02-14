"""
signatures.py — 311 Pattern Signature Vectors & Clustering

Implements the "311 pattern signature" methodology from:
  Wang et al. (2017) "Structure of 311 service requests as a signature of urban location"

A signature vector S(a) for area a is defined as:
  S(a) = (s(a,t) / s(a),  for t = 1..T)

Where:
  s(a, t) = count of requests of type t in area a
  s(a)    = Σ_t s(a, t) = total requests in area a
  T       = total number of unique request types

Each component is the *relative frequency* of that request type within the area,
so the vector sums to 1.0.  Two areas with similar signature vectors have similar
complaint/request profiles, which Wang et al. showed correlates with similar
socioeconomic characteristics.

Usage:
    from api311 import Year
    from signatures import SignatureAnalyzer

    y15 = Year("data/cleaned2015.csv")
    y25 = Year("data/cleaned2025.csv")

    sa = SignatureAnalyzer(area_col="neighborhood", type_col="type")

    # Build signature matrices
    sigs_15 = sa.build_signatures(y15.data)
    sigs_25 = sa.build_signatures(y25.data)

    # Cluster neighborhoods
    labels_15, centroids_15 = sa.cluster(sigs_15, k=4)

    # Compare two years
    drift = sa.compare_signatures(sigs_15, sigs_25)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine, jensenshannon
from typing import Optional, Tuple


class SignatureAnalyzer:
    """
    Builds and analyzes 311 pattern signature vectors following Wang et al. (2017).

    Parameters
    ----------
    area_col : str
        Column name defining the spatial unit (e.g., "neighborhood", "zipcode").
    type_col : str
        Column name for the request category (e.g., "type", "reason", "subject").
    """

    def __init__(self, area_col: str = "neighborhood", type_col: str = "type"):
        self.area_col = area_col
        self.type_col = type_col
        self.type_index_ = None  # fitted type labels, set by build_signatures

    def build_signatures(
        self, df: pd.DataFrame, min_requests: int = 30
    ) -> pd.DataFrame:
        """
        Compute the signature matrix from raw 311 data.

        For every unique area, counts requests by type and normalises to
        relative frequencies (each row sums to 1).

        Parameters
        ----------
        df : pd.DataFrame
            Raw (or cleaned) 311 data with at least `area_col` and `type_col`.
        min_requests : int
            Drop areas with fewer than this many total requests (avoids
            unstable frequency estimates from tiny samples).

        Returns
        -------
        pd.DataFrame
            Index = area names, Columns = request types, Values = relative
            frequencies.  Each row is one signature vector S(a).

        Example
        -------
        >>> sigs = sa.build_signatures(year15.data)
        >>> sigs.loc["Dorchester"]
        type
        Bed Bugs                 0.002
        Building Maintenance     0.041
        ...
        Street Lights            0.087
        Name: Dorchester, dtype: float64
        >>> sigs.loc["Dorchester"].sum()   # always 1.0
        1.0
        """
        # Step 1: cross-tabulate  s(a, t)
        # -------------------------------------------------
        # This gives us a matrix where rows = areas, cols = types,
        # and values = raw counts.
        ct = pd.crosstab(df[self.area_col], df[self.type_col])

        # Step 2: filter sparse areas
        # -------------------------------------------------
        totals = ct.sum(axis=1)
        ct = ct.loc[totals >= min_requests]

        # Step 3: normalise rows to relative frequencies   s(a,t) / s(a)
        # -------------------------------------------------
        signatures = ct.div(ct.sum(axis=1), axis=0)

        # Store the type ordering so we can align a second year's data
        self.type_index_ = signatures.columns

        return signatures

    def align_signatures(
        self, sig_a: pd.DataFrame, sig_b: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two signature matrices to the same set of columns (request types).

        Types present in one year but not the other get a column of zeros.
        This is essential for fair comparison because Boston may introduce or
        retire request categories between 2015 and 2025.

        Parameters
        ----------
        sig_a, sig_b : pd.DataFrame
            Signature matrices (output of build_signatures).

        Returns
        -------
        (sig_a_aligned, sig_b_aligned) : tuple of pd.DataFrame
            Both DataFrames now share identical column sets, sorted.
        """
        all_types = sorted(set(sig_a.columns) | set(sig_b.columns))
        a = sig_a.reindex(columns=all_types, fill_value=0.0)
        b = sig_b.reindex(columns=all_types, fill_value=0.0)

        # Re-normalise rows in case new zero-columns broke the sum
        a = a.div(a.sum(axis=1), axis=0).fillna(0)
        b = b.div(b.sum(axis=1), axis=0).fillna(0)
        return a, b

    # ------------------------------------------------------------------
    #  Clustering
    # ------------------------------------------------------------------

    def cluster(
        self,
        signatures: pd.DataFrame,
        k: int = 4,
        n_init: int = 100,
        random_state: int = 42,
    ) -> Tuple[pd.Series, np.ndarray]:
        """
        K-means clustering on signature vectors (Wang et al. §3).

        Parameters
        ----------
        signatures : pd.DataFrame
            Signature matrix (rows = areas).
        k : int
            Number of clusters.  Wang et al. found 4 appropriate for
            NYC, Boston, and Chicago via silhouette / elbow methods.
        n_init : int
            Number of random restarts (Wang et al. used 100).
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        labels : pd.Series
            Cluster label for each area (index = area name).
        centroids : np.ndarray
            (k, T) array of cluster centroid vectors.
        """
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(signatures.values)

        labels = pd.Series(km.labels_, index=signatures.index, name="cluster")
        return labels, km.cluster_centers_

    # ------------------------------------------------------------------
    #  Year-over-year comparison
    # ------------------------------------------------------------------

    def compare_signatures(
        self,
        sig_old: pd.DataFrame,
        sig_new: pd.DataFrame,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """
        Measure how each area's signature vector changed between two years.

        Parameters
        ----------
        sig_old, sig_new : pd.DataFrame
            Signature matrices (must share index for overlapping areas).
        metric : str
            "cosine"  — cosine distance  (0 = identical, 1 = orthogonal)
            "jsd"     — Jensen-Shannon divergence (information-theoretic)
            "l2"      — Euclidean distance

        Returns
        -------
        pd.DataFrame with columns:
            area          — neighborhood name
            distance      — chosen distance metric
            top_increase  — type with largest positive frequency shift
            top_decrease  — type with largest negative frequency shift

        Example
        -------
        >>> drift = sa.compare_signatures(sigs_15, sigs_25, metric="cosine")
        >>> drift.sort_values("distance", ascending=False).head()
        """
        old_aligned, new_aligned = self.align_signatures(sig_old, sig_new)

        # Only compare areas present in both years
        common = old_aligned.index.intersection(new_aligned.index)
        old_aligned = old_aligned.loc[common]
        new_aligned = new_aligned.loc[common]

        records = []
        for area in common:
            v_old = old_aligned.loc[area].values
            v_new = new_aligned.loc[area].values
            diff = v_new - v_old

            if metric == "cosine":
                d = cosine(v_old, v_new)
            elif metric == "jsd":
                d = jensenshannon(v_old, v_new)
            elif metric == "l2":
                d = np.linalg.norm(diff)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            cols = old_aligned.columns
            records.append(
                {
                    "area": area,
                    "distance": d,
                    "top_increase": cols[np.argmax(diff)],
                    "top_decrease": cols[np.argmin(diff)],
                    "max_increase_val": diff.max(),
                    "max_decrease_val": diff.min(),
                }
            )

        return pd.DataFrame(records).sort_values("distance", ascending=False)

    # ------------------------------------------------------------------
    #  Convenience: top types per cluster
    # ------------------------------------------------------------------

    def cluster_profiles(
        self, signatures: pd.DataFrame, labels: pd.Series, top_n: int = 5
    ) -> dict:
        """
        For each cluster, return the top-N request types by mean frequency.

        Useful for interpreting what distinguishes each cluster — mirroring
        the analysis in Wang et al. Fig 2 / Fig 7.

        Returns
        -------
        dict mapping cluster_id -> pd.Series of top types with mean freqs.
        """
        merged = signatures.copy()
        merged["cluster"] = labels
        profiles = {}
        for cid in sorted(labels.unique()):
            mean_freq = merged[merged["cluster"] == cid].drop(columns="cluster").mean()
            profiles[cid] = mean_freq.nlargest(top_n)
        return profiles


# ======================================================================
#  Quick demo / integration with Year class
# ======================================================================
if __name__ == "__main__":
    from api311 import Year

    # Load cleaned data
    y15 = Year("data/cleaned2015.csv")
    y25 = Year("data/cleaned2025.csv")

    # Initialise analyzer — using neighborhood as the spatial unit
    # and "type" as the request category column
    sa = SignatureAnalyzer(area_col="neighborhood", type_col="type")

    # ---- Build signatures ----
    sigs_15 = sa.build_signatures(y15.data, min_requests=30)
    sigs_25 = sa.build_signatures(y25.data, min_requests=30)

    print("=== 2015 Signature Matrix ===")
    print(f"  Shape: {sigs_15.shape}  (areas x types)")
    print(f"  Areas: {list(sigs_15.index)}")
    print(f"  Row sums (should be 1.0): {sigs_15.sum(axis=1).unique()}")
    print()

    # ---- Cluster ----
    labels_15, centroids_15 = sa.cluster(sigs_15, k=4)
    labels_25, centroids_25 = sa.cluster(sigs_25, k=4)

    print("=== 2015 Cluster Assignments ===")
    print(labels_15)
    print()

    # ---- Cluster profiles ----
    print("=== 2015 Cluster Profiles (top 5 types per cluster) ===")
    profiles_15 = sa.cluster_profiles(sigs_15, labels_15)
    for cid, top in profiles_15.items():
        print(f"\n  Cluster {cid}:")
        for ttype, freq in top.items():
            print(f"    {ttype:40s} {freq:.3f}")

    # ---- Compare years ----
    print("\n=== Signature Drift: 2015 → 2025 (cosine distance) ===")
    drift = sa.compare_signatures(sigs_15, sigs_25, metric="cosine")
    print(drift.to_string(index=False))
