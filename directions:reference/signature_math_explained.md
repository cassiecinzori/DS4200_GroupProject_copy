# 311 Pattern Signature Vectors — Mathematical Formulation

Based on Wang et al. (2017), "Structure of 311 service requests as a signature of urban location."

---

## 1. Raw Count Matrix

Let there be **A** spatial areas (neighborhoods) and **T** unique request types across the dataset.

Define **s(a, t)** as the raw count of 311 requests of type *t* in area *a*:

$$
s(a, t) = \left| \{ r \in \text{requests} : \text{area}(r) = a \text{ and } \text{type}(r) = t \} \right|
$$

This gives us a matrix of shape (A × T):

|              | Type 1 | Type 2 | ... | Type T |
|--------------|--------|--------|-----|--------|
| Dorchester   | 412    | 87     | ... | 203    |
| Roxbury      | 298    | 142    | ... | 176    |
| Back Bay     | 53     | 211    | ... | 88     |
| ...          | ...    | ...    | ... | ...    |

## 2. Row Totals

For each area *a*, sum across all types to get the total request count:

$$
s(a) = \sum_{t=1}^{T} s(a, t)
$$

## 3. Signature Vector (Normalization)

The **signature vector** for area *a* is a T-dimensional vector where each component is the **relative frequency** of that request type within the area:

$$
S(a) = \left( \frac{s(a, 1)}{s(a)},\ \frac{s(a, 2)}{s(a)},\ \ldots,\ \frac{s(a, T)}{s(a)} \right)
$$

Or more compactly, the *t*-th component is:

$$
S(a)_t = \frac{s(a, t)}{s(a)}
$$

Each component is bounded between 0 and 1, and every signature vector sums to exactly 1:

$$
\sum_{t=1}^{T} S(a)_t = \sum_{t=1}^{T} \frac{s(a, t)}{s(a)} = \frac{1}{s(a)} \sum_{t=1}^{T} s(a, t) = \frac{s(a)}{s(a)} = 1
$$

This means the signature is a **probability distribution** over request types — it tells you the likelihood that a randomly chosen request in area *a* belongs to type *t*.

### Concrete Example

Suppose Dorchester has 5,000 total requests distributed as:

| Type | Raw Count s(a,t) | Signature Component S(a)_t |
|------|----------------:|---------------------------:|
| Street Lights | 400 | 400 / 5000 = **0.080** |
| Code Enforcement | 650 | 650 / 5000 = **0.130** |
| Sanitation | 1200 | 1200 / 5000 = **0.240** |
| Noise | 310 | 310 / 5000 = **0.062** |
| ... (all others) | 2440 | 2440 / 5000 = **0.488** |
| **Total** | **5000** | **1.000** |

The resulting signature vector for Dorchester is:

$$
S(\text{Dorchester}) = (0.080,\ 0.130,\ 0.240,\ 0.062,\ \ldots)
$$

## 4. Signature Matrix

Stacking all signature vectors produces a normalized matrix **M** of shape (A × T):

$$
M_{a,t} = S(a)_t = \frac{s(a, t)}{s(a)}
$$

Every row sums to 1. Every cell is a relative frequency. This is the input to clustering and comparison.

## 5. K-Means Clustering

Following Wang et al., apply k-means clustering to the rows of **M** (each row is a point in T-dimensional space).

Given *k* clusters, k-means minimizes the within-cluster sum of squared Euclidean distances:

$$
\arg\min_{C_1, \ldots, C_k} \sum_{j=1}^{k} \sum_{a \in C_j} \left\| S(a) - \mu_j \right\|^2
$$

where **μ_j** is the centroid (mean signature) of cluster *C_j*:

$$
\mu_j = \frac{1}{|C_j|} \sum_{a \in C_j} S(a)
$$

Wang et al. used **k = 4** and ran the algorithm **100 times** with different random initializations, keeping the solution with the lowest total within-cluster sum of squares.

The centroid **μ_j** is itself a T-dimensional vector that represents the "average complaint profile" of its cluster. Examining which types have the highest values in each centroid tells you what distinguishes that cluster.

## 6. Year-over-Year Comparison

To compare how a neighborhood's profile changed between 2015 and 2025, we compute a distance between its two signature vectors.

### 6a. Alignment

First, align the type columns. If 2015 has types {A, B, C} and 2025 has types {B, C, D}, the union is {A, B, C, D}. Missing types get frequency 0, and each row is re-normalized to sum to 1.

### 6b. Cosine Distance

For area *a* with signature vectors $S_{2015}(a)$ and $S_{2025}(a)$:

$$
d_{\cos}(a) = 1 - \frac{S_{2015}(a) \cdot S_{2025}(a)}{\|S_{2015}(a)\| \cdot \|S_{2025}(a)\|}
$$

- **0** = identical complaint profiles (no change)
- **1** = completely orthogonal (maximum change)

### 6c. Jensen-Shannon Divergence (alternative)

Since signatures are probability distributions, Jensen-Shannon divergence is a natural information-theoretic alternative:

$$
\text{JSD}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)
$$

where $M = \frac{1}{2}(P + Q)$ and $D_{KL}$ is the Kullback-Leibler divergence:

$$
D_{KL}(P \| M) = \sum_{t=1}^{T} P_t \log \frac{P_t}{M_t}
$$

JSD is bounded [0, 1] (when using log base 2), symmetric, and well-defined even when some components are zero.

### 6d. Component-wise Difference

The raw shift for each type is simply:

$$
\Delta_t(a) = S_{2025}(a)_t - S_{2015}(a)_t
$$

- **Positive Δ_t** → type *t* became a larger share of complaints in that neighborhood
- **Negative Δ_t** → type *t* became a smaller share

The type with $\arg\max_t \Delta_t(a)$ is the "top increase" and $\arg\min_t \Delta_t(a)$ is the "top decrease" for that neighborhood.

---

## Code Mapping

| Math | Code (`signatures.py`) |
|------|----------------------|
| $s(a, t)$ | `pd.crosstab(df[area_col], df[type_col])` |
| $S(a)_t = s(a,t) / s(a)$ | `ct.div(ct.sum(axis=1), axis=0)` |
| Align columns across years | `sa.align_signatures(sigs_15, sigs_25)` |
| K-means on $M$ | `sa.cluster(signatures, k=4)` |
| $d_{\cos}(a)$ | `scipy.spatial.distance.cosine(v_old, v_new)` |
| $\text{JSD}(P \| Q)$ | `scipy.spatial.distance.jensenshannon(v_old, v_new)` |
| $\Delta_t(a)$ | `v_new - v_old` then `np.argmax` / `np.argmin` |