---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Hamming Clustering Guidelines

Hamming clustering is an error correction method that improves debarcoding accuracy by reassigning cells from low-count patterns to nearby high-count patterns. This tutorial explains how it works and provides parameter tuning guidelines.

## What is Hamming Distance?

X-Code barcodes can be represented as binary patterns, where each position indicates whether the cell carries (1) or does not carry (0) the n-th elementary DNA sequence. The Hamming distance between two such patterns is the number of positions where they differ. For example:

```text
Pattern A: 1 1 0 1 0 0 1 0 0
Pattern B: 1 0 0 1 0 1 1 0 0
             ^       ^
Hamming distance = 2 (differ at positions 2 and 6)
```

In debarcoding, patterns that are close in Hamming distance are likely to be related: one may be a noisy version of the other.

## Why Hamming Clustering?

After debarcoding, you typically see:
- **True barcodes**: High cell counts, these are the real barcode populations
- **Noisy children**: Low cell counts, usually within small Hamming distance of a true barcode (the "parent")

Noisy children arise from measurement errors: a few channels misclassified as ON instead of OFF (or vice versa), depending on the separation between the OFF and ON intensity modes. Hamming clustering identifies these noisy patterns and reassigns their cells to the likely parent barcode.

If two true barcodes are rarely Hamming neighbors (which we can compute from X-Code geometry), then a small pattern near a large pattern is almost certainly noise, not a real barcode.

<details>
<summary><b>Theoretical Background: X-Code Geometry</b></summary>

<br>

**The 4-of-9 Block Structure**

X-Code barcodes use a combinatorial scheme where each block of 9 channels has exactly 4 channels ON. This structure is useful for understanding error correction because it constrains which patterns are valid:

- Each block has $\binom{9}{4} = 126$ valid patterns
- For $k=2$ blocks (18 channels): $L_2 = 126^2 = 15,876$ valid barcodes
- For $k=3$ blocks (27 channels): $L_3 = 126^3 = 2,000,376$ valid barcodes

Since each block must maintain exactly 4 ON channels, the minimum non-zero Hamming distance between two valid patterns within a block is 2 (flip one ON to OFF and one OFF to ON). This means:

- Valid X-Codes always have even Hamming distances from each other
- An odd Hamming distance between a valid barcode and another pattern indicates that the latter is invalid

**Hamming Neighbors in X-Code Space**

For a single 4-of-9 block, the number of valid patterns at each Hamming distance from any pattern:

| Distance | Count | Formula |
|----------|-------|---------|
| 0 | 1 | (self) |
| 2 | 20 | $\binom{4}{1} \times \binom{5}{1}$ |
| 4 | 60 | $\binom{4}{2} \times \binom{5}{2}$ |
| 6 | 40 | $\binom{4}{3} \times \binom{5}{3}$ |
| 8 | 5 | $\binom{4}{4} \times \binom{5}{4}$ |

For multiple blocks, computing neighbors at radius > 2 is more complex because distances can be distributed across blocks. At radius 2, each barcode has exactly $D_k = 20k$ valid neighbors. Here are the neighbor counts for common configurations:

| Radius | 18ch (2 blocks) | 27ch (3 blocks) |
|--------|-----------------|-----------------|
| 2 | 40 | 60 |
| 4 | 520 | 1,380 |
| 6 | 2,480 | 15,320 |
| 8 | 5,210 | 87,615 |

<details>
<summary><b>Detailed Formula for Multiple Blocks</b></summary>

<br>

**Single block: distance distribution**

For one block (length-9 binary vector with exactly 4 ones), to get from one valid pattern to another at Hamming distance $2d$:
- Flip $d$ of the 1s to 0s
- Flip $d$ of the 0s to 1s

Number of ways:

$$
a_d = \binom{4}{d}\binom{5}{d}, \quad d = 0,1,2,3,4
$$

The distance-index polynomial is:

$$
P(x) = \sum_{d=0}^4 a_d x^d = 1 + 20x + 60x^2 + 40x^3 + 5x^4
$$

where the exponent $d$ corresponds to Hamming distance $2d$.

**Multiple blocks: distances**

For $k$ blocks (called `n_blocks` in the code), if a barcode is at total Hamming distance $2M$ from the reference:
- Let block $b$ contribute distance $2d_b$ where $d_b \in \{0,1,2,3,4\}$
- Total distance constraint: $\sum_{b=1}^{k} d_b = M$

For a fixed $(d_1, \ldots, d_k)$ with that sum, the count is:

$$
\prod_{b=1}^{k} a_{d_b} = \prod_{b=1}^{k} \binom{4}{d_b}\binom{5}{d_b}
$$

The total number of barcodes at distance $2M$ is:

$$
N_{2M}^{(k)} = \sum_{\substack{d_1+\cdots+d_k = M \\ 0\le d_b \le 4}} \prod_{b=1}^{k} \binom{4}{d_b}\binom{5}{d_b}
$$

The coefficient of $x^M$ in $P(x)^k$ is the number of barcodes at Hamming distance $2M$ from a given reference barcode in a $k$-block code.

</details>

<br>

**The Sublibrary Problem**

In practice, experiments use a small sublibrary of $S$ barcodes from the full library of $L_k = 126^k$ valid patterns (where $k$ is the number of blocks). When the sublibrary size is small compared to the full library size, two randomly selected true barcodes are unlikely to be Hamming neighbors.

**Expected Number of Neighbor Pairs**

Let:
- $k$ = number of 4-of-9 blocks
- $L_k = 126^k$ = total valid barcodes
- $S$ = sublibrary size (number of true barcodes)
- $D_k = 20k$ = number of Hamming-distance-2 neighbors per barcode

Then the expected number of unordered neighbor pairs in a random sublibrary of size $S$ is:

$$
\mathbb{E}[\#\text{neighbor pairs}] = \binom{S}{2} \times \frac{D_k}{L_k - 1}
$$

| Sublibrary Size (S) | 18ch ($L_2$=15,876) | 27ch ($L_3$=2,000,376) |
|---------------------|---------------------|------------------------|
| 30                  | 1.10                | 0.0130                 |
| 100                 | 12.47               | 0.1485                 |
| 200                 | 50.14               | 0.5969                 |
| 500                 | 314.33              | 3.7418                 |
| 1000                | 1258.58             | 14.98                  |
| 2000                | 5036.85             | 59.96                  |
| 3000                | 11334.80            | 134.93                 |
| 5000                | 31489.76            | 374.85                 |

**Conclusion**: Hamming clustering is much safer with 27-channel (2M library) barcodes. For 18-channel barcodes with large sublibraries (S > 1000), true barcode collisions become common.

**Safeguards Against Incorrect Remapping**

The situation is not as dramatic as the collision counts might suggest. Hamming clustering uses multiple safeguards:

1. **Count ratio test**: A pattern is only remapped if the neighbor has significantly more cells (controlled by `ratio` parameter)
2. **Confidence test**: By default, the absorbing neighbor must have higher median confidence than the pattern being absorbed

These safeguards ensure that even when true barcodes happen to be neighbors, the smaller one is only absorbed if it truly looks like noise.

</details>

---

## Clustering Methods

Both methods are inspired by [starcode](https://github.com/gui11aume/starcode), adapted for X-Code geometry.

### Message Passing (Default)

The message passing method (`method='msg'`) iteratively merges small patterns into larger neighbors:

1. Sort patterns by count (ascending), then by median confidence (ascending)
2. For each pattern p (smallest first):
   - Find valid neighbors within radius
   - **If p is invalid**: merge unconditionally into nearest valid neighbor (no ratio test)
   - **If p is valid**: check if neighbor q passes count ratio test:
     - Count ratio: $\mathrm{count}(q) \geq \mathrm{ratio} \times \mathrm{count}(p)$
   - If test passes, merge p into q (transfer all cells)
   - Update counts
3. Repeat until no more merges possible

The method is iterative: small patterns merge into medium-sized ones, which may then merge into larger ones. After all merges, chains are resolved by following each pattern to its final center.

### Sphere Clustering

The sphere method (`method='sphere'`) identifies local maxima as cluster centers:

1. Find all valid patterns that are local maxima (no neighbor has higher count)
2. Filter centers by minimum count threshold (`min_count_center`)
3. For each non-center pattern:
   - Find closest center within radius
   - **If pattern is invalid**: merge unconditionally into nearest valid center
   - **If pattern is valid**: merge only if ratio and confidence tests pass

Unlike message passing, sphere clustering assigns patterns directly to pre-identified centers without iterative chain formation. This is useful when message passing creates unwanted merge chains.

---

## Tie Breaking

When multiple neighbors are equidistant and pass all tests, the `tie_break` parameter controls resolution:

| Method | Description | When to use |
|--------|-------------|-------------|
| `'lda'` | Linear Discriminant Analysis on intensity data | Default; best accuracy |
| `'count'` | Choose neighbor with highest count | Fast and more aggressive |
| `'no_remap'` | Skip remapping for ties | Conservative; avoid errors |

Whereas starcode splits counts equally among tied neighbors, we use the optimal Gaussian LDA boundary based on the intensity distributions of the neighbors (including the source pattern distribution itself minus the cell event looking to be remapped) on the channels that differ between them. This provides a more informed tie-break decision using the actual measurement values. With LDA, it is possible that a cell event does not get remapped if it does not look likely that it arose from another pattern based on its intensities.

---

## Remapped Cell Confidence

After remapping, Mahalanobis confidence is recomputed for all cells based on the new cluster assignments. Each cell's confidence reflects its distance to the centroid of its (potentially new) pattern cluster:

$$
\text{confidence} = \frac{1}{1 + d^2 / 9k}
$$

where $d^2$ is the squared Mahalanobis distance and $9k$ is the number of channels.

---

## Simulation Results

The following results are from simulations varying sublibrary size, mean channel error rate, and barcode population structure (ground truth concentration of counts within the sublibrary).

### Message Passing Performance

The figure below shows message passing performance across different scenarios (all using PC-GMM debarcoding):

```{figure} ../_static/figures/fig1_msg_performance_summary.png
:align: center
:width: 100%

Message passing performance summary across simulations with varying sublibrary size, channel error rates, and population structures. Left panels show 18-channel (2-block) simulations; right panels show 27-channel (3-block) simulations. **a)** ΔF1: F1 score improvement over debarcoded baseline. **b)** ΔAUC: change in area under the coverage-accuracy curve. **c)** Δ Barcodes Detected: change in percentage of true barcodes detected. **d)** Δ Spearman: change in Spearman correlation between predicted and true barcode counts. **e)** Beneficial vs Harmful Remaps: beneficial remaps (incorrectly assigned cell remapped to correct barcode, shown above zero) vs harmful remaps (correctly assigned cell remapped to incorrect barcode, shown below zero).
```

Performance strongly depends on sublibrary size. The 27-channel (2M library) configuration is much safer across all metrics. For 18-channel barcodes with large S, benefits diminish and risks increase.

### Effect on Coverage-Accuracy Trade-off

The coverage-accuracy curve shows accuracy (fraction of correctly assigned cells among those included) as a function of coverage (fraction of cells included, sorted by confidence). Hamming clustering can improve this trade-off by correcting errors while maintaining high-confidence assignments.

```{figure} ../_static/figures/fig_roc_curves.png
:align: center
:width: 100%

Coverage-accuracy curves and neutral remaps analysis. Left panels show 18-channel simulations; right panels show 27-channel simulations. **a)** Coverage-accuracy curves averaged across all sublibrary sizes, channel error rates, and population structures. The dashed black line is the baseline (no Hamming clustering), colored lines show the effect of different ratio settings. Hamming clustering shifts the curve upward, meaning higher accuracy at the same coverage level. **b)** Neutral remaps as a percentage of total remaps (beneficial + harmful + neutral).
```

Despite a significant fraction of remaps being neutral (incorrectly assigned cells remapped to a different incorrect barcode), especially in the 18ch case, it doesn't affect accuracy because these cells were already incorrectly assigned before Hamming clustering; and it doesn't seem to negatively affect confidence calibration (AUC) either since it still improves. 

### Message Passing vs Sphere

```{figure} ../_static/figures/fig3_msg_vs_sphere_polished.png
:align: center
:width: 100%

Comparison of message passing and sphere methods across simulations with varying sublibrary size, channel error rates, and population structures. Left panels show 18-channel simulations (ratio=20); right panels show 27-channel simulations (ratio=2). **a)** ΔF1: F1 improvement. **b)** ΔAUC: change in coverage-accuracy AUC. **c)** Beneficial vs Harmful Remaps: beneficial (above zero) vs harmful (below zero) remappings for each method.
```

The two methods show similar performance in most cases, with message passing slightly better overall, especially for low barcode counts. The sphere method is useful when you want explicit center control or when message passing creates unwanted merge chains.

---

## Parameter Guidelines

### Recommended Settings

**18-channel (2 blocks):**
- Default: `ratio=20` (safe)
- S ≤ 100: can lower to `ratio=5-10` for more aggressive correction
- S ≥ 3000: marginal benefit, consider skipping Hamming clustering

**27-channel (3 blocks):**
- Default: `ratio=5` (safe)
- S ≤ 3000: can lower to `ratio=2` for maximum correction
- S > 3000: keep `ratio=5`

### Quick Reference

| Library | Sublibrary Size (S) | ratio | radius | Notes |
|---------|---------------------|-------|--------|-------|
| 18ch (2 blocks) | S ≤ 100 | 5-10 | 2 | Aggressive correction safe |
| 18ch (2 blocks) | 100 < S < 3000 | 15-20 | 2 | Default settings |
| 18ch (2 blocks) | S ≥ 3000 | - | - | Consider skipping |
| 27ch (3 blocks) | S ≤ 3000 | 2-5 | 2 | Aggressive correction safe |
| 27ch (3 blocks) | S > 3000 | 5 | 2 | Keep conservative |

### Other Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tie_break` | 'lda' | Use 'no_remap' for maximum safety |
| `min_count_center` | 1 | For sphere method, set based on expected population size |

---

## Example Workflow

```{code-cell} python
import warnings
warnings.filterwarnings('ignore')
import xcode_debarcode as xd
import plotly.io as pio
pio.renderers.default = 'notebook'

DATA_DIR = '../../data'

# Load 27-channel dataset
adata = xd.io.read_data(f'{DATA_DIR}/2M_test.fcs')
adata = xd.io.map_channels(adata, f'{DATA_DIR}/barcode_channel_mapping_27ch.csv')
print(f"Loaded {adata.n_obs:,} cells, {adata.n_vars} channels")
```

```{code-cell} python
# Standard pipeline
adata = xd.preprocessing.transform(adata, method='log', verbose=False)
adata = xd.debarcode.debarcode(adata, method='pc_gmm', layer='log', verbose=False)
```

### Apply Hamming Clustering

Since this is the 2M library (3 blocks, 27 channels) and we expect only 4 true barcodes, we can safely use an aggressive `ratio=2` setting.

```{code-cell} python
adata = xd.postprocessing.hamming_cluster(
    adata,
    assignment_col='pc_gmm_assignment',
    confidence_col='pc_gmm_confidence',
    method='msg',
    radius=2,
    ratio=2.0,
    verbose=True
)
```

### Visualizations

#### Barcode Rank Histogram

Before Hamming:

```{code-cell} python
fig = xd.plots.plot_barcode_rank_histogram(adata, assignment_col='pc_gmm_assignment')
fig
```

After Hamming:

```{code-cell} python
fig = xd.plots.plot_barcode_rank_histogram(adata, assignment_col='pc_gmm_hamming_assignment')
fig
```

#### Hamming Graph

The Hamming graph shows patterns as nodes connected by edges when they are within a given Hamming distance. Node size represents cell count, and color can represent confidence or validity.

Before Hamming (shows noisy children connected to true barcodes):

```{code-cell} python
fig = xd.plots.plot_hamming_graph(
    adata, 
    assignment_col='pc_gmm_assignment',
    confidence_col='pc_gmm_confidence',
    radius=2,
    min_count=50
)
fig
```

The visualization confirms that there are 4 real barcodes, each surrounded by a halo of noisy children within Hamming distance 2.

After Hamming:

```{code-cell} python
fig = xd.plots.plot_hamming_graph(
    adata, 
    assignment_col='pc_gmm_hamming_assignment',
    confidence_col='pc_gmm_confidence',
    radius=2,
    min_count=50
)
fig
```

Hamming clustering has considerably reduced the number of noisy children by reassigning them to their parent barcodes.

#### Hamming Heatmap

The Hamming heatmap shows pairwise distances between patterns. Patterns within the specified radius appear as colored cells.

Before Hamming:

```{code-cell} python
:tags: [wide-plot]

fig = xd.plots.plot_hamming_heatmap(
    adata,
    assignment_col='pc_gmm_assignment',
    hamming_radius=4,
    min_count=50
)
fig
```

After Hamming:

```{code-cell} python
fig = xd.plots.plot_hamming_heatmap(
    adata,
    assignment_col='pc_gmm_hamming_assignment',
    hamming_radius=4,
    min_count=50
)
fig
```

---

## Summary

1. **Hamming clustering corrects noise** by reassigning cells from low-count patterns to nearby high-count parents
2. **Safety depends on sublibrary size**: 27-channel is much safer than 18-channel for large S
3. **Multiple safeguards** (ratio, confidence) protect against incorrect remapping
4. **Start with recommended defaults**, adjust ratio based on your configuration
5. **Always compare before/after** using barcode rank histograms
