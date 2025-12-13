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

Hamming clustering is an error correction method that improves debarcoding accuracy by reassigning cells from low-count patterns to nearby high-count patterns.

## What is Hamming Distance?

X-Code barcodes can be represented as binary patterns. The Hamming distance between two patterns is the number of positions where they differ:

```text
Pattern A: 1 1 0 1 0 0 1 0 0
Pattern B: 1 0 0 1 0 1 1 0 0
             ^       ^
Hamming distance = 2 (differ at positions 2 and 6)
```

In debarcoding, patterns that are close in Hamming distance are likely related: one may be a noisy version of the other.

## Why Hamming Clustering?

After debarcoding, you typically see:
- **True barcodes**: High cell counts
- **Noisy children**: Low cell counts, usually within small Hamming distance of a true barcode

Noisy children arise from measurement errors. Hamming clustering identifies these noisy patterns and reassigns their cells to the likely parent barcode.

<details>
<summary><b>Theoretical Background: X-Code Geometry</b></summary>

<br>

**The 4-of-9 Block Structure**

X-Code barcodes use a combinatorial scheme where each block of 9 channels has exactly 4 channels ON:

- Each block has $\binom{9}{4} = 126$ valid patterns
- For $k=2$ blocks (18 channels): $L_2 = 126^2 = 15,876$ valid barcodes
- For $k=3$ blocks (27 channels): $L_3 = 126^3 = 2,000,376$ valid barcodes

Since each block must maintain exactly 4 ON channels, the minimum non-zero Hamming distance between two valid patterns within a block is 2. This means:

- Valid X-Codes always have even Hamming distances from each other
- An odd Hamming distance indicates an invalid pattern

**Hamming Neighbors in X-Code Space**

At radius 2, each barcode has exactly $D_k = 20k$ valid neighbors:

| Radius | 18ch (2 blocks) | 27ch (3 blocks) |
|--------|-----------------|-----------------|
| 2 | 40 | 60 |
| 4 | 520 | 1,380 |

**Expected Number of Neighbor Pairs**

The expected number of neighbor pairs in a random sublibrary of size $S$:

$$
\mathbb{E}[\#\text{neighbor pairs}] = \binom{S}{2} \times \frac{D_k}{L_k - 1}
$$

| Sublibrary Size (S) | 18ch | 27ch |
|---------------------|------|------|
| 100 | 12.5 | 0.15 |
| 500 | 314 | 3.7 |
| 1000 | 1259 | 15.0 |

**Conclusion**: Hamming clustering is much safer with 27-channel barcodes.

</details>

---

## Clustering Methods

Both methods are inspired by [starcode](https://github.com/gui11aume/starcode), adapted for X-Code geometry.

### Message Passing (Default)

The message passing method (`method='msg'`) iteratively merges small patterns into larger neighbors:

1. Sort patterns by count (ascending), then by median confidence
2. For each pattern p (smallest first):
   - Find valid neighbors within radius
   - **If p is invalid**: merge unconditionally into nearest valid neighbor
   - **If p is valid**: check if neighbor q passes count ratio test
   - If test passes, merge p into q
   - Update counts
3. Resolve chains

### Sphere Clustering

The sphere method (`method='sphere'`) identifies local maxima as cluster centers:

1. Find all valid patterns that are local maxima (no neighbor has higher count)
2. Filter centers by minimum count threshold
3. For each non-center pattern, merge into closest center within radius

---

## Performance Benchmarks

Performance was evaluated on simulated datasets with known ground truth.

### Beneficial vs Harmful Remaps

**This metric requires ground truth and is only available in simulation.**

- **Beneficial remap**: incorrectly assigned cell gets remapped to the correct barcode
- **Harmful remap**: correctly assigned cell gets remapped to an incorrect barcode

```{figure} ../_static/figures/fig1_msg_performance_summary.png
:align: center
:width: 100%

Message passing performance summary across simulations. Left panels show 18-channel simulations; right panels show 27-channel simulations. **a)** F1 improvement over baseline. **b)** Barcodes detected. **c)** Spearman correlation. **d)** Beneficial vs Harmful remaps heatmap: beneficial (above zero) vs harmful (below zero) remappings across sublibrary sizes and ratio settings.
```

Performance strongly depends on sublibrary size. The 27-channel (2M library) configuration is much safer across all metrics.

### Message Passing vs Sphere

```{figure} ../_static/figures/fig3_msg_vs_sphere_polished.png
:align: center
:width: 100%

Comparison of message passing and sphere methods. Left panels show 18-channel simulations (ratio=20); right panels show 27-channel simulations (ratio=2). **a)** F1 improvement. **b)** Beneficial vs Harmful remaps.
```

The two methods show similar performance, with message passing slightly better overall.

---

## Parameter Guidelines

### Recommended Settings

**18-channel (2 blocks):**
- Default: `ratio=20` (safe)
- S <= 100: can lower to `ratio=5-10` for more aggressive correction
- S >= 3000: marginal benefit, consider skipping Hamming clustering

**27-channel (3 blocks):**
- Default: `ratio=5` (safe)
- S <= 3000: can lower to `ratio=2` for maximum correction
- S > 3000: keep `ratio=5`

### Quick Reference

| Library | Sublibrary Size (S) | ratio | radius | Notes |
|---------|---------------------|-------|--------|-------|
| 18ch (2 blocks) | S <= 100 | 5-10 | 2 | Aggressive correction safe |
| 18ch (2 blocks) | 100 < S < 3000 | 15-20 | 2 | Default settings |
| 18ch (2 blocks) | S >= 3000 | - | - | Consider skipping |
| 27ch (3 blocks) | S <= 3000 | 2-5 | 2 | Aggressive correction safe |
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

adata = xd.io.read_data(f'{DATA_DIR}/2M_test.fcs')
adata = xd.io.map_channels(adata, f'{DATA_DIR}/barcode_channel_mapping_27ch.csv')
print(f"Loaded {adata.n_obs:,} cells, {adata.n_vars} channels")
```

```{code-cell} python
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

After Hamming clustering:
- `adata.obs['pc_gmm_hamming_assignment']`: corrected assignments
- `adata.obs['pc_gmm_hamming_remapped']`: boolean mask of remapped cells

Note: Confidence values are not modified. Continue to use `pc_gmm_confidence` for filtering.

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

---

## Summary

1. **Hamming clustering corrects noise** by reassigning cells from low-count patterns to nearby high-count parents
2. **Safety depends on sublibrary size**: 27-channel is much safer than 18-channel for large S
3. **Multiple safeguards** (ratio, confidence) protect against incorrect remapping
4. **Start with recommended defaults**, adjust ratio based on your configuration
5. **Always compare before/after** using barcode rank histograms
6. **Confidence values are not modified** by Hamming clustering
