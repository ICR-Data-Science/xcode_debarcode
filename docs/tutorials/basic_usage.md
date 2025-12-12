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

# Basic Usage

This tutorial walks through the complete xcode_debarcode workflow using a real dataset.

## Prerequisites

- CyTOF data file (FCS or H5AD)
- Channel mapping CSV file

## Example Dataset

| Item | Value |
|------|-------|
| Data file | `15K_test.fcs` |
| Mapping file | `barcode_channel_mapping_18ch.csv` |
| Configuration | 2-block barcode (18 channels) |

## Workflow Overview

```
Load Data -> Map Channels -> Transform -> Filter Intensity -> Debarcode 
    -> Inspect Distribution -> Hamming Clustering -> Confidence Filtering -> Save
```

---

## 1. Load and Map Data

### Load Data

```{code-cell} python
import warnings
warnings.filterwarnings('ignore')
import xcode_debarcode as xd
import plotly.io as pio
pio.renderers.default = 'notebook'  # Required for rendering plots in Jupyter notebooks

# Data path (relative to docs/tutorials/)
DATA_DIR = '../../data'

# Load FCS or H5AD file
adata = xd.io.read_data(f'{DATA_DIR}/15K_test.fcs')

print(f"Loaded {adata.n_obs} cells, {adata.n_vars} channels")
```

<details>
<summary><b>Supported formats</b></summary>

- `.fcs`: Flow Cytometry Standard (requires `readfcs` package)
- `.h5ad`: AnnData HDF5 format (preserves all metadata)

</details>

### Map Barcode Channels

Channel mapping identifies which CyTOF channels correspond to barcode sequences and renames them to a standardized format (`s_1`, `s_2`, ...).

**Mapping CSV structure:**

| bc_sequence | channel_name |
|-------------|--------------|
| 1 | 163Dy_1 |
| 2 | 158Gd_2 |
| ... | ... |

```{code-cell} python
adata = xd.io.map_channels(adata, f'{DATA_DIR}/barcode_channel_mapping_18ch.csv')
```

<details>
<summary><b>Alternative: Dictionary mapping</b></summary>

If you don't have a CSV file, use a dictionary directly:

```python
mapping = {
    '163Dy_1': 's_1',
    '158Gd_2': 's_2',
    '165Ho_3-Bead3': 's_3',
    # ... all 18 or 27 channels
}
adata = xd.io.map_channels(adata, mapping)
```

</details>

---

## 2. Transform Data

Apply log or arcsinh transformation to stabilize variance and normalize intensities.

| Method | Formula | Recommended for |
|--------|---------|-----------------|
| `log` | log(x + 1) | All methods (recommended) |
| `arcsinh` | arcsinh(x / cofactor) | PreMessa |

```{code-cell} python
adata = xd.preprocessing.transform(adata, method='log')
```

Transformed data is stored in `adata.layers['log']`.

<details>
<summary><b>Alternative: Arcsinh transform</b></summary>

Arcsinh transformation with adjustable cofactor:

```python
adata = xd.preprocessing.transform(adata, method='arcsinh', cofactor=10.0)
# Creates adata.layers['arcsinh_cf10.0']
```

The cofactor controls compression: lower values compress high intensities more.

</details>

### Visualize Channel Distributions

```{code-cell} python
:tags: [wide-plot]

fig = xd.plots.plot_channel_intensities(adata, layer='log')
fig
```

All plots in this library are interactive Plotly figures. Hover over data points for additional information.

Bimodal distributions indicate good separation between ON/OFF states—the typical case for most datasets. Since our channels show clear bimodality, we will use the PC-GMM method (recommended default). In rare cases where channels appear unimodal (e.g., very few barcodes where some channels are all-ON or all-OFF), PreMessa may be more appropriate.

---

## 3. Intensity Filtering

Remove debris (low intensity) and doublets (high intensity) before debarcoding.

### Visualize Cell Distribution

Use `plot_intensity_scatter` to visualize the sum vs variance distribution and preview filtering:

```{code-cell} python
fig = xd.plots.plot_intensity_scatter(adata, layer='log')
fig
```

### Preview Filter Boundaries

Preview filtering with the ellipsoidal method, which uses Mahalanobis distance and handles non-axis-aligned outliers well:

```{code-cell} python
fig = xd.plots.plot_intensity_scatter(adata, layer='log', method='ellipsoidal', percentile=95.0)
fig
```

### Apply Filter

```{code-cell} python
adata = xd.preprocessing.filter_cells_intensity(
    adata,
    layer='log',
    method='ellipsoidal',
    percentile=95.0,           # Keep cells within 95th percentile of Mahalanobis distance
    filter_or_flag='filter'    # Remove cells directly
)

print(f"Remaining cells: {adata.n_obs}")
```

The `filter_or_flag` parameter controls behavior:
- `'filter'`: Removes cells immediately (used here in the standard workflow)
- `'flag'`: Adds boolean column `adata.obs['intensity_pass']` without removing cells, useful for exploratory analysis

<details>
<summary><b>Alternative: Rectangular method</b></summary>

The rectangular method applies independent percentile thresholds on sum and variance. Useful when you want manual control over each axis:

```python
# Preview
fig = xd.plots.plot_intensity_scatter(
    adata, layer='log', method='rectangular',
    sum_low=1.0, sum_high=99.0, var_low=1.0, var_high=99.0
)

# Apply
adata = xd.preprocessing.filter_cells_intensity(
    adata,
    layer='log',
    method='rectangular',
    sum_low=1.0,               # Remove bottom 1% by channel sum
    sum_high=99.0,             # Remove top 1% by channel sum  
    var_low=1.0,               # Remove bottom 1% by variance
    var_high=99.0,             # Remove top 1% by variance
    filter_or_flag='filter'
)
```

</details>

---

## 4. Debarcoding

### Choose a Method

| Method | Best for | Notes |
|--------|----------|-------|
| **PC-GMM** | Bimodal regime | Recommended default; best coverage-accuracy trade-off in most situations |
| **PreMessa** | Unimodal regime | Iterative top-4 selection with per-channel normalization; use when channels lack bimodality |
| **GMM** | Exploration (bimodal regime) | Assignments not constrained to valid patterns; use only for exploring raw signal |
| **Manual** | Custom thresholds | User-defined per-channel thresholds |

Since we observed bimodal channel distributions above, we use PC-GMM (the recommended default):

### Apply Debarcoding

```{code-cell} python
adata = xd.debarcode.debarcode(
    adata,
    method='pc_gmm',
    layer='log'
)
```

<details>
<summary><b>Alternative: Manual thresholds</b></summary>

If you want to set the ON/OFF threshold for each channel yourself (e.g., from visual inspection):

```python
# One threshold per channel
thresholds = [1.5, 1.8, 1.6, 1.7, 1.5, 1.6, 1.8, 1.5, 1.7,   # Block 1
              1.6, 1.5, 1.7, 1.8, 1.6, 1.5, 1.7, 1.6, 1.8]   # Block 2

adata = xd.debarcode.debarcode(
    adata,
    method='manual',
    thresholds=thresholds
)
```

</details>

---

## 5. Inspect Barcode Distribution

### Barcode Rank Histogram

Use `plot_barcode_rank_histogram` to visualize the distribution of cell counts per barcode pattern:

```{code-cell} python
fig = xd.plots.plot_barcode_rank_histogram(adata, assignment_col='pc_gmm_assignment')
fig
```

True barcodes typically have high cell counts, while noisy patterns have low counts.

### Cumulative Barcode Rank

Use `plot_cumul_barcode_rank` to see how cells accumulate across patterns. A steep initial rise followed by a plateau indicates that a small number of real barcode patterns capture most cells, while the long tail represents noise:

```{code-cell} python
fig = xd.plots.plot_cumul_barcode_rank(adata, assignment_col='pc_gmm_assignment')
fig
```

A strong elbow suggests clear separation between real barcodes (high plateau) and junk patterns (tail). In this dataset, the separation is gradual, meaning real barcodes and noise patterns are not well separated by count alone.

---

## 6. Hamming Clustering

Hamming clustering corrects noisy assignments by merging small patterns into nearby valid patterns. The algorithms are inspired by [starcode](https://github.com/gui11aume/starcode).

### Why Hamming Clustering?

- True barcodes have high cell counts
- Noisy assignments create low-count patterns near true barcodes
- If barcode sublibrary << full library, collisions (true barcodes within Hamming radius) are rare

See [Hamming Guidelines](hamming_guidelines.md) for more about X-Code geometry theory. 

### Apply Hamming Clustering

The main method is message-passing (`msg`), which iteratively merges small patterns into dominant neighbors.

For this sample, we expect a sublibrary size between 3000 and 4000 barcodes. With the 15K library (18 channels) and such a large sublibrary, Hamming clustering benefits are negligible and could be skipped entirely. Using too low of a ratio could be harmful in this situation. Here we use an ultra-safe `ratio=25` to illustrate how to use the function. See [Hamming Guidelines](hamming_guidelines.md) for recommended settings in different situations.

```{code-cell} python
adata = xd.postprocessing.hamming_cluster(
    adata,
    assignment_col='pc_gmm_assignment',
    confidence_col='pc_gmm_confidence',
    method='msg',              # Message-passing algorithm
    radius=2,                  # Max Hamming distance for merging
    ratio=25.0,                # Ultra-safe ratio for large sublibrary
    tie_break='lda',           # LDA-based tie-breaking
    layer='log'
)
```

<details>
<summary><b>Alternative: Sphere method</b></summary>

The sphere method finds local maxima as cluster centers, then assigns all patterns within the radius to those centers:

```python
adata = xd.postprocessing.hamming_cluster(
    adata,
    assignment_col='pc_gmm_assignment',
    confidence_col='pc_gmm_confidence',
    method='sphere',
    radius=2,
    min_count_center=100       # Only patterns with 100+ cells become centers
)
```

</details>

### Barcode Rank After Clustering

```{code-cell} python
fig = xd.plots.plot_barcode_rank_histogram(adata, assignment_col='pc_gmm_hamming_assignment')
fig
```

Compare with the pre-clustering histogram: noisy low-count patterns should be absorbed into high-count valid patterns.

---

## 7. Filtering

Filtering can be applied either before or after Hamming clustering. Two approaches are available: cell-level confidence filtering and pattern-level filtering.

### Cell Confidence Filtering

Filter individual cells based on their confidence scores. This can be applied pre- or post-Hamming:

```{code-cell} python
adata = xd.postprocessing.filter_cells_conf(
    adata,
    confidence_col='pc_gmm_hamming_confidence',
    method='percentile',
    value=90,                  # Keep top 90% by confidence
    filter_or_flag='flag'
)
```

<details>
<summary><b>Alternative: Fixed threshold</b></summary>

```python
adata = xd.postprocessing.filter_cells_conf(
    adata,
    confidence_col='pc_gmm_hamming_confidence',
    method='threshold',
    value=0.5,                 # Keep cells with confidence >= 0.5
    filter_or_flag='flag'
)
```

</details>

### Pattern Filtering

Filter cells based on pattern-level statistics:

```{code-cell} python
:tags: [skip-execution]

adata = xd.postprocessing.filter_pattern(
    adata,
    assignment_col='pc_gmm_hamming_assignment',
    metric='count',
    method='threshold',
    value=50,                  # Minimum 50 cells per pattern
    filter_or_flag='flag'
)
```

<details>
<summary><b>Alternative: Filter by score (count x median confidence)</b></summary>

```python
adata = xd.postprocessing.filter_pattern(
    adata,
    assignment_col='pc_gmm_hamming_assignment',
    confidence_col='pc_gmm_hamming_confidence',
    metric='score',            # count * median_conf
    method='percentile',
    value=90,
    filter_or_flag='flag'
)
```

</details>

<details>
<summary><b>Recomputing confidence after filtering</b></summary>

If you filter patterns (removing cells), you can recompute Mahalanobis confidence based on updated cluster centroids:

```python
new_conf = xd.postprocessing.mahal_conf(
    adata,
    assignment_col='pc_gmm_assignment',
    layer='log',
    min_cells=5               # Adjust threshold if needed
)
adata.obs['pc_gmm_confidence'] = new_conf
```

</details>

---

## 8. Save Results

```{code-cell} python
:tags: [skip-execution]

xd.io.write_data(adata, 'debarcoded_15K.h5ad')
```

The H5AD file preserves:
- All channels (barcode + phenotypic markers)
- All layers (raw, transformed)
- All metadata (assignments, confidence, filtering flags)
- Processing parameters in `adata.uns`

<details>
<summary><b>Accessing results</b></summary>

After debarcoding, key results are in:

```python
# Assignments and confidence
adata.obs['pc_gmm_assignment']           # Barcode pattern strings
adata.obs['pc_gmm_confidence']           # Mahalanobis confidence [0, 1]

# After Hamming clustering
adata.obs['pc_gmm_hamming_assignment']   # Corrected assignments
adata.obs['pc_gmm_hamming_confidence']   # Mahalanobis confidence (recomputed)
adata.obs['pc_gmm_hamming_remapped']     # Boolean: was cell remapped?

# Filtering flags
adata.obs['intensity_pass']              # Intensity filter
adata.obs['pc_gmm_confidence_pass']      # Confidence filter
adata.obs['pc_gmm_hamming_confidence_pass'] # Confidence filter (after Hamming)

# Method parameters
adata.uns['debarcoding']['pc_gmm']       # Debarcoding parameters
adata.uns['hamming_clustering']          # Hamming parameters
```

</details>

---

## Alternative: Pipeline Function

For standard workflows, use the integrated pipeline:

```{code-cell} python
:tags: [skip-execution]

adata = xd.io.read_data(f'{DATA_DIR}/15K_test.fcs')
adata = xd.io.map_channels(adata, f'{DATA_DIR}/barcode_channel_mapping_18ch.csv')

adata = xd.debarcode.debarcoding_pipeline(
    adata,
    method='pc_gmm',
    transform_method='log',
    # Intensity filtering
    apply_intensity_filter=True,
    intensity_method='ellipsoidal',
    # Hamming clustering
    apply_hamming=True,
    hamming_ratio=25.0,
    # Confidence filtering  
    apply_confidence_filter=True,
    confidence_filter_method='percentile',
    confidence_value=90
)
```

The pipeline runs: Transform -> Intensity Filter -> Debarcode -> Hamming -> Confidence Filter

<details>
<summary><b>When to use manual steps vs pipeline</b></summary>

**Use the pipeline when:**
- Standard workflow with default parameters
- Batch processing multiple files
- Quick exploratory analysis

**Use manual steps when:**
- Custom parameters at each stage
- Visual inspection between steps
- Non-standard workflows
- Debugging issues

</details>

---

## Summary

| Step | Function | Key outputs |
|------|----------|-------------|
| Load | `io.read_data()` | `adata` |
| Map | `io.map_channels()` | `adata.uns['barcode_channels']` |
| Transform | `preprocessing.transform()` | `adata.layers['log']` |
| Intensity filter | `preprocessing.filter_cells_intensity()` | `adata.obs['intensity_pass']` |
| Debarcode | `debarcode.debarcode()` | `adata.obs['{method}_assignment']` |
| Hamming | `postprocessing.hamming_cluster()` | `adata.obs['{method}_hamming_assignment']` |
| Confidence filter | `postprocessing.filter_cells_conf()` | `adata.obs['{method}_confidence_pass']` |
| Save | `io.write_data()` | `.h5ad` file |

---

## Next Steps

- [Method Comparison](method_comparison.md): detailed comparison of debarcoding methods
- [Hamming Guidelines](hamming_guidelines.md): parameter tuning for Hamming clustering
- [Simulation](simulation.md): generate synthetic data for testing
