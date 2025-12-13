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
    -> Inspect Distribution -> Hamming Clustering -> Confidence Filtering -> (Optional) Cohesion -> Save
```

---

## 1. Load and Map Data

### Load Data

```{code-cell} python
import warnings
warnings.filterwarnings('ignore')
import xcode_debarcode as xd
import plotly.io as pio
pio.renderers.default = 'notebook'

DATA_DIR = '../../data'

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

### Visualize Channel Distributions

```{code-cell} python
:tags: [wide-plot]

fig = xd.plots.plot_channel_intensities(adata, layer='log')
fig
```

Bimodal distributions indicate good separation between ON/OFF states. Since our channels show clear bimodality, we will use the PC-GMM method (recommended default).

---

## 3. Intensity Filtering

Remove debris (low intensity) and doublets (high intensity) before debarcoding.

### Visualize Cell Distribution

```{code-cell} python
fig = xd.plots.plot_intensity_scatter(adata, layer='log')
fig
```

### Preview Filter Boundaries

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
    percentile=95.0,
    filter_or_flag='filter'
)

print(f"Remaining cells: {adata.n_obs}")
```

---

## 4. Debarcoding

### Choose a Method

| Method | Best for | Notes |
|--------|----------|-------|
| **PC-GMM** | Bimodal regime | Recommended default; best coverage-accuracy trade-off |
| **PreMessa** | Unimodal regime | For channels lacking bimodality |
| **GMM** | Exploration | Assignments not constrained to valid patterns |

### Apply Debarcoding

```{code-cell} python
adata = xd.debarcode.debarcode(
    adata,
    method='pc_gmm',
    layer='log'
)
```

After debarcoding:
- `adata.obs['pc_gmm_assignment']`: Assigned barcode patterns
- `adata.obs['pc_gmm_confidence']`: Raw method confidence score (product of per-block posteriors for PC-GMM)

The confidence score is the native score produced by the debarcoding method and is used for coverage-accuracy trade-off analysis by ranking cells and selecting the top N%.

---

## 5. Inspect Barcode Distribution

### Barcode Rank Histogram

```{code-cell} python
fig = xd.plots.plot_barcode_rank_histogram(adata, assignment_col='pc_gmm_assignment')
fig
```

True barcodes typically have high cell counts, while noisy patterns have low counts.

### Cumulative Barcode Rank

```{code-cell} python
fig = xd.plots.plot_cumul_barcode_rank(adata, assignment_col='pc_gmm_assignment')
fig
```

A strong elbow suggests clear separation between real barcodes (high plateau) and junk patterns (tail).

---

## 6. Hamming Clustering

Hamming clustering corrects noisy assignments by merging small patterns into nearby valid patterns.

### Why Hamming Clustering?

- True barcodes have high cell counts
- Noisy assignments create low-count patterns near true barcodes
- If barcode sublibrary << full library, collisions are rare

See [Hamming Guidelines](hamming_guidelines.md) for parameter tuning.

### Apply Hamming Clustering

```{code-cell} python
adata = xd.postprocessing.hamming_cluster(
    adata,
    assignment_col='pc_gmm_assignment',
    confidence_col='pc_gmm_confidence',
    method='msg',
    radius=2,
    ratio=25.0,
    tie_break='lda',
    layer='log'
)
```

After Hamming clustering:
- `adata.obs['pc_gmm_hamming_assignment']`: Corrected assignments
- `adata.obs['pc_gmm_hamming_remapped']`: Boolean mask of remapped cells

Note: Confidence values are not modified by Hamming clustering. Continue to use `pc_gmm_confidence` for filtering.

### Barcode Rank After Clustering

```{code-cell} python
fig = xd.plots.plot_barcode_rank_histogram(adata, assignment_col='pc_gmm_hamming_assignment')
fig
```

---

## 7. Confidence Filtering

Filter cells based on their confidence scores to select high-quality assignments:

```{code-cell} python
adata = xd.postprocessing.filter_cells_conf(
    adata,
    confidence_col='pc_gmm_confidence',
    method='percentile',
    value=90,
    filter_or_flag='flag'
)
```

This uses the raw confidence score from debarcoding to keep the top 90% of cells.

---

## 8. Cohesion (Optional)

Cohesion measures how close each cell is to its assigned cluster centroid. This is an optional end-of-workflow QC step to identify outlier cells within clusters.

```{code-cell} python
:tags: [skip-execution]

adata = xd.postprocessing.add_cohesion(
    adata,
    assignment_col='pc_gmm_hamming_assignment',
    layer='log',
    min_cells=5
)
```

After cohesion computation:
- `adata.obs['pc_gmm_hamming_cohesion']`: Cohesion score in [0, 1]
- 1.0 = at centroid, lower = farther from centroid
- 0.0 = unscored (cluster has < min_cells)

Use cohesion for optional post-hoc filtering of cluster outliers:

```{code-cell} python
:tags: [skip-execution]

# Example: filter cells with low cohesion
adata_filtered = adata[adata.obs['pc_gmm_hamming_cohesion'] > 0.3]
```

---

## 9. Save Results

```{code-cell} python
:tags: [skip-execution]

xd.io.write_data(adata, 'debarcoded_15K.h5ad')
```

The H5AD file preserves:
- All channels (barcode + phenotypic markers)
- All layers (raw, transformed)
- All metadata (assignments, confidence, filtering flags)

<details>
<summary><b>Accessing results</b></summary>

After debarcoding, key results are in:

```python
# Assignments and confidence
adata.obs['pc_gmm_assignment']           # Barcode pattern strings
adata.obs['pc_gmm_confidence']           # Raw method confidence

# After Hamming clustering
adata.obs['pc_gmm_hamming_assignment']   # Corrected assignments
adata.obs['pc_gmm_hamming_remapped']     # Was cell remapped?

# Filtering flags
adata.obs['intensity_pass']              # Intensity filter
adata.obs['pc_gmm_pass']                 # Confidence filter

# Optional cohesion (if computed)
adata.obs['pc_gmm_hamming_cohesion']     # Cluster cohesion

# Method parameters
adata.uns['debarcoding']['pc_gmm']       # Debarcoding parameters
adata.uns['hamming_clustering']          # Hamming parameters
```

</details>

```{code-cell} python
:tags: [skip-execution]

xd.io.write_data(adata, 'debarcoded_15K.h5ad')
```

The H5AD file preserves:
- All channels (barcode + phenotypic markers)
- All layers (raw, transformed)
- All metadata (assignments, confidence, filtering flags)

<details>
<summary><b>Accessing results</b></summary>

After debarcoding, key results are in:

```python
# Assignments and confidence
adata.obs['pc_gmm_assignment']           # Barcode pattern strings
adata.obs['pc_gmm_confidence']           # Raw method confidence

# After Hamming clustering
adata.obs['pc_gmm_hamming_assignment']   # Corrected assignments
adata.obs['pc_gmm_hamming_remapped']     # Was cell remapped?

# Optional cohesion (if computed)
adata.obs['pc_gmm_hamming_cohesion']     # Cluster cohesion

# Filtering flags
adata.obs['intensity_pass']              # Intensity filter
adata.obs['pc_gmm_pass']                 # Confidence filter

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
    apply_intensity_filter=True,
    intensity_method='ellipsoidal',
    apply_hamming=True,
    hamming_ratio=25.0,
    apply_cohesion=False,
    apply_confidence_filter=True,
    confidence_filter_method='percentile',
    confidence_value=90
)
```

---

## Summary

| Step | Function | Key outputs |
|------|----------|-------------|
| Load | `io.read_data()` | `adata` |
| Map | `io.map_channels()` | `adata.uns['barcode_channels']` |
| Transform | `preprocessing.transform()` | `adata.layers['log']` |
| Intensity filter | `preprocessing.filter_cells_intensity()` | `adata.obs['intensity_pass']` |
| Debarcode | `debarcode.debarcode()` | `{method}_assignment`, `{method}_confidence` (raw) |
| Hamming | `postprocessing.hamming_cluster()` | `{method}_hamming_assignment`, `{method}_hamming_remapped` |
| Confidence filter | `postprocessing.filter_cells_conf()` | `{method}_pass` |
| Cohesion (optional) | `postprocessing.add_cohesion()` | `{base}_cohesion` |
| Save | `io.write_data()` | `.h5ad` file |

---

## Next Steps

- [Method Comparison](method_comparison.md): detailed comparison of debarcoding methods
- [Hamming Guidelines](hamming_guidelines.md): parameter tuning for Hamming clustering
- [Simulation](simulation.md): generate synthetic data for testing
