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

# Simulation

The simulation module generates synthetic CyTOF barcode data with known ground truth, enabling method validation, performance benchmarking, and parameter optimization.

## Generative Model

The simulation generates realistic CyTOF barcode intensities by modeling each channel as a mixture of ON and OFF populations.

### Data Generation Process

1. **Pattern selection**: Randomly select `n_barcodes` valid 4-of-9 patterns from the full library
2. **Cell assignment**: Assign cells to patterns according to a Dirichlet distribution with concentration parameter `alpha`
3. **Intensity generation**: For each cell and channel, sample from the appropriate Gaussian distribution:
   - If channel is ON: sample from $\mathcal{N}(\mu_{\text{on}}, \sigma_{\text{on}})$
   - If channel is OFF: sample from $\mathcal{N}(\mu_{\text{off}}, \sigma_{\text{off}})$
4. **Transform to raw scale**: Apply exponential transformation $\max(0, e^x - 1)$ to produce realistic CyTOF intensities

### Channel Parameters

Each channel has four parameters controlling its ON and OFF distributions:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu_off` | ~1.0-1.8 | Mean of OFF distribution (log scale) |
| `sigma_off` | 0.5 | Standard deviation of OFF distribution |
| `mu_on` | ~2.0-3.0 | Mean of ON distribution (log scale) |
| `sigma_on` | 0.35 | Standard deviation of ON distribution |

**Parameter generation:**

- $\mu_{\text{off}} \sim \text{Uniform}(0.7, 1.8)$
- $\text{base_separation}$ is computed from `target_error_rate` (default 0.05)
- $\text{separation} = \text{base_separation} \times \text{Uniform}(0.7, 1.3)$
- $\mu_{\text{on}} = \min(\mu_{\text{off}} + \text{separation}, 3.0)$
- $\sigma_{\text{off}} = 0.5$, $\sigma_{\text{on}} = 0.35$ (fixed)

The ±30% multiplicative variation on separation introduces realistic differences between channels.

The separation between ON and OFF modes determines how easy or difficult debarcoding will be. Larger separation produces cleaner bimodal distributions; smaller separation creates more overlap and higher error rates.

---

## Key Parameters

### Population Distribution (alpha)

The `alpha` parameter controls how cells are distributed across barcodes using a Dirichlet distribution. Lower values create sparse distributions where a few barcodes dominate; higher values create more uniform distributions.

| alpha | Top barcodes for 50% of cells | Top barcodes for 80% of cells | Top barcodes for 95% of cells |
|-------|-------------------------------|-------------------------------|-------------------------------|
| 0.2 | 6% of barcodes | 17% of barcodes | 33% of barcodes |
| 0.5 | 12% of barcodes | 31% of barcodes | 55% of barcodes |
| 1.0 | 18% of barcodes | 43% of barcodes | 69% of barcodes |
| 2.0 | 25% of barcodes | 54% of barcodes | 79% of barcodes |
| 5.0 | 32% of barcodes | 63% of barcodes | 86% of barcodes |

With `alpha=0.2`, half of all cells come from just 6% of the barcodes, creating a highly skewed distribution. With `alpha=5.0`, the distribution is much more uniform, requiring 32% of barcodes to account for half the cells.

```python
# Sparse distribution (few dominant barcodes)
adata = xd.simulate.simulate_cytof_barcodes(n_barcodes=50, alpha=0.2)

# Uniform distribution (balanced across barcodes)
adata = xd.simulate.simulate_cytof_barcodes(n_barcodes=50, alpha=5.0)
```

### Channel Error Rate (target_error_rate)

The `target_error_rate` parameter (default 0.05) controls the expected per-channel classification error by adjusting the separation between ON and OFF distributions. This is the probability that a sample falls on the wrong side of the optimal GMM decision boundary.

```python
# Easy debarcoding (low error, high separation)
adata = xd.simulate.simulate_cytof_barcodes(target_error_rate=0.02)

# Default difficulty (5% error)
adata = xd.simulate.simulate_cytof_barcodes()  # target_error_rate=0.05

# Challenging debarcoding (high error, low separation)
adata = xd.simulate.simulate_cytof_barcodes(target_error_rate=0.12)
```

The effect on channel distributions can be visualized with `plot_channel_intensities()`:

```{code-cell} python
import warnings
warnings.filterwarnings('ignore')
import xcode_debarcode as xd
import plotly.io as pio
pio.renderers.default = 'notebook'

# Low error rate (2%) - clear separation
adata_easy = xd.simulate.simulate_cytof_barcodes(
    n_cells=10000, n_channels=18, n_barcodes=50,
    target_error_rate=0.02, random_seed=42, verbose=False
)
adata_easy = xd.preprocessing.transform(adata_easy, 'log', verbose=False)

fig = xd.plots.plot_channel_intensities(adata_easy, channels=['s_1', 's_2', 's_3'], layer='log', log_scale_x=False)
fig.update_layout(title='Low Error Rate (2%): Clear ON/OFF Separation', height=300)
fig
```

```{code-cell} python
# High error rate (12%) - overlapping distributions
adata_hard = xd.simulate.simulate_cytof_barcodes(
    n_cells=10000, n_channels=18, n_barcodes=50,
    target_error_rate=0.12, random_seed=42, verbose=False
)
adata_hard = xd.preprocessing.transform(adata_hard, 'log', verbose=False)

fig = xd.plots.plot_channel_intensities(adata_hard, channels=['s_1', 's_2', 's_3'], layer='log', log_scale_x=False)
fig.update_layout(title='High Error Rate (12%): Overlapping ON/OFF Distributions', height=300)
fig
```

Lower error rates produce clearly bimodal distributions that are easy to debarcode. Higher error rates create overlapping distributions where the boundary between ON and OFF states is ambiguous, leading to more misclassifications.

### Unbarcoded Cells (unbarcoded_fraction)

The `unbarcoded_fraction` parameter adds cells that lack a barcode, simulating debris or failed staining. Unbarcoded cells are generated with background-level intensities across all channels:

$$x_g \sim \mathcal{N}(0, \sigma_{\text{off},g})$$

With mean 0 in log scale, the exponential transformation produces raw intensities around $e^0 - 1 = 0$, representing true background noise. This ensures unbarcoded cells have consistently low signal and cannot be confused with barcoded cells.

```python
# 10% unbarcoded cells
adata = xd.simulate.simulate_cytof_barcodes(
    n_cells=10000,
    n_barcodes=50,
    unbarcoded_fraction=0.10
)
```

---

## Ground Truth Storage

Simulation results include complete ground truth information stored in the AnnData object:

### Cell-level metadata (adata.obs)

| Column | Description |
|--------|-------------|
| `ground_truth_pattern` | Binary pattern string (e.g., "111100001111000010000000100") |
| `ground_truth_barcode` | Barcode label (e.g., "BC_1-2-3-4-10-11-12-13-19") |
| `is_barcoded` | Boolean indicating whether the cell has a barcode |

### Simulation metadata (adata.uns['simulation'])

| Key | Description |
|-----|-------------|
| `n_cells` | Total cells simulated |
| `n_channels` | Number of barcode channels |
| `n_barcodes` | Number of unique barcodes |
| `alpha` | Dirichlet concentration parameter |
| `target_error_rate` | Target per-channel error rate |
| `mean_actual_error_rate` | Computed error rate from generated parameters |
| `channel_params` | Per-channel distribution parameters |
| `barcode_statistics` | Count statistics for each barcode |

---

## Basic Usage

```{code-cell} python
import xcode_debarcode as xd

# Generate simulated data
adata = xd.simulate.simulate_cytof_barcodes(
    n_cells=10000,
    n_channels=27,
    n_barcodes=50,
    alpha=1.0,
    random_seed=42
)
```

```{code-cell} python
# Standard debarcoding workflow
adata = xd.preprocessing.transform(adata, 'log', verbose=False)
adata = xd.debarcode.debarcode(adata, method='pc_gmm', layer='log', verbose=False)
```

---

## Analyzing Results

The `analyze_simulation()` function compares debarcoding results against ground truth:

```{code-cell} python
results = xd.simulate.analyze_simulation(adata, assignment_col='pc_gmm_assignment')
```

### Metrics

| Metric | Definition |
|--------|------------|
| **Accuracy** | Fraction of valid assignments that match ground truth |
| **Coverage** | Fraction of cells with valid pattern assignments |
| **F1 Score** | Harmonic mean of accuracy and coverage |

The function returns a dictionary with detailed breakdown including counts of correct assignments for barcoded vs unbarcoded cells.

---

## Summary

1. **Generative model**: Gaussian mixture per channel with exponential transformation to raw scale
2. **alpha** controls population skewness: lower values create sparse distributions, higher values create uniform distributions
3. **target_error_rate** (default 0.05) controls difficulty: lower values create clear separation, higher values create overlapping distributions
4. **Ground truth** is stored in `adata.obs` (patterns, labels) and `adata.uns['simulation']` (parameters, statistics)
5. **analyze_simulation()** computes accuracy, coverage, and F1 against ground truth
