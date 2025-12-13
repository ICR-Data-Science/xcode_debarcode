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

# Method Comparison

xcode_debarcode provides three debarcoding methods, each suited to different data regimes. This guide explains how each method works and provides performance benchmarks.

## Available Methods

### PC-GMM (Pattern-Constrained GMM)

**Recommended for most datasets (bimodal regime).**

PC-GMM fits a 2-component Gaussian Mixture Model to each barcode channel independently, then enforces valid 4-of-9 pattern constraints when assigning cells to barcodes.

**How it works:**

1. For each channel, fit a 2-component GMM to estimate ON and OFF distributions
2. For each cell, compute the log-likelihood of each channel being ON or OFF
3. For each 9-channel block, enumerate all 126 valid 4-of-9 patterns
4. Compute the total log-likelihood for each valid pattern
5. Select the maximum likelihood valid pattern for each block
6. Confidence = product of per-block posteriors (probability of selected pattern)

**Characteristics:**

- Always produces valid patterns (exactly 4 ON channels per block)
- Best coverage-accuracy trade-off in bimodal regime
- Confidence is the native posterior probability from the GMM

```python
adata = xd.debarcode.debarcode(adata, method='pc_gmm', layer='log')
# adata.obs['pc_gmm_confidence'] = product of per-block posteriors
```

### PreMessa

**Recommended for unimodal regime (rare edge case).**

PreMessa selects the top-4 highest intensity channels per block, with iterative per-channel normalization.

**How it works:**

1. For each 9-channel block, identify the 4 channels with highest intensity
2. Compute the separation score: difference between 4th and 5th highest intensities
3. Iteratively normalize each channel by the 95th percentile of its ON population
4. Repeat top-4 selection on normalized data
5. Confidence = minimum separation delta across blocks

**Characteristics:**

- Always produces valid patterns (top-4 selection guarantees 4 ON)
- Does not require bimodal distributions
- Less precise than PC-GMM when channels are bimodal

```python
adata = xd.debarcode.debarcode(adata, method='premessa', layer='log')
# adata.obs['premessa_confidence'] = min separation delta across blocks
```

### GMM

**For exploration only (assignments not constrained to valid patterns).**

GMM fits a 2-component Gaussian Mixture Model to each channel independently and classifies each channel as ON or OFF based on posterior probability.

**How it works:**

1. For each channel, fit a 2-component GMM
2. Classify channel as ON if posterior >= 0.5, OFF otherwise
3. Combine all channel calls to form the barcode pattern
4. Confidence = product of per-channel posteriors

**Characteristics:**

- Assignments not constrained to valid patterns (can produce patterns with != 4 ON per block)
- Lower effective coverage because invalid patterns cannot be used downstream

```python
adata = xd.debarcode.debarcode(adata, method='gmm', layer='log')
# adata.obs['gmm_confidence'] = product of per-channel posteriors
```

---

## Performance Comparison

We evaluated method performance on simulated datasets with known ground truth across varying sublibrary sizes, channel configurations (18ch and 27ch), and noise levels.

### Coverage-Accuracy Trade-off

Coverage-accuracy curves show accuracy (fraction of correctly assigned cells among those included) as a function of coverage (fraction of cells included). Cells are ranked by `{method}_confidence` (the raw score from each debarcoding method) and the top N% are selected.

```{figure} ../_static/figures/fig_method_comparison.png
:align: center
:width: 100%

Method comparison in bimodal regime (n_barcodes > 30). Left panels show 18-channel simulations; right panels show 27-channel simulations. **a1, a2)** Coverage-accuracy curves averaged across sublibrary sizes; cells ranked by raw confidence score. Solid lines show baseline, dashed lines show results after Hamming clustering. **b1, b2)** Percentage of true barcodes detected as a function of sublibrary size. **c1, c2)** Spearman correlation between predicted and true barcode counts. **d1, d2)** Accuracy at 80% coverage.
```

PC-GMM consistently outperforms PreMessa across all metrics in the bimodal regime.

### Unimodal Regime

The unimodal regime occurs when some channels lack either an ON or OFF population entirely. This happens with very small sublibraries.

```{figure} ../_static/figures/fig_method_uni_comparison.png
:align: center
:width: 100%

Method comparison in unimodal regime (n_barcodes <= 10). PreMessa (red) outperforms PC-GMM (blue) when sublibrary sizes are very small and some channels lack bimodal distributions.
```

PreMessa outperforms PC-GMM in the unimodal regime because PC-GMM struggles when GMM fitting fails on channels that lack bimodal structure.

---

## Method Selection Guidelines

### Quick Decision

1. **Visualize channel distributions** using `xd.plots.plot_channel_intensities()`
2. **If all channels show bimodal distributions**: use **PC-GMM**
3. **If some channels appear unimodal**: use **PreMessa**

### Summary Table

| Regime | Typical Sublibrary Size | Recommended Method | Rationale |
|--------|-------------------------|-------------------|-----------|
| Bimodal | >= 20 barcodes | PC-GMM | Best coverage-accuracy trade-off |
| Unimodal | < 10 barcodes | PreMessa | Works without bimodal structure |
| Exploration | Any | GMM | Examine raw signal without validity constraints |

---

## Summary

1. **PC-GMM is recommended for most datasets** where channels show bimodal ON/OFF distributions
2. **PreMessa is preferred for unimodal regimes** (very small sublibraries)
3. **GMM is for exploration only** since unconstrained assignments limit coverage flexibility
4. **Always inspect channel distributions** before choosing a method
5. **Coverage-accuracy curves** are computed by ranking cells by `{method}_confidence` (raw score) and selecting top N%
6. **Hamming clustering improves all methods** but cannot compensate for fundamentally mismatched method choice
