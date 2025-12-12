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

xcode_debarcode provides three debarcoding methods, each suited to different data regimes. This guide explains how each method works and provides performance benchmarks to help you choose the right method for your dataset.

## Available Methods

### PC-GMM (Pattern-Constrained GMM)

**Recommended for most datasets (bimodal regime).**

PC-GMM fits a 2-component Gaussian Mixture Model to each barcode channel independently, then enforces valid 4-of-9 pattern constraints when assigning cells to barcodes.

**How it works:**

1. For each channel, fit a 2-component GMM to estimate ON and OFF distributions (means, variances, weights)
2. For each cell, compute the log-likelihood of each channel being ON or OFF given the fitted GMM parameters
3. For each 9-channel block, enumerate all 126 valid 4-of-9 patterns
4. Compute the total log-likelihood for each valid pattern by summing channel log-likelihoods
5. Select the maximum likelihood valid pattern for each block
6. Compute Mahalanobis confidence based on distance to cluster centroids (raw per-block posteriors optionally saved)

**Characteristics:**

- Always produces valid patterns (exactly 4 ON channels per block)
- Best coverage-accuracy trade-off in bimodal regime
- Requires bimodal channel distributions for reliable GMM fitting

```python
adata = xd.debarcode.debarcode(adata, method='pc_gmm', layer='log')
```

### PreMessa

**Recommended for unimodal regime (rare edge case).**

PreMessa selects the top-4 highest intensity channels per block, with iterative per-channel normalization to handle intensity variation. The method is adapted from the [PreMessa R package](https://github.com/ParkerICI/premessa) for X-Code barcode design.

**How it works:**

1. For each 9-channel block, identify the 4 channels with highest intensity
2. Compute the separation score: difference between the 4th and 5th highest intensities
3. Iteratively normalize each channel by the 95th percentile of its ON population
4. Repeat top-4 selection on normalized data
5. Compute Mahalanobis confidence based on distance to cluster centroids (raw separation scores optionally saved)

**Characteristics:**

- Always produces valid patterns (top-4 selection guarantees 4 ON channels)
- Does not require bimodal distributions (works when some channels lack both ON and OFF populations)
- Less precise than PC-GMM when channels are bimodal

```python
adata = xd.debarcode.debarcode(adata, method='premessa', layer='log')
```

### GMM

**For exploration only (assignments not constrained to valid patterns).**

GMM fits a 2-component Gaussian Mixture Model to each channel independently and classifies each channel as ON or OFF based on posterior probability.

**How it works:**

1. For each channel, fit a 2-component GMM
2. For each cell, compute the posterior probability of the ON component
3. Classify channel as ON if posterior >= 0.5, OFF otherwise
4. Combine all channel calls to form the barcode pattern
5. Compute Mahalanobis confidence based on distance to cluster centroids (raw per-channel posteriors optionally saved)

**Characteristics:**

- Assignments not constrained to valid patterns (can produce patterns with != 4 ON channels per block)
- Good accuracy when channels are well-separated
- Lower effective coverage because invalid patterns cannot be used downstream

Because GMM does not enforce valid patterns, many cells receive invalid assignments. This makes the coverage-accuracy trade-off less flexible for the user to control via confidence filtering. Use GMM only for exploring the raw per-channel signal without forcing validity constraints on the CyTOF intensities.

```python
adata = xd.debarcode.debarcode(adata, method='gmm', layer='log')
```

---

## Performance Comparison

We evaluated method performance on simulated datasets with known ground truth across varying sublibrary sizes, channel configurations (18ch and 27ch), and noise levels.

### Bimodal Regime

The bimodal regime is the typical case where each barcode channel has distinct ON and OFF populations. This occurs when the sublibrary contains enough barcodes that every channel has some barcodes with it ON and some with it OFF.

**Simulation setup:** Sublibraries with more than 30 barcodes, where all channels exhibit bimodal intensity distributions. Hamming clustering applied with ratio=20 (18ch) and ratio=2 (27ch).

```{figure} ../_static/figures/fig_method_comparison.png
:align: center
:width: 100%

Method comparison in bimodal regime (n_barcodes > 30). Left panels show 18-channel simulations; right panels show 27-channel simulations. **a1, a2)** Coverage-accuracy curves averaged across sublibrary sizes; solid lines show baseline, dashed lines show results after Hamming clustering. **b1, b2)** Percentage of true barcodes detected as a function of sublibrary size. **c1, c2)** Spearman correlation between predicted and true barcode counts. **d1, d2)** Accuracy at 80% coverage; solid bars show baseline performance, hatched bars show improvement from Hamming clustering.
```

PC-GMM consistently outperforms PreMessa across all metrics in the bimodal regime. The coverage-accuracy curves show that PC-GMM achieves higher accuracy at any given coverage level. Both methods detect nearly all true barcodes and achieve high Spearman correlation with true counts.

The 27-channel configuration shows larger absolute accuracy values due to more dimensions to distinguish barcodes. Hamming clustering improves both methods, but PC-GMM maintains its lead throughout.

**Conclusion:** In the bimodal regime (the common case), PC-GMM provides the best coverage-accuracy trade-off.

### Unimodal Regime

The unimodal regime occurs when some channels lack either an ON or OFF population entirely. This happens with very small sublibraries where, by chance, certain channels may be always ON or always OFF across all barcodes in the sublibrary.

<details>
<summary><b>Expected Number of Bimodal Channels</b></summary>

<br>

For a randomly selected sublibrary of size S from the full X-Code library, we can compute the expected number of channels that have both ON and OFF populations (i.e., bimodal channels).

**Derivation of N_on and N_off:**

Let k = number of 4-of-9 blocks (k=2 for 18ch, k=3 for 27ch). The total number of valid barcodes is $L_k = 126^k$, and there are 9k total channels.

Pick one specific channel g; by symmetry all channels behave the same. The channel lives in some block. To count how many valid barcodes have that channel ON vs OFF:

- If that channel is ON, we need to choose 3 other ONs out of the remaining 8 channels in the block: $\binom{8}{3} = 56$ patterns
- If that channel is OFF, we choose 4 ONs among the other 8 channels: $\binom{8}{4} = 70$ patterns
- The other k-1 blocks can be any valid 4-of-9 pattern: $126^{k-1}$ possibilities each

So globally:

$$N_{\text{on}} = 56 \cdot 126^{k-1}, \quad N_{\text{off}} = 70 \cdot 126^{k-1}$$

Note that $L_k = N_{\text{on}} + N_{\text{off}} = 126^k$ as expected.

**Expected bimodal channels:**

The probability that a channel has both ON and OFF populations in a random sublibrary of size S is:

$$
\mathbb{P}(A_g) = 1 - \frac{\binom{N_{\text{on}}}{S}}{\binom{L_k}{S}} - \frac{\binom{N_{\text{off}}}{S}}{\binom{L_k}{S}}
$$

By linearity of expectation, the expected number of bimodal channels is:

$$
\mathbb{E}[X] = 9k \cdot \mathbb{P}(A_g)
$$

| Sublibrary Size (S) | 18ch (k=2) | 27ch (k=3) |
|---------------------|------------|------------|
| 1 | 0.00 | 0.00 |
| 2 | 8.89 | 13.33 |
| 3 | 13.33 | 20.00 |
| 5 | 16.74 | 25.10 |
| 10 | 17.94 | 26.92 |
| 15 | 17.99 | 26.99 |
| 20 | 18.00 | 27.00 |

**Interpretation:** With fewer than ~10 barcodes, a mix of unimodal and bimodal channels. Above ~10 barcodes, virtually all channels are bimodal.

</details>

<br>

**Simulation setup:** Sublibraries with 10 or fewer barcodes, representing the edge case where unimodal channels are expected. Hamming clustering applied with ratio=20 (18ch) and ratio=2 (27ch).

```{figure} ../_static/figures/fig_method_uni_comparison.png
:align: center
:width: 100%

Method comparison in unimodal regime (n_barcodes <= 10). Left panel shows 18-channel simulations; right panel shows 27-channel simulations. **a1, a2)** Accuracy at 80% coverage. PreMessa (red) outperforms PC-GMM (blue) when sublibrary sizes are very small and some channels lack bimodal distributions.
```

PreMessa outperforms PC-GMM in the unimodal regime, with the advantage most pronounced at very small sublibrary sizes (1-5 barcodes). This occurs because PC-GMM struggles when GMM fitting fails on channels that lack bimodal structure, while PreMessa's top-4 selection approach does not require bimodal distributions to function correctly.

**Conclusion:** In the rare unimodal regime (very small sublibraries), PreMessa provides better accuracy.

### GMM Specificities

GMM provides good per-channel classification but does not constrain assignments to valid patterns.

**Simulation setup:** Sublibraries with more than 10 barcodes. Hamming clustering applied with ratio=20 (18ch) and ratio=2 (27ch).

```{figure} ../_static/figures/fig_gmm_comparison.png
:align: center
:width: 100%

GMM method performance (n_barcodes > 10). Left panels show 18-channel simulations; right panels show 27-channel simulations. **a1, a2)** Accuracy: overall accuracy (correctly assigned cells / total cells). **b1, b2)** Coverage: fraction of cells assigned to valid patterns. Solid lines show baseline; dashed lines show results after Hamming clustering.
```

GMM achieves reasonable accuracy on cells with valid pattern assignments, but coverage is inherently limited because many cells receive invalid patterns. Hamming clustering helps by remapping invalid patterns to valid neighbors, but coverage remains lower than PC-GMM or PreMessa. This coverage limitation makes GMM less flexible for tuning the coverage-accuracy trade-off via confidence filtering.

**Conclusion:** While GMM produces reasonable per-channel classifications, its lack of validity constraints results in lower effective coverage. Use GMM only for exploration or when you specifically want to examine the raw per-channel signal without forcing valid pattern assignments.

---

## Method Selection Guidelines

### Quick Decision

1. **Visualize channel distributions** using `xd.plots.plot_channel_intensities()`
2. **If all channels show bimodal distributions** (distinct ON/OFF peaks): use **PC-GMM**
3. **If some channels appear unimodal** (single peak, typically with very small sublibraries): use **PreMessa**

### Summary Table

| Regime | Typical Sublibrary Size | Recommended Method | Rationale |
|--------|-------------------------|-------------------|-----------|
| Bimodal | >= 20 barcodes | PC-GMM | Best coverage-accuracy trade-off |
| Unimodal | < 10 barcodes | PreMessa | Works without bimodal structure |
| Exploration | Any | GMM | Examine raw signal without validity constraints |

---

## Common Workflow

For step-by-step usage examples including transformation, intensity filtering, debarcoding, Hamming clustering, and confidence filtering, see the [Basic Usage](basic_usage.md) tutorial.

---

## Summary

1. **PC-GMM is recommended for most datasets** where channels show bimodal ON/OFF distributions
2. **PreMessa is preferred for unimodal regimes** (very small sublibraries where some channels lack bimodal structure)
3. **GMM is for exploration only** since unconstrained assignments limit coverage flexibility
4. **Always inspect channel distributions** before choosing a method
5. **Hamming clustering improves all methods** but cannot compensate for fundamentally mismatched method choice
