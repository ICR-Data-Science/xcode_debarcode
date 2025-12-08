"""Simulation of synthetic CyTOF barcode data for testing and benchmarking."""
import numpy as np
import anndata as ad
from typing import Optional, Dict
from collections import Counter
from .barcode import _generate_4_of_9_patterns

__all__ = ["simulate_cytof_barcodes", "analyze_simulation"]


def _generate_channel_parameters(n_channels: int, rng: np.random.RandomState) -> Dict:
    """Generate random channel parameters for simulation."""
    params = {}
    for g in range(n_channels):
        mu_off = rng.uniform(0.7, 1.8)
        separation = rng.uniform(1.0, 1.8)
        params[f's_{g+1}'] = {
            'mu_off': mu_off,
            'sigma_off': 0.5,
            'mu_on': min(mu_off + separation, 3.0),
            'sigma_on': 0.35
        }
    return params


def simulate_cytof_barcodes(
    n_cells: int = 10000,
    n_channels: int = 27,
    n_barcodes: Optional[int] = None,
    channel_params: Optional[Dict] = None,
    unbarcoded_fraction: float = 0.00,
    alpha: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> ad.AnnData:
    """Simulate synthetic CyTOF barcode data.
    
    Generates CyTOF barcode data with known ground truth for testing
    and benchmarking debarcoding methods.
    
    Parameters:
    -----------
    n_cells : int
        Total number of cells to simulate (default: 10000)
    n_channels : int
        Number of barcode channels (default: 27)
    n_barcodes : int, optional
        Number of unique barcodes to use
    channel_params : dict, optional
        Custom channel parameters
    unbarcoded_fraction : float
        Fraction of cells without barcode (default: 0.0)
    alpha : float
        Dirichlet concentration parameter (default: 1.0)
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress messages (default: True)
    
    Returns:
    --------
    adata : AnnData
        Simulated raw intensity data with ground truth
    """
    if n_channels % 9 != 0:
        raise ValueError(f"n_channels must be multiple of 9, got {n_channels}")
    
    if not 0 <= unbarcoded_fraction < 1:
        raise ValueError(f"unbarcoded_fraction must be in [0, 1), got {unbarcoded_fraction}")
    
    rng = np.random.RandomState(random_seed)
    
    if verbose:
        print("="*80)
        print("SIMULATING CYTOF BARCODE DATA")
        print("="*80)
        print(f"\nCells: {n_cells:,}, channels: {n_channels}, Blocks: {n_channels // 9}")
        print(f"Unbarcoded: {unbarcoded_fraction:.1%}, Alpha: {alpha}, Seed: {random_seed}")
    
    channel_params = channel_params or _generate_channel_parameters(n_channels, rng)
    if len(channel_params) != n_channels:
        raise ValueError(f"channel_params must have {n_channels} channels, got {len(channel_params)}")
    
    n_blocks = n_channels // 9
    all_patterns = _generate_4_of_9_patterns(n_blocks=n_blocks)
    n_total_patterns = len(all_patterns)
    
    if n_barcodes is None:
        n_barcodes = rng.randint(min(10, n_total_patterns), min(500, n_total_patterns) + 1)
    elif n_barcodes > n_total_patterns:
        raise ValueError(f"n_barcodes ({n_barcodes}) exceeds valid patterns ({n_total_patterns})")
    
    if verbose:
        print(f"\nValid patterns: {n_total_patterns:,}, Using: {n_barcodes}")
    
    selected_indices = rng.choice(n_total_patterns, size=n_barcodes, replace=False)
    selected_patterns = all_patterns[selected_indices]
    
    pattern_probs = rng.dirichlet(alpha * np.ones(n_barcodes))
    
    n_unbarcoded = int(n_cells * unbarcoded_fraction)
    n_barcoded = n_cells - n_unbarcoded
    
    if verbose:
        print(f"Barcoded: {n_barcoded:,}, Unbarcoded: {n_unbarcoded:,}")
    
    cell_pattern_indices = rng.choice(n_barcodes, size=n_barcoded, p=pattern_probs)
    cell_patterns = selected_patterns[cell_pattern_indices]
    
    if verbose:
        unique_used = len(np.unique(cell_pattern_indices))
        print(f"Unique barcodes used: {unique_used}/{n_barcodes}")
    
    X = np.zeros((n_cells, n_channels))
    
    for i in range(n_barcoded):
        pattern = cell_patterns[i]
        for g in range(n_channels):
            params = channel_params[f's_{g+1}']
            mu = params['mu_on'] if pattern[g] == 1 else params['mu_off']
            sigma = params['sigma_on'] if pattern[g] == 1 else params['sigma_off']
            X[i, g] = rng.normal(mu, sigma)
    
    for i in range(n_barcoded, n_cells):
        for g in range(n_channels):
            params = channel_params[f's_{g+1}']
            X[i, g] = rng.normal(params['mu_off'], params['sigma_off'] * 1.5)
    
    X = np.maximum(np.exp(X) - 1, 0)
    
    if verbose:
        print(f"\nIntensity: [{X.min():.2f}, {X.max():.2f}], mean: {X.mean():.2f}")
    
    ground_truth_patterns = []
    ground_truth_labels = []
    is_barcoded = np.zeros(n_cells, dtype=bool)
    
    for i in range(n_cells):
        if i < n_barcoded:
            pattern = cell_patterns[i]
            pattern_str = "".join(map(str, pattern))
            on_indices = [j+1 for j, val in enumerate(pattern) if val == 1]
            ground_truth_patterns.append(pattern_str)
            ground_truth_labels.append(f"BC_{'-'.join(map(str, on_indices))}")
            is_barcoded[i] = True
        else:
            ground_truth_patterns.append("unbarcoded")
            ground_truth_labels.append("unbarcoded")
    
    barcode_channels = [f's_{i+1}' for i in range(n_channels)]
    
    adata = ad.AnnData(
        X=X,
        obs={
            'ground_truth_pattern': ground_truth_patterns,
            'ground_truth_barcode': ground_truth_labels,
            'is_barcoded': is_barcoded
        },
        var={'channel_name': barcode_channels}
    )
    adata.var_names = barcode_channels
    
    adata.uns['barcode_channels'] = barcode_channels
    adata.uns['channel_mapping'] = {
        'source': 'simulation',
        'n_barcode_channels': len(barcode_channels),
        'n_other_channels': 0,
        'barcode_channels': barcode_channels,
        'other_channels': []
    }
    
    barcode_counts = Counter(ground_truth_labels)
    barcode_counts.pop('unbarcoded', None)
    
    adata.uns['simulation'] = {
        'n_cells': n_cells,
        'n_channels': n_channels,
        'n_blocks': n_blocks,
        'n_total_valid_patterns': n_total_patterns,
        'n_barcodes': n_barcodes,
        'n_unique_barcodes_used': len(barcode_counts),
        'n_barcoded': n_barcoded,
        'n_unbarcoded': n_unbarcoded,
        'unbarcoded_fraction': unbarcoded_fraction,
        'alpha': alpha,
        'random_seed': random_seed,
        'channel_params': channel_params,
        'selected_pattern_indices': selected_indices.tolist(),
        'barcode_statistics': {
            'n_unique_used': len(barcode_counts),
            'top_10_barcodes': dict(barcode_counts.most_common(10)),
            'min_count': min(barcode_counts.values()) if barcode_counts else 0,
            'max_count': max(barcode_counts.values()) if barcode_counts else 0,
            'mean_count': np.mean(list(barcode_counts.values())) if barcode_counts else 0
        }
    }
    
    if verbose:
        print(f"\nShape: {adata.shape}")
        print("="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        print("Ground truth in: adata.obs['ground_truth_pattern']")
        print("Use preprocessing.transform() to apply arcsinh/log transformation")
    
    return adata


def analyze_simulation(adata: ad.AnnData,
                      assignment_col: str,
                      verbose: bool = True) -> Dict:
    """Analyze debarcoding results against ground truth.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData with simulation ground truth and debarcoding results
    assignment_col : str
        Column with debarcoding assignments to evaluate
    verbose : bool
        Print statistics (default: True)
    
    Returns:
    --------
    results : dict
        Performance metrics including accuracy, coverage, and F1
    """
    from .barcode import is_valid_pattern
    
    if 'simulation' not in adata.uns:
        raise ValueError("This AnnData does not contain simulation metadata")
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found")
    
    if 'ground_truth_pattern' not in adata.obs.columns:
        raise ValueError("Ground truth not found in adata.obs")
    
    ground_truth = adata.obs['ground_truth_pattern'].values
    predictions = adata.obs[assignment_col].values
    is_barcoded = adata.obs['is_barcoded'].values
    
    n_cells = len(adata)
    n_barcoded = is_barcoded.sum()
    
    valid_mask = np.array([is_valid_pattern(p) for p in predictions])
    n_valid_assigned = valid_mask.sum()
    
    correct_mask = ground_truth == predictions
    n_correct = (correct_mask & valid_mask).sum()
    
    accuracy = n_correct / n_valid_assigned if n_valid_assigned > 0 else 0.0
    coverage = n_valid_assigned / n_cells
    f1 = 2 * accuracy * coverage / (accuracy + coverage) if (accuracy + coverage) > 0 else 0.0
    
    correct_barcoded = (correct_mask & valid_mask & is_barcoded).sum()
    barcoded_with_valid = (valid_mask & is_barcoded).sum()
    unbarcoded_with_valid = (valid_mask & ~is_barcoded).sum()
    
    results = {
        'accuracy': accuracy,
        'coverage': coverage,
        'f1': f1,
        'n_cells': n_cells,
        'n_barcoded': int(n_barcoded),
        'n_unbarcoded': int(n_cells - n_barcoded),
        'n_valid_assigned': int(n_valid_assigned),
        'n_correct': int(n_correct),
        'n_correct_barcoded': int(correct_barcoded),
        'n_barcoded_with_valid': int(barcoded_with_valid),
        'n_unbarcoded_with_valid': int(unbarcoded_with_valid)
    }
    
    if verbose:
        print("="*80)
        print("SIMULATION ANALYSIS")
        print("="*80)
        print(f"\nDataset: {n_cells:,} cells ({n_barcoded:,} barcoded, {n_cells - n_barcoded:,} unbarcoded)")
        print(f"\nAssignments: {n_valid_assigned:,} valid ({100*coverage:.1f}%), {n_correct:,} correct")
        print(f"\nMetrics:")
        print(f"  Accuracy: {accuracy:.2%} (of valid assignments, % correct)")
        print(f"  Coverage: {coverage:.2%} (% cells with valid assignments)")
        print(f"  F1 Score: {f1:.4f}")
        print(f"\nBreakdown:")
        print(f"  Barcoded with valid: {barcoded_with_valid:,}/{n_barcoded:,} ({100*barcoded_with_valid/n_barcoded:.1f}%), {correct_barcoded:,} correct")
        print(f"  Unbarcoded with valid: {unbarcoded_with_valid:,} (false positives)")
        print("="*80)
    
    return results