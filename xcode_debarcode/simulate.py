"""Simulation of synthetic CyTOF barcode data for testing and benchmarking."""
import numpy as np
import anndata as ad
from typing import Optional, Dict
from collections import Counter
from scipy.stats import norm
from scipy.optimize import brentq


__all__ = ["simulate_cytof_barcodes", "analyze_simulation"]


def _compute_gmm_boundary(separation: float, sigma_off: float = 0.5, 
                          sigma_on: float = 0.35) -> float:
    """Compute GMM decision boundary assuming equal priors and mu_off=0."""
    a = 1/sigma_on**2 - 1/sigma_off**2
    b = -2*separation/sigma_on**2
    c = separation**2/sigma_on**2 - 2*np.log(sigma_off/sigma_on)
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return separation / 2
    
    x1 = (-b + np.sqrt(discriminant)) / (2*a)
    x2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # Return the boundary between the means
    if 0 < x1 < separation:
        return x1
    elif 0 < x2 < separation:
        return x2
    return separation / 2


def _compute_channel_error_rate(separation: float, sigma_off: float = 0.5,
                                sigma_on: float = 0.35) -> float:
    """Compute expected error rate for a channel with given separation.
    
    Assumes 4-of-9 pattern weighting (5/9 off, 4/9 on).
    """
    if separation <= 0:
        return 0.5
    
    boundary = _compute_gmm_boundary(separation, sigma_off, sigma_on)
    p_off_to_on = 1 - norm.cdf(boundary / sigma_off)
    p_on_to_off = norm.cdf((boundary - separation) / sigma_on)
    
    return (5/9) * p_off_to_on + (4/9) * p_on_to_off


def _separation_from_error_rate(target_error: float, sigma_off: float = 0.5,
                                sigma_on: float = 0.35) -> float:
    """Find separation that produces target error rate."""
    def objective(sep):
        return _compute_channel_error_rate(sep, sigma_off, sigma_on) - target_error
    
    # Check bounds
    min_err = _compute_channel_error_rate(10.0, sigma_off, sigma_on)
    max_err = _compute_channel_error_rate(0.1, sigma_off, sigma_on)
    
    if target_error <= min_err:
        return 10.0
    if target_error >= max_err:
        return 0.1
    
    return brentq(objective, 0.1, 10.0)


def _generate_channel_parameters(n_channels: int, rng: np.random.RandomState,
                                 target_error_rate: Optional[float] = None) -> Dict:
    """Generate random channel parameters for simulation."""
    params = {}
    
    if target_error_rate is not None:
        base_separation = _separation_from_error_rate(target_error_rate)
    
    for g in range(n_channels):
        mu_off = rng.uniform(0.7, 1.8)
        
        if target_error_rate is not None:
            # Add variation similar to default range (0.8 spread)
            separation = max(0.2, base_separation + rng.uniform(-0.4, 0.4))
        else:
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
    target_error_rate: Optional[float] = None,
    unbarcoded_fraction: float = 0.00,
    alpha: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> ad.AnnData:
    """Simulate synthetic CyTOF barcode data.
    
    Generates CyTOF barcode data with known ground truth for testing
    and benchmarking debarcoding methods.
    
    Parameters
    ----------
    n_cells : int, default 10000
        Total number of cells to simulate.
    n_channels : int, default 27
        Number of barcode channels. Must be a multiple of 9.
    n_barcodes : int, optional
        Number of unique barcodes to use. If None, randomly selected.
    channel_params : dict, optional
        Custom channel parameters for simulation. If None, randomly generated.
    target_error_rate : float, optional
        Target mean per-channel error rate (probability of a sample falling
        on the wrong side of the GMM decision boundary). Only used when
        channel_params is None. Separation is calibrated to achieve this
        error rate on average, with per-channel variation.
        Typical values: 0.01-0.15. Default behavior (~0.05) when None.
    unbarcoded_fraction : float, default 0.0
        Fraction of cells without barcode (in [0, 1)).
    alpha : float, default 1.0
        Dirichlet concentration parameter for barcode distribution.
    random_seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default True
        Print progress messages.
    
    Returns
    -------
    AnnData
        Simulated raw intensity data with ground truth:
        
        - ``adata.obs['ground_truth_pattern']``: true barcode patterns
        - ``adata.obs['ground_truth_barcode']``: true barcode labels
        - ``adata.obs['is_barcoded']``: whether cell has a barcode
        - ``adata.uns['simulation']``: simulation parameters and statistics
    
    Raises
    ------
    ValueError
        If n_channels is not a multiple of 9 or unbarcoded_fraction is invalid.
    
    Examples
    --------
    >>> adata = simulate_cytof_barcodes(n_cells=10000, n_channels=18)
    >>> adata = simulate_cytof_barcodes(n_cells=50000, random_seed=42)
    >>> adata = simulate_cytof_barcodes(target_error_rate=0.10)  # 10% error
    """
    # Import here to avoid circular dependency
    from .barcode import _generate_4_of_9_patterns
    
    if n_channels % 9 != 0:
        raise ValueError(f"n_channels must be multiple of 9, got {n_channels}")
    
    if not 0 <= unbarcoded_fraction < 1:
        raise ValueError(f"unbarcoded_fraction must be in [0, 1), got {unbarcoded_fraction}")
    
    if target_error_rate is not None and not 0 < target_error_rate < 0.5:
        raise ValueError(f"target_error_rate must be in (0, 0.5), got {target_error_rate}")
    
    rng = np.random.RandomState(random_seed)
    
    if verbose:
        print("="*80)
        print("SIMULATING CYTOF BARCODE DATA")
        print("="*80)
        print(f"\nCells: {n_cells:,}, channels: {n_channels}, Blocks: {n_channels // 9}")
        print(f"Unbarcoded: {unbarcoded_fraction:.1%}, Alpha: {alpha}, Seed: {random_seed}")
        if target_error_rate is not None:
            print(f"Target error rate: {target_error_rate:.1%}")
    
    if channel_params is None:
        channel_params = _generate_channel_parameters(n_channels, rng, target_error_rate)
    
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
    
    # Compute actual mean error rate from generated parameters
    actual_error_rates = []
    for g in range(n_channels):
        params = channel_params[f's_{g+1}']
        sep = params['mu_on'] - params['mu_off']
        actual_error_rates.append(_compute_channel_error_rate(sep))
    mean_actual_error = np.mean(actual_error_rates)
    
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
        'target_error_rate': target_error_rate,
        'mean_actual_error_rate': float(mean_actual_error),
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
        print(f"Mean actual error rate: {mean_actual_error:.2%}")
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
    
    Parameters
    ----------
    adata : AnnData
        AnnData with simulation ground truth and debarcoding results.
    assignment_col : str
        Column with debarcoding assignments to evaluate.
    verbose : bool, default True
        Print statistics.
    
    Returns
    -------
    dict
        Performance metrics:
        
        - accuracy : fraction of valid assignments that are correct
        - coverage : fraction of cells with valid assignments
        - f1 : harmonic mean of accuracy and coverage
        - n_cells : total number of cells
        - n_barcoded : number of barcoded cells
        - n_valid_assigned : number of cells with valid assignments
        - n_correct : number of correct assignments
    
    Raises
    ------
    ValueError
        If adata lacks simulation metadata or required columns.
    
    Examples
    --------
    >>> results = analyze_simulation(adata, 'pc_gmm_assignment')
    >>> print(f"Accuracy: {results['accuracy']:.2%}")
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
