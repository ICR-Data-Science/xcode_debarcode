"""Debarcoding methods for CyTOF barcode data."""
import numpy as np
import anndata as ad
import warnings
from typing import Optional, List, Dict, Tuple
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from .barcode import _generate_4_of_9_patterns

__all__ = ["debarcode", "debarcoding_pipeline"]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _get_data_matrix(adata: ad.AnnData, barcode_channels: List[str], layer: Optional[str] = None) -> np.ndarray:
    """Extract data matrix from AnnData."""
    if layer and layer in adata.layers:
        return np.column_stack([adata[:, ch].layers[layer].flatten() for ch in barcode_channels])
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    indices = [adata.var_names.get_loc(ch) for ch in barcode_channels]
    return X[:, indices]


def _get_unique_method_name(adata: ad.AnnData, method: str, custom_name: Optional[str] = None) -> str:
    """Get unique method name for saving results."""
    if custom_name:
        return custom_name
    
    name = method
    counter = 0
    while f'{name}_assignment' in adata.obs.columns:
        counter += 1
        name = f'{method}_{counter}'
    return name


# =============================================================================
# X-EM: EXPECTATION-MAXIMIZATION FOR X-CODE
# =============================================================================

def _x_em_block(X_block: np.ndarray, valid_4of9: np.ndarray, 
                max_iter: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Run X-EM on a single 9-channel block.
    
    Constrained EM that enforces valid 4-of-9 patterns throughout optimization.
    Uses PreMessa (top-4) initialization for robust warm start.
    """
    n_cells = X_block.shape[0]
    
    # PreMessa initialization: use top-4 channels to estimate ON/OFF means
    is_on = np.zeros((n_cells, 9), dtype=bool)
    for i in range(n_cells):
        top4 = np.argsort(X_block[i])[-4:]
        is_on[i, top4] = True
    
    on_vals = X_block[is_on]
    off_vals = X_block[~is_on]
    
    mu_on = np.full(9, on_vals.mean())
    mu_off = np.full(9, off_vals.mean())
    var_on = np.var(X_block, axis=0) + 1e-6
    var_off = np.var(X_block, axis=0) + 1e-6
    pi = np.ones(126) / 126
    
    prev_ll = -np.inf
    
    for _ in range(max_iter):
        # E-step: compute responsibilities (vectorized)
        log_p_off = -0.5 * np.log(2 * np.pi * var_off) - 0.5 * (X_block - mu_off)**2 / var_off
        log_p_on = -0.5 * np.log(2 * np.pi * var_on) - 0.5 * (X_block - mu_on)**2 / var_on
        
        # Pattern log-likelihoods: (n_cells, 126)
        log_lik = log_p_on @ valid_4of9.T + log_p_off @ (1 - valid_4of9).T
        log_lik += np.log(pi + 1e-300)
        
        log_norm = logsumexp(log_lik, axis=1, keepdims=True)
        resp = np.exp(log_lik - log_norm)
        
        # M-step: update parameters
        pi = np.maximum(resp.mean(axis=0), 1e-10)
        pi /= pi.sum()
        
        # Weighted sums for ON/OFF positions
        w_on = resp @ valid_4of9          # (n_cells, 9)
        w_off = resp @ (1 - valid_4of9)   # (n_cells, 9)
        W_on = w_on.sum(axis=0)
        W_off = w_off.sum(axis=0)
        
        mu_on = (w_on * X_block).sum(axis=0) / (W_on + 1e-10)
        mu_off = (w_off * X_block).sum(axis=0) / (W_off + 1e-10)
        var_on = np.maximum((w_on * (X_block - mu_on)**2).sum(axis=0) / (W_on + 1e-10), 1e-6)
        var_off = np.maximum((w_off * (X_block - mu_off)**2).sum(axis=0) / (W_off + 1e-10), 1e-6)
        
        # Check convergence
        ll = log_norm.sum()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    
    # Final assignment
    pattern_idx = np.argmax(resp, axis=1)
    conf = resp[np.arange(n_cells), pattern_idx]
    
    return pattern_idx, conf


def _x_em(X: np.ndarray, max_iter: int = 100, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """X-EM: Expectation-Maximization for X-Code debarcoding.
    
    Constrained EM that models each cell as arising from one of 126 valid 4-of-9
    patterns per block. Parameters (mu_on, mu_off, var_on, var_off) are shared
    across patterns within each block, enforcing the X-Code structure throughout.
    """
    n_cells, n_channels = X.shape
    n_blocks = n_channels // 9
    valid_4of9 = _generate_4_of_9_patterns(n_blocks=1).astype(np.float64)
    
    if verbose:
        print(f"X-EM: Processing {n_channels} channels, {n_cells} cells, {n_blocks} blocks")
    
    barcodes = np.zeros((n_cells, n_channels), dtype=np.int8)
    confidences = np.ones(n_cells)
    
    for b in range(n_blocks):
        start, end = b * 9, (b + 1) * 9
        X_block = X[:, start:end]
        
        pattern_idx, conf = _x_em_block(X_block, valid_4of9, max_iter=max_iter)
        barcodes[:, start:end] = valid_4of9[pattern_idx].astype(np.int8)
        confidences *= conf
    
    if verbose:
        print(f"X-EM: Complete. Mean confidence: {confidences.mean():.4f}")
    
    return barcodes, confidences, {'method': 'x_em', 'max_iter': max_iter}


# =============================================================================
# GMM: PER-CHANNEL GAUSSIAN MIXTURE MODEL
# =============================================================================

def _fit_gmm_channel(x: np.ndarray, min_int: float, n_init: int) -> Optional[Dict]:
    """Fit 2-component GMM to single channel and extract parameters."""
    valid_mask = x > min_int
    if valid_mask.sum() < 20:
        return None
    
    try:
        gmm = GaussianMixture(n_components=2, n_init=n_init, random_state=42)
        gmm.fit(x[valid_mask].reshape(-1, 1))
        
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_
        pos_idx = np.argmax(means)
        
        return {
            'mu_off': float(means[1-pos_idx]),
            'sigma_off': float(max(stds[1-pos_idx], 0.15)),
            'mu_on': float(means[pos_idx]),
            'sigma_on': float(max(stds[pos_idx], 0.15)),
            'w_off': float(weights[1-pos_idx]),
            'w_on': float(weights[pos_idx]),
            'gmm': gmm,
            'pos_idx': pos_idx
        }
    except:
        return None


def _gmm(X: np.ndarray, n_init: int = 5, min_int: float = 0.1, prob_thresh: float = 0.5, 
         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """GMM: Independent 2-component Gaussian Mixture Model per channel.
    
    Fits GMM per channel and classifies each channel independently.
    Can produce invalid patterns (not constrained to 4-of-9).
    """
    n_cells, n_channels = X.shape
    
    if verbose:
        print(f"GMM: Processing {n_channels} channels, {n_cells} cells (n_init={n_init})")
    
    channel_params = {}
    barcodes = np.zeros((n_cells, n_channels), dtype=int)
    channel_probs = np.ones((n_cells, n_channels))
    
    for g in range(n_channels):
        params = _fit_gmm_channel(X[:, g], min_int, n_init)
        if params:
            params['channel_index'] = g
            gmm, pos_idx = params.pop('gmm'), params.pop('pos_idx')
            params['threshold'] = float(params['mu_off'] + (params['mu_on'] - params['mu_off']) * prob_thresh)
            channel_params[str(g)] = params
            
            probs = gmm.predict_proba(X[:, g].reshape(-1, 1))
            pos_probs = probs[:, pos_idx]
            barcodes[:, g] = (pos_probs >= prob_thresh).astype(int)
            channel_probs[:, g] = np.where(barcodes[:, g] == 1, pos_probs, 1 - pos_probs)
        else:
            channel_params[str(g)] = None
    
    confidences = np.prod(channel_probs, axis=1)
    
    if verbose:
        print(f"GMM: Complete. Mean confidence: {confidences.mean():.4f}")
    
    return barcodes, confidences, channel_params


# =============================================================================
# PREMESSA: TOP-4 SELECTION
# =============================================================================

def _premessa(X: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """PreMessa: Top-4 channel selection method."""
    n_cells, n_channels = X.shape
    n_blocks = n_channels // 9
    
    if verbose:
        print(f"PreMessa: Processing {n_channels} channels, {n_cells} cells")
    
    barcodes = np.zeros((n_cells, n_channels), dtype=int)
    confidences = np.ones(n_cells)
    
    for b in range(n_blocks):
        start, end = b * 9, (b + 1) * 9
        X_block = X[:, start:end]
        
        for i in range(n_cells):
            intensities = X_block[i, :]
            top4_indices = np.argsort(intensities)[-4:]
            barcodes[i, start + top4_indices] = 1
            
            sorted_int = np.sort(intensities)
            if len(sorted_int) >= 5:
                separation = sorted_int[-4] - sorted_int[-5]
                confidences[i] *= min(1.0, separation / 2.0)
    
    if verbose:
        print(f"PreMessa: Complete. Mean confidence: {confidences.mean():.4f}")
    
    return barcodes, confidences, {'method': 'premessa'}


# =============================================================================
# MANUAL: THRESHOLD-BASED GATING
# =============================================================================

def _manual(X: np.ndarray, thresholds: List[float], verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Manual: Threshold-based gating."""
    n_cells, n_channels = X.shape
    
    if verbose:
        print(f"Manual: Processing {n_channels} channels, {n_cells} cells")
    
    if len(thresholds) != n_channels:
        raise ValueError(f"Number of thresholds ({len(thresholds)}) must match number of channels ({n_channels})")
    
    barcodes = (X > np.array(thresholds)).astype(int)
    
    from .barcode import is_valid_pattern
    confidences = np.array([1.0 if is_valid_pattern(b) else 0.5 for b in barcodes])
    channel_params = {str(g): {'channel_index': g, 'threshold': thresholds[g]} for g in range(n_channels)}
    
    if verbose:
        n_valid = confidences.sum()
        print(f"Manual: Complete. Valid patterns: {int(n_valid)}/{n_cells}")
    
    return barcodes, confidences, channel_params


# =============================================================================
# MAIN DEBARCODE FUNCTION
# =============================================================================

def debarcode(adata: ad.AnnData,
             method: str = 'x_em',
             layer: Optional[str] = None,
             n_init: int = 5,
             min_int: float = 0.1,
             prob_thresh: float = 0.5,
             max_iter: int = 100,
             thresholds: Optional[List[float]] = None,
             method_name: Optional[str] = None,
             inplace: bool = True,
             verbose: bool = True) -> ad.AnnData:
    """Apply debarcoding to assign barcodes to cells.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    method : {'x_em', 'gmm', 'premessa', 'manual'}, default 'x_em'
        Debarcoding method:
        
        - 'x_em': X-Code EM (default). Constrained EM that enforces valid 4-of-9
          patterns throughout optimization. Best overall performance across all
          barcode configurations.
        - 'gmm': Per-channel GMM. Independent 2-component GMM per channel.
          Can produce invalid patterns (not constrained to 4-of-9).
        - 'premessa': Top-4 selection per block with separation-based confidence.
        - 'manual': Fixed user-defined thresholds per channel.
    layer : str, optional
        Layer to use for debarcoding. Recommend using transformed layer ('log').
    n_init : int, default 5
        GMM initializations (for 'gmm' method only).
    min_int : float, default 0.1
        Minimum intensity for GMM fitting (for 'gmm' method only).
    prob_thresh : float, default 0.5
        GMM probability threshold (for 'gmm' method only).
    max_iter : int, default 100
        Maximum EM iterations (for 'x_em' method only).
    thresholds : list of float, optional
        Manual thresholds for 'manual' method (one per channel).
    method_name : str, optional
        Custom name for this debarcoding run. If None, uses method name
        and auto-increments if exists (e.g., 'x_em', 'x_em_1').
    inplace : bool, default True
        Modify adata in place.
    verbose : bool, default True
        Print progress messages.
    
    Returns
    -------
    AnnData
        Modified AnnData with debarcoding results:
        
        - ``adata.obs['{method_name}_assignment']``: Assigned barcode patterns
        - ``adata.obs['{method_name}_confidence']``: Confidence scores
        - ``adata.uns['debarcoding']['{method_name}']``: Method metadata
    
    Examples
    --------
    >>> adata = debarcode(adata, method='x_em', layer='log')
    >>> adata = debarcode(adata, method='gmm', layer='log')
    >>> adata = debarcode(adata, method='manual', thresholds=[1.5]*18)
    """
    from .io import get_barcode_channels
    barcode_channels = get_barcode_channels(adata)
    
    if len(barcode_channels) % 9 != 0:
        raise ValueError(f"Number of barcode channels ({len(barcode_channels)}) must be multiple of 9")
    
    if not inplace:
        adata = adata.copy()
    
    if layer is None:
        if method in ['x_em', 'gmm']:
            warnings.warn(
                f"Using method '{method}' without a transformed layer. "
                f"Recommend: preprocessing.transform(adata, method='log') and pass layer='log'.",
                UserWarning
            )
        elif method == 'premessa':
            warnings.warn(
                f"Using method 'premessa' without a transformed layer. "
                f"Recommend: preprocessing.transform(adata, method='arcsinh', cofactor=5) "
                f"and pass layer='arcsinh_cf5'.",
                UserWarning
            )
    
    X = _get_data_matrix(adata, barcode_channels, layer)
    final_method_name = _get_unique_method_name(adata, method, method_name)
    
    methods = {
        'x_em': lambda: _x_em(X, max_iter, verbose),
        'gmm': lambda: _gmm(X, n_init, min_int, prob_thresh, verbose),
        'premessa': lambda: _premessa(X, verbose),
        'manual': lambda: _manual(X, thresholds, verbose),
    }
    
    if method == 'manual' and thresholds is None:
        raise ValueError("Manual method requires 'thresholds' parameter")
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Valid options: {list(methods.keys())}")
    
    barcodes, confidences, method_params = methods[method]()
    
    pattern_strings = [''.join(map(str, b)) for b in barcodes]
    adata.obs[f'{final_method_name}_assignment'] = pattern_strings
    adata.obs[f'{final_method_name}_confidence'] = confidences

    if 'debarcoding' not in adata.uns:
        adata.uns['debarcoding'] = {}
    
    # Add channel names to params
    for channel_idx, params in method_params.items():
        if params and channel_idx not in ['method', 'max_iter']:
            try:
                idx = int(channel_idx)
                if idx < len(barcode_channels):
                    params['channel'] = barcode_channels[idx]
            except (ValueError, TypeError):
                pass

    metadata = {
        'method': method,
        'layer': layer,
        'n_init': n_init if method == 'gmm' else None,
        'min_int': min_int if method == 'gmm' else None,
        'prob_thresh': prob_thresh if method == 'gmm' else None,
        'max_iter': max_iter if method == 'x_em' else None,
        'thresholds': thresholds if method == 'manual' else (
            [method_params[str(g)]['threshold'] if method_params.get(str(g)) else None 
             for g in range(len(barcode_channels))] if method == 'gmm' else None
        ),
        'n_cells': len(adata),
        'n_channels': len(barcode_channels),
        'barcode_channels': barcode_channels,
        'mean_confidence': float(confidences.mean()),
        'channel_params': method_params
    }
    
    adata.uns['debarcoding'][final_method_name] = metadata
    
    if verbose:
        print(f"Debarcoding complete using {method} (saved as '{final_method_name}')")
        print(f"  Confidence: adata.obs['{final_method_name}_confidence']")
        print(f"  Assignment: adata.obs['{final_method_name}_assignment']")
        print(f"  [+] Metadata saved: adata.uns['debarcoding']['{final_method_name}']")
    
    return adata


# =============================================================================
# DEBARCODING PIPELINE
# =============================================================================

def debarcoding_pipeline(adata: ad.AnnData,
                        method: str = 'x_em',
                        transform_method: str = 'log',
                        cofactor: float = 5.0,
                        n_init: int = 5,
                        min_int: float = 0.1,
                        prob_thresh: float = 0.5,
                        max_iter: int = 100,
                        thresholds: Optional[List[float]] = None,
                        # Intensity filtering
                        apply_intensity_filter: bool = True,
                        intensity_method: str = 'rectangular',
                        intensity_percentile: float = 99.0,
                        intensity_sum_low: Optional[float] = 1.0,
                        intensity_sum_high: Optional[float] = 99.0,
                        intensity_var_low: Optional[float] = 1.0,
                        intensity_var_high: Optional[float] = 99.0,
                        intensity_filter_or_flag: str = 'filter',
                        # Hamming clustering
                        apply_hamming: bool = True,
                        hamming_method: str = 'msg',
                        hamming_radius: int = 2,
                        hamming_ratio: float = 15.0,
                        hamming_ratio_metric: str = 'count',
                        hamming_tie_break: str = 'lda',
                        hamming_test_conf: bool = True,
                        hamming_low_conf_perc: Optional[float] = None,
                        # Pattern filtering
                        apply_pattern_filter: bool = True,
                        pattern_filter_metric: str = 'median_conf',
                        pattern_filter_method: str = 'percentile',
                        pattern_filter_value: float = 90,
                        pattern_filter_exclude_remapped: bool = True,
                        pattern_filter_or_flag: str = 'flag',
                        inplace: bool = True,
                        verbose: bool = True) -> ad.AnnData:
    """Complete debarcoding pipeline with transformation and postprocessing.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with mapped barcode channels.
    method : str, default 'x_em'
        Debarcoding method: 'x_em', 'gmm', 'premessa', 'manual'.
    transform_method : str, default 'log'
        Transformation: 'log' or 'arcsinh'.
    cofactor : float, default 5.0
        Cofactor for arcsinh transformation.
    n_init : int, default 5
        GMM initializations (for 'gmm' method).
    min_int : float, default 0.1
        Minimum intensity for GMM fitting.
    prob_thresh : float, default 0.5
        GMM probability threshold (for 'gmm' method).
    max_iter : int, default 100
        Maximum EM iterations (for 'x_em' method).
    thresholds : list of float, optional
        Manual thresholds for 'manual' method (one per channel).
    apply_intensity_filter : bool, default True
        Apply intensity filtering before debarcoding.
    intensity_method : str, default 'rectangular'
        Intensity filter method: 'rectangular' or 'ellipsoidal'.
    intensity_percentile : float, default 99.0
        For 'ellipsoidal': percentile threshold.
    intensity_sum_low : float, optional, default 1.0
        For 'rectangular': lower percentile for channel sum.
    intensity_sum_high : float, optional, default 99.0
        For 'rectangular': upper percentile for channel sum.
    intensity_var_low : float, optional, default 1.0
        For 'rectangular': lower percentile for channel variance.
    intensity_var_high : float, optional, default 99.0
        For 'rectangular': upper percentile for channel variance.
    intensity_filter_or_flag : str, default 'filter'
        'filter' to remove cells or 'flag' to mark.
    apply_hamming : bool, default True
        Apply Hamming clustering.
    hamming_method : str, default 'msg'
        Hamming method: 'msg' or 'sphere'.
    hamming_radius : int, default 2
        Max Hamming distance for neighbor search.
    hamming_ratio : float, default 15.0
        Minimum ratio for remapping valid patterns.
    hamming_ratio_metric : str, default 'count'
        Ratio metric: 'count' or 'score'.
    hamming_tie_break : str, default 'lda'
        Tie-break method: 'no_remap', 'count', or 'lda'.
    hamming_test_conf : bool, default True
        Only remap to higher confidence neighbors.
    hamming_low_conf_perc : float, optional
        Only remap bottom N% confidence cells. Default: None (all).
    apply_pattern_filter : bool, default True
        Apply pattern-based filtering after hamming.
    pattern_filter_metric : str, default 'median_conf'
        Pattern metric: 'count', 'median_conf', or 'score'.
    pattern_filter_method : str, default 'percentile'
        Filter method: 'threshold' or 'percentile'.
    pattern_filter_value : float, default 90
        Threshold or percentile value.
    pattern_filter_exclude_remapped : bool, default True
        Exclude remapped cells from metric calculation.
    pattern_filter_or_flag : str, default 'flag'
        'flag' to mark or 'filter' to remove cells.
    inplace : bool, default True
        Modify adata in place.
    verbose : bool, default True
        Print progress messages.
    
    Returns
    -------
    AnnData
        AnnData with debarcoding results.
    
    Examples
    --------
    >>> adata = debarcoding_pipeline(adata, method='x_em')
    >>> adata = debarcoding_pipeline(adata, method='x_em', apply_hamming=True)
    """
    from .io import get_barcode_channels
    barcode_channels = get_barcode_channels(adata)
    
    if len(barcode_channels) % 9 != 0:
        raise ValueError(f"Number of barcode channels ({len(barcode_channels)}) must be multiple of 9")
    
    if not inplace:
        adata = adata.copy()
    
    from . import preprocessing, postprocessing
    
    if verbose:
        print("="*80)
        print("DEBARCODING PIPELINE")
        print("="*80)
    
    # Step 1: Transformation
    if verbose:
        print("\nStep 1: Transformation")
    adata = preprocessing.transform(adata, method=transform_method, cofactor=cofactor, verbose=verbose)
    layer_name = f'arcsinh_cf{cofactor}' if transform_method == 'arcsinh' else 'log'
    
    # Step 2: Intensity filtering (optional)
    step = 2
    if apply_intensity_filter:
        if verbose:
            print(f"\nStep {step}: Intensity filtering")
        adata = preprocessing.filter_cells_intensity(
            adata,
            layer=layer_name,
            method=intensity_method,
            percentile=intensity_percentile,
            sum_low=intensity_sum_low,
            sum_high=intensity_sum_high,
            var_low=intensity_var_low,
            var_high=intensity_var_high,
            filter_or_flag=intensity_filter_or_flag,
            verbose=verbose
        )
        step += 1
    
    # Step 3: Debarcoding
    if verbose:
        print(f"\nStep {step}: Debarcoding")
    adata = debarcode(adata, method=method, layer=layer_name, n_init=n_init,
                      min_int=min_int, prob_thresh=prob_thresh, max_iter=max_iter,
                      thresholds=thresholds, verbose=verbose)
    step += 1
    
    method_name = list(adata.uns['debarcoding'].keys())[-1]
    assignment_col = f'{method_name}_assignment'
    confidence_col = f'{method_name}_confidence'
    
    # Step 4: Hamming clustering
    if apply_hamming:
        if verbose:
            print(f"\nStep {step}: Hamming clustering")
        adata = postprocessing.hamming_cluster(
            adata,
            assignment_col=assignment_col,
            confidence_col=confidence_col,
            method=hamming_method,
            radius=hamming_radius,
            ratio=hamming_ratio,
            ratio_metric=hamming_ratio_metric,
            tie_break=hamming_tie_break,
            test_conf=hamming_test_conf,
            low_conf_perc=hamming_low_conf_perc,
            layer=layer_name,
            verbose=verbose
        )
        assignment_col = f'{method_name}_hamming_assignment'
        remapped_col = f'{method_name}_hamming_remapped'
        step += 1
    
    # Step 5: Pattern filtering
    if apply_pattern_filter:
        if verbose:
            print(f"\nStep {step}: Pattern filtering")
        
        exclude_col = remapped_col if (apply_hamming and pattern_filter_exclude_remapped) else None
        
        adata = postprocessing.filter_pattern(
            adata,
            assignment_col=assignment_col,
            confidence_col=confidence_col,
            metric=pattern_filter_metric,
            method=pattern_filter_method,
            value=pattern_filter_value,
            exclude=exclude_col,
            filter_or_flag=pattern_filter_or_flag,
            verbose=verbose
        )
    
    if verbose:
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nFinal assignment: {assignment_col}")
        print(f"Final confidence: {confidence_col}")
        if apply_intensity_filter and intensity_filter_or_flag == 'flag':
            print(f"Intensity filter flag: adata.obs['intensity_pass']")
        if apply_pattern_filter and pattern_filter_or_flag == 'flag':
            base_name = assignment_col.rsplit('_', 1)[0] if '_' in assignment_col else assignment_col
            print(f"Pattern filter flag: adata.obs['{base_name}_pattern_pass']")
    
    return adata
