"""Debarcoding methods for CyTOF barcode data."""
import numpy as np
import anndata as ad
import warnings
from typing import Optional, List, Dict, Tuple
from sklearn.mixture import GaussianMixture
from .barcode import _generate_4_of_9_patterns

__all__ = ["debarcode", "debarcoding_pipeline"]


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


def _pc_gmm(X: np.ndarray, n_init: int = 5, min_int: float = 0.1, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """PC-GMM: Pattern-Constrained Gaussian Mixture Model."""
    n_cells, n_channels = X.shape
    n_blocks = n_channels // 9
    
    if verbose:
        print(f"PC-GMM: Processing {n_channels} channels, {n_cells} cells (n_init={n_init})")
    
    channel_params = {}
    failed_channels = []
    
    for g in range(n_channels):
        params = _fit_gmm_channel(X[:, g], min_int, n_init)
        if params:
            params['channel_index'] = g
            del params['gmm'], params['pos_idx']
        else:
            failed_channels.append(g)
        channel_params[str(g)] = params
    
    valid_patterns = _generate_4_of_9_patterns(n_blocks=1)
    barcodes = np.zeros((n_cells, n_channels), dtype=int)
    confidences = np.ones(n_cells)
    
    for b in range(n_blocks):
        start, end = 9*b, 9*(b+1)
        
        if any(g in failed_channels for g in range(start, end)):
            continue
        
        X_block = X[:, start:end]
        log_like_off = np.zeros((n_cells, 9))
        log_like_on = np.zeros((n_cells, 9))
        
        for g in range(9):
            p = channel_params[str(start + g)]
            if p is None:
                continue
            
            log_like_off[:, g] = (
                np.log(p['w_off'] + 1e-10) - 
                0.5 * np.log(2 * np.pi * p['sigma_off']**2) - 
                0.5 * ((X_block[:, g] - p['mu_off']) / p['sigma_off'])**2
            )
            log_like_on[:, g] = (
                np.log(p['w_on'] + 1e-10) - 
                0.5 * np.log(2 * np.pi * p['sigma_on']**2) - 
                0.5 * ((X_block[:, g] - p['mu_on']) / p['sigma_on'])**2
            )
        
        patterns = valid_patterns[np.newaxis, :, :]
        log_likes = np.where(patterns == 0, log_like_off[:, np.newaxis, :], log_like_on[:, np.newaxis, :])
        pattern_log_likes = log_likes.sum(axis=2)
        pattern_probs = np.exp(pattern_log_likes - pattern_log_likes.max(axis=1, keepdims=True))
        pattern_probs /= pattern_probs.sum(axis=1, keepdims=True)
        
        best_idx = np.argmax(pattern_probs, axis=1)
        best_conf = pattern_probs[np.arange(n_cells), best_idx]
        
        barcodes[:, start:end] = valid_patterns[best_idx]
        confidences *= best_conf
    
    if verbose:
        print(f"PC-GMM: Complete. Mean confidence: {confidences.mean():.4f}")
    
    return barcodes, confidences, channel_params


def _gmm(X: np.ndarray, n_init: int = 5, min_int: float = 0.1, prob_thresh: float = 0.5, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """GMM: Gaussian Mixture Model."""
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


def _premessa_core(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Core PreMessa: vectorized top-4 selection per block."""
    n_cells, n_channels = X.shape
    n_blocks = n_channels // 9
    
    barcodes = np.zeros((n_cells, n_channels), dtype=int)
    block_deltas = np.zeros((n_cells, n_blocks))
    row_idx = np.arange(n_cells)
    
    for b in range(n_blocks):
        start = b * 9
        X_block = X[:, start:start + 9]
        
        sorted_vals = np.sort(X_block, axis=1)
        block_deltas[:, b] = sorted_vals[:, -4] - sorted_vals[:, -5]
        
        top4 = np.argpartition(X_block, -4, axis=1)[:, -4:]
        for j in range(4):
            barcodes[row_idx, start + top4[:, j]] = 1
    
    deltas = block_deltas.min(axis=1)
    return barcodes, deltas


def _premessa(X: np.ndarray, n_iter: int = 2, min_cells: int = 10, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """PreMessa: Top-4 selection with iterative per-channel normalization.
    
    Performs iterative refinement by normalizing each channel based on cells
    classified as ON, then re-running top-4 selection. This corrects for
    systematic intensity differences between channels.
    
    The returned delta (confidence) is the minimum separation (4th - 5th highest)
    across blocks. Use threshold-based filtering on deltas for quality control.
    """
    n_cells, n_channels = X.shape
    
    if verbose:
        print(f"PreMessa: Processing {n_channels} channels, {n_cells} cells (n_iter={n_iter})")
    
    X_norm = X.copy()
    
    for iteration in range(n_iter + 1):
        barcodes, deltas = _premessa_core(X_norm)
        
        if iteration < n_iter:
            for ch in range(n_channels):
                on_mask = barcodes[:, ch] == 1
                if on_mask.sum() >= min_cells:
                    norm_factor = np.percentile(X_norm[on_mask, ch], 95)
                    if norm_factor > 0:
                        X_norm[:, ch] /= norm_factor
    
    if verbose:
        print(f"PreMessa: Complete. Mean separation: {deltas.mean():.4f}")
    
    return barcodes, deltas, {'method': 'premessa', 'n_iter': n_iter}


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


def debarcode(adata: ad.AnnData,
             method: str = 'pc_gmm',
             layer: Optional[str] = None,
             n_init: int = 5,
             min_int: float = 0.1,
             prob_thresh: float = 0.5,
             n_iter: int = 2,
             thresholds: Optional[List[float]] = None,
             method_name: Optional[str] = None,
             inplace: bool = True,
             verbose: bool = True) -> ad.AnnData:
    """Apply debarcoding to assign barcodes to cells.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    method : {'gmm', 'premessa', 'pc_gmm', 'manual'}, default 'pc_gmm'
        Debarcoding method:
        
        - 'gmm': Gaussian Mixture Model per channel
        - 'premessa': Iterative top-4 selection per block
        - 'pc_gmm': Pattern-Constrained GMM
        - 'manual': Fixed thresholds
    layer : str, optional
        Layer to use for debarcoding. Recommend using transformed layer.
    n_init : int, default 5
        GMM initializations (for gmm/pc_gmm).
    min_int : float, default 0.1
        Minimum intensity for GMM fitting.
    prob_thresh : float, default 0.5
        GMM probability threshold (for 'gmm' method).
    n_iter : int, default 2
        Normalization iterations (for 'premessa' method).
    thresholds : list of float, optional
        Manual thresholds for 'manual' method (one per channel).
    method_name : str, optional
        Custom name for this debarcoding run. If None, uses method name
        and auto-increments if exists (e.g., 'pc_gmm', 'pc_gmm_1').
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
    >>> adata = debarcode(adata, method='pc_gmm', layer='log')
    >>> adata = debarcode(adata, method='premessa', layer='arcsinh_cf10.0')
    >>> adata = debarcode(adata, method='manual', thresholds=[1.5]*27)
    """
    from .io import get_barcode_channels
    barcode_channels = get_barcode_channels(adata)
    
    if len(barcode_channels) % 9 != 0:
        raise ValueError(f"Number of barcode channels ({len(barcode_channels)}) must be multiple of 9")
    
    if not inplace:
        adata = adata.copy()
    
    if layer is None:
        if method in ['gmm', 'pc_gmm']:
            warnings.warn(
                f"Using method '{method}' without a transformed layer. "
                f"Recommend: preprocessing.transform(adata, method='log') and pass layer='log'.",
                UserWarning
            )
        elif method == 'premessa':
            warnings.warn(
                f"Using method 'premessa' without a transformed layer. "
                f"Recommend: preprocessing.transform(adata, method='arcsinh', cofactor=10.0) "
                f"and pass layer='arcsinh_cf10.0'.",
                UserWarning
            )
    
    X = _get_data_matrix(adata, barcode_channels, layer)
    final_method_name = _get_unique_method_name(adata, method, method_name)
    
    methods = {
        'manual': lambda: (_manual(X, thresholds, verbose), None),
        'gmm': lambda: (_gmm(X, n_init, min_int, prob_thresh, verbose), None),
        'premessa': lambda: (_premessa(X, n_iter, verbose=verbose), None),
        'pc_gmm': lambda: (_pc_gmm(X, n_init, min_int, verbose), None),
    }
    
    if method == 'manual' and thresholds is None:
        raise ValueError("Manual method requires 'thresholds' parameter")
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Valid options: {list(methods.keys())}")
    
    result = methods[method]()
    (barcodes, confidences, channel_params), _ = result
    
    pattern_strings = [''.join(map(str, b)) for b in barcodes]
    adata.obs[f'{final_method_name}_assignment'] = pattern_strings
    adata.obs[f'{final_method_name}_confidence'] = confidences

    if 'debarcoding' not in adata.uns:
        adata.uns['debarcoding'] = {}
    
    for channel_idx, params in channel_params.items():
        if params and channel_idx != 'method':
            try:
                idx = int(channel_idx)
                if idx < len(barcode_channels):
                    params['channel'] = barcode_channels[idx]
            except (ValueError, TypeError):
                pass

    
    metadata = {
        'method': method,
        'layer': layer,
        'n_init': n_init if method in ['gmm', 'pc_gmm'] else None,
        'min_int': min_int if method in ['gmm', 'pc_gmm'] else None,
        'prob_thresh': prob_thresh if method == 'gmm' else None,
        'n_iter': n_iter if method == 'premessa' else None,
        'thresholds': thresholds if method == 'manual' else ([channel_params[str(g)]['threshold'] if channel_params.get(str(g)) else None for g in range(len(barcode_channels))] if method == 'gmm' else None),
        'n_cells': len(adata),
        'n_channels': len(barcode_channels),
        'barcode_channels': barcode_channels,
        'mean_confidence': float(confidences.mean()),
        'channel_params': channel_params
    }
    
    adata.uns['debarcoding'][final_method_name] = metadata
    
    if verbose:
        print(f"Debarcoding complete using {method} (saved as '{final_method_name}')")
        print(f"  Confidence: adata.obs['{final_method_name}_confidence']")
        print(f"  Assignment: adata.obs['{final_method_name}_assignment']")
        print(f"  [+] Metadata saved: adata.uns['debarcoding']['{final_method_name}']")
    
    return adata


def debarcoding_pipeline(adata: ad.AnnData,
                        method: str = 'pc_gmm',
                        transform_method: str = 'log',
                        cofactor: float = 10.0,
                        n_init: int = 5,
                        min_int: float = 0.1,
                        prob_thresh: float = 0.5,
                        n_iter: int = 2,
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
    method : str, default 'pc_gmm'
        Debarcoding method: 'gmm', 'premessa', 'pc_gmm', 'manual'.
    transform_method : str, default 'log'
        Transformation: 'log' or 'arcsinh'.
    cofactor : float, default 10.0
        Cofactor for arcsinh transformation.
    n_init : int, default 5
        GMM initializations for gmm/pc_gmm.
    min_int : float, default 0.1
        Minimum intensity for GMM fitting.
    prob_thresh : float, default 0.5
        GMM probability threshold for 'gmm' method.
    n_iter : int, default 2
        Normalization iterations for 'premessa' method.
    thresholds : list of float, optional
        Manual thresholds for 'manual' method (one per channel).
    apply_intensity_filter : bool, default False
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
    >>> adata = debarcoding_pipeline(adata, method='pc_gmm')
    >>> adata = debarcoding_pipeline(adata, method='pc_gmm', apply_hamming=False)
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
                      min_int=min_int, prob_thresh=prob_thresh, n_iter=n_iter,
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
