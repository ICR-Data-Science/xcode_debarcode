"""Debarcoding methods for CyTOF barcode data."""
import numpy as np
import anndata as ad
import warnings
from typing import Optional, List, Dict, Tuple
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
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


def _premessa(X: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """PreMessa: Top-4 selection method."""
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


def _scoring(X: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Scoring: Correlation-based pattern matching."""
    n_cells, n_channels = X.shape
    n_blocks = n_channels // 9
    
    if verbose:
        print(f"Scoring: Processing {n_channels} channels, {n_cells} cells")
    
    valid_patterns = _generate_4_of_9_patterns(n_blocks=1)
    barcodes = np.zeros((n_cells, n_channels), dtype=int)
    overall_confidences = np.ones(n_cells)
    
    for b in range(n_blocks):
        start, end = b * 9, (b + 1) * 9
        X_block = X[:, start:end]
        X_centered = X_block - X_block.mean(axis=1, keepdims=True)
        
        log_probs = np.sum(X_centered[:, np.newaxis, :] * (2 * valid_patterns[np.newaxis, :, :] - 1), axis=2)
        log_probs -= logsumexp(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        
        best_idx = np.argmax(probs, axis=1)
        barcodes[:, start:end] = valid_patterns[best_idx]
        overall_confidences *= probs[np.arange(n_cells), best_idx]
    
    if verbose:
        print(f"Scoring: Complete. Mean confidence: {overall_confidences.mean():.4f}")
    
    return barcodes, overall_confidences, {'method': 'scoring'}


def _auto(X: np.ndarray, n_init: int = 5, min_int: float = 0.1, mean_conf_switch: float = 0.5, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List, str]:
    """Auto: PC-GMM with fallback to Scoring."""
    if verbose:
        print("Auto: Trying PC-GMM first...")
    
    barcodes, confidences, channel_params = _pc_gmm(X, n_init, min_int, verbose=False)
    mean_conf = confidences.mean()
    
    if mean_conf >= mean_conf_switch:
        if verbose:
            print(f"Auto: Using PC-GMM (mean confidence: {mean_conf:.4f})")
        return barcodes, confidences, channel_params, 'pc_gmm'
    else:
        if verbose:
            print(f"Auto: PC-GMM low confidence ({mean_conf:.4f}), using Scoring...")
        barcodes, confidences, channel_params = _scoring(X, verbose=False)
        return barcodes, confidences, channel_params, 'scoring'


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
             method: str = 'auto',
             layer: Optional[str] = None,
             n_init: int = 5,
             min_int: float = 0.1,
             prob_thresh: float = 0.5,
             mean_conf_switch: float = 0.5,
             thresholds: Optional[List[float]] = None,
             method_name: Optional[str] = None,
             inplace: bool = True,
             verbose: bool = True) -> ad.AnnData:
    """Apply debarcoding to assign barcodes to cells.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    method : str
        Debarcoding method: 'gmm', 'premessa', 'pc_gmm', 'scoring', 'auto', 'manual'
    layer : str, optional
        Layer to use for debarcoding
    n_init : int
        GMM initializations (default: 5)
    min_int : float
        Minimum intensity (default: 0.1)
    prob_thresh : float
        GMM probability threshold (default: 0.5)
    mean_conf_switch : float
        Auto method switch threshold (default: 0.5)
    thresholds : list of float, optional
        Manual thresholds for 'manual' method
    method_name : str, optional
        Custom name for this debarcoding run. If None, uses method name
        and auto-increments if exists (e.g., 'pc_gmm', 'pc_gmm_1', 'pc_gmm_2').
    inplace : bool
        Modify adata in place (default: True)
    verbose : bool
        Print progress messages (default: True)
    
    Returns:
    --------
    adata : AnnData
        Modified AnnData with debarcoding results:
        - adata.obs['{method_name}_assignment']: Assigned barcode patterns
        - adata.obs['{method_name}_confidence']: Confidence scores
        - adata.uns['debarcoding']['{method_name}']: Method metadata
    """
    from .io import get_barcode_channels
    barcode_channels = get_barcode_channels(adata)
    
    if len(barcode_channels) % 9 != 0:
        raise ValueError(f"Number of barcode channels ({len(barcode_channels)}) must be multiple of 9")
    
    if not inplace:
        adata = adata.copy()
    
    if layer is None:
        if method in ['gmm', 'pc_gmm', 'scoring', 'auto']:
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
        'manual': lambda: (_manual(X, thresholds, verbose), None),
        'gmm': lambda: (_gmm(X, n_init, min_int, prob_thresh, verbose), None),
        'premessa': lambda: (_premessa(X, verbose), None),
        'pc_gmm': lambda: (_pc_gmm(X, n_init, min_int, verbose), None),
        'scoring': lambda: (_scoring(X, verbose), None),
        'auto': lambda: _auto(X, n_init, min_int, mean_conf_switch, verbose)
    }
    
    if method == 'manual' and thresholds is None:
        raise ValueError("Manual method requires 'thresholds' parameter")
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Valid options: {list(methods.keys())}")
    
    result = methods[method]()
    if method == 'auto':
        barcodes, confidences, channel_params, auto_method_used = result
    else:
        (barcodes, confidences, channel_params), _ = result
        auto_method_used = None
    
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
                pass  # Skip non-numeric keys
            

    
    metadata = {
        'method': method,
        'layer': layer,
        'n_init': n_init if method in ['gmm', 'pc_gmm', 'auto'] else None,
        'min_int': min_int if method in ['gmm', 'pc_gmm', 'auto'] else None,
        'prob_thresh': prob_thresh if method == 'gmm' else None,
        'mean_conf_switch': mean_conf_switch if method == 'auto' else None,
        'thresholds': thresholds if method == 'manual' else ([channel_params[str(g)]['threshold'] if channel_params.get(str(g)) else None for g in range(len(barcode_channels))] if method == 'gmm' else None),
        'n_cells': len(adata),
        'n_channels': len(barcode_channels),
        'barcode_channels': barcode_channels,
        'mean_confidence': float(confidences.mean()),
        'channel_params': channel_params
    }
    
    if auto_method_used:
        metadata['auto_method_used'] = auto_method_used
    
    adata.uns['debarcoding'][final_method_name] = metadata
    
    if verbose:
        print(f"Debarcoding complete using {method} (saved as '{final_method_name}')")
        print(f"  Confidence: adata.obs['{final_method_name}_confidence']")
        print(f"  Assignment: adata.obs['{final_method_name}_assignment']")
    
    return adata


def debarcoding_pipeline(adata: ad.AnnData,
                        method: str = 'auto',
                        transform_method: str = 'log',
                        cofactor: float = 5.0,
                        n_init: int = 5,
                        min_int: float = 0.1,
                        prob_thresh: float = 0.5,
                        mean_conf_switch: float = 0.5,
                        thresholds: Optional[List[float]] = None,
                        apply_hamming: bool = True,
                        hamming_radius: int = 2,
                        hamming_min_count: int = 0,
                        hamming_low_confidence_percentile: float = 100.0,
                        hamming_min_mean_conf: float = 0.0,
                        hamming_score_metric: str = 'count',
                        apply_confidence_filter: bool = True,
                        confidence_filter_method: Optional[str] = 'percentile',
                        confidence_value: Optional[float] = 90,
                        confidence_filter_or_flag: str = 'flag',
                        inplace: bool = True,
                        verbose: bool = True) -> ad.AnnData:
    """Complete debarcoding pipeline with transformation and postprocessing.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with mapped barcode channels
    method : str
        Debarcoding method: 'gmm', 'premessa', 'pc_gmm', 'scoring', 'auto' (default: 'auto')
    transform_method : str
        Transformation: 'log' or 'arcsinh' (default: 'log')
    cofactor : float
        Cofactor for arcsinh transformation (default: 5.0)
    n_init : int
        GMM initializations for gmm/pc_gmm/auto (default: 5)
    min_int : float
        Minimum intensity for GMM fitting (default: 0.1)
    prob_thresh : float
        GMM probability threshold for 'gmm' method (default: 0.5)
    mean_conf_switch : float
        Auto method switch threshold (default: 0.5)
    thresholds : list of float, optional
        Manual thresholds for 'manual' method (one per channel)
    
    apply_hamming : bool
        Apply Hamming clustering (default: True)
    hamming_radius : int
        Hamming distance radius for clustering (default: 2)
    hamming_min_count : int
        Minimum pattern count for Hamming clustering (default: 0)
    hamming_low_confidence_percentile : float
        Percentile for low-confidence filtering in Hamming (default: 100.0)
    hamming_min_mean_conf : float
        Minimum mean confidence for cluster centers (default: 0.0)
    hamming_score_metric : str
        Scoring metric for Hamming: 'count', 'mean_conf' (default: 'count')
    
    apply_confidence_filter : bool
        Apply confidence filtering (default: True)
    confidence_filter_method : str, optional
        Filter method: 'threshold', 'percentile', 'adaptive'
    confidence_value : float, optional
        Value for filtering (interpretation depends on method)
    confidence_filter_or_flag : str
        Action: 'filter' (remove) or 'flag' (mark) (default: 'filter')
    
    inplace : bool
        Modify adata in place (default: True)
    verbose : bool
        Print progress messages (default: True)
    
    Returns:
    --------
    adata : AnnData
        Processed AnnData with debarcoding results
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
    
    if verbose:
        print("\nStep 1: Transformation")
    adata = preprocessing.transform(adata, method=transform_method, cofactor=cofactor, verbose=verbose)
    layer_name = f'arcsinh_cf{cofactor}' if transform_method == 'arcsinh' else 'log'
    
    if verbose:
        print("\nStep 2: Debarcoding")
    adata = debarcode(adata, method=method,layer=layer_name, n_init=n_init,
                      min_int=min_int, prob_thresh=prob_thresh, mean_conf_switch=mean_conf_switch,
                     thresholds=thresholds, verbose=verbose)

    method_name = list(adata.uns['debarcoding'].keys())[-1]
    assignment_col = f'{method_name}_assignment'
    confidence_col = f'{method_name}_confidence'
    
    if apply_hamming:
        if verbose:
            print("\nStep 3: Hamming clustering")
        adata = postprocessing.hamming_cluster(
            adata, assignment_col=assignment_col, confidence_col=confidence_col,
            radius=hamming_radius, min_valid_count=hamming_min_count,
            low_confidence_percentile=hamming_low_confidence_percentile,
            min_mean_conf=hamming_min_mean_conf, score_metric=hamming_score_metric,
            verbose=verbose
        )
        assignment_col = f'{method_name}_hamming_assignment'
        confidence_col = f'{method_name}_hamming_confidence'
    
    if apply_confidence_filter:
        if verbose:
            print(f"\nStep 4: Confidence filtering ({confidence_filter_or_flag})")
        
        if not confidence_filter_method:
            raise ValueError("apply_confidence_filter=True requires confidence_filter_method")
        
        if confidence_filter_method in ['threshold', 'percentile'] and confidence_value is None:
            raise ValueError(f"confidence_filter_method='{confidence_filter_method}' requires confidence_value")
        
        adata = postprocessing.filter_by_confidence(
            adata, confidence_col=confidence_col, method=confidence_filter_method,
            value=confidence_value, filter_or_flag=confidence_filter_or_flag, verbose=verbose
        )
    
    if verbose:
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nFinal assignment: {assignment_col}")
        print(f"Final confidence: {confidence_col}")
        if apply_confidence_filter and confidence_filter_or_flag == 'flag':
            flag_method = confidence_col[:confidence_col.rfind('_')]
            print(f"Low confidence flag: adata.obs['{flag_method}_pass']")
    
    return adata
