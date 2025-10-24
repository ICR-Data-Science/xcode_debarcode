"""Postprocessing methods for confidence filtering and Hamming clustering."""
import numpy as np
import anndata as ad
from sklearn.mixture import GaussianMixture
from collections import Counter
from typing import Optional

from .barcode import is_valid_pattern

__all__ = ["filter_by_confidence", "hamming_cluster", "mahalanobis_filter"]


def _adaptive_threshold_gmm(values, n_init=5):
    """Compute adaptive threshold using GMM."""
    values = np.array(values)[np.isfinite(values)]
    
    if len(values) == 0:
        return 0.0, None
    
    # Check if there's enough variance for 2-component GMM
    if np.std(values) < 1e-6 or len(np.unique(values)) < 2:
        threshold = float(np.median(values))
        return threshold, {'threshold': threshold, 'note': 'insufficient variance, using median'}
    
    gmm = GaussianMixture(n_components=2, n_init=n_init, random_state=42)
    gmm.fit(values.reshape(-1, 1))
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    
    threshold = means.mean()
    threshold = np.clip(threshold, np.percentile(values, 10), np.percentile(values, 90))
    
    return threshold, {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'weights': weights.tolist(),
        'threshold': float(threshold),
        'n_init': n_init
    }


def filter_by_confidence(adata: ad.AnnData,
                        confidence_col: str,
                        method: str = 'percentile',
                        value: Optional[float] = 90,
                        filter_or_flag: str = 'flag',
                        adaptive_n_init: int = 5,
                        filter_name: Optional[str] = None,
                        inplace: bool = True, 
                        verbose: bool = True) -> ad.AnnData:
    """Filter cells by confidence score.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    confidence_col : str
        Name of confidence column in adata.obs
    method : str
        Filtering method: 'adaptive', 'threshold', 'percentile' (default: 'percentile')
    value : float, optional
        Threshold value (usage depends on method)
    filter_or_flag : str
        'flag' or 'filter' (default: 'flag')
    adaptive_n_init : int
        GMM initializations for adaptive method (default: 5)
    filter_name : str, optional
        Custom name for this filtering run. If None, auto-generates unique name.
    inplace : bool
        Modify adata in place (default: True)
    
    Returns:
    --------
    adata : AnnData
        Modified AnnData with filtering results
    """
    if not inplace:
        adata = adata.copy()
    
    if confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found in adata.obs")
    
    if method not in ['adaptive', 'threshold', 'percentile']:
        raise ValueError(f"method must be 'adaptive', 'threshold', or 'percentile', got '{method}'")
    
    if filter_or_flag not in ['flag', 'filter']:
        raise ValueError(f"filter_or_flag must be 'flag' or 'filter', got '{filter_or_flag}'")
    
    if method in ['threshold', 'percentile'] and value is None:
        raise ValueError(f"method='{method}' requires 'value' parameter")
    
    confidences = adata.obs[confidence_col].values
    gmm_params = None
    
    if method == 'adaptive':
        if verbose:
            print(f"Computing adaptive threshold using GMM (n_init={adaptive_n_init})...")
        threshold, gmm_params = _adaptive_threshold_gmm(confidences, adaptive_n_init)
        if verbose:
            print(f"  Adaptive threshold: {threshold:.4f}")
            if gmm_params and 'means' in gmm_params:
                print(f"  GMM means: {gmm_params['means']}")
    elif method == 'threshold':
        threshold = value
        if verbose:
            print(f"Using manual threshold: {threshold:.4f}")
    else:  # percentile
        threshold = np.percentile(confidences, 100 - value)
        if verbose:
            print(f"Percentile threshold (keep top {value}%): {threshold:.4f}")
    
    pass_mask = confidences >= threshold
    n_pass = pass_mask.sum()
    n_total = len(adata)
    pct_pass = 100 * n_pass / n_total

    if verbose:
        print(f"  Pass: {n_pass}/{n_total} cells ({pct_pass:.1f}%)")
    
    # Handle flag column naming with auto-increment
    if filter_or_flag == 'flag':
        method_name = confidence_col[:confidence_col.rfind('_')]
        base_flag_col = f"{method_name}_pass"
        
        # Auto-increment if column exists
        flag_col = base_flag_col
        counter = 1
        while flag_col in adata.obs.columns:
            flag_col = f"{base_flag_col}_{counter}"
            counter += 1
        
        adata.obs[flag_col] = pass_mask
        
        # Use flag column name as filter_name if not provided
        if filter_name is None:
            filter_name = flag_col
        
        if verbose:
            print(f"  [+] Added flag column: adata.obs['{flag_col}']")
    else:
        adata = adata[pass_mask].copy()
        
        # Use default filter_name if not provided
        if filter_name is None:
            filter_name = confidence_col
        
        if verbose:
            print(f"  [+] Filtered to {len(adata)} cells")
    
    # Store metadata with unique filter_name
    if 'confidence_filtering' not in adata.uns:
        adata.uns['confidence_filtering'] = {}
    
    adata.uns['confidence_filtering'][filter_name] = {
        'confidence_col': confidence_col,
        'method': method,
        'value': value,
        'threshold': float(threshold),
        'filter_or_flag': filter_or_flag,
        'n_pass': int(n_pass),
        'n_total': int(n_total),
        'pct_pass': float(pct_pass),
        'adaptive_n_init': adaptive_n_init if method == 'adaptive' else None
    }
    
    if gmm_params:
        adata.uns['confidence_filtering'][filter_name]['gmm_params'] = gmm_params

    if verbose:
        print(f"  [+] Metadata saved: adata.uns['confidence_filtering']['{filter_name}']")
    
    return adata


def hamming_cluster(adata: ad.AnnData,
                   assignment_col: str,
                   confidence_col: str,
                   radius: int = 2,
                   min_valid_count: int = 20,
                   low_confidence_percentile: Optional[float] = None,
                   min_mean_conf: float = 0.0,
                   score_metric: str = 'count',
                   save_results: bool = True,
                   clustering_name: Optional[str] = None,
                   inplace: bool = True,
                   verbose: bool = True) -> ad.AnnData:
    """Apply Hamming clustering to correct barcode assignments.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    assignment_col : str
        Column with barcode assignments
    confidence_col : str
        Column with confidence scores
    radius : int
        Maximum Hamming distance (default: 2)
    min_valid_count : int
        Minimum count for cluster centers (default: 20)
    low_confidence_percentile : float or None
        Bottom N% eligible for clustering (default: None)
    min_mean_conf : float
        Minimum mean confidence for cluster centers (default: 0.0)
    score_metric : str
        Scoring metric: 'count' or 'mean_conf' (default: 'count')
    save_results : bool
        Save clustering results (default: True)
    clustering_name : str, optional
        Custom name for this clustering run
    inplace : bool
        Modify adata in place (default: True)
    
    Returns:
    --------
    adata : AnnData
        Modified AnnData with clustering results
    """
    if not inplace:
        adata = adata.copy()
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found")
    
    if confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found")
    
    if score_metric not in ['count', 'mean_conf']:
        raise ValueError(f"score_metric must be 'count' or 'mean_conf'")
    
    clustering_name = clustering_name or assignment_col

    if verbose:
        print(f"Hamming clustering: radius={radius}, min_count={min_valid_count}")
        if low_confidence_percentile:
            print(f"  Eligible: bottom {low_confidence_percentile}% by confidence")
        else:
            print(f"  Eligible: all cells")
    
    patterns_str = adata.obs[assignment_col].values
    confidences = adata.obs[confidence_col].values
    n_cells = len(adata)
    
    barcodes = np.array([[int(c) for c in p] for p in patterns_str])
    
    if low_confidence_percentile is None:
        low_conf_mask = np.ones(n_cells, dtype=bool)
    else:
        conf_threshold = np.percentile(confidences, low_confidence_percentile)
        low_conf_mask = confidences <= conf_threshold
    
    n_eligible = low_conf_mask.sum()
    if verbose:
        print(f"  Eligible cells: {n_eligible}/{n_cells}")
    
    patterns = [tuple(bc) for bc in barcodes]
    pattern_counts = Counter(patterns)
    
    pattern_stats = {}
    for pattern in pattern_counts:
        pattern_mask = np.array([patterns[i] == pattern for i in range(n_cells)])
        pattern_confs = confidences[pattern_mask]
        pattern_stats[pattern] = {
            'count': pattern_counts[pattern],
            'mean': np.mean(pattern_confs),
            'median': np.median(pattern_confs)
        }
    
    absorbing_patterns = [p for p, stats in pattern_stats.items() 
                         if is_valid_pattern(p) and stats['count'] > min_valid_count and stats['mean'] > min_mean_conf]

    if verbose:
        print(f"  Absorbing patterns: {len(absorbing_patterns)}")
        
    method_name = assignment_col[:assignment_col.rfind('_')]
    
    if len(absorbing_patterns) == 0:
        print("  [!] No valid absorbing patterns found. Skipping clustering.")
        adata.obs[f"{method_name}_hamming_assignment"] = adata.obs[assignment_col]
        adata.obs[f"{method_name}_hamming_confidence"] = adata.obs[confidence_col]
        adata.obs[f"{method_name}_hamming_remapped"] = False
        return adata
    
    pattern_scores = {p: (pattern_stats[p]['count'] if score_metric == 'count' else pattern_stats[p]['mean'])
                     for p in absorbing_patterns}
    
    absorbing_array = np.array([list(p) for p in absorbing_patterns], dtype=np.int8)
    absorbing_scores = np.array([pattern_scores[p] for p in absorbing_patterns])
    
    clustered_barcodes = barcodes.copy()
    updated_confidences = confidences.copy()
    remapped = np.zeros(n_cells, dtype=bool)
    
    for i in np.where(low_conf_mask)[0]:
        current_pattern = patterns[i]
        current_score = pattern_scores.get(current_pattern, -1)
        
        hamming_dists = np.sum(absorbing_array != barcodes[i], axis=1)
        valid_mask = (hamming_dists > 0) & (hamming_dists <= radius) & (hamming_dists % 2 == 0)
        
        if not valid_mask.any():
            continue
        
        valid_indices = np.where(valid_mask)[0]
        valid_dists = hamming_dists[valid_mask]
        valid_scores = absorbing_scores[valid_mask]
        
        min_dist = valid_dists.min()
        at_min_dist = valid_dists == min_dist
        best_idx = valid_indices[at_min_dist][np.argmax(valid_scores[at_min_dist])]
        
        if absorbing_scores[best_idx] > current_score:
            best_pattern = absorbing_patterns[best_idx]
            clustered_barcodes[i] = np.array(best_pattern)
            updated_confidences[i] = pattern_stats[best_pattern]['median']
            remapped[i] = True
    
    n_remapped = remapped.sum()
    pct_remapped = 100 * n_remapped / n_cells
    if verbose:
        print(f"  Remapped: {n_remapped}/{n_cells} cells ({pct_remapped:.1f}%)")
    
    adata.obs[f"{method_name}_hamming_assignment"] = ["".join(map(str, row)) for row in clustered_barcodes]
    adata.obs[f"{method_name}_hamming_confidence"] = updated_confidences
    adata.obs[f"{method_name}_hamming_remapped"] = remapped
    
    if save_results:
        if 'hamming_clustering' not in adata.uns:
            adata.uns['hamming_clustering'] = {}
        
        adata.uns['hamming_clustering'][clustering_name] = {
            'clustering_name': clustering_name,
            'assignment_col': assignment_col,
            'confidence_col': confidence_col,
            'radius': radius,
            'min_valid_count': min_valid_count,
            'low_confidence_percentile': low_confidence_percentile,
            'min_mean_conf': min_mean_conf,
            'score_metric': score_metric,
            'n_absorbing_patterns': len(absorbing_patterns),
            'n_eligible': int(n_eligible),
            'n_remapped': int(n_remapped),
            'pct_remapped': float(pct_remapped),
            'output_assignment_col': f"{method_name}_hamming_assignment",
            'output_confidence_col': f"{method_name}_hamming_confidence",
            'output_remapped_col': f"{assignment_col}_hamming_remapped"
        }
        if verbose:
            print(f"  [+] Clustering saved in adata.uns['hamming_clustering']['{clustering_name}']")

    if verbose:
        print(f"  [+] Results stored in:")
        print(f"    - adata.obs['{method_name}_hamming_assignment']")
        print(f"    - adata.obs['{method_name}_hamming_confidence']")
        print(f"    - adata.obs['{method_name}_hamming_remapped']")
    
    return adata


def mahalanobis_filter(adata: ad.AnnData,
                      assignment_col: str,
                      confidence_col: Optional[str] = None,
                      layer: Optional[str] = None,
                      mahal_threshold: Optional[float] = None,
                      confidence_threshold: Optional[float] = None,
                      filter_or_flag: str = 'flag',
                      min_cells_for_cov: int = 10,
                      filter_name: Optional[str] = None,
                      inplace: bool = True,
                      verbose: bool = True) -> ad.AnnData:
    """Filter cells by Mahalanobis distance from barcode pattern centroid.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    assignment_col : str
        Column name with barcode assignments (string patterns)
    confidence_col : str, optional
        Column name with confidence scores
    layer : str, optional
        Layer to use for distance calculation
    mahal_threshold : float, optional
        Maximum Mahalanobis distance threshold
    confidence_threshold : float, optional
        Minimum confidence threshold
    filter_or_flag : str
        'flag' or 'filter' (default: 'flag')
    min_cells_for_cov : int
        Minimum cells per pattern to compute covariance (default: 10)
    filter_name : str, optional
        Custom name for this filtering run
    inplace : bool
        Modify adata in place (default: True)
    
    Returns:
    --------
    adata : AnnData
        Modified AnnData with filtering results
    """
    if not inplace:
        adata = adata.copy()
    
    from .io import get_barcode_channels
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    
    if confidence_col and confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found in adata.obs")
    
    if filter_or_flag not in ['flag', 'filter']:
        raise ValueError(f"filter_or_flag must be 'flag' or 'filter', got '{filter_or_flag}'")
    
    barcode_channels = get_barcode_channels(adata)
    filter_name = filter_name or assignment_col

    if verbose:
        print(f"Mahalanobis distance filtering...")
        print(f"  Assignment: {assignment_col}")
        if confidence_col:
            print(f"  Confidence: {confidence_col}")
        print(f"  Layer: {layer or '.X (raw)'}")
    
    if layer and layer in adata.layers:
        X = np.column_stack([adata[:, ch].layers[layer].flatten() for ch in barcode_channels])
    else:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        indices = [adata.var_names.get_loc(ch) for ch in barcode_channels]
        X = X[:, indices]
    
    n_cells = len(adata)
    patterns_str = adata.obs[assignment_col].values
    mahal_distances = np.full(n_cells, np.nan)
    
    unique_patterns = np.unique(patterns_str)

    if verbose:
        print(f"  Computing Mahalanobis distances for {len(unique_patterns)} unique patterns...")
    
    patterns_computed = 0
    patterns_skipped = 0
    
    from scipy.spatial.distance import mahalanobis as sp_mahal
    
    for pattern_str in unique_patterns:
        pattern_mask = patterns_str == pattern_str
        n_pattern_cells = pattern_mask.sum()
        
        if n_pattern_cells < min_cells_for_cov:
            patterns_skipped += 1
            continue
        
        X_pattern = X[pattern_mask, :]
        
        try:
            centroid = X_pattern.mean(axis=0)
            cov_inv = np.linalg.inv(np.cov(X_pattern.T))
            distances = np.array([sp_mahal(X_pattern[i], centroid, cov_inv) for i in range(n_pattern_cells)])
            mahal_distances[pattern_mask] = np.minimum(distances, 30.0)
            patterns_computed += 1
        except Exception as e:
            print(f"    Warning: Could not compute Mahalanobis for pattern {pattern_str[:20]}... ({e})")
            patterns_skipped += 1
    
    if verbose:
        print(f"  Computed distances for {patterns_computed} patterns, skipped {patterns_skipped}")
    
    valid_distances = mahal_distances[~np.isnan(mahal_distances)]
    
    if mahal_threshold is None:
        mahal_threshold = float(np.percentile(valid_distances, 95)) if len(valid_distances) > 0 else 30.0
        if verbose:
            print(f"  {'Auto' if len(valid_distances) > 0 else 'Default'} Mahalanobis threshold: {mahal_threshold:.2f}")
    elif verbose:
        print(f"  Manual Mahalanobis threshold: {mahal_threshold:.2f}")
    
    if confidence_col:
        confidences = adata.obs[confidence_col].values
        
        if confidence_threshold is None:
            confidence_threshold = float(np.percentile(confidences, 25))
            if verbose:
                print(f"  Auto confidence threshold (25th percentile): {confidence_threshold:.4f}")
        elif verbose:
            print(f"  Manual confidence threshold: {confidence_threshold:.4f}")
        
        fail_mask = ((confidences < confidence_threshold) & (mahal_distances > mahal_threshold)) | np.isnan(mahal_distances)
    else:
        confidence_threshold = None
        fail_mask = (mahal_distances > mahal_threshold) | np.isnan(mahal_distances)
    
    pass_mask = ~fail_mask
    n_pass = pass_mask.sum()
    pct_pass = 100 * n_pass / n_cells

    if verbose:
        print(f"  Pass: {n_pass}/{n_cells} cells ({pct_pass:.1f}%)")

    method_name = assignment_col[:assignment_col.rfind('_')]
    mahal_col = f"{method_name}_mahal_dist"
    adata.obs[mahal_col] = mahal_distances
    
    if filter_or_flag == 'flag':
        flag_col = f"{method_name}_mahal_pass"
        adata.obs[flag_col] = pass_mask
        if verbose:
            print(f"  [+] Added flag: adata.obs['{flag_col}']")
            print(f"  [+] Added distance: adata.obs['{mahal_col}']")
    else:
        adata = adata[pass_mask].copy()
        if verbose:
            print(f"  [+] Filtered to {len(adata)} cells")
            print(f"  [+] Distance: adata.obs['{mahal_col}']")
    
    if 'mahalanobis_filtering' not in adata.uns:
        adata.uns['mahalanobis_filtering'] = {}
    
    adata.uns['mahalanobis_filtering'][filter_name] = {
        'filter_name': filter_name,
        'assignment_col': assignment_col,
        'confidence_col': confidence_col,
        'layer': layer,
        'mahal_threshold': float(mahal_threshold),
        'confidence_threshold': float(confidence_threshold) if confidence_threshold else None,
        'filter_or_flag': filter_or_flag,
        'min_cells_for_cov': int(min_cells_for_cov),
        'n_patterns_computed': int(patterns_computed),
        'n_patterns_skipped': int(patterns_skipped),
        'n_pass': int(n_pass),
        'n_fail': int(n_cells - n_pass),
        'pct_pass': float(pct_pass),
        'mahal_dist_col': mahal_col
    }

    if verbose:
        print(f"  [+] Filter parameters saved in adata.uns['mahalanobis_filtering']['{filter_name}']")
    
    return adata