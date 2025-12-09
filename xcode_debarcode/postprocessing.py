"""Postprocessing methods for confidence filtering and Hamming clustering."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import anndata as ad
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import Counter, defaultdict
from typing import Optional, Dict, Tuple, List
from .barcode import is_valid_pattern

__all__ = ["filter_cells_conf", "filter_pattern", "hamming_cluster", "mahalanobis_filter"]


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


def _build_hamming_neighbors(unique_patterns, max_radius=2):
    """Build Hamming neighbor graph up to max_radius."""
    if not unique_patterns:
        return {}
    
    arr = np.array([[int(c) for c in p] for p in unique_patterns], dtype=np.int8)
    dists = (arr[:, None] != arr).sum(2)
    
    neighbors = {p: {} for p in unique_patterns}
    for i, j in zip(*np.where(np.triu((dists >= 1) & (dists <= max_radius), k=1))):
        d = int(dists[i, j])
        neighbors[unique_patterns[i]][unique_patterns[j]] = d
        neighbors[unique_patterns[j]][unique_patterns[i]] = d
    
    return neighbors


def _lda_classify(p: str, candidates: List[str], pattern_cells: Dict[str, set],
                  X: np.ndarray, min_n: int = 5) -> Optional[Dict[str, List[int]]]:
    """LDA tie-breaking using sklearn. Returns cell assignments or None if insufficient data."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    cells_p = pattern_cells.get(p, set())
    if not cells_p:
        return None
    
    cells_p_arr = np.array(list(cells_p))
    
    # Include source only if valid
    p_valid = is_valid_pattern(p)
    class_list = [p] + list(candidates) if p_valid else list(candidates)
    
    X_train_idx, y_train = [], []
    label_map = {}
    for c in class_list:
        cells_c = pattern_cells.get(c, set())
        if len(cells_c) >= min_n:
            label = len(label_map)
            label_map[label] = c
            for idx in cells_c:
                X_train_idx.append(idx)
                y_train.append(label)
    
    if len(label_map) < 2:
        return None
    
    # Get informative features
    p_bits = np.array([int(c) for c in p], dtype=np.int8)
    info_mask = np.zeros(X.shape[1], dtype=bool)
    for c in candidates:
        info_mask |= (p_bits != np.array([int(ch) for ch in c], dtype=np.int8))
    info_idx = np.where(info_mask)[0]
    if len(info_idx) == 0:
        return None
    
    X_train = X[X_train_idx][:, info_idx]
    y_train = np.array(y_train)
    
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(X_train, y_train)
    best_labels = lda.predict(X[cells_p_arr][:, info_idx])
    
    result = defaultdict(list)
    for idx, label in enumerate(best_labels):
        result[label_map[label]].append(cells_p_arr[idx])
    return dict(result)


def _msg_cluster(patterns_str: np.ndarray, confidences: np.ndarray,
                 X: Optional[np.ndarray], radius: int, ratio: float, test_conf: bool,
                 tie_break: str, ratio_metric: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Message passing clustering."""
    n_cells = len(patterns_str)
    
    pattern_to_cells = defaultdict(list)
    for i, p in enumerate(patterns_str):
        pattern_to_cells[p].append(i)
    
    pattern_stats, pattern_cells = {}, {}
    for p, indices in pattern_to_cells.items():
        arr = np.array(indices)
        count = len(arr)
        median = float(np.median(confidences[arr]))
        pattern_stats[p] = {'count': count, 'median': median, 'is_valid': is_valid_pattern(p)}
        pattern_cells[p] = set(indices)
    
    neighbors = _build_hamming_neighbors(list(pattern_stats.keys()), max_radius=radius)
    all_patterns = list(pattern_stats.keys())
    if not all_patterns:
        return patterns_str.copy(), confidences.copy(), {'n_remapped': 0}
    
    sorted_patterns = sorted(all_patterns, key=lambda p: (pattern_stats[p]['count'], pattern_stats[p]['median']))
    mass = {p: pattern_stats[p]['count'] for p in pattern_stats}
    center_of = {p: p for p in all_patterns}
    stats = {'n_ties': 0, 'n_lda': 0, 'n_stayed': 0, 'n_moved': 0}
    
    for p in sorted_patterns:
        cells_p = pattern_cells.get(p, set())
        if not cells_p:
            continue
        
        N_p, conf_p = len(cells_p), pattern_stats[p]['median']
        score_p = N_p * conf_p
        p_valid = pattern_stats[p]['is_valid']
        
        candidates_with_dist = []
        for q, dist in neighbors.get(p, {}).items():
            if not pattern_stats.get(q, {}).get('is_valid'):
                continue
            if p_valid and dist % 2 != 0:
                continue
            if p_valid:
                if ratio_metric == 'score':
                    if mass.get(q, 0) * pattern_stats[q]['median'] < ratio * score_p:
                        continue
                else:
                    if mass.get(q, 0) < ratio * N_p:
                        continue
                if test_conf and pattern_stats[q]['median'] <= conf_p:
                    continue
            candidates_with_dist.append((q, dist))
        
        if not candidates_with_dist:
            continue
        
        # Pick candidates at minimum distance
        min_dist = min(d for _, d in candidates_with_dist)
        candidates = [q for q, d in candidates_with_dist if d == min_dist]
        
        if len(candidates) > 1:
            stats['n_ties'] += 1
            if tie_break == 'no_remap':
                continue
            if tie_break == 'lda' and X is not None:
                result = _lda_classify(p, candidates, pattern_cells, X)
                if result:
                    stats['n_lda'] += 1
                    for tgt, cells in result.items():
                        if tgt == p:
                            stats['n_stayed'] += len(cells)
                        else:
                            stats['n_moved'] += len(cells)
                            for c in cells:
                                pattern_cells[p].discard(c)
                                pattern_cells.setdefault(tgt, set()).add(c)
                            mass[p], mass[tgt] = len(pattern_cells[p]), len(pattern_cells[tgt])
                    continue
        
        best_q = max(candidates, key=lambda q: mass.get(q, 0))
        stats['n_moved'] += len(cells_p)
        for c in list(cells_p):
            pattern_cells[p].discard(c)
            pattern_cells.setdefault(best_q, set()).add(c)
        mass[p], mass[best_q] = 0, len(pattern_cells[best_q])
        center_of[p] = best_q
    
    # Resolve chains
    for p in all_patterns:
        curr, visited = p, set()
        while center_of.get(curr, curr) != curr and curr not in visited:
            visited.add(curr)
            curr = center_of[curr]
        center_of[p] = curr
    
    cell_to_pat = {c: p for p, cells in pattern_cells.items() for c in cells}
    new_patterns = np.array(patterns_str, dtype=object)
    new_confidences = confidences.copy()
    n_remapped = 0
    
    # Compute median confidence per final pattern (non-remapped cells only)
    final_pattern_cells = defaultdict(list)
    for i in range(n_cells):
        final = center_of.get(cell_to_pat.get(i, patterns_str[i]), patterns_str[i])
        if final == patterns_str[i]:
            final_pattern_cells[final].append(i)
    
    pattern_median_conf = {}
    for p, cells in final_pattern_cells.items():
        pattern_median_conf[p] = float(np.median(confidences[cells]))
    
    for i in range(n_cells):
        final = center_of.get(cell_to_pat.get(i, patterns_str[i]), patterns_str[i])
        if final != patterns_str[i]:
            n_remapped += 1
            new_patterns[i] = final
            new_confidences[i] = pattern_median_conf.get(final, confidences[i])
    
    stats['n_remapped'] = n_remapped
    return new_patterns, new_confidences, stats


def _sphere_cluster(patterns_str: np.ndarray, confidences: np.ndarray,
                    X: Optional[np.ndarray], radius: int, ratio: float, test_conf: bool,
                    tie_break: str, min_count_center: int, ratio_metric: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Sphere clustering: local maxima centers, remap within radius."""
    n_cells = len(patterns_str)
    
    pattern_to_cells = defaultdict(list)
    for i, p in enumerate(patterns_str):
        pattern_to_cells[p].append(i)
    
    pattern_stats, pattern_cells = {}, {}
    for p, indices in pattern_to_cells.items():
        arr = np.array(indices)
        count = len(arr)
        median = float(np.median(confidences[arr]))
        pattern_stats[p] = {'count': count, 'median': median, 'is_valid': is_valid_pattern(p)}
        pattern_cells[p] = set(indices)
    
    unique = list(pattern_stats.keys())
    neighbors = _build_hamming_neighbors(unique, max_radius=radius)
    
    # Centers: valid local maxima
    centers = [p for p, s in pattern_stats.items()
               if s['is_valid'] and s['count'] >= min_count_center
               and all(pattern_stats.get(q, {}).get('count', 0) <= s['count'] 
                      for q, d in neighbors.get(p, {}).items() if d == 2)]
    
    if not centers:
        return patterns_str.copy(), confidences.copy(), {'n_centers': 0, 'n_remapped': 0, 'centers': []}
    
    center_arr = np.array([[int(c) for c in p] for p in centers], dtype=np.int8)
    center_set = set(centers)
    
    non_centers = sorted([p for p in unique if p not in center_set],
                        key=lambda p: (pattern_stats[p]['count'], pattern_stats[p]['median']))
    
    mass = {p: pattern_stats[p]['count'] for p in pattern_stats}
    parent = {}
    stats = {'n_centers': len(centers), 'n_ties': 0, 'n_lda': 0, 'n_stayed': 0, 'n_moved': 0, 'centers': centers}
    
    for p in non_centers:
        cells_p = pattern_cells.get(p, set())
        if not cells_p:
            continue
        
        N_p, conf_p = len(cells_p), pattern_stats[p]['median']
        score_p = N_p * conf_p
        p_valid = pattern_stats[p]['is_valid']
        p_arr = np.array([[int(c) for c in p]], dtype=np.int8)
        dists = (center_arr != p_arr).sum(1)
        
        valid_idx = []
        for i in range(len(centers)):
            d = dists[i]
            if d == 0 or d > radius:
                continue
            if p_valid and d % 2 != 0:
                continue
            if p_valid:
                if ratio_metric == 'score':
                    if mass.get(centers[i], 0) * pattern_stats[centers[i]]['median'] < ratio * score_p:
                        continue
                else:
                    if mass.get(centers[i], 0) < ratio * N_p:
                        continue
                if test_conf and pattern_stats[centers[i]]['median'] <= conf_p:
                    continue
            valid_idx.append((i, d))
        
        if not valid_idx:
            continue
        
        min_dist = min(d for _, d in valid_idx)
        closest = [centers[i] for i, d in valid_idx if d == min_dist]
        
        if len(closest) > 1:
            stats['n_ties'] += 1
            if tie_break == 'no_remap':
                continue
            if tie_break == 'lda' and X is not None:
                result = _lda_classify(p, closest, pattern_cells, X)
                if result:
                    stats['n_lda'] += 1
                    for tgt, cells in result.items():
                        if tgt == p:
                            stats['n_stayed'] += len(cells)
                        else:
                            stats['n_moved'] += len(cells)
                            parent[p] = tgt
                            for c in cells:
                                pattern_cells[p].discard(c)
                                pattern_cells.setdefault(tgt, set()).add(c)
                            mass[p], mass[tgt] = len(pattern_cells[p]), len(pattern_cells[tgt])
                    continue
        
        best = max(closest, key=lambda c: mass.get(c, 0))
        parent[p] = best
        stats['n_moved'] += len(cells_p)
        for c in list(cells_p):
            pattern_cells[p].discard(c)
            pattern_cells.setdefault(best, set()).add(c)
        mass[p], mass[best] = 0, len(pattern_cells[best])
    
    new_patterns = np.array(patterns_str, dtype=object)
    new_confidences = confidences.copy()
    n_remapped = 0
    
    # Compute median confidence per final pattern (non-remapped cells only)
    final_pattern_cells = defaultdict(list)
    for i in range(n_cells):
        final = parent.get(patterns_str[i], patterns_str[i])
        if final == patterns_str[i]:
            final_pattern_cells[final].append(i)
    
    pattern_median_conf = {}
    for p, cells in final_pattern_cells.items():
        pattern_median_conf[p] = float(np.median(confidences[cells]))
    
    for i in range(n_cells):
        final = parent.get(patterns_str[i], patterns_str[i])
        if final != patterns_str[i]:
            n_remapped += 1
            new_patterns[i] = final
            new_confidences[i] = pattern_median_conf.get(final, confidences[i])
    
    stats['n_remapped'] = n_remapped
    return new_patterns, new_confidences, stats


def filter_cells_conf(adata: ad.AnnData,
                        confidence_col: str,
                        method: str = 'percentile',
                        value: Optional[float] = 90,
                        filter_or_flag: str = 'flag',
                        adaptive_n_init: int = 5,
                        filter_name: Optional[str] = None,
                        inplace: bool = True, 
                        verbose: bool = True) -> ad.AnnData:
    """Filter cells by confidence score.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    confidence_col : str
        Name of confidence column in adata.obs.
    method : str
        Filtering method. Default: 'percentile'.
        - 'threshold': keep cells with confidence >= value
        - 'percentile': keep top value% of cells by confidence
        - 'adaptive': GMM-based automatic threshold
    value : float, optional
        For 'threshold': minimum confidence to pass.
        For 'percentile': percentage of top cells to keep (e.g., 90 keeps top 90%).
        Default: 90.
    filter_or_flag : str
        'flag' to add boolean column, 'filter' to remove cells. Default: 'flag'.
    adaptive_n_init : int
        GMM initializations for adaptive method. Default: 5.
    filter_name : str, optional
        Custom name for this filtering run. If None, auto-generates unique name.
    inplace : bool
        Modify adata in place. Default: True.
    
    Returns
    -------
    AnnData with filtering results.
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

def filter_pattern(adata: ad.AnnData,
                   assignment_col: str,
                   confidence_col: Optional[str] = None,
                   metric: str = 'count',
                   method: str = 'threshold',
                   value: float = 10,
                   filter_or_flag: str = 'flag',
                   filter_name: Optional[str] = None,
                   inplace: bool = True,
                   verbose: bool = True) -> ad.AnnData:
    """Filter cells by barcode pattern statistics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    assignment_col : str
        Name of barcode assignment column in adata.obs.
    confidence_col : str, optional
        Name of confidence column in adata.obs. Required when metric is
        'median_conf' or 'score'.
    metric : str
        Pattern metric. Default: 'count'.
        - 'count': number of cells per pattern
        - 'median_conf': median confidence per pattern
        - 'score': count * median_conf
    method : str
        Filtering method. Default: 'threshold'.
        - 'threshold': keep patterns with metric >= value
        - 'percentile': keep top value% of patterns by metric
    value : float
        For 'threshold': minimum metric value to pass.
        For 'percentile': percentage of top patterns to keep (e.g., 90 keeps top 90%).
        Default: 10.
    filter_or_flag : str
        'flag' to add boolean column, 'filter' to remove cells. Default: 'flag'.
    filter_name : str, optional
        Custom name for this filtering run. If None, auto-generates unique name.
    inplace : bool
        Modify adata in place. Default: True.
    verbose : bool
        Print progress. Default: True.
    
    Returns
    -------
    AnnData with filtering results.
    """
    if not inplace:
        adata = adata.copy()
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    if metric not in {'count', 'median_conf', 'score'}:
        raise ValueError(f"metric must be 'count', 'median_conf', or 'score', got {metric!r}")
    if metric in {'median_conf', 'score'} and confidence_col is None:
        raise ValueError(f"confidence_col is required when metric is '{metric}'")
    if confidence_col is not None and confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found in adata.obs")
    if method not in {'threshold', 'percentile'}:
        raise ValueError(f"method must be 'threshold' or 'percentile', got {method!r}")
    if filter_or_flag not in {'flag', 'filter'}:
        raise ValueError(f"filter_or_flag must be 'flag' or 'filter', got {filter_or_flag!r}")
    
    patterns = adata.obs[assignment_col].values.astype(str)
    confidences = adata.obs[confidence_col].values.astype(np.float64) if confidence_col else None
    
    # Compute per-pattern statistics
    pattern_to_cells = {}
    for i, p in enumerate(patterns):
        if p not in pattern_to_cells:
            pattern_to_cells[p] = []
        pattern_to_cells[p].append(i)
    
    pattern_stats = {}
    for p, indices in pattern_to_cells.items():
        count = len(indices)
        stats = {'count': count}
        if confidences is not None:
            median_conf = float(np.median(confidences[indices]))
            stats['median_conf'] = median_conf
            stats['score'] = count * median_conf
        pattern_stats[p] = stats
    
    metric_values = np.array([pattern_stats[p][metric] for p in pattern_stats])
    
    if method == 'threshold':
        threshold = value
        if verbose:
            print(f"Filter by pattern {metric} >= {threshold}")
    else:  
        threshold = np.percentile(metric_values, 100 - value)
        if verbose:
            print(f"Filter by pattern {metric} (top {value}%): threshold = {threshold:.4f}")
    
    passing_patterns = {p for p, stats in pattern_stats.items() if stats[metric] >= threshold}
    n_patterns_pass = len(passing_patterns)
    n_patterns_total = len(pattern_stats)
    
    pass_mask = np.array([p in passing_patterns for p in patterns])
    n_pass = pass_mask.sum()
    n_total = len(adata)
    pct_pass = 100 * n_pass / n_total
    
    if verbose:
        print(f"  Patterns passing: {n_patterns_pass}/{n_patterns_total}")
        print(f"  Cells passing: {n_pass}/{n_total} ({pct_pass:.1f}%)")
    
    # Handle flag column naming with auto-increment
    if filter_or_flag == 'flag':
        base_name = assignment_col.rsplit('_', 1)[0] if '_' in assignment_col else assignment_col
        base_flag_col = f"{base_name}_pattern_pass"
        
        flag_col = base_flag_col
        counter = 1
        while flag_col in adata.obs.columns:
            flag_col = f"{base_flag_col}_{counter}"
            counter += 1
        
        adata.obs[flag_col] = pass_mask
        
        if filter_name is None:
            filter_name = flag_col
        
        if verbose:
            print(f"  [+] Added flag column: adata.obs['{flag_col}']")
    else:
        adata = adata[pass_mask].copy()
        
        if filter_name is None:
            filter_name = assignment_col
        
        if verbose:
            print(f"  [+] Filtered to {len(adata)} cells")
    
    if 'pattern_filtering' not in adata.uns:
        adata.uns['pattern_filtering'] = {}
    
    adata.uns['pattern_filtering'][filter_name] = {
        'assignment_col': assignment_col,
        'confidence_col': confidence_col,
        'metric': metric,
        'method': method,
        'value': value,
        'threshold': float(threshold),
        'filter_or_flag': filter_or_flag,
        'n_patterns_pass': int(n_patterns_pass),
        'n_patterns_total': int(n_patterns_total),
        'n_cells_pass': int(n_pass),
        'n_cells_total': int(n_total),
        'pct_cells_pass': float(pct_pass),
    }
    
    if verbose:
        print(f"  [+] Metadata saved: adata.uns['pattern_filtering']['{filter_name}']")
    
    return adata


def hamming_cluster(adata: ad.AnnData,
                    assignment_col: str,
                    confidence_col: str,
                    method: str = 'msg',
                    radius: int = 2,
                    ratio: float = 15.0,
                    ratio_metric: str = 'count',
                    tie_break: str = 'lda',
                    test_conf: bool = True,
                    low_conf_perc: Optional[float] = None,
                    min_count_center: int = 1,
                    layer: str = 'log',
                    save_results: bool = True,
                    inplace: bool = True,
                    verbose: bool = True) -> ad.AnnData:
    """Apply Hamming clustering to correct barcode assignments.
    
    Iteratively merges small/low-confidence patterns into larger neighbors.
    Handles both valid patterns (4-of-9 in each block) and invalid patterns
    (which are remapped unconditionally to nearest valid neighbor within radius).
    
    Recommended settings
    --------------------
    18-channel (2 blocks):
        Default settings work well: method='msg', ratio=15, tie_break='lda', test_conf=True
    
    27-channel (3 blocks):
        Lower ratio recommended due to sparser barcode space: ratio=5
    
    Small K (few barcodes):
        Consider lowering ratio (e.g., 5-10) as patterns are more isolated.
    
    When to use 'sphere':
        For more local control. Set min_count_center to a threshold above which 
        patterns are likely real barcodes. Adjust radius based on expected Hamming 
        distance between true barcodes (typically 2-4).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with debarcoding results.
    assignment_col : str
        Column in adata.obs with barcode assignments.
    confidence_col : str
        Column in adata.obs with confidence scores (used for ratio_metric='score' 
        and test_conf).
    method : str
        'msg' (message passing) or 'sphere' (local maxima). Default: 'msg'.
    radius : int
        Max Hamming distance for neighbor search. Default: 2.
    ratio : float
        Minimum ratio for remapping valid patterns. Default: 15.0.
        Use ~15 for 18ch, ~5 for 27ch or small K.
    ratio_metric : str
        'count' (N_q >= ratio * N_p) or 'score' (count*conf_q >= ratio * count*conf_p). 
        Default: 'count'.
    tie_break : str
        'no_remap', 'count', or 'lda'. Default: 'lda'.
    test_conf : bool
        Only remap valid patterns to higher confidence neighbors. Default: True.
    low_conf_perc : float or None
        Only remap bottom N% confidence cells. Default: None (all).
    min_count_center : int
        Min count for centers (sphere only). Default: 1.
    layer : str
        Data layer for LDA. Default: 'log'.
    save_results : bool
        Save metadata to adata.uns. Default: True.
    inplace : bool
        Modify in place. Default: True.
    verbose : bool
        Print progress. Default: True.
    
    Returns
    -------
    AnnData with columns added:
        - `{base}_hamming_assignment`: corrected assignments
        - `{base}_hamming_confidence`: confidence scores (remapped cells receive the
          median confidence of their target pattern)
        - `{base}_hamming_remapped`: boolean mask of remapped cells
    where `base` is `assignment_col` with trailing suffix (e.g. '_assignment') stripped.
    
    Notes
    -----
    Invalid patterns (not 4-of-9 per block) are always remapped to the nearest
    valid neighbor within radius without ratio/confidence tests. Valid patterns 
    are remapped only to even-distance valid neighbors that pass ratio and 
    confidence tests.
    
    Remapped cells are assigned the median confidence of the pattern they are
    remapped to (computed from non-remapped cells only).
    """
    if method not in {"msg", "sphere"}:
        raise ValueError(f"method must be 'msg' or 'sphere', got {method!r}")
    if ratio_metric not in {"count", "score"}:
        raise ValueError(f"ratio_metric must be 'count' or 'score', got {ratio_metric!r}")
    if tie_break not in {"no_remap", "count", "lda"}:
        raise ValueError(f"tie_break must be 'no_remap', 'count', or 'lda', got {tie_break!r}")
    
    if not inplace:
        adata = adata.copy()
    
    base_name = assignment_col.rsplit('_', 1)[0] if '_' in assignment_col else assignment_col
    patterns_str = adata.obs[assignment_col].values.astype(str)
    confidences = adata.obs[confidence_col].values.astype(np.float64)
    n_cells = len(adata)
    
    if verbose:
        print(f"Hamming clustering: method={method}, radius={radius}, ratio={ratio}, "
              f"ratio_metric={ratio_metric}, tie_break={tie_break}, test_conf={test_conf}")
    
    X = None
    if tie_break == 'lda' and layer in adata.layers:
        X = adata.layers[layer].astype(np.float64)
        if 'barcode_channels' in adata.uns:
            bc_idx = [list(adata.var_names).index(ch) for ch in adata.uns['barcode_channels']]
            X = X[:, bc_idx]
    
    if low_conf_perc is not None:
        eligible = confidences <= np.percentile(confidences, low_conf_perc)
        eligible_idx = np.where(eligible)[0]
        n_eligible = len(eligible_idx)
        if verbose:
            print(f"  Eligible: bottom {low_conf_perc}% ({n_eligible}/{n_cells} cells)")
    else:
        eligible_idx = np.arange(n_cells)
        n_eligible = n_cells
        if verbose:
            print(f"  Eligible: all cells ({n_cells})")
    
    p_sub, c_sub = patterns_str[eligible_idx], confidences[eligible_idx]
    X_sub = X[eligible_idx] if X is not None else None
    
    if method == 'msg':
        new_p, new_c, stats = _msg_cluster(p_sub, c_sub, X_sub, radius, ratio, test_conf, tie_break, ratio_metric)
    else:
        new_p, new_c, stats = _sphere_cluster(p_sub, c_sub, X_sub, radius, ratio, test_conf, tie_break, min_count_center, ratio_metric)
    
    new_patterns = patterns_str.copy()
    new_confidences = confidences.copy()
    new_patterns[eligible_idx] = new_p
    new_confidences[eligible_idx] = new_c
    remapped = new_patterns != patterns_str
    n_remapped = remapped.sum()
    pct_remapped = 100 * n_remapped / n_cells
    
    if verbose:
        print(f"  Remapped: {n_remapped}/{n_cells} cells ({pct_remapped:.1f}%)")
        if stats.get('n_ties', 0) > 0:
            print(f"  Ties encountered: {stats['n_ties']}, LDA applied: {stats.get('n_lda', 0)}")
        if 'n_centers' in stats:
            print(f"  Centers found: {stats['n_centers']}")
    
    adata.obs[f"{base_name}_hamming_assignment"] = new_patterns
    adata.obs[f"{base_name}_hamming_confidence"] = new_confidences
    adata.obs[f"{base_name}_hamming_remapped"] = remapped
    
    if save_results:
        if 'hamming_clustering' not in adata.uns:
            adata.uns['hamming_clustering'] = {}
        
        metadata = {
            'method': method,
            'radius': radius,
            'ratio': ratio,
            'ratio_metric': ratio_metric,
            'tie_break': tie_break,
            'test_conf': test_conf,
            'low_conf_perc': low_conf_perc,
            'min_count_center': min_count_center if method == 'sphere' else None,
            'n_eligible': int(n_eligible),
            'n_remapped': int(n_remapped),
            'pct_remapped': float(pct_remapped),
            'output_assignment_col': f"{base_name}_hamming_assignment",
            'output_confidence_col': f"{base_name}_hamming_confidence",
            'output_remapped_col': f"{base_name}_hamming_remapped",
            'n_ties': stats.get('n_ties', 0),
            'n_lda': stats.get('n_lda', 0),
            'n_stayed': stats.get('n_stayed', 0),
            'n_moved': stats.get('n_moved', 0),
        }
        if method == 'sphere':
            metadata['n_centers'] = stats.get('n_centers', 0)
            metadata['centers'] = stats.get('centers', [])
        
        adata.uns['hamming_clustering'][assignment_col] = metadata
        if verbose:
            print(f"  [+] Clustering saved in adata.uns['hamming_clustering']['{assignment_col}']")
    
    if verbose:
        print(f"  [+] Results stored in:")
        print(f"      - adata.obs['{base_name}_hamming_assignment']")
        print(f"      - adata.obs['{base_name}_hamming_confidence']")
        print(f"      - adata.obs['{base_name}_hamming_remapped']")
    
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