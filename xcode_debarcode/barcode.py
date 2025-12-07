"""Barcode pattern analysis and validation utilities."""
import numpy as np
import anndata as ad
from typing import Optional, List, Union, Tuple 
from itertools import combinations

__all__ = ["is_valid_pattern", "add_barcode_validity", "pattern_to_indices", 
           "indices_to_pattern", "hamming_distance", "get_pattern_statistics"]


def _generate_4_of_9_patterns(n_blocks: int = 1) -> np.ndarray:
    """Generate all valid 4-of-9 patterns for n blocks."""
    # Generate single block patterns
    patterns = []
    for combo in combinations(range(9), 4):
        pattern = np.zeros(9, dtype=np.int8)
        pattern[list(combo)] = 1
        patterns.append(pattern)
    single_block = np.array(patterns)
    
    if n_blocks == 1:
        return single_block
    
    # Generate all combinations for multiple blocks from the single block patterns
    n_patterns_per_block = len(single_block)
    total_patterns = n_patterns_per_block ** n_blocks
    n_channels = 9 * n_blocks
    all_patterns = np.zeros((total_patterns, n_channels), dtype=np.int8)
    
    indices = np.zeros(n_blocks, dtype=int)
    for i in range(total_patterns):
        for block in range(n_blocks):
            start = block * 9
            end = (block + 1) * 9
            all_patterns[i, start:end] = single_block[indices[block]]
        
        indices[0] += 1
        for j in range(n_blocks - 1):
            if indices[j] >= n_patterns_per_block:
                indices[j] = 0
                indices[j + 1] += 1

    return all_patterns


def is_valid_pattern(pattern: Union[str, List[int], Tuple[int], np.ndarray]) -> bool:
    """Check if a barcode pattern is valid (4-of-9 in each block).
    
    Parameters
    ----------
    pattern : str, list, tuple, or ndarray
        Binary pattern to check. Can be a string of 0s and 1s, a list/tuple
        of integers, or a numpy array.
    
    Returns
    -------
    bool
        True if pattern is valid (all blocks sum to 4).
    
    Raises
    ------
    ValueError
        If pattern length is not a multiple of 9.
    
    Examples
    --------
    >>> is_valid_pattern("111100000111100000")
    True
    >>> is_valid_pattern([1,1,1,1,0,0,0,0,0])
    True
    """
    if isinstance(pattern, str):
        pattern = [int(c) for c in pattern]
    elif isinstance(pattern, tuple):
        pattern = list(pattern)
    elif isinstance(pattern, np.ndarray):
        pattern = pattern.tolist()

    # Check if length is multiple of 9
    if len(pattern) % 9 != 0:
        raise ValueError(f"Pattern length {len(pattern)} is not a multiple of 9")
    
    n_groups = len(pattern) // 9
    
    group_sums = []
    for i in range(n_groups):
        start = i * 9
        end = (i + 1) * 9
        group_sum = sum(pattern[start:end])
        group_sums.append(group_sum)
    
    # All groups must sum to 4 for valid barcode
    return all(s == 4 for s in group_sums)


def add_barcode_validity(adata: ad.AnnData,
                        assignment_col: str,
                        validity_col: Optional[str] = None,
                        inplace: bool = True) -> ad.AnnData:
    """Add boolean barcode validity column to adata.obs.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with debarcoding results.
    assignment_col : str
        Column name in adata.obs containing barcode patterns.
    validity_col : str, optional
        Name for the new validity column. If None, auto-generates as
        ``'{assignment_col}_validity'``.
    inplace : bool, default True
        Modify adata in place.
    
    Returns
    -------
    AnnData
        Modified AnnData with new validity column in adata.obs.
    """
    if not inplace:
        adata = adata.copy()
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Column '{assignment_col}' not found in adata.obs")
    
    if validity_col is None:
        validity_col = f'{assignment_col}_validity'
    
    adata.obs[validity_col] = adata.obs[assignment_col].apply(is_valid_pattern)
        
    return adata


def pattern_to_indices(pattern: Union[str, List[int], Tuple[int], np.ndarray],
                      one_based: bool = True) -> List[int]:
    """Convert binary pattern to list of indices where value is 1.
    
    Parameters
    ----------
    pattern : str, list, tuple, or ndarray
        Binary pattern.
    one_based : bool, default True
        Return 1-based indices. If False, returns 0-based indices.
    
    Returns
    -------
    list of int
        Indices where pattern is 1.
    
    Examples
    --------
    >>> pattern_to_indices("110010000")
    [1, 2, 4]
    >>> pattern_to_indices("110010000", one_based=False)
    [0, 1, 3]
    """
    if isinstance(pattern, str):
        pattern = [int(c) for c in pattern]
    
    offset = 1 if one_based else 0
    return [i + offset for i, val in enumerate(pattern) if val == 1]


def indices_to_pattern(indices: List[int],
                      n_channels: int = 27,
                      one_based: bool = True) -> np.ndarray:
    """Convert list of indices to binary pattern.
    
    Parameters
    ----------
    indices : list of int
        Indices where pattern should be 1.
    n_channels : int, default 27
        Total number of channels.
    one_based : bool, default True
        Indices are 1-based. If False, assumes 0-based indices.
    
    Returns
    -------
    ndarray
        Binary pattern array.
    
    Raises
    ------
    ValueError
        If any index is out of range.
    
    Examples
    --------
    >>> indices_to_pattern([1, 2, 4, 5], n_channels=9)
    array([1, 1, 0, 1, 1, 0, 0, 0, 0])
    """
    pattern = np.zeros(n_channels, dtype=int)
    
    if one_based:
        # Convert to 0-based
        indices = [i - 1 for i in indices]
    
    for idx in indices:
        if 0 <= idx < n_channels:
            pattern[idx] = 1
        else:
            raise ValueError(f"Index {idx} out of range [0, {n_channels})")
    
    return pattern


def hamming_distance(pattern1: Union[str, List, np.ndarray],
                    pattern2: Union[str, List, np.ndarray]) -> int:
    """Compute Hamming distance between two patterns.
    
    Parameters
    ----------
    pattern1 : str, list, or ndarray
        First binary pattern.
    pattern2 : str, list, or ndarray
        Second binary pattern.
    
    Returns
    -------
    int
        Number of positions where patterns differ.
    
    Raises
    ------
    ValueError
        If patterns have different lengths.
    
    Examples
    --------
    >>> hamming_distance("111100000", "111010000")
    2
    """
    if isinstance(pattern1, str):
        pattern1 = [int(c) for c in pattern1]
    if isinstance(pattern2, str):
        pattern2 = [int(c) for c in pattern2]
    
    pattern1 = np.array(pattern1)
    pattern2 = np.array(pattern2)
    
    if len(pattern1) != len(pattern2):
        raise ValueError("Patterns must have same length")
    
    return int(np.sum(pattern1 != pattern2))


def get_pattern_statistics(adata: ad.AnnData, 
                          assignment_col: str,
                          verbose: bool = True) -> dict:
    """Compute statistics for debarcoding assignment patterns.
    
    Analyzes the distribution and validity of barcode patterns from
    debarcoding results. 
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with debarcoding results.
    assignment_col : str
        Column name in adata.obs containing barcode patterns
        (e.g., 'pc_gmm_assignment', 'premessa_assignment').
    verbose : bool, default True
        Print summary statistics.
    
    Returns
    -------
    dict
        Dictionary with pattern statistics:
        
        - n_cells : total number of cells
        - n_unique_patterns : number of unique patterns
        - n_valid : number of cells with valid patterns
        - n_invalid : number of cells with invalid patterns
        - pct_valid : percentage of cells with valid patterns
        - most_common : list of (pattern, count) tuples for top 10
    
    Raises
    ------
    ValueError
        If assignment_col not found in adata.obs.
    """
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Column '{assignment_col}' not found in adata.obs")
    
    patterns = adata.obs[assignment_col]
    
    n_cells = len(patterns)
    pattern_counts = patterns.value_counts()
    n_unique_patterns = len(pattern_counts)
    
    unique_patterns = pattern_counts.index.tolist()
    classifications = {p: is_valid_pattern(p) for p in unique_patterns}
    
    n_valid = sum(count for pattern, count in pattern_counts.items() if classifications[pattern])
    n_invalid = n_cells - n_valid
    pct_valid = 100 * n_valid / n_cells if n_cells > 0 else 0
    
    most_common = list(pattern_counts.head(10).items())
    
    stats = {
        'assignment_col': assignment_col,
        'n_cells': int(n_cells),
        'n_unique_patterns': int(n_unique_patterns),
        'n_valid': int(n_valid),
        'n_invalid': int(n_invalid),
        'pct_valid': float(pct_valid),
        'most_common': most_common,
    }
    
    if verbose:
        print(f"Pattern Statistics for '{assignment_col}':")
        print(f"  Total cells: {n_cells:,}")
        print(f"  Unique patterns: {n_unique_patterns:,}")
        print(f"  Valid patterns: {n_valid:,} ({pct_valid:.1f}%)")
        print(f"  Invalid patterns: {n_invalid:,} ({100-pct_valid:.1f}%)")
        
        print(f"\n  Top 5 patterns:")
        for i, (pattern, count) in enumerate(most_common[:5], 1):
            pct = 100 * count / n_cells
            valid = classifications[pattern]
            status = "valid" if valid else "invalid"
            print(f"    {i}. [{status}] {pattern[:20]}... : {count:,} cells ({pct:.1f}%)")
    
    return stats
