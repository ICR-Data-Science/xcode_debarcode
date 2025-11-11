"""Barcode pattern analysis and validation utilities."""
import numpy as np
import anndata as ad
from typing import Optional, List, Union, Tuple 
from itertools import combinations

__all__ = ["classify_pattern", "is_valid_pattern", "pattern_to_indices", 
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


def classify_pattern(pattern: Union[str, List[int], Tuple[int], np.ndarray]) -> str:
    """Classify barcode pattern as valid/invalid based on 4-4-...-4 rule.
    
    Supports arbitrary number of 4-plexes (groups of 9 channels).
    Each group must sum to exactly 4 for a valid barcode.
    
    Parameters:
    -----------
    pattern : str, list, tuple, or array
        Binary pattern to classify
    
    Returns:
    --------
    classification : str
        One of: 'valid', 
        'invalid_doublet' (too many ON channels), 
        'invalid_debris' (too few ON channels), 
        'invalid_misc' (valid ON channels number but still invalid)
    """
    # Convert to list if needed
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
    
    total = sum(group_sums)
    expected_total = 4 * n_groups
    
    # All groups must sum to 4 for valid barcode
    if all(s == 4 for s in group_sums):
        return "valid"
    elif total > expected_total:
        return "invalid_doublet"
    elif total < expected_total:
        return "invalid_debris"
    else:
        return "invalid_misc"

def add_barcode_validity(adata: ad.AnnData,
                        assignment_col: str,
                        validity_col: Optional[str] = None,
                        inplace: bool = True) -> ad.AnnData:
    """Add barcode validity classification to adata.obs.
    
    Classifies each barcode pattern as 'valid', 'invalid_doublet', 
    'invalid_debris', or 'invalid_misc' and adds the result as a new column.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with debarcoding results
    assignment_col : str
        Column name in adata.obs containing barcode patterns
    validity_col : str, optional
        Name for the new validity column. If None, auto-generates:
        '{assignment_col}_validity'
    inplace : bool
        Modify adata in place (default: True)
    verbose : bool
        Print validity statistics (default: True)
    
    Returns:
    --------
    adata : AnnData
        Modified AnnData with new column in adata.obs
    """
    if not inplace:
        adata = adata.copy()
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Column '{assignment_col}' not found in adata.obs")
    
    if validity_col is None:
        validity_col = f'{assignment_col}_validity'
    
    adata.obs[validity_col] = adata.obs[assignment_col].apply(classify_pattern)
        
    return adata


def is_valid_pattern(pattern: Union[str, List[int], Tuple[int], np.ndarray]) -> bool:
    """Check if pattern is valid (4-4-...-4).
    
    Parameters:
    -----------
    pattern : str, list, tuple, or array
        Binary pattern to check
    
    Returns:
    --------
    is_valid : bool
        True if pattern is valid (all blocks sum to 4)
    """
    return classify_pattern(pattern) == "valid"


def pattern_to_indices(pattern: Union[str, List[int], Tuple[int], np.ndarray],
                      one_based: bool = True) -> List[int]:
    """Convert binary pattern to list of indices where value is 1.
    
    Parameters:
    -----------
    pattern : str, list, tuple, or array
        Binary pattern
    one_based : bool
        Return 1-based indices (default: True)
        If False, returns 0-based indices
    
    Returns:
    --------
    indices : list of int
        Indices where pattern is 1
    """
    if isinstance(pattern, str):
        pattern = [int(c) for c in pattern]
    
    offset = 1 if one_based else 0
    return [i + offset for i, val in enumerate(pattern) if val == 1]


def indices_to_pattern(indices: List[int],
                      n_channels: int = 27,
                      one_based: bool = True) -> np.ndarray:
    """Convert list of indices to binary pattern.
    
    Parameters:
    -----------
    indices : list of int
        Indices where pattern should be 1
    n_channels : int
        Total number of channels (default: 27)
    one_based : bool
        Indices are 1-based (default: True)
        If False, assumes 0-based indices
    
    Returns:
    --------
    pattern : np.ndarray
        Binary pattern array
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
    
    Parameters:
    -----------
    pattern1, pattern2 : str, list, or array
        Binary patterns to compare
    
    Returns:
    --------
    distance : int
        Number of positions where patterns differ
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
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with debarcoding results
    assignment_col : str
        Column name in adata.obs containing barcode patterns
        (e.g., 'pc_gmm_assignment', 'premessa_assignment')
    verbose : bool
        Print summary statistics (default: True)
    
    Returns:
    --------
    stats : dict
        Dictionary with pattern statistics:
        - n_cells: total number of cells
        - n_unique_patterns: number of unique patterns
        - n_valid: number of cells with valid patterns
        - n_invalid: number of cells with invalid patterns
        - pct_valid: percentage of cells with valid patterns
        - most_common: list of (pattern, count) tuples for top 10
        - validity_breakdown: counts by validity type
        - pattern_counts: pandas Series with all pattern counts
    """
    import pandas as pd
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Column '{assignment_col}' not found in adata.obs")
    
    patterns = adata.obs[assignment_col]
    
    n_cells = len(patterns)
    pattern_counts = patterns.value_counts()
    n_unique_patterns = len(pattern_counts)
    
    unique_patterns = pattern_counts.index.tolist()
    classifications = {p: classify_pattern(p) for p in unique_patterns}
    
    # Count validity types
    validity_counts = {}
    for pattern, count in pattern_counts.items():
        validity = classifications[pattern]
        validity_counts[validity] = validity_counts.get(validity, 0) + count
    
    # Calculate valid/invalid counts
    n_valid = validity_counts.get('valid', 0)
    n_invalid = n_cells - n_valid
    pct_valid = 100 * n_valid / n_cells if n_cells > 0 else 0
    
    # Most common patterns
    most_common = list(pattern_counts.head(10).items())
    
    stats = {
        'assignment_col': assignment_col,
        'n_cells': int(n_cells),
        'n_unique_patterns': int(n_unique_patterns),
        'n_valid': int(n_valid),
        'n_invalid': int(n_invalid),
        'pct_valid': float(pct_valid),
        'most_common': most_common,
        'validity_breakdown': validity_counts,
        'pattern_counts': pattern_counts  
    }
    
    if verbose:
        print(f"Pattern Statistics for '{assignment_col}':")
        print(f"  Total cells: {n_cells:,}")
        print(f"  Unique patterns: {n_unique_patterns:,}")
        print(f"  Valid patterns: {n_valid:,} ({pct_valid:.1f}%)")
        print(f"  Invalid patterns: {n_invalid:,} ({100-pct_valid:.1f}%)")
        
        if n_invalid > 0:
            print(f"\n  Validity breakdown:")
            for validity_type, count in validity_counts.items():
                pct = 100 * count / n_cells
                print(f"    {validity_type}: {count:,} ({pct:.1f}%)")
        
        print(f"\n  Top 5 patterns:")
        for i, (pattern, count) in enumerate(most_common[:5], 1):
            pct = 100 * count / n_cells
            validity = classifications[pattern]
            status = "valid" if validity == "valid" else "invalid"
            print(f"    {i}. [{status}] {pattern[:20]}... : {count:,} cells ({pct:.1f}%)")
    
    return stats