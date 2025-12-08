"""Data preprocessing and transformation for CyTOF data."""
import numpy as np
import anndata as ad
from typing import Optional

__all__ = ["transform", "filter_cells_intensity"]


def transform(adata: ad.AnnData,
             method: str = 'log',
             cofactor: float = 10.0,
             layer_name: Optional[str] = None,
             inplace: bool = True,
             verbose: bool = True) -> ad.AnnData:
    """Apply transformation to intensity data.
    
    Creates a new layer in adata with transformed values. The transformation
    is applied to ALL channels (barcode and non-barcode), but the function
    tracks which channels are barcode channels using adata.uns['barcode_channels'].
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object (should have barcode channels mapped via map_channels)
    method : str
        Transformation method: 'arcsinh' or 'log' (default: 'arcsinh')
        - 'arcsinh': inverse hyperbolic sine, arcsinh(x/cofactor)
        - 'log': natural logarithm, log(x + 1)
    cofactor : float
        Cofactor for arcsinh transformation (default: 10.0)
    layer_name : str, optional
        Name for the transformed layer. If None, auto-generates:
        - 'arcsinh_cf{cofactor}' for arcsinh
        - 'log' for log
    inplace : bool
        Modify adata in place (default: True)
    verbose : bool
        Print progress messages (default: True)
    
    Returns:
    --------
    adata : AnnData
        Modified AnnData with new layer containing transformed data
        - adata.layers[layer_name]: transformed data for all channels
        - adata.uns['transformations'][layer_name]: transformation metadata
    """
    if not inplace:
        adata = adata.copy()
    
    # Check transform
    if method not in ['arcsinh', 'log']:
        raise ValueError(
            f"Unknown transformation method: '{method}'. "
            "Valid options: 'arcsinh', 'log'"
        )
    
    # Determine layer name
    if layer_name is None:
        if method == 'arcsinh':
            layer_name = f'arcsinh_cf{cofactor}'
        else:  
            layer_name = 'log'
    
    barcode_channels = None
    if 'barcode_channels' in adata.uns:
        barcode_channels = adata.uns['barcode_channels']
    
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    
    # Apply transformation to all channels
    if method == 'arcsinh':
        X_transformed = np.arcsinh(X / cofactor)
    elif method == 'log':
        X_transformed = np.log(X + 1)
    
    adata.layers[layer_name] = X_transformed
    
    if 'transformations' not in adata.uns:
        adata.uns['transformations'] = {}
    
    adata.uns['transformations'][layer_name] = {
        'method': method,
        'cofactor': cofactor if method == 'arcsinh' else None,
        'n_channels': adata.n_vars,
        'n_cells': adata.n_obs,
        'all_channels': adata.var_names.tolist(),
        'barcode_channels': barcode_channels,
        'n_barcode_channels': len(barcode_channels) if barcode_channels else None
    }
    
    if verbose:
        print(f"Transformation complete")
        print(f"  Method: {method}")
        if method == 'arcsinh':
            print(f"  Cofactor: {cofactor}")
        print(f"  Layer: '{layer_name}'")
        print(f"  Transformed: {adata.n_vars} channels x {adata.n_obs} cells")
        if barcode_channels:
            print(f"  Barcode channels: {len(barcode_channels)}")
        else:
            print(f"  Warning: Barcode channels not set. Run map_channels() first.")
    
    return adata

def filter_cells_intensity(adata: ad.AnnData,
                           layer: Optional[str] = 'log',
                           method: str = 'rectangular',
                           percentile: float = 99.0,
                           sum_low: Optional[float] = 1.0,
                           sum_high: Optional[float] = 99.0,
                           var_low: Optional[float] = 1.0,
                           var_high: Optional[float] = 99.0,
                           filter_or_flag: str = 'flag',
                           filter_name: Optional[str] = None,
                           inplace: bool = True,
                           verbose: bool = True) -> ad.AnnData:
    """Filter cells by barcode channel intensity statistics.
    
    Removes debris (low sum/variance) and doublets (high sum/variance)
    based on channel sum (L1 norm) and channel variance.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with mapped barcode channels.
    layer : str, optional
        Data layer to use. None for adata.X. Default: 'log'.
    method : str
        Filtering method: 'rectangular' or 'ellipsoidal'. Default: 'rectangular'.
        - 'rectangular': Independent percentile thresholds on sum and variance.
        - 'ellipsoidal': Mahalanobis distance with robust covariance estimation.
    percentile : float
        For 'ellipsoidal' method, keep cells within this percentile of
        Mahalanobis distance. Default: 99.0.
    sum_low : float, optional
        For 'rectangular': lower percentile bound for channel sum.
        Cells below this percentile are removed. Set to None to disable. Default: 1.0.
    sum_high : float, optional
        For 'rectangular': upper percentile bound for channel sum.
        Cells above this percentile are removed. Set to None to disable. Default: 99.0.
    var_low : float, optional
        For 'rectangular': lower percentile bound for channel variance.
        Cells below this percentile are removed. Set to None to disable. Default: 1.0.
    var_high : float, optional
        For 'rectangular': upper percentile bound for channel variance.
        Cells above this percentile are removed. Set to None to disable. Default: 99.0.
    filter_or_flag : str
        'flag' to add boolean column, 'filter' to remove cells. Default: 'flag'.
    filter_name : str, optional
        Custom name for this filtering run. Default: auto-generated.
    inplace : bool
        Modify adata in place. Default: True.
    verbose : bool
        Print progress. Default: True.
    
    Returns
    -------
    AnnData with filtering results.
    
    Notes
    -----
    For rectangular method:
        - sum_low=1, sum_high=99 removes bottom 1% and top 1% by channel sum
        - var_low=1, var_high=99 removes bottom 1% and top 1% by variance
    
    For ellipsoidal method:
        - Uses Minimum Covariance Determinant for robust covariance estimation
        - Filters based on Mahalanobis distance in (sum, variance) space
        - Better handles correlated sum/variance and non-axis-aligned outliers
    """
    if not inplace:
        adata = adata.copy()
    
    if method not in {'rectangular', 'ellipsoidal'}:
        raise ValueError(f"method must be 'rectangular' or 'ellipsoidal', got {method!r}")
    if filter_or_flag not in {'flag', 'filter'}:
        raise ValueError(f"filter_or_flag must be 'flag' or 'filter', got {filter_or_flag!r}")
    if 'barcode_channels' not in adata.uns:
        raise ValueError("Barcode channels not mapped. Run map_channels first.")
    
    # Get barcode channel data
    bc_channels = adata.uns['barcode_channels']
    bc_idx = [list(adata.var_names).index(ch) for ch in bc_channels]
    X = adata.layers[layer][:, bc_idx] if layer else adata.X[:, bc_idx]
    
    # Compute statistics
    channel_sum = np.abs(X).sum(axis=1)
    channel_var = np.var(X, axis=1)
    
    if verbose:
        print(f"Intensity filtering: method={method}, layer={layer}")
        print(f"  Channel sum: min={channel_sum.min():.2f}, max={channel_sum.max():.2f}, "
              f"mean={channel_sum.mean():.2f}")
        print(f"  Channel var: min={channel_var.min():.4f}, max={channel_var.max():.2f}, "
              f"mean={channel_var.mean():.2f}")
    
    if method == 'rectangular':
        thresholds = {
            'sum_low': np.percentile(channel_sum, sum_low) if sum_low is not None else None,
            'sum_high': np.percentile(channel_sum, sum_high) if sum_high is not None else None,
            'var_low': np.percentile(channel_var, var_low) if var_low is not None else None,
            'var_high': np.percentile(channel_var, var_high) if var_high is not None else None,
        }
        
        if verbose:
            if thresholds['sum_low'] is not None or thresholds['sum_high'] is not None:
                sum_low_str = f"{thresholds['sum_low']:.2f}" if thresholds['sum_low'] is not None else "None"
                sum_high_str = f"{thresholds['sum_high']:.2f}" if thresholds['sum_high'] is not None else "None"
                print(f"  Sum thresholds: [{sum_low_str}, {sum_high_str}]")
            if thresholds['var_low'] is not None or thresholds['var_high'] is not None:
                var_low_str = f"{thresholds['var_low']:.4f}" if thresholds['var_low'] is not None else "None"
                var_high_str = f"{thresholds['var_high']:.4f}" if thresholds['var_high'] is not None else "None"
                print(f"  Var thresholds: [{var_low_str}, {var_high_str}]")
        
        pass_mask = np.ones(len(adata), dtype=bool)
        
        if thresholds['sum_low'] is not None:
            mask = channel_sum >= thresholds['sum_low']
            if verbose and (~mask).sum() > 0:
                print(f"  Sum low: {(~mask).sum()} cells removed")
            pass_mask &= mask
        
        if thresholds['sum_high'] is not None:
            mask = channel_sum <= thresholds['sum_high']
            if verbose and (~mask).sum() > 0:
                print(f"  Sum high: {(~mask).sum()} cells removed")
            pass_mask &= mask
        
        if thresholds['var_low'] is not None:
            mask = channel_var >= thresholds['var_low']
            if verbose and (~mask).sum() > 0:
                print(f"  Var low: {(~mask).sum()} cells removed")
            pass_mask &= mask
        
        if thresholds['var_high'] is not None:
            mask = channel_var <= thresholds['var_high']
            if verbose and (~mask).sum() > 0:
                print(f"  Var high: {(~mask).sum()} cells removed")
            pass_mask &= mask
        
        metadata = {
            'layer': layer,
            'method': method,
            'sum_low': sum_low,
            'sum_high': sum_high,
            'var_low': var_low,
            'var_high': var_high,
            'thresholds': {k: float(v) if v is not None else None for k, v in thresholds.items()},
        }
    
    else:  # ellipsoidal
        from sklearn.covariance import MinCovDet
        
        if verbose:
            print(f"  Using robust Mahalanobis distance (percentile={percentile})")
        
        # Robust covariance estimation
        data = np.column_stack([channel_sum, channel_var])
        mcd = MinCovDet(random_state=42, support_fraction=0.75)
        mcd.fit(data)
        
        mahal_dist = np.sqrt(mcd.mahalanobis(data))
        threshold = np.percentile(mahal_dist, percentile)
        pass_mask = mahal_dist <= threshold
        
        if verbose:
            print(f"  Mahalanobis threshold: {threshold:.2f}")
            print(f"  Center: sum={mcd.location_[0]:.2f}, var={mcd.location_[1]:.4f}")
        
        metadata = {
            'layer': layer,
            'method': method,
            'percentile': percentile,
            'mahal_threshold': float(threshold),
            'center': {'sum': float(mcd.location_[0]), 'var': float(mcd.location_[1])},
        }
    
    n_pass = pass_mask.sum()
    n_total = len(adata)
    pct_pass = 100 * n_pass / n_total
    
    if verbose:
        print(f"  Pass: {n_pass}/{n_total} cells ({pct_pass:.1f}%)")
    
    if filter_or_flag == 'flag':
        base_flag_col = "intensity_pass"
        
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
            filter_name = "intensity"
        
        if verbose:
            print(f"  [+] Filtered to {len(adata)} cells")
    
    if 'intensity_filtering' not in adata.uns:
        adata.uns['intensity_filtering'] = {}
    
    metadata.update({
        'filter_or_flag': filter_or_flag,
        'n_pass': int(n_pass),
        'n_total': int(n_total),
        'pct_pass': float(pct_pass),
    })
    
    adata.uns['intensity_filtering'][filter_name] = metadata
    
    if verbose:
        print(f"  [+] Metadata saved: adata.uns['intensity_filtering']['{filter_name}']")
    
    return adata