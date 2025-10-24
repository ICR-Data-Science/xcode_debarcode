"""Data preprocessing and transformation for CyTOF data."""
import numpy as np
import anndata as ad
from typing import Optional

__all__ = ["transform"]


def transform(adata: ad.AnnData,
             method: str = 'log',
             cofactor: float = 5.0,
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
        Cofactor for arcsinh transformation (default: 5.0)
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