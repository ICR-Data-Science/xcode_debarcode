"""Data I/O and channel mapping utilities."""
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Union, Dict, List

__all__ = ["read_data", "write_data", "load_mapping", "map_channels", "get_barcode_channels"]


def read_data(path: str) -> ad.AnnData:
    """Read data from FCS or H5AD file.
    
    Parameters:
    -----------
    path : str
        Path to the data file (.h5ad or .fcs)
    
    Returns:
    --------
    adata : AnnData
        Loaded data as AnnData object
    """
    path = Path(path)
    
    if path.suffix.lower() == '.h5ad':
        return ad.read_h5ad(path)
    elif path.suffix.lower() == '.fcs':
        try:
            from pytometry.io import read_fcs
            return read_fcs(str(path))
        except ImportError:
            raise ImportError(
                "pytometry is required to read FCS files. "
                "Install it with: pip install pytometry"
            )
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            "Supported formats: .h5ad, .fcs"
        )


def write_data(adata: ad.AnnData, path: str, **kwargs):
    """Write data to FCS or H5AD file.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    path : str
        Output file path (.h5ad or .fcs)
    **kwargs
        Additional arguments passed to write function
    """
    path = Path(path)
    
    if path.suffix.lower() == '.h5ad':
        adata.write_h5ad(path, **kwargs)
    elif path.suffix.lower() == '.fcs':
        try:
            from pytometry.io import write_fcs
            write_fcs(adata, str(path), **kwargs)
        except ImportError:
            raise ImportError(
                "pytometry is required to write FCS files. "
                "Install it with: pip install pytometry"
            )
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            "Supported formats: .h5ad, .fcs"
        )


def load_mapping(path: str) -> pd.DataFrame:
    """Load channel mapping from CSV file.
    
    Expected columns: Element, Isotope, Sequence No., FCS_Channel_Name
    
    Parameters:
    -----------
    path : str
        Path to mapping CSV file
    
    Returns:
    --------
    mapping_df : DataFrame
        Channel mapping dataframe
    """
    df = pd.read_csv(path)
    required = {"Element", "Isotope", "Sequence No.", "FCS_Channel_Name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping file missing columns: {missing}")
    return df


def map_channels(adata: ad.AnnData, 
                mapping: Union[str, pd.DataFrame, Dict[str, str]],
                inplace: bool = True,
                verbose: bool = True) -> ad.AnnData:
    """Map and rename barcode channels, storing barcode channel list in adata.uns.
    
    Renames channels to s_1, s_2, ..., s_N and reorders them to the end.
    The list of barcode channels is stored in adata.uns['barcode_channels'].
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    mapping : str, DataFrame, or dict
        Channel mapping specification:
        - str: path to CSV file
        - DataFrame: mapping dataframe with required columns
        - dict: {old_name: new_name} mapping
    inplace : bool
        Modify adata in place (default: True)
    verbose : bool
        Print progress messages (default: True)
    
    Returns:
    --------
    adata : AnnData
        Modified AnnData object with:
        - Renamed barcode channels (s_1, s_2, ...)
        - Reordered channels (non-barcode first, then barcodes)
        - adata.uns['barcode_channels']: list of barcode channel names
        - adata.uns['channel_mapping']: mapping information
    """
    if not inplace:
        adata = adata.copy()
    
    # Convert mapping to dictionary
    if isinstance(mapping, str):
        mapping_df = load_mapping(mapping)
        rename_dict = {
            row["FCS_Channel_Name"]: f"s_{row['Sequence No.']}"
            for _, row in mapping_df.iterrows()
        }
        mapping_source = 'csv_file'
    elif isinstance(mapping, pd.DataFrame):
        rename_dict = {
            row["FCS_Channel_Name"]: f"s_{row['Sequence No.']}"
            for _, row in mapping.iterrows()
        }
        mapping_source = 'dataframe'
    elif isinstance(mapping, dict):
        rename_dict = mapping
        mapping_source = 'dict'
    else:
        raise TypeError("mapping must be str (path), DataFrame, or dict")
    
    # Store original channel names for barcode channels
    original_barcode_channels = [k for k in rename_dict.keys() if k in adata.var_names]
    
    # Rename channels
    adata.var_names = [rename_dict.get(v, v) for v in adata.var_names]
    
    barcode_channels = sorted(
        [v for v in adata.var_names if v.startswith("s_")],
        key=lambda x: int(x.split("_")[1])
    )
    
    other_channels = [v for v in adata.var_names if not v.startswith("s_")]
    
    adata = adata[:, other_channels + barcode_channels]
    
    adata.uns['barcode_channels'] = barcode_channels
    adata.uns['channel_mapping'] = {
        'source': mapping_source,
        'n_barcode_channels': len(barcode_channels),
        'n_other_channels': len(other_channels),
        'barcode_channels': barcode_channels,
        'other_channels': other_channels,
        'original_barcode_names': original_barcode_channels,
    }
    
    if verbose:
        print(f"Channel mapping complete")
        print(f"  Barcode channels: {len(barcode_channels)} (stored in adata.uns['barcode_channels'])")
        print(f"  Other channels: {len(other_channels)}")
        print(f"  Total channels: {len(adata.var_names)}")
        
        if len(barcode_channels) % 9 != 0:
            print(f"  Warning: Number of barcode channels ({len(barcode_channels)}) is not a multiple of 9")
    
    return adata


def get_barcode_channels(adata: ad.AnnData) -> List[str]:
    """Get list of barcode channel names.
    
    First checks adata.uns['barcode_channels'] (set by map_channels).
    If not found, detects channels with s_* pattern.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    
    Returns:
    --------
    channels : list
        List of barcode channel names
    
    Raises:
    -------
    ValueError
        If no barcode channels are found
    """

    if 'barcode_channels' in adata.uns:
        return adata.uns['barcode_channels']
    
    barcode_channels = sorted(
        [v for v in adata.var_names if v.startswith("s_")],
        key=lambda x: int(x.split("_")[1])
    )
    
    if len(barcode_channels) == 0:
        raise ValueError(
            "No barcode channels found. "
            "Please run map_channels() first to set up barcode channels, "
            "or ensure channels follow s_* naming pattern."
        )
    
    return barcode_channels