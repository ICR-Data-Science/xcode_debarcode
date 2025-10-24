"""xcode_debarcode - CyTOF barcode gating toolkit

Multi-method debarcoding with preprocessing, debarcoding, and postprocessing.

Public sub-modules
------------------
    io             : Data I/O and channel mapping
    preprocessing  : Data transformation (log, arcsinh)
    debarcode      : Debarcoding methods (GMM, PreMessa, PC-GMM, Scoring, Auto)
    postprocessing : Confidence filtering, Hamming clustering, Mahalanobis filtering
    barcode        : Pattern analysis utilities
    plots          : Interactive visualizations
    simulate       : Synthetic data generation
    
Methods Overview
---------------
    GMM (Gaussian Mixture Model):
        - Gate-level 2-component GMM
        - Independent per-gate classification
        - Confidence = product of gate probabilities
        
    PreMessa:
        - Top-4 gate selection per barcode block
        - Separation-based confidence scoring
        - Confidence based on separation between ON and OFF intensities
        
    PC-GMM (Pattern-Constrained GMM):
        - Gate-level GMMs + valid pattern constraints
        - Maximum likelihood pattern selection
        - Block-wise probabilistic decoding
        
    Scoring:
        - Score-based pattern matching
        - Mean-centered intensity scoring
        - Confidence based on score
        
    Auto:
        - Adaptive method selection
        - PC-GMM with fallback to Scoring
        - Automatic quality assessment based on PC-GMM mean confidence

Typical Workflow
----------------
1. Load data: `adata = xd.io.read_data('data.fcs')`
2. Map channels: `adata = xd.io.map_channels(adata, mapping_dict)`
3. Transform: `adata = xd.preprocessing.transform(adata, method='arcsinh')`
4. Debarcode: `adata = xd.debarcode.debarcode(adata, method='pc_gmm')`
5. Hamming cluster: `adata = xd.postprocessing.hamming_cluster(adata)`
6. Filter: `adata = xd.postprocessing.filter_by_confidence(adata)`
7. Visualize: `xd.plots.plot_confidence_distribution(adata)`

Or use the automated pipeline:
`adata = xd.debarcode.debarcoding_pipeline(adata, method='auto')`

Simulation and Testing
---------------------
Generate synthetic data for testing and benchmarking:
`adata = xd.simulate.simulate_cytof_barcodes(n_cells=10000)`
"""
from importlib import metadata
from . import io, preprocessing, debarcode, postprocessing, barcode, plots, simulate

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.dev0"

__all__ = [
    "io", "preprocessing", "debarcode", "postprocessing", 
    "barcode", "plots", "simulate", "__version__",
]