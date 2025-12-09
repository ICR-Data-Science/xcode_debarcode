"""
xcode_debarcode - CyTOF barcode gating toolkit.

Multi-method debarcoding with preprocessing, debarcoding, and postprocessing.

Submodules
----------
io
    Data I/O and channel mapping.
preprocessing
    Data transformation (log, arcsinh) and intensity outlier filtering.
debarcode
    Debarcoding methods (GMM, PreMessa, PC-GMM, GMM-BIC, Manual).
postprocessing
    Filtering, Hamming clustering, Mahalanobis filtering.
barcode
    Pattern analysis utilities.
plots
    Interactive visualizations.
simulate
    Synthetic data generation.

Methods Overview
----------------
GMM (Gaussian Mixture Model)
    Channel-level 2-component GMM with independent per-channel classification.
PreMessa
    Iterative top-4 channel selection with per-channel normalization.
PC-GMM (Pattern-Constrained GMM)
    Channel-level GMMs with valid pattern constraints and maximum likelihood selection.
GMM-BIC (Cluster-based GMM)
    Global clustering in PCA space for low barcode regimes with unimodal channels.
Manual
    Manual user-fixed per-channel thresholding.
"""
from importlib import metadata
from . import io, preprocessing, debarcode, postprocessing, barcode, plots, simulate

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "io", "preprocessing", "debarcode", "postprocessing", 
    "barcode", "plots", "simulate", "__version__",
]
