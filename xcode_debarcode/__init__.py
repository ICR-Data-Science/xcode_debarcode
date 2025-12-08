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
    Debarcoding methods (X-EM, GMM, PreMessa, Manual).
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
X-EM (X-Code EM)
    Constrained Expectation-Maximization for X-Code. Enforces valid 4-of-9
    patterns throughout optimization. Best overall performance across all
    barcode configurations. Recommended default method.
GMM (Per-Channel GMM)
    Independent 2-component GMM per channel. Can produce invalid patterns.
    Useful for per-channel threshold estimation.
PreMessa
    Top-4 channel selection per barcode block with separation-based confidence.
    Published baseline method.
Manual
    User-defined per-channel thresholding.
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
