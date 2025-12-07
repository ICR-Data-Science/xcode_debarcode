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
    Debarcoding methods (GMM-CH, CGMM-CH, GMM-B, PreMessa, Manual).
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
GMM-CH (Per-Channel GMM)
    Independent 2-component GMM per channel. Can produce invalid patterns.
CGMM-CH (Constrained Per-Channel GMM)
    Per-channel GMMs with valid pattern constraints and maximum likelihood selection.
    Best for bimodal channel distributions.
GMM-B (Per-Barcode GMM)
    Blockwise 126-component GMM using all valid 4-of-9 patterns.
    Best when some channels have unimodal distributions.
PreMessa
    Top-4 channel selection per barcode block with separation-based confidence.
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
