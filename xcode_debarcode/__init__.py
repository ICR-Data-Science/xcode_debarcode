"""xcode_debarcode - CyTOF barcode gating toolkit

Multi-method debarcoding with preprocessing, debarcoding, and postprocessing.

Public sub-modules
------------------
    io             : Data I/O and channel mapping
    preprocessing  : Data transformation (log, arcsinh), and intensity outlier filtering
    debarcode      : Debarcoding methods (GMM, PreMessa, PC-GMM, Scoring, Auto)
    postprocessing : Filtering, Hamming clustering, Mahalanobis filtering
    barcode        : Pattern analysis utilities
    plots          : Interactive visualisations
    simulate       : Synthetic data generation
    
Methods Overview
---------------
    GMM (Gaussian Mixture Model):
        - Channel-level 2-component GMM
        - Independent per-channel classification
        
    PreMessa:
        - Top-4 channel selection per barcode block
        - Separation-based confidence scoring
        
    PC-GMM (Pattern-Constrained GMM):
        - Channel-level GMMs + valid pattern constraints
        - Maximum likelihood pattern selection
        
    Scoring:
        - Score-based pattern matching
        - Mean-centered intensity scoring
        
    Auto:
        - Adaptive method selection based on PC-GMM mean confidence
        - PC-GMM with fallback to Scoring

    Manual:
        - Manual user-fixed per-channel thresholding
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
