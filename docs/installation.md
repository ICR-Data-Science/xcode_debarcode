# Installation

## Requirements

Python 3.8 or higher with the following packages:

- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- anndata >= 0.8.0
- scikit-learn >= 1.0.0
- plotly >= 5.0.0
- kaleido >= 0.2.0
- readfcs

## Install from GitHub
```bash
pip install git+https://github.com/ICR-Data-Science/xcode_debarcode.git
```

## Verify Installation
```python
import xcode_debarcode as xd
print(xd.__version__)

adata = xd.simulate.simulate_cytof_barcodes(100, 27, 10, verbose=False)
print("Installation successful")
```

Check out the [basic usage tutorial](tutorials/basic_usage.md) to start using the library. 

## Jupyter Notebook Setup

For visualisation in Jupyter notebooks/Jupyter lab, add this at the beginning:
```python
import plotly.io as pio
pio.renderers.default = "notebook"
```