# xcode_debarcode

Multi-method CyTOF barcode debarcoding with pattern-constrained GMM, adaptive filtering, and interactive visualisations.

**[Full Documentation & Tutorials](https://icr-data-science.github.io/xcode_debarcode/)**

## Installing xcode_debarcode

We recommend setting up a fresh conda environment with a Python version >= 3.8:
```
conda create --name xcode python=3.11
conda activate xcode
```

Then one can either use:
```
pip install git+https://github.com/ICR-Data-Science/xcode_debarcode/
```

Or:
```
git clone https://github.com/ICR-Data-Science/xcode_debarcode/
cd xcode_debarcode
pip install .
```

Verify that the installation went well with:
```python
import xcode_debarcode as xd
print(xd.__version__) 
```
