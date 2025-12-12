from setuptools import setup, find_packages

setup(
    name='xcode_debarcode',
    version='0.1.0',
    description='Multi-method CyTOF barcode debarcoding toolkit with pattern-constrained GMM, adaptive filtering, and interactive visualisations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Marwane Bourdim',  
    author_email='marwane.bourdim@icr.ac.uk',  
    url='https://github.com/ICR-Data-Science/xcode_debarcode',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'anndata>=0.8.0',
        'scikit-learn>=1.0.0',
        'plotly>=5.0.0',
        'kaleido>=0.2.0',
        'readfcs',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
