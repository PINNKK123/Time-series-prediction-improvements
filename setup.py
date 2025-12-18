"""
Setup script for GNN-STAR package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gnn-star",
    version="1.0.0",
    author="PINNKK123",
    description="Enhanced GNN-STAR Model for InSAR Time Series Prediction with KCL Constraints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PINNKK123/Time-series-prediction-improvements",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "optuna>=3.1.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "vmd": ["vmdpy>=0.2"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
)
