from setuptools import find_packages, setup
from setuptools import setup, Extension
import shutil
import os 
from pathlib import Path
from setuptools import setup, find_packages

BASEPATH = Path(__file__).resolve().parent


setup(
    name="temporis",
    packages=find_packages(),
    version="0.3.1",
    description="Time-series utilities for predictive maintenance",
    author="",
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn",
        "seaborn",
        "xgboost",
        "gwpy",
        "emd",
        "numba",
        "dill",
        "mmh3",
        "pyinform",
        "antropy"
    ],
    license="MIT",
    include_package_data=True,
    package_data={"": ["RUL*.txt", "train*.txt", "test*.txt"]},


)
