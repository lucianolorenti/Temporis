from setuptools import find_packages, setup
from setuptools import setup, Extension
import shutil
import os 
from pathlib import Path
from setuptools import setup, find_packages
from Cython.Build import cythonize

BASEPATH = Path(__file__).resolve().parent


setup(
    name="temporis",
    packages=find_packages(),
    version="0.2.0",
    description="Time series utilities for machine learning",
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
        "mmh3"
    ],
    license="MIT",
    include_package_data=True,
    package_data={"": ["RUL*.txt", "train*.txt", "test*.txt"]},


)
