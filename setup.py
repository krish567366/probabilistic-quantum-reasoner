#!/usr/bin/env python3
"""
Setup script for probabilistic-quantum-reasoner.

This script is maintained for backwards compatibility.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements
def read_requirements(filename):
    """Read requirements from file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Base requirements
install_requires = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "networkx>=2.6.0",
    "typing-extensions>=4.0.0",
    "quantummeta-license>=1.0.0"
]

# Optional dependencies
extras_require = {
    "qiskit": [
        "qiskit>=0.45.0",
        "qiskit-aer>=0.12.0",
        "qiskit-ibm-runtime>=0.15.0"
    ],
    "pennylane": [
        "pennylane>=0.30.0",
        "pennylane-qiskit>=0.30.0"
    ],
    "visualization": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0"
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.8.0",
        "black>=22.0.0",
        "pylint>=2.15.0",
        "mypy>=0.991",
        "isort>=5.10.0",
        "pre-commit>=2.20.0",
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "mkdocs>=1.4.0",
        "mkdocs-material>=8.5.0",
        "mkdocstrings[python]>=0.19.0"
    ],
    "docs": [
        "mkdocs>=1.4.0",
        "mkdocs-material>=8.5.0",
        "mkdocstrings[python]>=0.19.0",
        "mkdocs-autorefs>=0.4.0"
    ]
}

# Add "all" extras that includes everything except dev
extras_require["quantum"] = (
    extras_require["qiskit"] + 
    extras_require["pennylane"]
)

extras_require["all"] = (
    extras_require["quantum"] + 
    extras_require["visualization"]
)

setup(
    name="probabilistic-quantum-reasoner",
    version="0.1.0",
    author="Quantum AI Research Team",
    author_email="quantum-reasoner@example.com",
    description="Quantum-classical hybrid reasoning engine for uncertainty-aware AI inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-ai/probabilistic-quantum-reasoner",
    project_urls={
        "Bug Tracker": "https://github.com/quantum-ai/probabilistic-quantum-reasoner/issues",
        "Documentation": "https://quantum-reasoner.readthedocs.io",
        "Source Code": "https://github.com/quantum-ai/probabilistic-quantum-reasoner",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "probabilistic_quantum_reasoner": [
            "py.typed",
        ],
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "pqr-example=probabilistic_quantum_reasoner.examples.cli:main",
        ],
    },
    keywords=[
        "quantum computing",
        "machine learning", 
        "probabilistic reasoning",
        "bayesian networks",
        "quantum machine learning",
        "causal inference",
        "artificial intelligence",
        "uncertainty quantification"
    ],
)
