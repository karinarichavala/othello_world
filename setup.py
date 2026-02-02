"""
Othello World - Setup Configuration
Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="othello-world",
    version="1.0.0",
    author="Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg",
    description="Mechanistic interpretability research on Othello-GPT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/likenneth/othello_world",
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
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
        "python-chess>=1.0.0",
        "pgn==0.1.0",
    ],
    extras_require={
        "interpretability": [
            "transformer-lens>=1.0.0",
            "einops>=0.6.0",
            "fancy-einsum>=0.0.3",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "all": [
            "transformer-lens>=1.0.0",
            "einops>=0.6.0",
            "fancy-einsum>=0.0.3",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "othello-gui=gui.run:main",
        ],
    },
)
