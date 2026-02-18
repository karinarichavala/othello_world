"""
SAE (Sparse Autoencoder) module for Othello-GPT
Provides tools for training and analyzing sparse autoencoders on model activations
"""

from .models.sae import SparseAutoencoder, load_sae

__all__ = ['SparseAutoencoder', 'load_sae']
