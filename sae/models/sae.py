"""
Sparse Autoencoder (SAE) para activaciones de Othello-GPT

Uso típico:
    # Cargar modelo pre-entrenado
    from sae.model import load_sae
    model = load_sae('sae/model/saved_model/sae_othello_final.pt')
    
    # Usar para análisis
    hidden = model.encode(activations)
    features = (hidden > 0).nonzero()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class SparseAutoencoder(nn.Module):
    """
    Autoencoder disperso para descomponer activaciones en features interpretables
    
    Arquitectura:
        Encoder: input_dim -> hidden_dim (con ReLU)
        Decoder: hidden_dim -> input_dim (lineal)
    
    La sparsity se logra mediante:
        1. ReLU en el encoder (activaciones no negativas → sparsity natural)
        2. Penalización L1 en las activaciones durante el entrenamiento
    """
    
    def __init__(self, input_dim=512, hidden_dim=16384, tied_weights=False):
        """
        Args:
            input_dim: Dimensión de las activaciones de entrada (512 para Othello-GPT capa 5)
            hidden_dim: Dimensión del espacio latente disperso (16384 → 32x expansión)
            tied_weights: Si True, W_dec = W_enc^T (reduce parámetros a la mitad)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights
        
        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: hidden_dim -> input_dim
        if tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Inicialización Xavier para estabilidad
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización de pesos (Xavier/Glorot)"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x):
        """
        Proyecta activaciones al espacio latente disperso
        
        Args:
            x: Tensor (batch_size, input_dim) o (input_dim,)
        
        Returns:
            hidden: Activaciones dispersas (batch_size, hidden_dim)
        """
        return F.relu(self.encoder(x))
    
    def decode(self, hidden):
        """
        Reconstruye activaciones desde el espacio latente
        
        Args:
            hidden: Tensor (batch_size, hidden_dim)
        
        Returns:
            reconstruction: Activaciones reconstruidas (batch_size, input_dim)
        """
        if self.tied_weights:
            return F.linear(hidden, self.encoder.weight.t(), self.decoder_bias)
        else:
            return self.decoder(hidden)
    def normalize_decoder(self):          
        """
        Normaliza columnas del decoder a norma unitaria.
        Llamar después de cada optimizer.step() durante entrenamiento.
        """
        if not self.tied_weights:
            with torch.no_grad():
                norms = self.decoder.weight.data.norm(dim=0, keepdim=True)
                self.decoder.weight.data.div_(norms.clamp(min=1e-8))
    
    def forward(self, x):
        """
        Forward pass completo: encode -> decode
        
        Args:
            x: Tensor (batch_size, input_dim)
        
        Returns:
            reconstruction: Reconstrucción (batch_size, input_dim)
            hidden: Activaciones latentes (batch_size, hidden_dim)
        """
        hidden = self.encode(x)
        reconstruction = self.decode(hidden)
        return reconstruction, hidden
    
    def get_active_features(self, x, threshold=0.0):
        """
        Obtiene las features activas y sus valores
        
        Args:
            x: Tensor de activaciones (batch_size, input_dim) o (input_dim,)
            threshold: Umbral de activación (default: 0.0)
        
        Returns:
            dict con:
                - 'indices': Índices de features activas
                - 'values': Valores de activación
                - 'count': Número de features activas (L0)
        """
        # Asegurar batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        hidden = self.encode(x)
        
        # Encontrar features activas
        active_mask = hidden > threshold
        
        if squeeze_output:
            # Retornar para un solo ejemplo
            indices = active_mask[0].nonzero(as_tuple=False).squeeze(-1)
            values = hidden[0, indices]
            return {
                'indices': indices,
                'values': values,
                'count': len(indices)
            }
        else:
            # Retornar para batch
            results = []
            for i in range(hidden.size(0)):
                indices = active_mask[i].nonzero(as_tuple=False).squeeze(-1)
                values = hidden[i, indices]
                results.append({
                    'indices': indices,
                    'values': values,
                    'count': len(indices)
                })
            return results


def load_sae(checkpoint_path, device='cpu'):
    """
    Carga un SAE desde un checkpoint guardado
    
    Args:
        checkpoint_path: Ruta al archivo .pt
        device: 'cuda' o 'cpu'
    
    Returns:
        model: SAE cargado en modo evaluación
        checkpoint: Dict con config e historial (opcional)
    
    Ejemplo:
        model = load_sae('sae/model/saved_model/sae_othello_final.pt')
        hidden = model.encode(activations)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extraer configuración
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 512)
    hidden_dim = config.get('hidden_dim', 16384)
    tied_weights = config.get('tied_weights', False)
    
    # Crear modelo
    model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        tied_weights=tied_weights
    )
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f'✓ SAE cargado desde: {checkpoint_path}')
    print(f'  Arquitectura: {input_dim} → {hidden_dim} → {input_dim}')
    if 'history' in checkpoint:
        history = checkpoint['history']
        if 'l0_sparsity' in history and len(history['l0_sparsity']) > 0:
            final_l0 = history['l0_sparsity'][-1]
            final_fraction = history['fraction_active'][-1]
            print(f'  Sparsity final: {final_l0:.1f} features ({final_fraction:.2%} activas)')
    
    return model, checkpoint
