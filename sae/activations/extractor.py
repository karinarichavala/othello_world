"""
Extractor de activaciones con hooks de PyTorch
"""

import torch
import numpy as np
from tqdm import tqdm


class ActivationExtractor:
    """
    Extrae activaciones de una capa específica del modelo GPT usando hooks
    """
    
    def __init__(self, model, target_layer, device='cpu'):
        """
        Args:
            model: Modelo GPT cargado
            target_layer: Índice de la capa a extraer (0-7 para Othello-GPT)
            device: 'cuda' o 'cpu'
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = []
        self.hook_handle = None
        
        # Registrar hook
        self._register_hook()
    
    def _register_hook(self):
        """Registra un forward hook en la capa objetivo"""
        def hook_fn(module, input, output):
            # output shape: (batch_size, sequence_length, hidden_dim)
            # Guardamos en CPU para ahorrar memoria GPU
            self.activations.append(output.detach().cpu())
        
        # Registrar el hook en el bloque específico
        self.hook_handle = self.model.blocks[self.target_layer].register_forward_hook(hook_fn)
        print(f"✓ Hook registrado en Bloque {self.target_layer + 1} (índice {self.target_layer})")
    
    def remove_hook(self):
        """Elimina el hook registrado"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def extract_from_sequence(self, sequence):
        """
        Extrae activaciones de una secuencia de movimientos
        
        Args:
            sequence: Tensor o lista con tokens
        
        Returns:
            activations: Tensor de forma (sequence_length, hidden_dim)
        """
        self.activations = []
        
        # Convertir a tensor si es lista
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.long)
        
        # Truncar secuencia si excede block_size (59 para Othello-GPT)
        max_length = 59
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        # Preparar input (agregar batch dimension)
        x = sequence.unsqueeze(0).to(self.device)  # (1, sequence_length)
        
        # Forward pass (sin calcular gradientes)
        with torch.no_grad():
            self.model(x)
        
        # Obtener las activaciones capturadas
        if len(self.activations) > 0:
            # Shape: (1, sequence_length, hidden_dim) -> (sequence_length, hidden_dim)
            acts = self.activations[0].squeeze(0)
            return acts
        else:
            return None
    
    def extract_from_games(self, games, desc="Extrayendo activaciones"):
        """
        Extrae activaciones de múltiples partidas
        
        Args:
            games: Lista de partidas (cada una es una lista de tokens)
            desc: Descripción para la barra de progreso
        
        Returns:
            all_activations: Array numpy con todas las activaciones (N, hidden_dim)
        """
        all_activations = []
        
        for game in tqdm(games, desc=desc):
            # Extraer activaciones de esta partida
            acts = self.extract_from_sequence(game)
            
            if acts is not None:
                # Agregar todas las activaciones de esta secuencia
                all_activations.append(acts.numpy())
        
        # Concatenar todas las activaciones
        if all_activations:
            return np.vstack(all_activations)
        else:
            return np.array([])
    
    def __del__(self):
        """Limpieza: remover hook al destruir el objeto"""
        self.remove_hook()
