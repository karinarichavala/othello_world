"""
Configuración para extracción de activaciones del modelo Othello-GPT
"""
import torch

class ExtractionConfig:
    """Configuración para extracción de activaciones"""
    
    # === GENERACIÓN DE PARTIDAS ===
    num_games = 25000           # Número de partidas sintéticas a generar
    memory_threshold = 1000        # Umbral para usar archivo (no modificar)
    
    # === EXTRACCIÓN DE ACTIVACIONES ===
    target_layer = 5               # Bloque 6 (índice 5 en Python, 0-7)
    checkpoint_path = "ckpts/gpt_championship.ckpt"  # Checkpoint del GPT
    batch_size_games = 100         # Partidas por lote (si > 1000 games)
    
    # === RUTAS DE SALIDA ===
    activations_dir = "sae/activations/data"
    
    # === HARDWARE ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __repr__(self):
        return (f"ExtractionConfig(num_games={self.num_games}, target_layer={self.target_layer})")
    
    def get_activations_filename(self):
        """Genera el nombre del archivo de activaciones basado en la configuración"""
        return f"layer{self.target_layer + 1}_{self.num_games}games.npy"
    
    def get_activations_path(self):
        """Retorna la ruta completa para guardar las activaciones"""
        import os
        return os.path.join(self.activations_dir, self.get_activations_filename())
