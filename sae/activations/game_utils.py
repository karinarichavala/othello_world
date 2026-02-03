"""
Utilidades para guardar partidas sintéticas
"""

import os


def save_games_to_file(sequences, filepath):
    """
    Guarda partidas en formato texto (sobreescribe si existe)
    
    Args:
        sequences: Lista de partidas (cada una es lista de enteros)
        filepath: Ruta completa del archivo
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for seq in sequences:
            f.write(' '.join(map(str, seq)) + '\n')
    
    file_size = os.path.getsize(filepath) / 1024
    print(f"💾 Partidas guardadas: {filepath} ({len(sequences)} partidas, {file_size:.1f} KB)")


def get_games_filepath(config):
    """
    Genera la ruta del archivo de partidas basado en num_games
    
    Args:
        config: ExtractionConfig
    
    Returns:
        str: Ruta completa (ej: 'sae/activations/data/games_200.txt')
    """
    filename = f"games_{config.num_games}.txt"
    return os.path.join(config.activations_dir, filename)
