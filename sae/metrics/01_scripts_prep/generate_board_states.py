"""
Genera estados de tablero a partir de las secuencias de movimientos.
Convierte games_*.txt (secuencias de movimientos) en board_states_*.npy (estados de tablero).
"""

import numpy as np
from pathlib import Path
import sys

# Añadir el directorio raíz al path para importar módulos del proyecto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data.othello import OthelloBoardState


def reconstruct_board_states_from_games(games_filepath: str, output_filepath: str = None):
    """
    Reconstruye los estados de tablero desde un archivo de partidas.
    
    Args:
        games_filepath: Ruta al archivo games_*.txt con secuencias de movimientos
        output_filepath: Ruta donde guardar board_states_*.npz (opcional)
        
    Returns:
        dict: {'boards': array (n_games, n_moves, 8, 8), 'colors': array (n_games, n_moves)}
    """
    games_path = Path(games_filepath)
    
    if not games_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {games_filepath}")
    
    print(f"Leyendo partidas desde: {games_path}")
    
    # Leer todas las partidas
    with open(games_path, 'r') as f:
        games_lines = f.readlines()
    
    n_games = len(games_lines)
    print(f"Número de partidas: {n_games}")
    
    # Procesar cada partida
    all_board_states = []
    all_hand_colors = []
    
    for game_idx, line in enumerate(games_lines):
        # Parsear los movimientos (secuencia de enteros 0-63)
        moves = [int(m) for m in line.strip().split()]
        n_moves = len(moves)
        
        # Reconstruir la partida movimiento por movimiento
        board_state = OthelloBoardState()
        game_boards = []
        game_colors = []
        
        for move in moves:
            # Capturar el color del jugador ANTES de aplicar el movimiento
            game_colors.append(board_state.next_hand_color)
            
            board_state.update([move])
            # Guardar el estado del tablero después de este movimiento
            # state es un array 8x8 con 1=black, -1=white, 0=empty
            game_boards.append(board_state.state.copy())
        
        all_board_states.append(game_boards)
        all_hand_colors.append(game_colors)
        
        if (game_idx + 1) % 50 == 0:
            print(f"Procesadas {game_idx + 1}/{n_games} partidas...")
    
    # Verificar longitudes de las partidas
    game_lengths = [len(game) for game in all_board_states]
    min_len, max_len = min(game_lengths), max(game_lengths)
    print(f"Longitudes de partidas: min={min_len}, max={max_len}")
    
    # Convertir a array numpy
    # Si hay diferentes longitudes, rellenar con ceros o truncar a la longitud mínima
    if min_len != max_len:
        print(f"  Las partidas tienen diferentes longitudes, usando longitud mínima: {min_len}")
        all_board_states = [game[:min_len] for game in all_board_states]
        all_hand_colors = [colors[:min_len] for colors in all_hand_colors]
    
    board_states = np.array(all_board_states)  # Shape: (n_games, n_moves, 8, 8)
    hand_colors = np.array(all_hand_colors, dtype=np.int8)  # Shape: (n_games, n_moves)
    
    print(f"Forma del array de estados: {board_states.shape}")
    print(f"Forma del array de colores: {hand_colors.shape}")
    print(f"Rango de valores (tableros): [{board_states.min()}, {board_states.max()}]")
    print(f"Valores únicos (tableros): {np.unique(board_states)}")
    print(f"Valores únicos (colores): {np.unique(hand_colors)}")
    
    # Guardar si se especificó un archivo de salida
    if output_filepath is not None:
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar ambos arrays en un archivo .npz
        np.savez(output_path, boards=board_states, colors=hand_colors)
        print(f"Estados guardados en: {output_path}")
        print(f"Tamaño del archivo: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return {'boards': board_states, 'colors': hand_colors}


def main():
    """Script principal para generar estados de tablero."""
    
    # Rutas por defecto
    project_root = Path(__file__).parent.parent.parent.parent
    games_file = project_root / "sae" / "activations" / "data" / "games_200.txt"
    output_file = project_root / "sae" / "metrics" / "02_data" / "board_states_200games.npz"
    
    print("=" * 60)
    print("Generación de Estados de Tablero desde Partidas Sintéticas")
    print("=" * 60)
    print()
    
    # Verificar si el archivo de entrada existe
    if not games_file.exists():
        print(f"ERROR: No se encontró {games_file}")
        print("Por favor verifica la ruta del archivo de partidas.")
        return
    
    # Generar estados
    board_states = reconstruct_board_states_from_games(
        games_filepath=str(games_file),
        output_filepath=str(output_file)
    )
    
    print()
    print("=" * 60)
    print("✓ Reconstrucción completada exitosamente")
    print("=" * 60)
    print()
    print("Siguiente paso: Usa estos estados con bsp_identifier.py")
    print("para generar la ground truth de BSPs.")


if __name__ == "__main__":
    main()
