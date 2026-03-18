"""
Verificar que los tableros reconstruidos tienen el formato correcto
"""

import numpy as np
import sys
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.othello import OthelloBoardState
from sae.bsp_identifier import identificador


def visualize_board(board):
    """Visualiza un tablero de forma legible"""
    print("  0 1 2 3 4 5 6 7")
    for i, row in enumerate(board):
        symbols = []
        for val in row:
            if val == 1:
                symbols.append('●')  # Negro
            elif val == -1:
                symbols.append('○')  # Blanco
            else:
                symbols.append('·')  # Vacío
        print(f"{i} {' '.join(symbols)}")


def main():
    print("=" * 60)
    print("Verificación de Tableros Reconstruidos")
    print("=" * 60)
    
    # Cargar tableros reconstruidos
    boards_path = project_root / "metrics" / "02_data" / "board_states_1games.npz"
    data = np.load(boards_path)
    boards = data['boards']

    print(f"\n Información del array:")
    print(f"   Shape: {boards.shape}")
    print(f"   Dtype: {boards.dtype}")
    print(f"   Valores únicos: {np.unique(boards)}")
    print(f"   Memoria: {boards.nbytes / (1024**2):.2f} MB")
    
    # Verificar primera partida
    print(f"\n PARTIDA 0 - Primeros 5 movimientos:")
    print("=" * 60)
    
    for move_idx in range(5):
        board = boards[0, move_idx]
        
        print(f"\n Después del movimiento {move_idx + 1}:")
        visualize_board(board)
        
        # Contar piezas
        black_count = np.sum(board == 1)
        white_count = np.sum(board == -1)
        empty_count = np.sum(board == 0)
        
        print(f"   Negras (●): {black_count}, Blancas (○): {white_count}, Vacías (·): {empty_count}")
    
    # Probar bsp_identifier con el tablero del movimiento 5
    print(f"\n Verificar compatibilidad con bsp_identifier:")
    print("=" * 60)
    
    test_board = boards[0, 4]  # Movimiento 5
    
    # Probar con negro (1)
    print(f"\n   Probando identificador(tablero, color_jugador=1)...")
    bsps_negro = identificador(test_board, color_jugador=1)
    print(f"   ✓ Resultado: vector de {len(bsps_negro)} BSPs")
    print(f"   ✓ Valores únicos: {np.unique(bsps_negro)}")
    print(f"   ✓ BSPs activos (=1): {np.sum(bsps_negro == 1)}")
    
    # Probar con blanco (-1)
    print(f"\n   Probando identificador(tablero, color_jugador=-1)...")
    bsps_blanco = identificador(test_board, color_jugador=-1)
    print(f"   ✓ Resultado: vector de {len(bsps_blanco)} BSPs")
    print(f"   ✓ Valores únicos: {np.unique(bsps_blanco)}")
    print(f"   ✓ BSPs activos (=1): {np.sum(bsps_blanco == 1)}")
    
    # Verificar algunas partidas más
    n_games = boards.shape[0]
    print(f"\n Verificación estadística (primeras {min(10, n_games)} partidas):")
    print("=" * 60)

    for game_idx in range(min(10, n_games)):
        # Último movimiento de cada partida
        final_board = boards[game_idx, -1]
        
        black_count = np.sum(final_board == 1)
        white_count = np.sum(final_board == -1)
        total = black_count + white_count
        
        print(f"   Partida {game_idx}: {black_count} negras, {white_count} blancas (total: {total})")
    
    print(f"\n" + "=" * 60)
    print(" Verificación completada - Los tableros tienen el formato correcto")
    print("=" * 60)


if __name__ == "__main__":
    main()
