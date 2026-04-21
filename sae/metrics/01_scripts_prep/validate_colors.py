"""
Script de validación: Verifica que next_hand_color se capturó correctamente.
Compara el método antiguo (move_idx % 2) vs el nuevo (capturado del motor).
"""

import numpy as np
from pathlib import Path
import sys

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def validate_colors():
    """Valida que los colores capturados sean diferentes al método move_idx % 2."""
    
    # Cargar datos
    data_path = project_root / "sae" / "metrics" / "02_data" / "board_states_1games.npz"
    
    print("=" * 60)
    print("Validación de Captura de Colores")
    print("=" * 60)
    print()
    
    print(f"Cargando: {data_path}")
    data = np.load(data_path)
    boards = data['boards']
    colors_real = data['colors']
    
    n_games, n_moves = colors_real.shape
    print(f"✓ Shape de colores: {colors_real.shape}")
    print(f"✓ Valores únicos: {np.unique(colors_real)}")
    print()
    
    # Simular el método antiguo (asume alternancia perfecta)
    colors_old_method = np.zeros_like(colors_real)
    for game_idx in range(n_games):
        for move_idx in range(n_moves):
            colors_old_method[game_idx, move_idx] = 1 if move_idx % 2 == 0 else -1
    
    # Comparar
    differences = (colors_real != colors_old_method)
    n_differences = np.sum(differences)
    total_positions = n_games * n_moves
    
    print(f"Comparación con método antiguo (move_idx % 2):")
    print(f"  Total de posiciones: {total_positions:,}")
    print(f"  Diferencias encontradas: {n_differences:,} ({100 * n_differences / total_positions:.2f}%)")
    print()
    
    if n_differences > 0:
        print("✓ ÉXITO: Los colores capturados son diferentes al método antiguo.")
        print("  Esto indica que se están manejando correctamente los forfeits.")
        print()
        
        # Mostrar algunos ejemplos de diferencias
        print("Ejemplos de posiciones con forfeits detectados:")
        print("-" * 60)
        
        example_count = 0
        for game_idx in range(n_games):
            for move_idx in range(n_moves):
                if differences[game_idx, move_idx]:
                    old_color = colors_old_method[game_idx, move_idx]
                    real_color = colors_real[game_idx, move_idx]
                    color_name_old = "Negro" if old_color == 1 else "Blanco"
                    color_name_real = "Negro" if real_color == 1 else "Blanco"
                    
                    print(f"  Partida {game_idx}, Movimiento {move_idx}:")
                    print(f"    Método antiguo: {color_name_old} ({old_color})")
                    print(f"    Método nuevo:   {color_name_real} ({real_color})")
                    
                    example_count += 1
                    if example_count >= 5:
                        break
            if example_count >= 5:
                break
        
        print("-" * 60)
    else:
        print(" ADVERTENCIA: No se encontraron diferencias con el método antiguo.")
        print("  Esto podría indicar que:")
        print("  1. No hay forfeits en las 200 partidas (muy improbable)")
        print("  2. La captura no está funcionando correctamente")
    
    print()
    print("=" * 60)
    print("Validación completada")
    print("=" * 60)


if __name__ == "__main__":
    validate_colors()
