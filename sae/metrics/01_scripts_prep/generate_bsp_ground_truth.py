"""
Genera la ground truth de Board State Properties (BSPs) para todas las posiciones.
Convierte los tableros reconstruidos en vectores de 198 BSPs usando bsp_identifier.py
"""

import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sae.bsp_identifier import identificador


def boards_to_bsp_ground_truth(boards_filepath: str, output_filepath: str = None):
    """
    Convierte tableros en ground truth de BSPs.
    
    Args:
        boards_filepath: Ruta a board_states_*.npz con 'boards' y 'colors'
        output_filepath: Ruta donde guardar bsp_ground_truth_*.npy (opcional)
        
    Returns:
        bsp_ground_truth: Array de forma (n_positions, 198)
    """
    boards_path = Path(boards_filepath)
    
    if not boards_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {boards_filepath}")
    
    print("=" * 60)
    print("Generación de Ground Truth de BSPs")
    print("=" * 60)
    print()
    
    # Cargar tableros y colores desde archivo .npz
    print(f"Cargando tableros y colores desde: {boards_path}")
    data = np.load(boards_path)
    boards = data['boards']
    colors = data['colors']
    n_games, n_moves, board_h, board_w = boards.shape
    n_positions = n_games * n_moves
    
    print(f"✓ Shape de tableros: {boards.shape}")
    print(f"✓ Shape de colores: {colors.shape}")
    print(f"✓ Total de posiciones: {n_positions:,}")
    print()
    
    # Obtener nombres de BSPs desde el primer tablero
    print("Identificando estructura de BSPs...")
    sample_bsps = identificador(boards[0, 0], color_jugador=1)
    bsp_names = list(sample_bsps.keys())
    n_bsps = len(bsp_names)
    
    print(f"✓ Total de BSPs por posición: {n_bsps}")
    print()
    
    # Preparar array de output
    bsp_ground_truth = np.zeros((n_positions, n_bsps), dtype=np.int8)
    
    # Procesar cada posición
    print("Generando BSPs para cada posición...")
    position_idx = 0
    
    for game_idx in tqdm(range(n_games), desc="Procesando partidas"):
        for move_idx in range(n_moves):
            tablero = boards[game_idx, move_idx]
            
            # Usar el color REAL capturado del motor de juego (maneja forfeits correctamente)
            color_jugador = colors[game_idx, move_idx]
            
            # Generar BSPs
            bsps_dict = identificador(tablero, color_jugador)
            
            # Convertir diccionario a array numérico
            bsp_values = [int(bsps_dict[name]) for name in bsp_names]
            bsp_ground_truth[position_idx] = bsp_values
            
            position_idx += 1
    
    print(f"✓ Generadas BSPs para {position_idx:,} posiciones")
    print()
    
    # Estadísticas
    print("Estadísticas de BSPs generadas:")
    print(f"  Shape: {bsp_ground_truth.shape}")
    print(f"  Dtype: {bsp_ground_truth.dtype}")
    print(f"  Valores únicos: {np.unique(bsp_ground_truth)}")
    print(f"  Memoria: {bsp_ground_truth.nbytes / (1024**2):.2f} MB")
    
    # Calcular densidad (porcentaje de BSPs activas)
    active_bsps_per_position = np.sum(bsp_ground_truth, axis=1)
    avg_active = np.mean(active_bsps_per_position)
    print(f"  BSPs activas promedio por posición: {avg_active:.1f}")
    print()
    
    # Guardar si se especificó un archivo de salida
    if output_filepath is not None:
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, bsp_ground_truth)
        print(f"✓ Ground truth guardada en: {output_path}")
        print(f"  Tamaño del archivo: {output_path.stat().st_size / (1024**2):.2f} MB")
        print()
        
        # Guardar también los nombres de las BSPs para referencia
        names_path = output_path.with_suffix('.names.npy')
        np.save(names_path, bsp_names)
        print(f"✓ Nombres de BSPs guardados en: {names_path}")
    
    return bsp_ground_truth, bsp_names


def main():
    """Script principal para generar ground truth de BSPs."""
    
    # Rutas por defecto
    project_root = Path(__file__).parent.parent.parent.parent
    boards_file = project_root / "sae" / "metrics" / "02_data" / "board_states_200games.npz"
    output_file = project_root / "sae" / "metrics" / "02_data" / "bsp_ground_truth_200games.npy"
    
    print()
    print("=" * 60)
    print("Generación de Ground Truth de BSPs")
    print("Paper: Karvonen et al., NeurIPS 2024")
    print("=" * 60)
    print()
    
    # Verificar si el archivo de entrada existe
    if not boards_file.exists():
        print(f"ERROR: No se encontró {boards_file}")
        print("Por favor ejecuta primero generate_board_states.py")
        return
    
    # Generar ground truth
    bsp_ground_truth, bsp_names = boards_to_bsp_ground_truth(
        boards_filepath=str(boards_file),
        output_filepath=str(output_file)
    )
    
    print()
    print("=" * 60)
    print("✓ Generación de ground truth completada exitosamente")
    print("=" * 60)
    print()
    print("Siguiente paso:")
    print("  1. Cargar activaciones SAE: layer5_200games.npy")
    print("  2. Cargar modelo SAE: sae/model/saved_model/sae_othello_best.pt")
    print("  3. Calcular Coverage y Reconstruction metrics")


if __name__ == "__main__":
    main()
