# run.py
# Script para ejecutar la interfaz de Othello desde la carpeta gui

import os
import sys

# Aseguramos que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Importamos los módulos necesarios desde gui
from gui.game_gui import GameGUI
from gui.probs_plot import ProbsPlot
from gui.probability_heatmap import ProbabilityHeatmap
from gui.model_handler import ModelHandler

if __name__ == "__main__":
    # Ruta al checkpoint del modelo pre-entrenado
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ckpts", "gpt_championship.ckpt")
    
    # Inicializar ambas visualizaciones
    probs_plot = ProbsPlot()  # Gráfico de barras tradicional
    probability_heatmap = ProbabilityHeatmap()  # Nuevo heatmap del tablero
    
    # Crear el manejador del modelo con ambas visualizaciones
    model_handler = ModelHandler(checkpoint_path, probs_plot, probability_heatmap)
    
    # Inicializa la GUI del juego con el manejador del modelo completo
    game_gui = GameGUI(callback=model_handler)
    
    # Ejecutar la interfaz
    game_gui.run()
