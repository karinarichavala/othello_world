# run.py
# Script para ejecutar la interfaz de Othello desde la carpeta gui

import os

# Importamos los módulos necesarios desde gui
from gui.game_gui import GameGUI
from gui.probs_plot import ProbsPlot
from gui.probability_heatmap import ProbabilityHeatmap
from gui.game_heatmap import GameHeatmap
from gui.model_handler import ModelHandler


def main():
    """Entry point for GUI application"""
    # Ruta al checkpoint del modelo pre-entrenado
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ckpts", "gpt_championship.ckpt")
    
    # Inicializar todas las visualizaciones
    probs_plot = ProbsPlot()  # Gráfico de barras tradicional
    probability_heatmap = ProbabilityHeatmap()  # Heatmap original
    game_heatmap = GameHeatmap()  # Nuevo heatmap integrado con fichas
    
    # Crear el manejador del modelo con todas las visualizaciones
    model_handler = ModelHandler(checkpoint_path, probs_plot, probability_heatmap, game_heatmap)
    
    # Inicializa la GUI del juego con el manejador del modelo completo
    game_gui = GameGUI(callback=model_handler)
    
    # Ejecutar la interfaz
    game_gui.run()


if __name__ == "__main__":
    main()
