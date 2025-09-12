# run_gui.py
# Script para ejecutar la interfaz de Othello desde la raíz del proyecto

import os
import sys

# Aseguramos que el directorio actual esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importamos los módulos necesarios
from gui.game_gui import GameGUI
from gui.probs_plot import ProbsPlot
from gui.model_handler import ModelHandler

if __name__ == "__main__":
    # Ruta al checkpoint del modelo pre-entrenado
    checkpoint_path = "ckpts/battery_othello/checkpoint.pt"
    
    # Inicializa la GUI del gráfico de probabilidades
    probs_plot = ProbsPlot()
    
    # Crear el manejador del modelo
    model_handler = ModelHandler(checkpoint_path, probs_plot)
    
    # Inicializa la GUI del juego con el callback para actualizar el gráfico
    game_gui = GameGUI(callback=model_handler.update_probabilities)
    
    # Ejecutar la interfaz
    game_gui.run()
