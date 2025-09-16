# run.py
# Script para ejecutar la interfaz de Othello desde la carpeta gui

import os
import sys

# Aseguramos que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Importamos los módulos necesarios desde gui
from gui.game_gui import GameGUI
from gui.probs_plot import ProbsPlot
from gui.model_handler import ModelHandler

if __name__ == "__main__":
    # Ruta al checkpoint del modelo pre-entrenado
    checkpoint_path = "../ckpts/battery_othello/gpt_championship.ckpt"  # Cambiado a gpt_championship.ckpt
    
    # Inicializa la GUI del gráfico de probabilidades
    probs_plot = ProbsPlot()
    
    # Crear el manejador del modelo
    model_handler = ModelHandler(checkpoint_path, probs_plot)
    
    # Inicializa la GUI del juego con el callback para actualizar el gráfico
    game_gui = GameGUI(callback=model_handler.update_probabilities)
    
    # Ejecutar la interfaz
    game_gui.run()
    game_gui.run()
