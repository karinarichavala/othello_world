# run.py
# Script simple para ejecutar la interfaz directamente desde la carpeta gui

import os
import sys

# Agregamos el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Importamos los módulos necesarios
from game_gui import GameGUI
from probs_plot import ProbsPlot
from model_handler import ModelHandler

if __name__ == "__main__":
    # Ruta al checkpoint del modelo pre-entrenado
    checkpoint_path = os.path.join(parent_dir, "ckpts/battery_othello/checkpoint.pt")
    
    # Inicializa la GUI del gráfico de probabilidades
    probs_plot = ProbsPlot()
    
    # Crear el manejador del modelo
    model_handler = ModelHandler(checkpoint_path, probs_plot)
    
    # Inicializa la GUI del juego con el callback para actualizar el gráfico
    game_gui = GameGUI(callback=model_handler.update_probabilities)
    
    # Ejecutar la interfaz
    game_gui.run()
