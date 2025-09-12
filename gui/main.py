# main.py
# Archivo principal para ejecutar la interfaz de Othello y el gráfico de probabilidades

import os
import sys

# Agregamos el directorio raíz al path para poder importar módulos del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Importamos nuestros módulos con la ruta correcta
from gui.game_gui import GameGUI
from gui.probs_plot import ProbsPlot
from gui.model_handler import ModelHandler

if __name__ == "__main__":
    # Ruta al checkpoint del modelo pre-entrenado
    # Ajusta esta ruta a la ubicación de tu checkpoint
    checkpoint_path = "ckpts/battery_othello/checkpoint.pt"
    
    # Inicializa la GUI del gráfico de probabilidades
    probs_plot = ProbsPlot()
    
    # Crear el manejador del modelo
    model_handler = ModelHandler(checkpoint_path, probs_plot)
    
    # Inicializa la GUI del juego con el callback para actualizar el gráfico
    game_gui = GameGUI(callback=model_handler.update_probabilities)
    
    # Ejecutar la interfaz
    game_gui.run()
