import sys
import os

# Agregar el directorio raíz del proyecto al sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from gui.model_handler import ModelHandler

# Ruta absoluta al checkpoint del modelo
checkpoint_path = os.path.join(root_dir, "ckpts", "gpt_championship.ckpt")

# Inicializar el manejador del modelo
model_handler = ModelHandler(checkpoint_path=checkpoint_path)

# Verificar si el modelo se cargó correctamente
if model_handler.model:
    print("Modelo cargado exitosamente.")
else:
    print("Error al cargar el modelo.")

# Probar el método get_move_probabilities con un historial de movimientos ficticio
move_history = [0, 1, 2, 3, 4, 5]  # Ejemplo de movimientos
move_probs = model_handler.get_move_probabilities(move_history)

# Imprimir las probabilidades calculadas
print("Probabilidades de movimiento:")
for move, prob in move_probs.items():
    print(f"{move}: {prob:.4f}")