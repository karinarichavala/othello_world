# model_handler.py
# Encapsula la lógica de cargar y consultar el modelo Othello-GPT

import os
import torch
import numpy as np
import torch.nn.functional as F
import sys

# Agregamos el directorio raíz al path para poder importar módulos del proyecto
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importamos las clases y funciones necesarias del proyecto
from mingpt.model import GPT, GPTConfig
from data.othello import permit, permit_reverse

class ModelHandler:
    def __init__(self, checkpoint_path=None, probs_plot=None):
        """
        Inicializa el manejador del modelo Othello-GPT.
        
        Args:
            checkpoint_path: Ruta al checkpoint del modelo pre-entrenado.
                           Si es None, no se cargará ningún modelo.
            probs_plot: Referencia al objeto ProbsPlot para actualizar el gráfico.
        """
        self.model = None
        self.probs_plot = probs_plot
        
        if checkpoint_path:
            try:
                self.model = self.load_model(checkpoint_path)
                print(f"Modelo cargado correctamente desde {checkpoint_path}")
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                print("Continuando sin el modelo (no se mostrarán probabilidades)")
    
    def load_model(self, checkpoint_path):
        """
        Carga el modelo Othello-GPT desde un checkpoint.
        
        Args:
            checkpoint_path: Ruta al checkpoint del modelo pre-entrenado.
        
        Returns:
            El modelo cargado.
        """
        # Configuración del modelo - ajusta estos parámetros según tu modelo entrenado
        model_config = GPTConfig(
            vocab_size=64,  # Tamaño del vocabulario (número de posiciones en el tablero)
            block_size=60,  # Tamaño máximo de secuencia
            n_layer=8,      # Número de capas transformer
            n_head=8,       # Número de cabezas de atención
            n_embd=512      # Dimensión de embedding
        )
        
        # Crear el modelo
        model = GPT(model_config)
        
        # Cargar los pesos del checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        
        # Poner el modelo en modo de evaluación
        model.eval()
        
        return model
    
    def get_move_probabilities(self, board_state, move_history):
        """
        Obtiene las probabilidades de cada jugada según el modelo Othello-GPT.
        
        Args:
            board_state: Estado actual del tablero (numpy array)
            move_history: Historial de movimientos hasta ahora
        
        Returns:
            Diccionario con las coordenadas de las jugadas y sus probabilidades.
            Si el modelo no está cargado, retorna un diccionario vacío.
        """
        if self.model is None:
            return {}
        
        # Convertir la secuencia de movimientos a un tensor
        x = torch.tensor(move_history, dtype=torch.long).unsqueeze(0)  # shape [1, seq_len]
        
        # Obtener los logits (valores pre-softmax) del modelo
        with torch.no_grad():
            logits, _ = self.model(x)
        
        # Obtener las probabilidades para la última posición
        logits = logits[:, -1, :]  # shape [1, vocab_size]
        probs = F.softmax(logits, dim=-1)  # shape [1, vocab_size]
        
        # Convertir a numpy para facilitar el procesamiento
        probs = probs[0].numpy()  # shape [vocab_size]
        
        # Crear diccionario de jugadas y probabilidades
        # Asumimos que los índices de 0 a 63 corresponden a las posiciones del tablero
        move_probs = {}
        for pos in range(64):
            row, col = pos // 8, pos % 8
            coord = f"{chr(97 + row)}{col + 1}"  # Por ejemplo, "a1", "b2", etc.
            move_probs[coord] = probs[pos]
        
        return move_probs
        
    def update_probabilities(self, board_state, move_history):
        """
        Actualiza el gráfico de probabilidades basado en el estado actual del tablero.
        
        Args:
            board_state: Estado actual del tablero
            move_history: Historial de movimientos hasta ahora
        """
        # Obtener las probabilidades del modelo
        move_probs = self.get_move_probabilities(board_state, move_history)
        
        # Actualizar el gráfico si está disponible
        if self.probs_plot is not None:
            self.probs_plot.update(move_probs)
