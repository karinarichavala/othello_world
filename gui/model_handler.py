# model_handler.py
# Encapsula la lógica de cargar y consultar el modelo Othello-GPT

import torch
import torch.nn.functional as F

from mingpt.model import GPT, GPTConfig

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
            except (FileNotFoundError, RuntimeError) as e:  # Excepciones específicas
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
        # Configuración del modelo - 
        model_config = GPTConfig(
            vocab_size=61,  # Tamaño del vocabulario con el checkpoint
            block_size=59,  # Tamaño máximo de secuencia 
            n_layer=8,     # Número de capas transformer 
            n_head=8,      # Número de cabezas de atención
            n_embd=512     # Dimensión de embedding 
        )
        
        # Crear el modelo
        model = GPT(model_config)
        
        # Cargar los pesos del checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)  # Cargar directamente el checkpoint

        # Poner el modelo en modo de evaluación
        model.eval()

        return model
    
    def get_move_probabilities(self, move_history):  # Eliminamos board_state no utilizado
        """
        Obtiene las probabilidades de cada jugada según el modelo Othello-GPT.
        
        Args:
            move_history: Historial de movimientos hasta ahora
        
        Returns:
            Diccionario con las coordenadas de las jugadas y sus probabilidades.
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
        # Ajustamos el rango para que coincida con el tamaño del vocabulario
        move_probs = {}
        for pos in range(len(probs)):  # Usamos len(probs) para evitar índices fuera de rango
            row, col = pos // 8, pos % 8
            coord = f"{chr(97 + row)}{col + 1}"  # Por ejemplo, "a1", "b2", etc.
            move_probs[coord] = probs[pos]
        
        return move_probs
        
    def update_probabilities(self, move_history):
        """
        Actualiza el gráfico de probabilidades basado en el estado actual del tablero.
        
        Args:
            move_history: Historial de movimientos hasta ahora
        """
        # Obtener las probabilidades del modelo
        move_probs = self.get_move_probabilities(move_history)
        
        # Actualizar el gráfico si está disponible
        if self.probs_plot is not None:
            self.probs_plot.update(move_probs)
