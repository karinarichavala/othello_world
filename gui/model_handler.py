# model_handler.py
# Encapsula la lógica de cargar y consultar el modelo Othello-GPT

import torch
import torch.nn.functional as F

from mingpt.model import GPT, GPTConfig
from data.othello import permit, permit_reverse

def board_to_model_index(board_pos):
    """
    Convierte un índice del tablero (0-63) a un índice del modelo (1-60).
    Las casillas centrales (D3, D4, E3, E4) no tienen índice en el modelo.
    """
    # Las casillas centrales no tienen índice en el modelo
    center_squares = {27, 28, 35, 36}  # D3, D4, E3, E4 en formato tablero
    if board_pos in center_squares:
        return None
        
    # Calcular el desplazamiento basado en cuántas casillas centrales hay antes de esta posición
    offset = 0
    if board_pos > 36:  # Después de E4
        offset = 4
    elif board_pos > 35:  # Después de E3
        offset = 3
    elif board_pos > 28:  # Después de D4
        offset = 2
    elif board_pos > 27:  # Después de D3
        offset = 1
        
    # El índice del modelo empieza en 1 (0 es para "pass")
    return board_pos - offset + 1

def model_to_board_index(model_pos):
    """
    Convierte un índice del modelo (1-60) a un índice del tablero (0-63).
    """
    if model_pos == 0:  # "pass" no tiene equivalente en el tablero
        return None
        
    # Ajustar por el offset de las casillas centrales
    board_pos = model_pos - 1  # -1 porque el modelo empieza en 1
    
    # Agregar offset para las casillas centrales
    if board_pos >= 33:  # Después de E4 en el modelo
        board_pos += 4
    elif board_pos >= 27:  # Después de E3 en el modelo
        board_pos += 3
    elif board_pos >= 26:  # Después de D4 en el modelo
        board_pos += 2
    elif board_pos >= 25:  # Después de D3 en el modelo
        board_pos += 1
        
    return board_pos

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
            
        try:
            # Convertir movimientos del formato tablero al formato del modelo
            model_moves = []
            for move in move_history:
                model_move = board_to_model_index(move)
                if model_move is not None:
                    model_moves.append(model_move)
            
            # Si tenemos más de 59 movimientos, usar solo los últimos 59
            # (el modelo está entrenado para recibir 59 movimientos y predecir el siguiente)
            if len(model_moves) > 59:
                print("Usando los últimos 59 movimientos del historial")
                model_moves = model_moves[-59:]
                    
            # Convertir la secuencia de movimientos a un tensor
            x = torch.tensor(model_moves, dtype=torch.long).unsqueeze(0)  # shape [1, seq_len]
            
            # Obtener los logits (valores pre-softmax) del modelo
            with torch.no_grad():
                logits, _ = self.model(x)
        except Exception as e:
            print(f"Error al procesar movimientos: {e}")
            print("Movimientos:", move_history)
            return {}
        # Obtener las probabilidades para la última posición
        logits = logits[:, -1, :]  # shape [1, vocab_size]
        probs = F.softmax(logits, dim=-1)  # shape [1, vocab_size]
        
        # Convertir a numpy para facilitar el procesamiento
        probs = probs[0].numpy()  # shape [vocab_size]
        
        # Crear diccionario de jugadas y probabilidades
        move_probs = {}
        
        # Convertir las probabilidades del formato del modelo al formato del tablero
        # Ignoramos la posición 0 que representa "pass"
        for model_pos in range(1, len(probs)):  # Empezamos en 1 para saltar el "pass"
            board_pos = model_to_board_index(model_pos)
            if board_pos is not None:
                row = board_pos // 8
                col = board_pos % 8
                coord = f"{chr(97 + row)}{col + 1}"  # Por ejemplo, "a1", "b2", etc.
                move_probs[coord] = float(probs[model_pos])  # Convertir a float para evitar problemas con numpy
        
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
    
    def get_best_move(self, move_probs, valid_moves):
        """
        Obtiene la jugada con mayor probabilidad entre los movimientos válidos.
        
        Args:
            move_probs: Diccionario con las coordenadas de las jugadas y sus probabilidades.
            valid_moves: Lista de movimientos válidos (índices 0-63).
        
        Returns:
            La jugada válida con mayor probabilidad como índice numérico (0-63).
        """
        # Convertir movimientos válidos a coordenadas
        valid_coords = [permit_reverse(move) for move in valid_moves]
        
        # Filtrar probabilidades para solo considerar movimientos válidos
        valid_probs = {coord: prob for coord, prob in move_probs.items() if coord in valid_coords}
        
        if not valid_probs:
            return None
            
        best_coord = max(valid_probs, key=valid_probs.get)
        return permit(best_coord)
