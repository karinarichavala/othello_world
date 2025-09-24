# game_state.py
# Manejo del estado del juego y historial de movimientos

import os
import sys

# Agregamos el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data.othello import permit_reverse


class GameState:
    """Maneja el estado del juego, historial de movimientos y navegación"""
    
    def __init__(self):
        # Historial de jugadas
        self.move_history = []
        self.recent_moves = []  # Para mostrar los últimos movimientos con detalles
        
        # Estados del tablero guardados
        self.game_states = []
        
        # Variables para el modo observador
        self.is_observer_mode = False
        self.current_view_index = -1  # -1 significa vista actual del juego
        
        # Variable para trackear el último movimiento de la IA
        self.last_ai_move = None
        self.highlighted_move = None  # Para resaltar movimientos en modo observador
    
    def save_game_state(self, board_state):
        """Guarda el estado actual del juego"""
        self.game_states.append(board_state.copy())
    
    def add_move_to_history(self, move, player):
        """Agrega un movimiento al historial detallado"""
        coord = permit_reverse(move).upper()
        turn_number = len(self.recent_moves) + 1
        
        move_info = {
            'player': player,
            'coord': coord,
            'turn': turn_number,
            'move': move
        }
        
        self.recent_moves.append(move_info)
        # Asegurar que estamos en vista actual cuando se agrega un nuevo movimiento
        self.current_view_index = -1
    
    def record_move(self, move):
        """
        Registra una jugada en el historial.
        
        Args:
            move: La jugada realizada (índice de 0 a 63).
        """
        # Validar que el movimiento esté en el rango correcto
        if not (0 <= move < 64):
            print(f"Error: Movimiento fuera de rango: {move}")
            return
            
        self.move_history.append(move)
    
    def get_game_state_at_index(self, index):
        """Obtiene el estado del juego en un índice específico"""
        if 0 <= index < len(self.game_states):
            return self.game_states[index]
        return None
    
    def get_move_info_at_index(self, index):
        """Obtiene la información del movimiento en un índice específico"""
        if 0 <= index < len(self.recent_moves):
            return self.recent_moves[index]
        return None
    
    def get_total_moves(self):
        """Retorna el número total de movimientos registrados"""
        return len(self.recent_moves)
    
    def clear_ai_highlight(self):
        """Limpia el resaltado del último movimiento de la IA"""
        if not self.is_observer_mode:
            self.last_ai_move = None
    
    def set_ai_move(self, move):
        """Establece el último movimiento de la IA"""
        self.last_ai_move = move
    
    def reset(self):
        """Reinicia el estado del juego"""
        self.move_history = []
        self.recent_moves = []
        self.game_states = []
        self.is_observer_mode = False
        self.current_view_index = -1
        self.last_ai_move = None
        self.highlighted_move = None