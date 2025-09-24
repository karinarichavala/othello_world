# game_controller.py
# Controlador principal que coordina toda la lógica del juego

import os
import sys

# Agregamos el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data.othello import OthelloBoardState
from gui.logic.game_state import GameState
from gui.logic.navigation import NavigationController
from gui.logic.move_handler import MoveHandler


class GameController:
    """Controlador principal que coordina toda la lógica del juego"""
    
    def __init__(self, callback=None):
        # Estado del tablero
        self.board_state = OthelloBoardState()
        
        # Estado del juego (historial, navegación, etc.)
        self.game_state = GameState()
        
        # Controladores
        self.navigation = NavigationController(self.game_state)
        self.move_handler = MoveHandler(self.board_state, self.game_state, callback)
        
        # Callback para actualizar el gráfico de probabilidades
        self.callback = callback
    
    def handle_player_click(self, row, col):
        """Maneja el clic del jugador en el tablero"""
        # No permitir movimientos en modo observador
        if self.game_state.is_observer_mode:
            return {'success': False, 'reason': 'observer_mode'}
        
        # No permitir nuevos clics si la IA está "pensando"
        if self.move_handler.is_ai_thinking():
            return {'success': False, 'reason': 'ai_thinking'}
        
        # Verificar si la posición está dentro del tablero
        if not (0 <= row < 8 and 0 <= col < 8):
            return {'success': False, 'reason': 'invalid_position'}
        
        move = row * 8 + col
        
        # Verificar si el movimiento es válido
        if not self.move_handler.is_valid_move(move):
            return {'success': False, 'reason': 'invalid_move'}
        
        # Realizar el movimiento del jugador
        if self.move_handler.make_player_move(move):
            result = {
                'success': True,
                'move': move,
                'player': 'Jugador',
                'last_move_info': self.move_handler.get_last_move_info(move, "Jugador"),
                'needs_ai_move': self.callback is not None
            }
            
            # Verificar si el juego ha terminado después del movimiento del jugador
            game_over_info = self.move_handler.check_game_over()
            result.update(game_over_info)
            
            return result
        
        return {'success': False, 'reason': 'move_failed'}
    
    def get_ai_move_async(self):
        """Inicia el proceso de obtener el movimiento de la IA"""
        ai_move = self.move_handler.get_ai_move()
        if ai_move is not None:
            self.move_handler.set_ai_thinking(True)
            return {'has_move': True, 'move': ai_move}
        else:
            # Verificar si el juego ha terminado
            game_over_info = self.move_handler.check_game_over()
            return {'has_move': False, 'game_over_info': game_over_info}
    
    def complete_ai_move(self, move):
        """Completa el movimiento de la IA"""
        if self.move_handler.make_ai_move(move):
            result = {
                'success': True,
                'move': move,
                'player': 'IA',
                'last_move_info': self.move_handler.get_last_move_info(move, "IA")
            }
            
            # Verificar si el juego ha terminado después de la jugada del modelo
            game_over_info = self.move_handler.check_game_over()
            result.update(game_over_info)
            
            # Permitir nuevos clics del jugador
            self.move_handler.set_ai_thinking(False)
            
            return result
        else:
            # Permitir nuevos clics del jugador incluso si hay error
            self.move_handler.set_ai_thinking(False)
            game_over_info = self.move_handler.check_game_over()
            return {'success': False, 'game_over_info': game_over_info}
    
    def clear_ai_highlight(self):
        """Limpia el resaltado del último movimiento de la IA"""
        self.game_state.clear_ai_highlight()
    
    def get_current_player(self):
        """Retorna el jugador actual"""
        return self.move_handler.get_current_player()
    
    def get_valid_moves(self):
        """Retorna los movimientos válidos actuales"""
        return self.board_state.get_valid_moves()
    
    def get_board_state(self):
        """Retorna el estado actual del tablero"""
        return self.board_state.state
    
    def get_game_state(self):
        """Retorna el estado del juego para la GUI"""
        return self.game_state
    
    # Métodos de navegación
    def go_to_first(self):
        """Va al primer movimiento"""
        return self.navigation.go_to_first()
    
    def go_to_previous(self):
        """Va al movimiento anterior"""
        return self.navigation.go_to_previous()
    
    def go_to_next(self):
        """Va al siguiente movimiento"""
        return self.navigation.go_to_next()
    
    def go_to_current(self):
        """Vuelve al estado actual del juego"""
        return self.navigation.go_to_current()
    
    def get_button_states(self):
        """Retorna el estado de los botones de navegación"""
        return self.navigation.get_button_states()
    
    def get_navigation_info(self):
        """Retorna información de navegación para la GUI"""
        return self.navigation.get_navigation_info()
    
    def reset_game(self):
        """Reinicia el juego"""
        self.board_state = OthelloBoardState()
        self.game_state.reset()
        self.move_handler = MoveHandler(self.board_state, self.game_state, self.callback)