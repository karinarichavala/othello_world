# move_handler.py
# Lógica para el manejo de movimientos del jugador y la IA

import numpy as np

from data.othello import permit_reverse


class MoveHandler:
    """Maneja la lógica de movimientos del jugador y la IA"""
    
    def __init__(self, board_state, game_state, callback=None):
        self.board_state = board_state
        self.game_state = game_state
        self.callback = callback
        self._ai_thinking = False
    
    def is_valid_move(self, move):
        """Verifica si un movimiento es válido"""
        return move in self.board_state.get_valid_moves()
    
    def make_player_move(self, move):
        """Realiza un movimiento del jugador"""
        if not self.is_valid_move(move):
            print(f"Movimiento inválido: {move}")
            return False
        
        # Actualizar el estado del tablero con la jugada del jugador
        self.board_state.update([move])
        self.game_state.record_move(move)
        
        # Guardar estado del juego y agregar al historial detallado
        self.game_state.save_game_state(self.board_state.state)
        self.game_state.add_move_to_history(move, "Jugador")
        
        # Actualizar probabilidades después del movimiento del jugador
        if self.callback:
            self.callback.update_probabilities(self.game_state.move_history, self.board_state.state)
        
        return True
    
    def get_ai_move(self):
        """Obtiene el movimiento de la IA"""
        if not self.callback:
            return None
        
        # Guardar el color original
        original_color = self.board_state.next_hand_color
        
        # Obtener las jugadas válidas para el modelo
        self.board_state.next_hand_color = -1  # Cambiar a blanco para obtener sus movimientos válidos
        valid_moves = self.board_state.get_valid_moves()
        
        if not valid_moves:
            # Restaurar el color original antes de retornar
            self.board_state.next_hand_color = original_color
            return None
        
        # Obtener y realizars la jugada del modelo
        move_probs = self.callback.get_move_probabilities(self.game_state.move_history)
        best_move = self.callback.get_best_move(move_probs, valid_moves)
        
        # Guardar las probabilidades para reutilizar en make_ai_move si es necesario
        self._last_move_probs = move_probs
        
        return best_move
    
    def make_ai_move(self, move):
        """Realiza un movimiento de la IA con validaciones"""
        # Verificar que el movimiento sigue siendo válido
        current_valid_moves = self.board_state.get_valid_moves()
        if move not in current_valid_moves:
            print(f"Error: El movimiento de la IA {move} ya no es válido")
            print(f"Movimientos válidos actuales: {current_valid_moves}")
            # Obtener un nuevo movimiento válido
            if current_valid_moves:
                # Reutilizar las probabilidades ya calculadas si están disponibles
                if hasattr(self, '_last_move_probs') and self._last_move_probs is not None:
                    move_probs = self._last_move_probs
                else:
                    move_probs = self.callback.get_move_probabilities(self.game_state.move_history)
                move = self.callback.get_best_move(move_probs, current_valid_moves)
                if move is None:
                    print("No se pudo obtener un movimiento válido para la IA")
                    return False
            else:
                print("No hay movimientos válidos para la IA")
                return False
        
        # Verificar que la posición está realmente libre
        row, col = move // 8, move % 8
        if self.board_state.state[row, col] != 0:
            print(f"Error: La posición {row}-{col} ya está ocupada")
            print(f"Estado actual del tablero en esa posición: {self.board_state.state[row, col]}")
            # Buscar un movimiento alternativo
            current_valid_moves = self.board_state.get_valid_moves()
            if current_valid_moves:
                move = current_valid_moves[0]  # Tomar el primer movimiento válido
                print(f"Usando movimiento alternativo: {move}")
            else:
                return False
        
        # Realizar el movimiento
        try:
            self.board_state.update([move])
            self.game_state.record_move(move)
            
            # Marcar este movimiento como el último de la IA
            self.game_state.set_ai_move(move)
            
            # Guardar estado del juego y agregar al historial detallado
            self.game_state.save_game_state(self.board_state.state)
            self.game_state.add_move_to_history(move, "IA")
            
            # Actualizar probabilidades si hay callback
            if self.callback:
                self.callback.update_probabilities(self.game_state.move_history, self.board_state.state)
            
            # Limpiar las probabilidades guardadas
            self._last_move_probs = None
            
            return True
                
        except AssertionError as e:
            print(f"Error al realizar el movimiento de la IA: {e}")
            print(f"Movimiento intentado: {move}")
            print(f"Estado del tablero: {self.board_state.state}")
            return False
    
    def check_game_over(self):
        """Verifica si el juego ha terminado y retorna información del ganador"""
        # Verificar si el jugador actual tiene movimientos
        current_player_moves = self.board_state.get_valid_moves()
        
        # Cambiar al otro jugador para ver si tiene movimientos
        original_color = self.board_state.next_hand_color
        self.board_state.next_hand_color *= -1
        other_player_moves = self.board_state.get_valid_moves()
        
        # Restaurar el color original
        self.board_state.next_hand_color = original_color
        
        # Si ningún jugador puede mover, el juego ha terminado
        if not current_player_moves and not other_player_moves:
            black_count = np.sum(self.board_state.state == 1)
            white_count = np.sum(self.board_state.state == -1)
            
            if black_count > white_count:
                winner = "Negro"
            elif white_count > black_count:
                winner = "Blanco"
            else:
                winner = "Empate"
            
            return {
                'game_over': True,
                'winner': winner,
                'black_count': black_count,
                'white_count': white_count
            }
        
        # Si el jugador actual no puede moverse pero el otro sí, cambiar turno
        if not current_player_moves and other_player_moves:
            self.board_state.next_hand_color *= -1
            current_player = "Negro" if self.board_state.next_hand_color == 1 else "Blanco"
            print(f"El jugador pasa el turno. Ahora juega: {current_player}")
        
        return {'game_over': False}
    
    def get_current_player(self):
        """Retorna el jugador actual"""
        return "Negro" if self.board_state.next_hand_color == 1 else "Blanco"
    
    def get_last_move_info(self, move, player):
        """Retorna información formateada del último movimiento"""
        move_coord = permit_reverse(move)
        return f"Último movimiento: {player} - {move_coord.upper()}"
    
    def set_ai_thinking(self, thinking):
        """Establece el estado de 'IA pensando'"""
        self._ai_thinking = thinking
    
    def is_ai_thinking(self):
        """Retorna si la IA está pensando"""
        return self._ai_thinking