# navigation.py
# Lógica de navegación entre movimientos y modo observador


class NavigationController:
    """Controla la navegación entre movimientos y el modo observador"""
    
    def __init__(self, game_state):
        self.game_state = game_state
    
    def go_to_first(self):
        """Va al primer movimiento"""
        if self.game_state.get_total_moves() > 0:
            self.game_state.current_view_index = 0
            self.game_state.is_observer_mode = True
            return self._get_navigation_state()
        return None
    
    def go_to_previous(self):
        """Va al movimiento anterior"""
        if self.game_state.current_view_index == -1:
            # Si estamos en vista actual, ir al movimiento MÁS RECIENTE
            if self.game_state.get_total_moves() > 0:
                self.game_state.current_view_index = self.game_state.get_total_moves() - 1
                self.game_state.is_observer_mode = True
                return self._get_navigation_state()
        elif self.game_state.current_view_index > 0:
            # Retroceder en el historial hacia movimientos más antiguos
            self.game_state.current_view_index -= 1
            self.game_state.is_observer_mode = True
            return self._get_navigation_state()
        return None
    
    def go_to_next(self):
        """Va al siguiente movimiento"""
        if self.game_state.current_view_index < self.game_state.get_total_moves() - 1:
            # Avanzar hacia movimientos más recientes
            self.game_state.current_view_index += 1
            self.game_state.is_observer_mode = True
            return self._get_navigation_state()
        elif self.game_state.current_view_index == self.game_state.get_total_moves() - 1:
            # Si estamos en el último movimiento, volver al modo juego actual
            return self.go_to_current()
        return None
    
    def go_to_current(self):
        """Vuelve al estado actual del juego"""
        self.game_state.current_view_index = -1
        self.game_state.is_observer_mode = False
        self.game_state.highlighted_move = None
        return self._get_navigation_state()
    
    def _get_navigation_state(self):
        """Retorna el estado actual de navegación para actualizar la GUI"""
        if self.game_state.current_view_index == -1:
            return {
                'type': 'current_game',
                'board_state': None,
                'highlighted_move': None,
                'is_observer_mode': False
            }
        else:
            # Obtener el estado del tablero histórico
            board_state = self.game_state.get_game_state_at_index(self.game_state.current_view_index)
            move_info = self.game_state.get_move_info_at_index(self.game_state.current_view_index)
            
            # Resaltar el movimiento correspondiente
            highlighted_move = move_info['move'] if move_info else None
            self.game_state.highlighted_move = highlighted_move
            
            return {
                'type': 'historical_state',
                'board_state': board_state,
                'highlighted_move': highlighted_move,
                'is_observer_mode': True,
                'move_info': move_info
            }
    
    def get_button_states(self):
        """Retorna el estado que deben tener los botones de navegación"""
        total_moves = self.game_state.get_total_moves()
        
        if self.game_state.current_view_index == -1:
            # Estamos en vista actual del juego
            return {
                'first': total_moves > 0,
                'prev': total_moves > 0,
                'next': False,
                'last': False
            }
        else:
            # Estamos en modo observador
            return {
                'first': self.game_state.current_view_index > 0,
                'prev': self.game_state.current_view_index > 0,
                'next': True,
                'last': True
            }
    
    def get_navigation_info(self):
        """Retorna información para mostrar en la GUI"""
        if self.game_state.current_view_index == -1:
            return {
                'info_text': "Juego actual",
                'detail_text': "",
                'mode': "JUEGO",
                'mode_color': "green"
            }
        else:
            move_num = self.game_state.current_view_index + 1
            total = self.game_state.get_total_moves()
            move_info = self.game_state.get_move_info_at_index(self.game_state.current_view_index)
            
            detail_text = ""
            if move_info:
                detail_text = f"{move_info['player']}: {move_info['coord']}"
            
            return {
                'info_text': f"Mov {move_num}/{total}",
                'detail_text': detail_text,
                'mode': "OBSERVADOR",
                'mode_color': "orange"
            }