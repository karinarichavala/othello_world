# game_gui.py
# Interfaz gráfica para jugar Othello

import tkinter as tk
import sys
import os
import numpy as np

# Agregamos el directorio raíz al path para poder importar módulos del proyecto
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importamos las clases y funciones necesarias
from data.othello import OthelloBoardState, permit, permit_reverse

class GameGUI:
    def __init__(self, callback=None):
        """
        Inicializa la interfaz gráfica del juego Othello.
        
        Args:
            callback: Función que se llamará después de cada movimiento
                     para actualizar el gráfico de probabilidades.
        """
        self.window = tk.Tk()
        self.window.title("Othello Game")
        self.window.configure(background="forest green")
        self.window.geometry("650x650")  # Más compacto y cuadrado
        
        # Estado del tablero
        self.board_state = OthelloBoardState()
        
        # Tamaño de las casillas
        self.cell_size = 60
        
        # Callback para actualizar el gráfico de probabilidades
        self.callback = callback
        
        # Colores
        self.board_color = "forest green"
        self.line_color = "black"
        self.valid_move_color = "yellow"
        
        # Crear el tablero
        self.create_board()
        
        # Etiqueta para mostrar mensajes
        self.message_label = tk.Label(self.window, text="Turno: Negro", 
                                     font=("Arial", 14), bg=self.board_color)
        self.message_label.pack(pady=10)
        
        # Etiqueta para mostrar el último movimiento
        self.last_move_label = tk.Label(self.window, text="Último movimiento: Ninguno", 
                                       font=("Arial", 12), bg=self.board_color, fg="white")
        self.last_move_label.pack(pady=5)
        
        # Historial de jugadas
        self.move_history = []
        self.recent_moves = []  # Para mostrar los últimos 5 movimientos con detalles
        
        # Variables para el modo observador
        self.is_observer_mode = False
        self.current_view_index = -1  # -1 significa vista actual del juego
        self.game_states = []  # Guardar estados del tablero después de cada movimiento
        
        # Variable para trackear el último movimiento de la IA
        self.last_ai_move = None
        self.ai_highlight_timer = None
        self.highlighted_move = None  # Para resaltar movimientos en modo observador
        
        # Crear el layout principal
        self.create_layout()
        
        # Actualiza la visualización del tablero
        self.update_board()
    
    def create_layout(self):
        """Crea el layout principal con tablero centrado y controles compactos en la parte inferior"""
        # Frame principal vertical
        main_frame = tk.Frame(self.window, bg=self.board_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame superior para el tablero (centrado)
        board_frame = tk.Frame(main_frame, bg=self.board_color)
        board_frame.pack(side=tk.TOP, pady=(0, 10))
        
        # Mover el canvas del tablero al frame superior
        self.canvas.pack_forget()  # Remover del pack anterior
        self.canvas.master = board_frame  # Cambiar el padre
        self.canvas.pack()
        
        # Mover las etiquetas de mensaje al frame superior
        self.message_label.pack_forget()
        self.message_label.master = board_frame
        self.message_label.pack(pady=(5, 0))
        
        self.last_move_label.pack_forget()
        self.last_move_label.master = board_frame
        self.last_move_label.pack(pady=(2, 0))
        
        # Frame inferior COMPACTO para controles de navegación
        controls_frame = tk.Frame(main_frame, bg=self.board_color)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Línea separadora sutil
        separator = tk.Frame(controls_frame, height=1, bg="darkgreen")
        separator.pack(fill=tk.X, pady=(0, 5))
        
        # Frame para los botones (más compacto)
        nav_frame = tk.Frame(controls_frame, bg=self.board_color)
        nav_frame.pack()
        
        # Botones de navegación más pequeños
        self.btn_first = tk.Button(nav_frame, text="⏮", command=self.go_to_first,
                                  font=("Arial", 12), width=2, height=1)
        self.btn_first.pack(side=tk.LEFT, padx=3)
        
        self.btn_prev = tk.Button(nav_frame, text="⏪", command=self.go_to_previous,
                                 font=("Arial", 12), width=2, height=1)
        self.btn_prev.pack(side=tk.LEFT, padx=3)
        
        self.btn_next = tk.Button(nav_frame, text="⏩", command=self.go_to_next,
                                 font=("Arial", 12), width=2, height=1)
        self.btn_next.pack(side=tk.LEFT, padx=3)
        
        self.btn_last = tk.Button(nav_frame, text="⏭", command=self.go_to_current,
                                 font=("Arial", 12), width=2, height=1)
        self.btn_last.pack(side=tk.LEFT, padx=3)
        
        # Información del movimiento actual (MÁS COMPACTA en una sola línea)
        info_frame = tk.Frame(controls_frame, bg=self.board_color)
        info_frame.pack(pady=(5, 0))
        
        # Información en línea horizontal
        status_frame = tk.Frame(info_frame, bg=self.board_color)
        status_frame.pack()
        
        self.move_info_label = tk.Label(status_frame, text="Vista: Juego actual", 
                                       font=("Arial", 9), bg=self.board_color, fg="white")
        self.move_info_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.move_detail_label = tk.Label(status_frame, text="", 
                                         font=("Arial", 9), bg="lightgreen", fg="black",
                                         relief=tk.RAISED, bd=1, width=15)
        self.move_detail_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Indicador de modo (más pequeño)
        self.mode_label = tk.Label(status_frame, text="MODO: JUEGO", 
                                  font=("Arial", 8, "bold"), bg="green", fg="white", 
                                  relief=tk.RAISED, bd=1)
        self.mode_label.pack(side=tk.LEFT)
        
        # Actualizar estado de botones
        self.update_navigation_buttons()
    
    def update_navigation_buttons(self):
        """Actualiza el estado de los botones de navegación"""
        total_moves = len(self.recent_moves)
        
        if self.current_view_index == -1:
            # Estamos en vista actual del juego
            self.btn_first.config(state=tk.NORMAL if total_moves > 0 else tk.DISABLED)
            self.btn_prev.config(state=tk.NORMAL if total_moves > 0 else tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)  # No hay "siguiente" desde vista actual
            self.btn_last.config(state=tk.DISABLED)  # Ya estamos en actual
        else:
            # Estamos en modo observador
            self.btn_first.config(state=tk.NORMAL if self.current_view_index > 0 else tk.DISABLED)
            self.btn_prev.config(state=tk.NORMAL if self.current_view_index > 0 else tk.DISABLED)
            self.btn_next.config(state=tk.NORMAL)  # Siempre podemos ir hacia adelante
            self.btn_last.config(state=tk.NORMAL)  # Siempre podemos volver al actual
        
        # Actualizar información del movimiento actual
        if self.current_view_index == -1:
            self.move_info_label.config(text="Juego actual")
            self.move_detail_label.config(text="")
            self.mode_label.config(text="JUEGO", bg="green")
        else:
            move_num = self.current_view_index + 1
            total = len(self.recent_moves)
            self.move_info_label.config(text=f"Mov {move_num}/{total}")
            
            if self.current_view_index < len(self.recent_moves):
                move_info = self.recent_moves[self.current_view_index]
                detail_text = f"{move_info['player']}: {move_info['coord']}"
                self.move_detail_label.config(text=detail_text)
            
            self.mode_label.config(text="OBSERVADOR", bg="orange")
    
    def go_to_first(self):
        """Va al primer movimiento"""
        if len(self.recent_moves) > 0:
            self.current_view_index = 0
            self.is_observer_mode = True
            self.show_game_state_at_index(0)
    
    def go_to_previous(self):
        """Va al movimiento anterior (más reciente cuando vienes del juego, o retrocede en el historial)"""
        if self.current_view_index == -1:
            # Si estamos en vista actual, ir al movimiento MÁS RECIENTE (último jugado)
            if len(self.recent_moves) > 0:
                self.current_view_index = len(self.recent_moves) - 1
                self.is_observer_mode = True
                self.show_game_state_at_index(self.current_view_index)
        elif self.current_view_index > 0:
            # Retroceder en el historial hacia movimientos más antiguos
            self.current_view_index -= 1
            self.is_observer_mode = True
            self.show_game_state_at_index(self.current_view_index)
    
    def go_to_next(self):
        """Va al siguiente movimiento (avanza hacia el presente)"""
        if self.current_view_index < len(self.recent_moves) - 1:
            # Avanzar hacia movimientos más recientes
            self.current_view_index += 1
            self.is_observer_mode = True
            self.show_game_state_at_index(self.current_view_index)
        elif self.current_view_index == len(self.recent_moves) - 1:
            # Si estamos en el último movimiento, volver al modo juego actual
            self.go_to_current()
    
    def go_to_current(self):
        """Vuelve al estado actual del juego"""
        self.current_view_index = -1
        self.is_observer_mode = False
        self.highlighted_move = None
        
        # Cancelar cualquier temporizador activo
        if self.ai_highlight_timer:
            try:
                self.window.after_cancel(self.ai_highlight_timer)
            except:
                pass  # Si el temporizador ya expiró, ignorar
            self.ai_highlight_timer = None
            
        self.update_board()
        self.update_navigation_buttons()
    
    def show_game_state_at_index(self, index):
        """Muestra el estado del juego después del movimiento en el índice dado"""
        if 0 <= index < len(self.game_states):
            # Cancelar temporizadores de la IA al entrar en modo observador
            if self.ai_highlight_timer:
                try:
                    self.window.after_cancel(self.ai_highlight_timer)
                except:
                    pass
                self.ai_highlight_timer = None
            
            # Guardar el estado actual antes de cambiar
            current_state = self.board_state.state.copy()
            current_next_hand = self.board_state.next_hand_color
            
            # Mostrar el estado histórico
            self.board_state.state = self.game_states[index].copy()
            
            # Resaltar el movimiento correspondiente
            if index < len(self.recent_moves):
                self.highlighted_move = self.recent_moves[index]['move']
            
            self.update_board()
            self.update_navigation_buttons()
            
            # Restaurar el estado actual (sin mostrarlo)
            self.board_state.state = current_state
            self.board_state.next_hand_color = current_next_hand
    
    def save_game_state(self):
        """Guarda el estado actual del juego"""
        self.game_states.append(self.board_state.state.copy())
    
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
        self.update_navigation_buttons()
    
    def create_board(self):
        """Crea el canvas del tablero con etiquetas de filas y columnas"""
        board_width = 8 * self.cell_size
        board_height = 8 * self.cell_size

        self.canvas = tk.Canvas(self.window, width=board_width + 40, height=board_height + 40, 
                              background=self.board_color)
        self.canvas.pack(padx=20, pady=20)

        # Vincular el evento de clic al canvas
        self.canvas.bind("<Button-1>", self.handle_click)

        # Dibujar las líneas del tablero
        for i in range(9):
            # Líneas horizontales
            self.canvas.create_line(40, i * self.cell_size + 40, 
                                  board_width + 40, i * self.cell_size + 40, 
                                  fill=self.line_color)
            # Líneas verticales
            self.canvas.create_line(i * self.cell_size + 40, 40, 
                                  i * self.cell_size + 40, board_height + 40, 
                                  fill=self.line_color)

        # Etiquetas de filas y columnas
        rows = ["a", "b", "c", "d", "e", "f", "g", "h"]
        columns = ["1", "2", "3", "4", "5", "6", "7", "8"]

        # Etiquetas de filas (letras)
        for i, row in enumerate(rows):
            self.canvas.create_text(20, i * self.cell_size + self.cell_size // 2 + 40, 
                                    text=row, font=("Arial", 12), fill="white")

        # Etiquetas de columnas (números)
        for i, col in enumerate(columns):
            self.canvas.create_text(i * self.cell_size + self.cell_size // 2 + 40, 20, 
                                    text=col, font=("Arial", 12), fill="white")
    
    def update_board(self):
        """Actualiza la visualización del tablero según el estado actual"""
        # Limpiar fichas existentes
        self.canvas.delete("piece")
        
        # Dibujar las fichas
        for i in range(8):
            for j in range(8):
                cell_value = self.board_state.state[i, j]
                if cell_value != 0:  # Si hay una ficha
                    color = "black" if cell_value == 1 else "white"
                    
                    # Verificar qué tipo de resaltado aplicar
                    current_pos = i * 8 + j
                    outline_color = "black"
                    outline_width = 1
                    
                    if self.is_observer_mode and current_pos == self.highlighted_move:
                        # Resaltar movimiento en modo observador con amarillo
                        outline_color = "yellow"
                        outline_width = 5
                    elif not self.is_observer_mode and current_pos == self.last_ai_move:
                        # Resaltar último movimiento de la IA con celeste (solo en modo juego)
                        outline_color = "cyan"
                        outline_width = 4
                    
                    x = j * self.cell_size + self.cell_size // 2 + 40
                    y = i * self.cell_size + self.cell_size // 2 + 40
                    self.canvas.create_oval(
                        x - self.cell_size * 0.4, 
                        y - self.cell_size * 0.4,
                        x + self.cell_size * 0.4, 
                        y + self.cell_size * 0.4,
                        fill=color, outline=outline_color, width=outline_width, tags="piece"
                    )

        # Mostrar movimientos válidos
        valid_moves = self.board_state.get_valid_moves()
        for move in valid_moves:
            row, col = move // 8, move % 8
            x = col * self.cell_size + self.cell_size // 2 + 40
            y = row * self.cell_size + self.cell_size // 2 + 40
            self.canvas.create_oval(
                x - self.cell_size * 0.1, 
                y - self.cell_size * 0.1,
                x + self.cell_size * 0.1, 
                y + self.cell_size * 0.1,
                fill=self.valid_move_color, tags="piece"
            )
        
        # Actualizar mensaje del turno
        current_player = "Negro" if self.board_state.next_hand_color == 1 else "Blanco"
        self.message_label.config(text=f"Turno: {current_player}")
        
        # Verificar si el juego ha terminado
        if not valid_moves:
            self.check_game_over()
    
    def handle_click(self, event):
        """Maneja el clic en el tablero para realizar una jugada."""
        # No permitir movimientos en modo observador
        if self.is_observer_mode:
            return
        
        # No permitir nuevos clics si la IA está "pensando"
        if hasattr(self, '_ai_thinking') and self._ai_thinking:
            return
            
        # Calcular la fila y columna según la posición del clic
        col = (event.x - 40) // self.cell_size
        row = (event.y - 40) // self.cell_size

        # Verificar si la posición está dentro del tablero
        if 0 <= row < 8 and 0 <= col < 8:
            move = row * 8 + col

            # Verificar si el movimiento es válido
            if move in self.board_state.get_valid_moves():
                # Usar el método make_move que maneja la alternancia con la IA
                self.make_move(move)

    def make_move(self, move):
        """Realiza un movimiento en la posición dada y obtiene la respuesta del modelo"""
        # Verificar si es un movimiento válido
        valid_moves = self.board_state.get_valid_moves()
        if move not in valid_moves:
            print(f"Movimiento inválido: {move}")
            return
            
        # Actualizar el estado del tablero con la jugada del jugador (negro)
        self.board_state.update([move])
        self.record_move(move)
        
        # Guardar estado del juego y agregar al historial detallado
        self.save_game_state()
        self.add_move_to_history(move, "Jugador")
        
        # Actualizar etiqueta del último movimiento
        move_coord = permit_reverse(move)
        self.last_move_label.config(text=f"Último movimiento: Jugador - {move_coord.upper()}")
        
        self.update_board()
        
        # Si no hay modelo o el juego ha terminado, no hacer nada más
        if not self.callback:
            return
            
        # Obtener las jugadas válidas para el modelo
        self.board_state.next_hand_color = -1  # Cambiar a blanco para obtener sus movimientos válidos
        valid_moves = self.board_state.get_valid_moves()
        
        # Si no hay movimientos válidos, verificar si el juego ha terminado
        if not valid_moves:
            self.check_game_over()
            return
            
        # Obtener y realizar la jugada del modelo
        move_probs = self.callback.get_move_probabilities(self.move_history)
        best_move = self.callback.get_best_move(move_probs, valid_moves)
        
        if best_move is not None:
            # Mostrar mensaje de "IA pensando..." y bloquear nuevos clics
            self._ai_thinking = True
            self.message_label.config(text="IA pensando...")
            self.window.update()  # Forzar actualización de la GUI
            
            # Pausa para simular que la IA está "pensando"
            self.window.after(1000, lambda: self._complete_ai_move(best_move))
        else:
            # Si no hay movimiento válido, continuar
            self.check_game_over()
    
    def _complete_ai_move(self, best_move):
        """Completa el movimiento de la IA después de la pausa"""
        # Verificar que el movimiento sigue siendo válido
        current_valid_moves = self.board_state.get_valid_moves()
        if best_move not in current_valid_moves:
            print(f"Error: El movimiento de la IA {best_move} ya no es válido")
            print(f"Movimientos válidos actuales: {current_valid_moves}")
            # Obtener un nuevo movimiento válido
            if current_valid_moves:
                move_probs = self.callback.get_move_probabilities(self.move_history)
                best_move = self.callback.get_best_move(move_probs, current_valid_moves)
                if best_move is None:
                    print("No se pudo obtener un movimiento válido para la IA")
                    self.check_game_over()
                    return
            else:
                print("No hay movimientos válidos para la IA")
                self.check_game_over()
                return
        
        # Verificar que la posición está realmente libre
        row, col = best_move // 8, best_move % 8
        if self.board_state.state[row, col] != 0:
            print(f"Error: La posición {row}-{col} ya está ocupada")
            print(f"Estado actual del tablero en esa posición: {self.board_state.state[row, col]}")
            # Buscar un movimiento alternativo
            current_valid_moves = self.board_state.get_valid_moves()
            if current_valid_moves:
                best_move = current_valid_moves[0]  # Tomar el primer movimiento válido
                print(f"Usando movimiento alternativo: {best_move}")
            else:
                self.check_game_over()
                return
        
        # Realizar el movimiento
        try:
            self.board_state.update([best_move])
            self.record_move(best_move)
            
            # Marcar este movimiento como el último de la IA
            self.last_ai_move = best_move
            
            # Guardar estado del juego y agregar al historial detallado
            self.save_game_state()
            self.add_move_to_history(best_move, "IA")
            
            # Actualizar etiqueta del último movimiento
            move_coord = permit_reverse(best_move)
            self.last_move_label.config(text=f"Último movimiento: IA - {move_coord.upper()}")
            
            # Actualizar el tablero
            self.update_board()
            self.callback.update_probabilities(self.move_history)
            
            # Programar quitar el resaltado después de 3 segundos
            if self.ai_highlight_timer:
                try:
                    self.window.after_cancel(self.ai_highlight_timer)
                except:
                    pass  # Si el temporizador ya expiró, ignorar
            self.ai_highlight_timer = self.window.after(3000, self._clear_ai_highlight)
            
            # Verificar si el juego ha terminado después de la jugada del modelo
            if not self.board_state.get_valid_moves():
                self.check_game_over()
            
            # Permitir nuevos clics del jugador
            self._ai_thinking = False
                
        except AssertionError as e:
            print(f"Error al realizar el movimiento de la IA: {e}")
            print(f"Movimiento intentado: {best_move}")
            print(f"Estado del tablero: {self.board_state.state}")
            # Permitir nuevos clics del jugador incluso si hay error
            self._ai_thinking = False
            # Intentar continuar el juego
            self.check_game_over()
    
    def _clear_ai_highlight(self):
        """Quita el resaltado del último movimiento de la IA"""
        # Verificar que aún estemos en modo juego antes de limpiar
        if not self.is_observer_mode:
            self.last_ai_move = None
            self.update_board()
        # Limpiar la referencia del temporizador
        self.ai_highlight_timer = None
    
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
    
    def check_game_over(self):
        """Verifica si el juego ha terminado y actualiza el mensaje correspondiente"""
        # Cambiar al otro jugador para ver si tiene movimientos
        self.board_state.next_hand_color *= -1
        other_player_moves = self.board_state.get_valid_moves()
        
        if not other_player_moves:
            # Si ningún jugador puede mover, el juego ha terminado
            black_count = np.sum(self.board_state.state == 1)
            white_count = np.sum(self.board_state.state == -1)
            
            if black_count > white_count:
                winner = "Negro"
            elif white_count > black_count:
                winner = "Blanco"
            else:
                winner = "Empate"
                
            self.message_label.config(text=f"Juego terminado. Ganador: {winner} ({black_count}-{white_count})")
            return True
            
        # Si el otro jugador tiene movimientos, actualizar el mensaje
        current_player = "Negro" if self.board_state.next_hand_color == 1 else "Blanco"
        self.message_label.config(text=f"Turno: {current_player}")
        return False
    
    def run(self):
        """Ejecuta el bucle principal de la interfaz"""
        self.window.mainloop()
