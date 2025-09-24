# game_gui.py
# Interfaz gráfica para jugar Othello (refactorizada)

import tkinter as tk
import sys
import os

# Agregamos el directorio raíz al path para poder importar módulos del proyecto
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importamos las clases y funciones necesarias
from gui.logic.game_controller import GameController
from gui.components.board_renderer import BoardRenderer
from gui.config.settings import *


class GameGUI:
    def __init__(self, callback=None):
        """
        Inicializa la interfaz gráfica del juego Othello.
        
        Args:
            callback: Función que se llamará después de cada movimiento
                     para actualizar el gráfico de probabilidades.
        """
        # Crear el controlador del juego
        self.game_controller = GameController(callback)
        
        # Configurar la ventana
        self.window = tk.Tk()
        self.window.title(WINDOW_TITLE)
        self.window.configure(background=BOARD_COLOR)
        self.window.geometry(WINDOW_GEOMETRY)
        
        # Configurar el cierre de la ventana para limpiar timers
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Callback para actualizar el gráfico de probabilidades
        self.callback = callback
        
        # Variables para el temporizador de resaltado de la IA
        self.ai_highlight_timer = None
        
        # Crear el tablero
        self.create_board()
        
        # Etiqueta para mostrar mensajes
        self.message_label = tk.Label(self.window, text="Turno: Negro", 
                                     font=DEFAULT_FONT, bg=BOARD_COLOR)
        self.message_label.pack(pady=10)
        
        # Etiqueta para mostrar el último movimiento
        self.last_move_label = tk.Label(self.window, text="Último movimiento: Ninguno", 
                                       font=LABEL_FONT, bg=BOARD_COLOR, fg="white")
        self.last_move_label.pack(pady=5)
        
        # Crear el layout principal
        self.create_layout()
        
        # Actualiza la visualización del tablero
        self.update_board()
    
    def create_layout(self):
        """Crea el layout principal con tablero centrado y controles compactos en la parte inferior"""
        # Frame principal vertical
        main_frame = tk.Frame(self.window, bg=BOARD_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame superior para el tablero (centrado)
        board_frame = tk.Frame(main_frame, bg=BOARD_COLOR)
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
        controls_frame = tk.Frame(main_frame, bg=BOARD_COLOR)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Línea separadora sutil
        separator = tk.Frame(controls_frame, height=1, bg="darkgreen")
        separator.pack(fill=tk.X, pady=(0, 5))
        
        # Frame para los botones (más compacto)
        nav_frame = tk.Frame(controls_frame, bg=BOARD_COLOR)
        nav_frame.pack()
        
        # Botones de navegación más pequeños
        self.btn_first = tk.Button(nav_frame, text="⏮", command=self.go_to_first,
                                  font=LABEL_FONT, width=2, height=1)
        self.btn_first.pack(side=tk.LEFT, padx=3)
        
        self.btn_prev = tk.Button(nav_frame, text="⏪", command=self.go_to_previous,
                                 font=LABEL_FONT, width=2, height=1)
        self.btn_prev.pack(side=tk.LEFT, padx=3)
        
        self.btn_next = tk.Button(nav_frame, text="⏩", command=self.go_to_next,
                                 font=LABEL_FONT, width=2, height=1)
        self.btn_next.pack(side=tk.LEFT, padx=3)
        
        self.btn_last = tk.Button(nav_frame, text="⏭", command=self.go_to_current,
                                 font=LABEL_FONT, width=2, height=1)
        self.btn_last.pack(side=tk.LEFT, padx=3)
        
        # Información del movimiento actual (MÁS COMPACTA en una sola línea)
        info_frame = tk.Frame(controls_frame, bg=BOARD_COLOR)
        info_frame.pack(pady=(5, 0))
        
        # Información en línea horizontal
        status_frame = tk.Frame(info_frame, bg=BOARD_COLOR)
        status_frame.pack()
        
        self.move_info_label = tk.Label(status_frame, text="Vista: Juego actual", 
                                       font=SMALL_FONT, bg=BOARD_COLOR, fg="white")
        self.move_info_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.move_detail_label = tk.Label(status_frame, text="", 
                                         font=SMALL_FONT, bg="lightgreen", fg="black",
                                         relief=tk.RAISED, bd=1, width=15)
        self.move_detail_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Indicador de modo (más pequeño)
        self.mode_label = tk.Label(status_frame, text="MODO: JUEGO", 
                                  font=TINY_FONT, bg="green", fg="white", 
                                  relief=tk.RAISED, bd=1)
        self.mode_label.pack(side=tk.LEFT)
        
        # Actualizar estado de botones
        self.update_navigation_buttons()
    
    def update_navigation_buttons(self):
        """Actualiza el estado de los botones de navegación"""
        button_states = self.game_controller.get_button_states()
        nav_info = self.game_controller.get_navigation_info()
        
        # Actualizar estados de botones
        self.btn_first.config(state=tk.NORMAL if button_states['first'] else tk.DISABLED)
        self.btn_prev.config(state=tk.NORMAL if button_states['prev'] else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if button_states['next'] else tk.DISABLED)
        self.btn_last.config(state=tk.NORMAL if button_states['last'] else tk.DISABLED)
        
        # Actualizar información de navegación
        self.move_info_label.config(text=nav_info['info_text'])
        self.move_detail_label.config(text=nav_info['detail_text'])
        self.mode_label.config(text=nav_info['mode'], bg=nav_info['mode_color'])
    
    def go_to_first(self):
        """Va al primer movimiento"""
        nav_state = self.game_controller.go_to_first()
        if nav_state:
            self._update_view_from_navigation(nav_state)
    
    def go_to_previous(self):
        """Va al movimiento anterior"""
        nav_state = self.game_controller.go_to_previous()
        if nav_state:
            self._update_view_from_navigation(nav_state)
    
    def go_to_next(self):
        """Va al siguiente movimiento"""
        nav_state = self.game_controller.go_to_next()
        if nav_state:
            self._update_view_from_navigation(nav_state)
    
    def go_to_current(self):
        """Vuelve al estado actual del juego"""
        nav_state = self.game_controller.go_to_current()
        if nav_state:
            self._update_view_from_navigation(nav_state)
    
    def _update_view_from_navigation(self, nav_state):
        """Actualiza la vista basándose en el estado de navegación"""
        # Cancelar cualquier temporizador activo
        if self.ai_highlight_timer:
            try:
                self.window.after_cancel(self.ai_highlight_timer)
            except:
                pass  # Si el temporizador ya expiró, ignorar
            self.ai_highlight_timer = None
        
        self.update_board()
        self.update_navigation_buttons()
    
    def create_board(self):
        """Crea el canvas del tablero con etiquetas de filas y columnas"""
        self.canvas = tk.Canvas(self.window, width=BOARD_WIDTH + 2*BOARD_PADDING, 
                              height=BOARD_HEIGHT + 2*BOARD_PADDING, 
                              background=BOARD_COLOR)
        self.canvas.pack(padx=PADDING, pady=PADDING)

        # Crear el renderizador del tablero
        self.board_renderer = BoardRenderer(self.canvas)
        self.board_renderer.create_board_grid()

        # Vincular el evento de clic al canvas
        self.canvas.bind("<Button-1>", self.handle_click)
    
    def update_board(self):
        """Actualiza la visualización del tablero según el estado actual"""
        game_state = self.game_controller.get_game_state()
        
        # Obtener el estado del tablero y movimientos válidos
        if game_state.is_observer_mode:
            # En modo observador, usar el estado histórico del tablero
            board_state = game_state.get_game_state_at_index(game_state.current_view_index)
            valid_moves = []  # No mostrar movimientos válidos en modo observador
            highlighted_move = game_state.highlighted_move
        else:
            # En modo juego, usar el estado actual
            board_state = self.game_controller.get_board_state()
            valid_moves = self.game_controller.get_valid_moves()
            highlighted_move = None
        
        # Renderizar el tablero
        self.board_renderer.render_board(
            board_state=board_state,
            valid_moves=valid_moves,
            last_ai_move=game_state.last_ai_move,
            highlighted_move=highlighted_move,
            is_observer_mode=game_state.is_observer_mode
        )
        
        # Actualizar mensaje del turno (solo en modo juego)
        if not game_state.is_observer_mode:
            current_player = self.game_controller.get_current_player()
            self.message_label.config(text=f"Turno: {current_player}")
            
            # Verificar si el juego ha terminado cuando no hay movimientos válidos
            if not valid_moves:
                self._check_game_over()
    
    def handle_click(self, event):
        """Maneja el clic en el tablero para realizar una jugada."""
        # Obtener la posición del clic
        row, col = self.board_renderer.get_clicked_position(event)
        
        # Intentar realizar el movimiento a través del controlador
        result = self.game_controller.handle_player_click(row, col)
        
        if result['success']:
            # Actualizar la GUI con el resultado del movimiento
            self.last_move_label.config(text=result['last_move_info'])
            self.update_board()
            self.update_navigation_buttons()  # Actualizar botones de navegación
            
            # Verificar si el juego ha terminado
            if result.get('game_over', False):
                winner_info = f"Juego terminado. Ganador: {result['winner']} ({result['black_count']}-{result['white_count']})"
                self.message_label.config(text=winner_info)
                return
            
            # Si necesita movimiento de la IA, procesarlo
            if result.get('needs_ai_move', False):
                self._process_ai_move()
    
    def _process_ai_move(self):
        """Procesa el movimiento de la IA"""
        ai_result = self.game_controller.get_ai_move_async()
        
        if ai_result['has_move']:
            # Mostrar mensaje de "IA pensando..." 
            self.message_label.config(text="IA pensando...")
            self.window.update()  # Forzar actualización de la GUI
            
            # Pausa para simular que la IA está "pensando"
            self.window.after(AI_THINKING_DELAY, lambda: self._complete_ai_move(ai_result['move']))
        else:
            # No hay movimiento de IA, verificar si el juego terminó
            game_over_info = ai_result.get('game_over_info', {})
            if game_over_info.get('game_over', False):
                winner_info = f"Juego terminado. Ganador: {game_over_info['winner']} ({game_over_info['black_count']}-{game_over_info['white_count']})"
                self.message_label.config(text=winner_info)
    
    def _complete_ai_move(self, ai_move):
        """Completa el movimiento de la IA después de la pausa"""
        result = self.game_controller.complete_ai_move(ai_move)
        
        if result['success']:
            # Actualizar la GUI con el resultado del movimiento de la IA
            self.last_move_label.config(text=result['last_move_info'])
            self.update_board()
            self.update_navigation_buttons()  # Actualizar botones de navegación
            
            # Programar quitar el resaltado después de 3 segundos
            if self.ai_highlight_timer:
                try:
                    self.window.after_cancel(self.ai_highlight_timer)
                except:
                    pass
            self.ai_highlight_timer = self.window.after(AI_HIGHLIGHT_DURATION, self._clear_ai_highlight)
            
            # Verificar si el juego ha terminado
            if result.get('game_over', False):
                winner_info = f"Juego terminado. Ganador: {result['winner']} ({result['black_count']}-{result['white_count']})"
                self.message_label.config(text=winner_info)
        else:
            # Manejar error en el movimiento de la IA
            game_over_info = result.get('game_over_info', {})
            if game_over_info.get('game_over', False):
                winner_info = f"Juego terminado. Ganador: {game_over_info['winner']} ({game_over_info['black_count']}-{game_over_info['white_count']})"
                self.message_label.config(text=winner_info)
    
    def _clear_ai_highlight(self):
        """Quita el resaltado del último movimiento de la IA"""
        try:
            # Verificar que la ventana aún existe
            if not hasattr(self, 'window') or not self.window.winfo_exists():
                return
            
            game_state = self.game_controller.get_game_state()
            # Verificar que aún estemos en modo juego antes de limpiar
            if not game_state.is_observer_mode:
                self.game_controller.clear_ai_highlight()
                self.update_board()
        except tk.TclError:
            # La ventana fue destruida, ignorar silenciosamente
            pass
        except Exception as e:
            print(f"Error al limpiar highlight de IA: {e}")
        finally:
            # Limpiar la referencia del temporizador
            self.ai_highlight_timer = None
    
    def _check_game_over(self):
        """Verifica si el juego ha terminado y actualiza el mensaje correspondiente"""
        # Obtener el estado del tablero directamente
        board_state = self.game_controller.board_state
        
        # Cambiar al otro jugador para ver si tiene movimientos
        original_color = board_state.next_hand_color
        board_state.next_hand_color *= -1
        other_player_moves = board_state.get_valid_moves()
        
        if not other_player_moves:
            # Si ningún jugador puede mover, el juego ha terminado
            import numpy as np
            black_count = np.sum(board_state.state == 1)
            white_count = np.sum(board_state.state == -1)
            
            if black_count > white_count:
                winner = "Negro"
            elif white_count > black_count:
                winner = "Blanco"
            else:
                winner = "Empate"
                
            self.message_label.config(text=f"Juego terminado. Ganador: {winner} ({black_count}-{white_count})")
            return True
            
        # Si el otro jugador tiene movimientos, actualizar el mensaje
        current_player = "Negro" if board_state.next_hand_color == 1 else "Blanco"
        self.message_label.config(text=f"Turno: {current_player}")
        return False
    
    def _on_closing(self):
        """Maneja el cierre de la ventana cancelando todos los timers activos"""
        try:
            # Cancelar timer de highlight de IA si está activo
            if self.ai_highlight_timer:
                self.window.after_cancel(self.ai_highlight_timer)
                self.ai_highlight_timer = None
        except:
            pass  # Ignorar errores al cancelar timers
        
        # Destruir la ventana
        self.window.destroy()
    
    def run(self):
        """Ejecuta el bucle principal de la interfaz"""
        self.window.mainloop()