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
        
        # Historial de jugadas
        self.move_history = []
        
        # Actualiza la visualización del tablero
        self.update_board()
    
    def create_board(self):
        """Crea el canvas del tablero"""
        board_width = 8 * self.cell_size
        board_height = 8 * self.cell_size
        
        self.canvas = tk.Canvas(self.window, width=board_width, height=board_height, 
                              background=self.board_color)
        self.canvas.pack(padx=20, pady=20)
        
        # Dibujar las líneas del tablero
        for i in range(9):
            # Líneas horizontales
            self.canvas.create_line(0, i * self.cell_size, 
                                  board_width, i * self.cell_size, 
                                  fill=self.line_color)
            # Líneas verticales
            self.canvas.create_line(i * self.cell_size, 0, 
                                  i * self.cell_size, board_height, 
                                  fill=self.line_color)
        
        # Manejar clics del usuario
        self.canvas.bind("<Button-1>", self.handle_click)
    
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
                    x = j * self.cell_size + self.cell_size // 2
                    y = i * self.cell_size + self.cell_size // 2
                    self.canvas.create_oval(
                        x - self.cell_size * 0.4, 
                        y - self.cell_size * 0.4,
                        x + self.cell_size * 0.4, 
                        y + self.cell_size * 0.4,
                        fill=color, tags="piece"
                    )
        
        # Mostrar movimientos válidos
        valid_moves = self.board_state.get_valid_moves()
        for move in valid_moves:
            row, col = move // 8, move % 8
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
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
            # Cambiar de jugador para ver si hay movimientos
            self.board_state.next_hand_color *= -1
            if not self.board_state.get_valid_moves():
                # Si ningún jugador puede mover, el juego ha terminado
                black_count = np.sum(self.board_state.state == 1)
                white_count = np.sum(self.board_state.state == -1)
                if black_count > white_count:
                    winner = "Negro"
                elif white_count > black_count:
                    winner = "Blanco"
                else:
                    winner = "Empate"
                self.message_label.config(text=f"Juego terminado. Ganador: {winner}")
            else:
                self.message_label.config(text=f"No hay movimientos. Turno: {current_player}")
    
    def handle_click(self, event):
        """Maneja el clic del usuario en el tablero"""
        # Convertir las coordenadas del clic a posición del tablero
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        if 0 <= col < 8 and 0 <= row < 8:
            move = row * 8 + col
            valid_moves = self.board_state.get_valid_moves()
            
            if move in valid_moves:
                # Realizar el movimiento
                self.make_move(move)
    
    def make_move(self, move):
        """Realiza un movimiento en la posición dada y obtiene la respuesta del modelo"""
        # Actualizar el estado del tablero con la jugada del jugador (negro)
        self.board_state.update([move])
        
        # Registrar la jugada en el historial
        self.record_move(move)
        
        # Actualizar la visualización
        self.update_board()
        
        # Si hay un modelo, obtener y realizar su jugada (blanco)
        if self.callback:
            # Obtener las jugadas válidas para el modelo
            self.board_state.next_hand_color = -1  # Cambiar a blanco para obtener sus movimientos válidos
            valid_moves = self.board_state.get_valid_moves()
            
            if valid_moves:  # Si hay movimientos válidos disponibles
                # Obtener probabilidades y la mejor jugada
                move_probs = self.callback.get_move_probabilities(self.move_history)
                best_move_index = self.callback.get_best_move(move_probs, valid_moves)
                
                if best_move_index is not None:
                    # Realizar la jugada del modelo
                    self.board_state.update([best_move_index])
                    self.record_move(best_move_index)
                    
                    # Actualizar la visualización y las probabilidades
                    self.update_board()
                    self.callback.update_probabilities(self.move_history)
    
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
    
    def player_move(self, move):
        """
        Maneja la jugada del jugador y actualiza el tablero.
        
        Args:
            move: La jugada realizada por el jugador (índice de 0 a 63).
        """
        # Verificar si es un movimiento válido
        valid_moves = self.board_state.get_valid_moves()
        if move not in valid_moves:
            print(f"Movimiento inválido: {move}")
            return
            
        # Registrar la jugada del jugador
        self.record_move(move)
        
        # Actualizar el tablero con la jugada del jugador
        self.board_state.update([move])
        self.update_board()
        
        # Llamar al modelo para obtener la siguiente jugada
        if self.callback:
            # Obtener las jugadas válidas para el modelo
            self.board_state.next_hand_color = -1  # Cambiar a blanco para obtener sus movimientos válidos
            valid_moves = self.board_state.get_valid_moves()
            
            if valid_moves:  # Si hay movimientos válidos disponibles
                # Obtener probabilidades y la mejor jugada
                move_probs = self.callback.get_move_probabilities(self.move_history)
                best_move = self.callback.get_best_move(move_probs, valid_moves)
                
                if best_move is not None:
                    # Registrar la jugada del modelo
                    self.record_move(best_move)
                    
                    # Actualizar el tablero con la jugada del modelo
                    self.board_state.update([best_move])
                    self.update_board()
                    
                    # Actualizar las probabilidades
                    if hasattr(self.callback, 'update_probabilities'):
                        self.callback.update_probabilities(self.move_history)
    
    def run(self):
        """Ejecuta el bucle principal de la interfaz"""
        self.window.mainloop()
