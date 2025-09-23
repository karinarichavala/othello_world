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
        
        # Variable para trackear el último movimiento de la IA
        self.last_ai_move = None
        self.ai_highlight_timer = None
        
        # Actualiza la visualización del tablero
        self.update_board()
    
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
                    
                    # Verificar si esta posición es el último movimiento de la IA
                    current_pos = i * 8 + j
                    if current_pos == self.last_ai_move:
                        # Resaltar el último movimiento de la IA con celeste
                        outline_color = "cyan"
                        outline_width = 4
                    else:
                        outline_color = "black"
                        outline_width = 1
                    
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
            # Mostrar mensaje de "IA pensando..."
            current_player = "Negro" if self.board_state.next_hand_color == 1 else "Blanco"
            self.message_label.config(text="IA pensando...")
            self.window.update()  # Forzar actualización de la GUI
            
            # Pausa para simular que la IA está "pensando"
            self.window.after(1000, lambda: self._complete_ai_move(best_move))
        else:
            # Si no hay movimiento válido, continuar
            self.check_game_over()
    
    def _complete_ai_move(self, best_move):
        """Completa el movimiento de la IA después de la pausa"""
        # Realizar el movimiento
        self.board_state.update([best_move])
        self.record_move(best_move)
        
        # Marcar este movimiento como el último de la IA
        self.last_ai_move = best_move
        
        # Actualizar el tablero
        self.update_board()
        self.callback.update_probabilities(self.move_history)
        
        # Programar quitar el resaltado después de 3 segundos
        if self.ai_highlight_timer:
            self.window.after_cancel(self.ai_highlight_timer)
        self.ai_highlight_timer = self.window.after(3000, self._clear_ai_highlight)
        
        # Verificar si el juego ha terminado después de la jugada del modelo
        if not self.board_state.get_valid_moves():
            self.check_game_over()
    
    def _clear_ai_highlight(self):
        """Quita el resaltado del último movimiento de la IA"""
        self.last_ai_move = None
        self.update_board()
    
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
