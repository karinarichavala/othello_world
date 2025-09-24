# board_renderer.py
# Componente para el renderizado del tablero

import tkinter as tk
import os
import sys

# Agregamos el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from gui.config.settings import *


class BoardRenderer:
    """Maneja el renderizado del tablero de Othello"""
    
    def __init__(self, canvas):
        self.canvas = canvas
    
    def create_board_grid(self):
        """Crea las líneas del tablero con etiquetas de filas y columnas"""
        # Dibujar las líneas del tablero
        for i in range(9):
            # Líneas horizontales
            self.canvas.create_line(BOARD_PADDING, i * CELL_SIZE + BOARD_PADDING, 
                                  BOARD_WIDTH + BOARD_PADDING, i * CELL_SIZE + BOARD_PADDING, 
                                  fill=LINE_COLOR)
            # Líneas verticales
            self.canvas.create_line(i * CELL_SIZE + BOARD_PADDING, BOARD_PADDING, 
                                  i * CELL_SIZE + BOARD_PADDING, BOARD_HEIGHT + BOARD_PADDING, 
                                  fill=LINE_COLOR)

        # Etiquetas de filas (letras)
        for i, row in enumerate(BOARD_ROWS):
            self.canvas.create_text(20, i * CELL_SIZE + CELL_SIZE // 2 + BOARD_PADDING, 
                                    text=row, font=LABEL_FONT, fill="white")

        # Etiquetas de columnas (números)
        for i, col in enumerate(BOARD_COLUMNS):
            self.canvas.create_text(i * CELL_SIZE + CELL_SIZE // 2 + BOARD_PADDING, 20, 
                                    text=col, font=LABEL_FONT, fill="white")
    
    def render_board(self, board_state, valid_moves, last_ai_move=None, highlighted_move=None, is_observer_mode=False):
        """Renderiza el estado actual del tablero"""
        # Limpiar fichas existentes
        self.canvas.delete("piece")
        
        # Dibujar las fichas
        for i in range(8):
            for j in range(8):
                cell_value = board_state[i, j]
                if cell_value != 0:  # Si hay una ficha
                    color = "black" if cell_value == 1 else "white"
                    
                    # Verificar qué tipo de resaltado aplicar
                    current_pos = i * 8 + j
                    outline_color = "black"
                    outline_width = 1
                    
                    if is_observer_mode and current_pos == highlighted_move:
                        # Resaltar movimiento en modo observador con amarillo
                        outline_color = OBSERVER_HIGHLIGHT_COLOR
                        outline_width = 5
                    elif not is_observer_mode and current_pos == last_ai_move:
                        # Resaltar último movimiento de la IA con celeste (solo en modo juego)
                        outline_color = AI_HIGHLIGHT_COLOR
                        outline_width = 4
                    
                    x = j * CELL_SIZE + CELL_SIZE // 2 + BOARD_PADDING
                    y = i * CELL_SIZE + CELL_SIZE // 2 + BOARD_PADDING
                    self.canvas.create_oval(
                        x - CELL_SIZE * 0.4, 
                        y - CELL_SIZE * 0.4,
                        x + CELL_SIZE * 0.4, 
                        y + CELL_SIZE * 0.4,
                        fill=color, outline=outline_color, width=outline_width, tags="piece"
                    )

        # Mostrar movimientos válidos
        for move in valid_moves:
            row, col = move // 8, move % 8
            x = col * CELL_SIZE + CELL_SIZE // 2 + BOARD_PADDING
            y = row * CELL_SIZE + CELL_SIZE // 2 + BOARD_PADDING
            self.canvas.create_oval(
                x - CELL_SIZE * 0.1, 
                y - CELL_SIZE * 0.1,
                x + CELL_SIZE * 0.1, 
                y + CELL_SIZE * 0.1,
                fill=VALID_MOVE_COLOR, tags="piece"
            )
    
    def get_clicked_position(self, event):
        """Convierte las coordenadas del clic en posición del tablero"""
        col = (event.x - BOARD_PADDING) // CELL_SIZE
        row = (event.y - BOARD_PADDING) // CELL_SIZE
        return row, col