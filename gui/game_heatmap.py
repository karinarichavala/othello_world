# game_heatmap.py
# Heatmap de probabilidades integrado con las fichas del juego

import tkinter as tk
import numpy as np
from gui.config.settings import *

class GameHeatmap:
    def __init__(self):
        """Inicializa la ventana del heatmap integrado con fichas"""
        # Crear ventana
        self.window = tk.Toplevel()
        self.window.title("Heatmap Integrado - Othello")
        self.window.configure(background="white")
        self.window.geometry("650x700")  # Más ancho y alto para que no se corte
        
        # Variables
        self.cell_size = 60
        self.board_size = 8 * self.cell_size
        self.current_probabilities = np.zeros(64)
        self.current_board_state = np.zeros(64, dtype=int)  # Inicializar como array plano de enteros
        self._grid_drawn = False  # Flag para evitar redibujar el grid constantemente
        
        # Crear la interfaz
        self.create_interface()
        self.draw_empty_board()
        
        # Configurar tooltip
        self.tooltip_label = None
        
    def create_interface(self):
        """Crea la interfaz de la ventana"""
        # Título
        title_label = tk.Label(self.window, 
                              text="Probabilidades y Estado del Juego", 
                              font=("Arial", 16, "bold"),
                              bg="white", 
                              fg="black")
        title_label.pack(pady=10)
        
        # Etiqueta informativa
        self.info_label = tk.Label(self.window,
                                  text="Pase el mouse sobre las casillas para ver probabilidades",
                                  font=LABEL_FONT,
                                  bg="white",
                                  fg="black")
        self.info_label.pack(pady=5)
        
        # Marco principal
        main_frame = tk.Frame(self.window, bg="white")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Canvas del tablero con padding para las etiquetas
        canvas_padding = 30
        canvas_width = self.board_size + 2 * canvas_padding
        canvas_height = self.board_size + 2 * canvas_padding
        
        self.canvas = tk.Canvas(main_frame,
                               width=canvas_width,
                               height=canvas_height,
                               bg="white",
                               highlightthickness=0)
        self.canvas.pack(expand=True)
        
        # Guardar el padding para usarlo en el dibujo
        self.canvas_padding = canvas_padding
        
        # Asegurar que el canvas mantenga el fondo blanco
        self.canvas.configure(bg="white")
        
        # Bind del mouse para mostrar probabilidades
        self.canvas.bind("<Motion>", self.on_mouse_motion)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        
        # Etiqueta para mostrar información de la casilla
        self.hover_info = tk.Label(self.window,
                                  text="",
                                  font=DEFAULT_FONT,
                                  bg="white",
                                  fg="blue")
        self.hover_info.pack(pady=5)
        
    def update(self, move_probabilities, board_state=None):
        """
        Actualiza el heatmap con nuevas probabilidades y estado del tablero.
        
        Args:
            move_probabilities: Diccionario con coordenadas como claves (ej: "a1") y probabilidades como valores
            board_state: Array de 64 elementos con el estado del tablero (0=vacío, 1=negro, 2=blanco)
        """
        # Convertir diccionario a array de 64 elementos
        probs_array = np.zeros(64)
        
        for coord, prob in move_probabilities.items():
            # Convertir coordenada (ej: "a1") a índice del tablero (0-63)
            row = ord(coord[0]) - ord('a')  # 'a' -> 0, 'b' -> 1, etc.
            col = int(coord[1]) - 1         # '1' -> 0, '2' -> 1, etc.
            board_index = row * 8 + col
            probs_array[board_index] = prob
        
        self.current_probabilities = probs_array
        
        # Actualizar estado del tablero si se proporciona
        if board_state is not None:
            # Convertir a array plano de enteros
            if isinstance(board_state, (list, tuple)):
                self.current_board_state = np.array(board_state, dtype=int).flatten()
            else:
                self.current_board_state = np.array(board_state, dtype=int).flatten()
            
            # Asegurar que tenga 64 elementos
            if len(self.current_board_state) < 64:
                self.current_board_state = np.pad(self.current_board_state, (0, 64 - len(self.current_board_state)), mode='constant', constant_values=0)
            elif len(self.current_board_state) > 64:
                self.current_board_state = self.current_board_state[:64]
        
        self.draw_heatmap_with_pieces(probs_array, self.current_board_state)
    
    def draw_heatmap_with_pieces(self, probabilities, board_state):
        """
        Dibuja el heatmap en el tablero con las fichas superpuestas.
        
        Args:
            probabilities: Array de numpy con 64 probabilidades
            board_state: Array con el estado del tablero (0=vacío, 1=negro, 2=blanco)
        """
        # SOLUCIÓN EXTREMA: No limpiar todo, solo limpiar elementos específicos
        self.canvas.delete("cell")
        self.canvas.delete("text")
        self.canvas.delete("piece")  # Nueva tag para las fichas
        
        # Si es la primera vez o hay problemas, recrear todo
        if not hasattr(self, '_grid_drawn') or not self._grid_drawn:
            self.canvas.delete("all")
            self.canvas.configure(bg="white")
            # Crear un fondo blanco masivo que cubra todo
            self.canvas.create_rectangle(0, 0, 1000, 1000,
                                       fill="white", outline="", width=0, tags="background")
            self._draw_grid_and_labels()
            self._grid_drawn = True
        
        # Normalizar probabilidades para el color (0-1)
        max_prob = np.max(probabilities)
        if max_prob > 0:
            normalized_probs = probabilities / max_prob
        else:
            normalized_probs = probabilities
        
        # Convertir board_state a array plano de enteros para evitar problemas
        if isinstance(board_state, (list, tuple)):
            board_flat = np.array(board_state, dtype=int).flatten()
        else:
            board_flat = np.array(board_state, dtype=int).flatten()
        
        # Asegurar que tenga 64 elementos
        if len(board_flat) < 64:
            board_flat = np.pad(board_flat, (0, 64 - len(board_flat)), mode='constant', constant_values=0)
        elif len(board_flat) > 64:
            board_flat = board_flat[:64]
        
        # Dibujar cada casilla con intensidad según probabilidad y fichas
        for i in range(8):
            for j in range(8):
                board_index = i * 8 + j
                prob = probabilities[board_index]
                piece = board_flat[board_index]
                normalized_prob = normalized_probs[board_index] if max_prob > 0 else 0
                
                # Coordenadas de la casilla con padding
                x1 = j * self.cell_size + self.canvas_padding
                y1 = i * self.cell_size + self.canvas_padding
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Color basado en la intensidad de la probabilidad
                if prob > 0:
                    # Gradiente de blanco a rojo oscuro
                    # normalized_prob = 0 → blanco (255, 255, 255)
                    # normalized_prob = 1 → rojo oscuro (139, 0, 0)
                    
                    # Calcular componentes RGB
                    red = int(255 - (255 - 139) * normalized_prob)      # De 255 a 139
                    green = int(255 * (1 - normalized_prob))             # De 255 a 0  
                    blue = int(255 * (1 - normalized_prob))              # De 255 a 0
                    
                    color = f"#{red:02x}{green:02x}{blue:02x}"
                else:
                    # Si no hay probabilidad, usar un color muy claro de fondo
                    color = "#f8f8f8"  # Gris muy claro para casillas sin probabilidad
                
                # Dibujar casilla con color
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                           fill=color,
                                           outline=LINE_COLOR,
                                           tags="cell")
                
                # Dibujar ficha si hay una (ANTES del texto para que quede de fondo)
                if piece != 0:
                    self._draw_piece(x1, y1, self.cell_size, piece)
                
                # Mostrar valor de probabilidad si es significativo (DESPUÉS de la ficha para que quede encima)
                if prob >= 0.01:  # Solo mostrar si es >= 1%
                    prob_text = f"{prob:.2%}"
                    
                    # Color del texto según el tipo de ficha para mejor legibilidad
                    if piece == 1:  # Ficha negra - texto blanco para contraste
                        text_color = "white"
                    elif piece == -1:  # Ficha blanca - texto negro para contraste
                        text_color = "black"
                    else:  # Sin ficha - color según intensidad de probabilidad
                        text_color = "white" if normalized_prob > 0.5 else "black"
                    
                    self.canvas.create_text(x1 + self.cell_size/2,
                                          y1 + self.cell_size/2,
                                          text=prob_text,
                                          font=TINY_FONT,
                                          fill=text_color,
                                          tags="text")
    
    def _draw_piece(self, x, y, cell_size, piece_type):
        """
        Dibuja una ficha en la casilla especificada.
        
        Args:
            x, y: Coordenadas de la esquina superior izquierda de la casilla
            cell_size: Tamaño de la casilla
            piece_type: 1 para negro, -1 para blanco
        """
        # Calcular el centro y radio de la ficha
        center_x = x + cell_size // 2
        center_y = y + cell_size // 2
        radius = cell_size // 2.5  # Ficha más grande para mayor visibilidad
        
        # Coordenadas del círculo
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius
        
        if piece_type == 1:  # Ficha negra (semitransparente como marca de agua)
            fill_color = "#606060"  # Gris oscuro en lugar de negro puro
            outline_color = "#404040"  # Gris muy oscuro para el borde
            width = 2
        elif piece_type == -1:  # Ficha blanca (valor -1 en lugar de 2)
            fill_color = "#e0e0e0"  # Gris claro para fichas blancas como marca de agua
            outline_color = "#b0b0b0"  # Gris medio para el borde
            width = 2  # Borde visible pero sutil
        else:
            return  # No dibujar nada para piece_type == 0 u otros valores
        
        # Dibujar la ficha con efecto de marca de agua
        self.canvas.create_oval(x1, y1, x2, y2,
                               fill=fill_color,
                               outline=outline_color,
                               width=width,
                               tags="piece")
    
    def draw_heatmap(self, probabilities):
        """
        Dibuja el heatmap en el tablero (método de compatibilidad).
        
        Args:
            probabilities: Array de numpy con 64 probabilidades
        """
        # Llamar al método principal con el estado actual del tablero
        self.draw_heatmap_with_pieces(probabilities, self.current_board_state)
    
    def _draw_grid_and_labels(self):
        """Dibuja solo el grid y las etiquetas (función auxiliar)"""
        # Dibujar líneas de grid con padding
        for i in range(9):  # 9 líneas para crear 8 casillas
            # Líneas horizontales
            self.canvas.create_line(self.canvas_padding, i * self.cell_size + self.canvas_padding, 
                                  self.board_size + self.canvas_padding, i * self.cell_size + self.canvas_padding,
                                  fill=LINE_COLOR, width=1)
            # Líneas verticales  
            self.canvas.create_line(i * self.cell_size + self.canvas_padding, self.canvas_padding,
                                  i * self.cell_size + self.canvas_padding, self.board_size + self.canvas_padding,
                                  fill=LINE_COLOR, width=1)
        
        # Dibujar etiquetas de filas (letras A-H) - centradas en cada fila
        for i in range(8):
            self.canvas.create_text(15, i * self.cell_size + self.cell_size // 2 + self.canvas_padding, 
                                  text=chr(97 + i).upper(), font=SMALL_FONT, fill="black")
        
        # Dibujar etiquetas de columnas (números 1-8) - centradas en cada columna  
        for i in range(8):
            self.canvas.create_text(i * self.cell_size + self.cell_size // 2 + self.canvas_padding, 15, 
                                  text=str(i + 1), font=SMALL_FONT, fill="black")
    
    def draw_empty_board(self):
        """Dibuja el tablero vacío con líneas de grid y etiquetas"""
        # Limpiar solo las casillas y texto, NO el fondo
        self.canvas.delete("cell")
        self.canvas.delete("text")
        self.canvas.delete("piece")
        
        # Solo dibujar todo si no se ha hecho antes
        if not hasattr(self, '_grid_drawn') or not self._grid_drawn:
            self.canvas.delete("all")
            self.canvas.configure(bg="white")
            # Crear fondo blanco masivo
            self.canvas.create_rectangle(0, 0, 1000, 1000,
                                       fill="white", outline="", width=0, tags="background")
            self._draw_grid_and_labels()
            self._grid_drawn = True
    
    def on_mouse_motion(self, event):
        """Maneja el movimiento del mouse para mostrar probabilidades"""
        # Calcular qué casilla está bajo el mouse (ajustando por el padding)
        col = (event.x - self.canvas_padding) // self.cell_size
        row = (event.y - self.canvas_padding) // self.cell_size
        
        if 0 <= row < 8 and 0 <= col < 8:
            board_index = row * 8 + col
            prob = self.current_probabilities[board_index]
            piece = self.current_board_state[board_index] if len(self.current_board_state) > board_index else 0
            coord = f"{chr(97 + row)}{col + 1}".upper()
            
            # Información de la casilla
            piece_text = ""
            if piece == 1:
                piece_text = " (Negra)"
            elif piece == -1:  # Ficha blanca tiene valor -1
                piece_text = " (Blanca)"
            
            # Actualizar información
            if prob > 0:
                info_text = f"Casilla {coord}: {prob:.4%}{piece_text}"
            else:
                info_text = f"Casilla {coord}: 0%{piece_text}"
            
            self.hover_info.config(text=info_text)
    
    def on_mouse_leave(self, event):
        """Limpia la información cuando el mouse sale del canvas"""
        self.hover_info.config(text="")
    
    def show(self):
        """Muestra la ventana"""
        self.window.deiconify()
    
    def hide(self):
        """Oculta la ventana"""
        self.window.withdraw()