# probability_heatmap.py
# Ventana separada para mostrar un heatmap de probabilidades en formato tablero

import tkinter as tk
import numpy as np
from gui.config.settings import *

class ProbabilityHeatmap:
    def __init__(self):
        """Inicializa la ventana del heatmap de probabilidades"""
        # Crear ventana
        self.window = tk.Toplevel()
        self.window.title("Heatmap de Probabilidades - Othello")
        self.window.configure(background="white")
        self.window.geometry("650x700")  # Más ancho y alto para que no se corte
        
        # Variables
        self.cell_size = 60
        self.board_size = 8 * self.cell_size
        self.current_probabilities = np.zeros(64)
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
                              text="Probabilidades por Casilla", 
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
        
    def update(self, move_probabilities):
        """
        Actualiza el heatmap con nuevas probabilidades.
        
        Args:
            move_probabilities: Diccionario con coordenadas como claves (ej: "a1") y probabilidades como valores
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
        self.draw_heatmap(probs_array)
    
    def draw_heatmap(self, probabilities):
        """
        Dibuja el heatmap en el tablero.
        
        Args:
            probabilities: Array de numpy con 64 probabilidades
        """
        # SOLUCIÓN EXTREMA: No limpiar todo, solo limpiar elementos específicos
        self.canvas.delete("cell")
        self.canvas.delete("text")
        
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
        if max_prob == 0:
            return
        
        normalized_probs = probabilities / max_prob
        
        # Dibujar cada casilla con intensidad según probabilidad
        for i in range(8):
            for j in range(8):
                board_index = i * 8 + j
                prob = probabilities[board_index]
                normalized_prob = normalized_probs[board_index]
                
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
                    
                    # Dibujar casilla con color
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                                               fill=color,
                                               outline=LINE_COLOR,
                                               tags="cell")
                    
                    # Mostrar valor de probabilidad si es significativo
                    if prob >= 0.01:  # Solo mostrar si es >= 1%
                        prob_text = f"{prob:.2%}"
                        # Color del texto: blanco para probabilidades altas, negro para bajas
                        text_color = "white" if normalized_prob > 0.5 else "black"
                        
                        self.canvas.create_text(x1 + self.cell_size/2,
                                              y1 + self.cell_size/2,
                                              text=prob_text,
                                              font=TINY_FONT,
                                              fill=text_color,
                                              tags="text")
                # Si prob == 0, no dibujar nada sobre la casilla (queda blanca)
    
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
            coord = f"{chr(97 + row)}{col + 1}".upper()
            
            # Actualizar información
            info_text = f"Casilla {coord}: {prob:.4%}" if prob > 0 else f"Casilla {coord}: 0%"
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