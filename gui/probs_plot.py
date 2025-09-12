# probs_plot.py
# Interfaz gráfica para mostrar el gráfico de probabilidades de jugadas

import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')  # Usar TkAgg como backend para mostrar dentro de Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sys
import os

# Agregamos el directorio raíz al path para poder importar módulos del proyecto
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importamos las funciones necesarias
from data.othello import permit, permit_reverse

class ProbsPlot:
    def __init__(self):
        """
        Inicializa la ventana y el gráfico para mostrar las probabilidades de las jugadas.
        """
        self.window = tk.Toplevel()
        self.window.title("Probabilidades de Jugadas")
        self.window.geometry("800x600")
        
        # Crear la figura de matplotlib
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Inicializar el canvas de matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Configurar el gráfico inicial
        self.ax.set_title("Probabilidades de Jugadas")
        self.ax.set_xlabel("Posición en el tablero")
        self.ax.set_ylabel("Probabilidad (%)")
        
        # Mensaje inicial
        self.ax.text(0.5, 0.5, "Esperando primer movimiento...", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=14)
        
        # Actualizar el canvas
        self.canvas.draw()

    def update(self, probs):
        """
        Actualiza el gráfico con nuevas probabilidades.
        
        Args:
            probs: Diccionario con las coordenadas de las jugadas y sus probabilidades.
                  Formato: {"a1": 0.05, "b2": 0.1, ...}
        """
        if not probs:
            return
            
        # Limpiar el gráfico
        self.ax.clear()
        
        # Ordenar las jugadas por probabilidad (de mayor a menor)
        sorted_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # Tomar las 10 jugadas con mayor probabilidad
        top_moves = sorted_moves[:10]
        
        # Preparar los datos para el gráfico
        positions = [move for move, _ in top_moves]
        probabilities = [prob * 100 for _, prob in top_moves]  # Convertir a porcentaje
        
        # Crear el gráfico de barras
        bars = self.ax.bar(positions, probabilities, color='skyblue')
        
        # Resaltar la jugada con mayor probabilidad
        if top_moves:
            bars[0].set_color('orange')
        
        # Configurar el gráfico
        self.ax.set_title("Probabilidades de las 10 Mejores Jugadas")
        self.ax.set_xlabel("Posición en el tablero")
        self.ax.set_ylabel("Probabilidad (%)")
        self.ax.set_ylim(0, max(probabilities) * 1.2)  # Dejar espacio en la parte superior
        
        # Añadir los valores numéricos sobre cada barra
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', rotation=0)
        
        # Ajustar las etiquetas del eje x para evitar solapamiento
        self.ax.set_xticklabels(positions, rotation=45, ha='right')
        
        # Actualizar el canvas
        self.fig.tight_layout()
        self.canvas.draw()
