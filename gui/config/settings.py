# settings.py
# Configuraciones para la interfaz gráfica del juego Othello

# Configuración de la ventana
WINDOW_TITLE = "Othello Game"
WINDOW_GEOMETRY = "650x650"  # Más compacto y cuadrado

# Configuración del tablero
CELL_SIZE = 60
BOARD_WIDTH = 8 * CELL_SIZE
BOARD_HEIGHT = 8 * CELL_SIZE

# Colores
BOARD_COLOR = "forest green"
LINE_COLOR = "black"
VALID_MOVE_COLOR = "yellow"
AI_HIGHLIGHT_COLOR = "cyan"
OBSERVER_HIGHLIGHT_COLOR = "yellow"

# Configuración de la interfaz
PADDING = 20
BOARD_PADDING = 40

# Fuentes
DEFAULT_FONT = ("Arial", 14)
LABEL_FONT = ("Arial", 12)
SMALL_FONT = ("Arial", 9)
TINY_FONT = ("Arial", 8, "bold")

# Coordenadas del tablero
BOARD_ROWS = ["a", "b", "c", "d", "e", "f", "g", "h"]
BOARD_COLUMNS = ["1", "2", "3", "4", "5", "6", "7", "8"]

# Configuración de la IA
AI_THINKING_DELAY = 5500  # milisegundos (aumentado para apreciar mejor el gráfico)
AI_HIGHLIGHT_DURATION = 3000  # milisegundos