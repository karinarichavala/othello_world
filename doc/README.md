# Othello GPT - Documentación

## Descripción 

Este proyecto implementa un juego de Othello (Reversi) basada en un modelo GPT entrenado específicamente para predecir movimientos en el juego. El sistema incluye una interfaz gráfica que visualiza tanto el tablero de juego y las probabilidades de cada jugada posible.

## Arquitectura del Sistema

### Modelo GPT Othello

El modelo utilizado es un **Transformer Decoder** con las siguientes características:

#### Configuración del Modelo
- **Vocabulario**: 61 tokens (60 movimientos posibles + "pass")
- **Contexto**: Secuencias de hasta 59 movimientos
- **Capas**: 8 bloques Transformer
- **Atención**: 8 cabezas por bloque (64D cada una)
- **Embeddings**: 512 dimensiones

#### Arquitectura Detallada
```
Entrada: Secuencia de jugadas (tokens: int 0–60)
                    ↓
┌────────────────────────────────────┐
│  1. Token Embedding Layer          │ ← (Convierte cada token en vector de 512D)
└────────────────────────────────────┘
                    ↓
┌────────────────────────────────────┐
│  2. Positional Encoding             │ ← (Agrega posición al vector)
└────────────────────────────────────┘
                    ↓
                ▼▼▼ x8 ▼▼▼
┌────────────────────────────────────┐
│         Bloque Decoder #i           │
│  ┌──────────────────────────────┐  │
│  │  LayerNorm                   │  │
│  │  Multi-Head Self-Attention   │  │
│  │  (8 cabezas de 64D)          │  │
│  │  Add & Residual              │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │  LayerNorm                   │  │
│  │  Feed Forward (MLP)          │  │
│  │  512 → 2048 → 512            │  │
│  │  Add & Residual              │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
                    ↓
┌────────────────────────────────────┐
│  7. Proyección Lineal Final         │ ← (512 → 61 logits)
└────────────────────────────────────┘
                    ↓
┌────────────────────────────────────┐
│  8. Softmax                         │ ← (Convierte logits en probabilidades)
└────────────────────────────────────┘
                    ↓
    Distribución de probabilidad de la siguiente jugada
```

### Componentes del Sistema

#### 1. Modelo y Lógica (`mingpt/`)
- **`model.py`**: Implementación del modelo GPT
- **`dataset.py`**: Manejo de datos de entrenamiento

#### 2. Datos del Juego (`data/`)
- **`othello.py`**: Lógica del juego Othello, reglas y validaciones

#### 3. Interfaz Gráfica (`gui/`)
- **`run.py`**: Punto de entrada principal
- **`game_gui.py`**: Interfaz principal del tablero
- **`model_handler.py`**: Conexión entre modelo y GUI
- **`probability_heatmap.py`**: Visualización original de probabilidades
- **`game_heatmap.py`**: Heatmap integrado con fichas
- **`probs_plot.py`**: Gráfico de barras de probabilidades

#### 4. Lógica de Control (`gui/logic/`)
- **`game_controller.py`**: Controlador principal del juego
- **`move_handler.py`**: Manejo de movimientos y turnos

## Adicciones realizadas

### Heatmap Integrado (`game_heatmap.py`)
Creación de un heatmap integrado:

### Características Principales

- Probabilidades mostradas como colores de fondo (gradiente blanco → rojo oscuro)
- Se actualiza automáticamente después de cada movimiento
- Sincronizado con el estado actual del juego

### Ejecución

1. **Ejecutar el juego dessde el root del proyecto**:
   ```bash
   python gui/run.py
   ```