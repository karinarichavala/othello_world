# Othello GUI - Tutorial de Instalación Completo

##  Requisitos Previos

### 1. Python
- **Versión**: Python 3.8+ (probado con Python 3.13.5)
- **Instalación**: Descargar desde [python.org](https://www.python.org/downloads/)
- **Importante**: Marcar "Add Python to PATH" durante la instalación

### 2. Verificar Instalación
```bash
python --version
pip --version
```

##  Instalación de Dependencias

### Método 1: Requirements.txt (Recomendado)
```bash
pip install -r requirements.txt
```

### Método 2: Instalación Manual
```bash
pip install torch numpy matplotlib python-chess pgn psutil seaborn
```

##  Ejecución


### Desde el directorio raíz del proyecto:
```bash
cd othello_world
python gui/run.py
```

##  Funcionamiento

### Al ejecutar se abren 4 ventanas:
1. **Ventana Principal**: Tablero de juego interactivo
2. **Gráfico de Barras**: Top 10 movimientos con mayor probabilidad
3. **Heatmap Simple**: Probabilidades por casilla
4. **Heatmap Integrado**: Probabilidades + fichas del juego

### Características:
-  **Juego vs IA**: El modelo GPT actúa como oponente inteligente
-  **Visualización en tiempo real**: Probabilidades de cada movimiento
-  **Navegación temporal**: Revisar movimientos anteriores
-  **Modelo pre-entrenado**: Usa el checkpoint en `ckpts/gpt_championship.ckpt`

##  Dependencias Completas

```
torch>=1.8.0          # Modelo GPT y carga del checkpoint
numpy>=1.20.0          # Arrays y manipulación de datos
matplotlib>=3.3.0      # Gráficos de probabilidades
python-chess           # Utilidades de juegos de tablero
pgn                    # Formato PGN para partidas
psutil                 # Información del sistema
seaborn               # Visualizaciones estadísticas
```

**Nota**: `tkinter` está incluido con Python (no requiere instalación adicional)



