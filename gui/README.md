# Othello GUI - Tutorial de Instalación

## Ejecución

Para ejecutar la interfaz gráfica del juego Othello:

```bash
python gui/run.py
```

## Dependencias

### Setuptools

Asegúrate de tener setuptools instalado:

```bash
pip install setuptools
```

### Environment.yml

El proyecto usa conda para el manejo de dependencias. Las dependencias están definidas en el archivo `environment.yml` del directorio raíz. Para instalar el entorno:

```bash
conda env create -f environment.yml
conda activate othello
```

#### Dependencias principales:
- tkinter (incluido con Python)
- matplotlib
- numpy
- torch
- pandas
- jupyter
- tqdm
