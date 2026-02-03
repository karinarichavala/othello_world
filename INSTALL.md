# Guía de Instalación - Othello World

Esta guía proporciona instrucciones paso a paso para instalar y configurar el proyecto **Othello World** después del refactoring.

##  Requisitos Previos

- **Python 3.8+** (probado con Python 3.8-3.11)
- **pip** (administrador de paquetes de Python)
- **Git** (opcional, para clonar el repositorio)
- **CUDA** (opcional, para entrenamiento con GPU)

### Verificar Instalación de Python

```bash
python --version
pip --version
```

##  Instalación Rápida

### Opción 1: Instalación Completa (Recomendada)

```bash
# 1. Navegar al directorio del proyecto
cd othello_world

# 2. Instalar el proyecto en modo editable con todas las dependencias
pip install -e ".[all]"

# 3. (Opcional) Instalar kernel de Jupyter
python -m ipykernel install --user --name othello --display-name "othello"
```

### Opción 2: Instalación Básica

```bash
# Solo dependencias básicas (sin interpretabilidad mecanística)
pip install -e .
```

### Opción 3: Instalación con requirements.txt

```bash
# Todas las dependencias básicas desde requirements.txt
pip install -r requirements.txt

# Luego instalar el paquete en modo editable
pip install -e .
```

##  Instalaciones Opcionales

### Para Interpretabilidad Mecanística

```bash
pip install -e ".[interpretability]"
```

Incluye:
- `transformer-lens` - Para análisis de circuitos
- `einops` - Operaciones de tensores
- `fancy-einsum` - Einstein summation notation

### Para Desarrollo

```bash
pip install -e ".[dev]"
```

Incluye:
- `pytest` - Testing
- `black` - Formateo de código
- `flake8` - Linting
- `mypy` - Type checking

### Para Gráficos de Publicación

```bash
pip install scienceplots
```

 **Nota**: `scienceplots` requiere LaTeX instalado en el sistema. Ver [FAQ de SciencePlots](https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-latex).

##  Descargar Datasets

Los datasets no están incluidos en el repositorio. Descárgalos desde:

1. **Championship Dataset**: [Google Drive](https://drive.google.com/drive/folders/1KFtP7gfrjmaoCV-WFC4XrdVeOxy1KmXe?usp=sharing)
2. **Synthetic Dataset**: [Google Drive](https://drive.google.com/drive/folders/1pDMdMrnxMRiDnUd-CNfRNvZCi7VXFRtv?usp=sharing)

Colócalos en:
```
othello_world/
├── data/
│   ├── othello_championship/
│   └── othello_synthetic/
```

##  Descargar Checkpoints (Opcional)

Si quieres evitar el entrenamiento, descarga los checkpoints:

- **Modelo GPT**: [Google Drive - GPT Checkpoints](https://drive.google.com/drive/folders/1bpnwJnccpr9W-N_hzXSm59hT7Lij4HxZ?usp=sharing)
- **Probes**: [Google Drive - Probe Checkpoints](https://drive.google.com/drive/folders/1uvj_M9ekHDJVdVOvMq828Z23AE7jZ01H?usp=sharing)

Colócalos en:
```
othello_world/
└── ckpts/
    ├── gpt_championship.ckpt
    └── battery_othello/
```

##  Verificar Instalación

### Probar Imports

```bash
python -c "from mingpt.model import GPT; from data.othello import Othello; print('✓ Imports funcionando correctamente')"
```

### Ejecutar la GUI

```bash
# Desde cualquier directorio
othello-gui

# O directamente con Python
python -m gui.run
```

Deberían abrirse 4 ventanas:
1. Tablero de juego principal
2. Gráfico de barras de probabilidades
3. Heatmap simple
4. Heatmap integrado con fichas

##  Ejecutar Notebooks

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir cualquier notebook, por ejemplo:
# - train_gpt_othello.ipynb
# - Othello_GPT_Circuits.ipynb
# - sae/model/train_sae_othello.ipynb
```

##  Solución de Problemas

### Error: "No module named 'mingpt'"

**Solución**: Asegúrate de haber instalado el paquete en modo editable:
```bash
pip install -e .
```

### Error: "No module named 'transformer_lens'"

**Solución**: Instala las dependencias de interpretabilidad:
```bash
pip install -e ".[interpretability]"
```

### Error: "FileNotFoundError: checkpoint not found"

**Solución**: Descarga el checkpoint `gpt_championship.ckpt` y colócalo en `ckpts/`

### La GUI no se abre

**Solución**: Verifica que tkinter esté instalado:
```bash
python -m tkinter
# Debería abrir una ventana de prueba
```

En Ubuntu/Debian:
```bash
sudo apt-get install python3-tk
```

### Error de memoria (OOM) durante entrenamiento

**Solución**: Reduce el batch size en la configuración del entrenamiento o usa una GPU con más memoria.

##  Uso Básico

### 1. Jugar contra el Modelo

```bash
othello-gui
```

### 2. Entrenar el Modelo GPT

```bash
jupyter notebook train_gpt_othello.ipynb
```

### 3. Entrenar Probes

```bash
# Probe lineal en capa 6
python train_probe_othello.py --layer 6 --championship

# Probe no lineal (2 capas) en capa 6
python train_probe_othello.py --layer 6 --twolayer --mid_dim 64 --championship
```

### 4. Extraer Activaciones para SAE

```bash
python sae/activations/run_extraction.py
```

### 5. Entrenar SAE

```bash
jupyter notebook sae/model/train_sae_othello.ipynb
```

##  Estructura del Proyecto

```
othello_world/
├── data/                    # Lógica del juego Othello
├── mingpt/                  # Modelo GPT y entrenamiento
├── gui/                     # Interfaz gráfica
├── sae/                     # Sparse Autoencoders
├── mechanistic_interpretability/  # Análisis de circuitos
├── ckpts/                   # Checkpoints del modelo
├── setup.py                 # Configuración del paquete
├── pyproject.toml           # Configuración moderna
└── requirements.txt         # Dependencias del proyecto
```

##  Actualizar Instalación

Si haces cambios en el código:

```bash
# No necesitas reinstalar, los cambios se reflejan automáticamente con -e
# Solo reinstala si cambias dependencias en setup.py:
pip install -e ".[all]" --upgrade
```

##  Documentación Adicional

- **README principal**: [README.md](README.md)
- **Informe técnico**: [gui/doc/Informe_Tecnico.md](gui/doc/Informe_Tecnico.md)
- **Paper original**: [arXiv:2210.13382](https://arxiv.org/abs/2210.13382)

##  Mejoras Implementadas en esta Versión

✅ **Instalación como paquete Python estándar**
✅ **Eliminadas manipulaciones de `sys.path`**
✅ **Imports limpios y consistentes**
✅ **Configuración con `pyproject.toml` (PEP 621)**
✅ **Comando CLI: `othello-gui`**
✅ **Soporte para instalación con pip desde cualquier directorio**
✅ **Mejor organización de dependencias opcionales**

##  Contribuir

Si encuentras problemas o quieres contribuir:

1. Abre un issue en GitHub
2. Envía un pull request
3. Contacta a los autores (ver paper)

---

