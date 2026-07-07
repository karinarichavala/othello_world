# Guía de Instalación - Othello World

Esta guía proporciona instrucciones paso a paso para instalar y configurar el proyecto **Othello World**.

**Alcance**: Esta guía cubre la instalación del proyecto y el uso de los módulos agregados en esta tesis (`gui/` y `sae/`), a partir del checkpoint ya entrenado de Othello-GPT (`ckpts/gpt_championship.ckpt`, obtenido de Kenneth Li e incluido en el repositorio). **No cubre el entrenamiento de Othello-GPT desde cero**, esto ya lo realizaron Kenneth Li et al.

##  Requisitos Previos

- **Python 3.8+** (declarado como compatible en `pyproject.toml`; verificado en esta guía únicamente con Python 3.11.9)
- **pip** (administrador de paquetes de Python)
- **Git** (opcional, para clonar el repositorio)
- **CUDA** (opcional, para entrenamiento con GPU)

### Verificar Instalación de Python

```bash
python --version
pip --version
```

### Entorno de GPU usado durante el desarrollo

El proyecto fue desarrollado y probado con la siguiente configuración de GPU:

| Componente | Versión |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 |
| Driver NVIDIA | 591.59 (soporta hasta CUDA 13.1) |
| PyTorch | 2.8.0+cu128 |

**Nota**: La RTX 5060 es arquitectura Blackwell (compute capability `sm_120`), que requiere como mínimo builds de PyTorch con soporte CUDA 12.8 (`+cu128`) para usar aceleración GPU.

`pyproject.toml`/`requirements.txt` solo piden `torch>=1.8.0` (sin techo de versión), así que `pip install -e ".[all]"` instalará automáticamente **la última versión de PyTorch disponible**, no necesariamente 2.8.0+cu128. Esto es intencional para no atar el proyecto a una versión fija, pero según tu GPU debes ajustar el paso de PyTorch:

- **Si tienes una GPU Blackwell (serie RTX 50, como la RTX 5060)**: instala la misma versión usada en desarrollo para asegurar compatibilidad:
  ```bash
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
  ```
- **Si tienes una GPU NVIDIA más antigua (series RTX 30/40 o anteriores)**: usa el build de CUDA que soporte tu driver (`cu121`, `cu124`, etc.), no es necesario que coincida con `cu128`.
- **Si no tienes GPU**: no necesitas hacer nada especial — la versión que instala `pip install -e ".[all]"` por defecto (build CPU) funciona igual, solo el entrenamiento será más lento.

Verifica tu propio driver y GPU con:

```bash
nvidia-smi
```

##  Instalación Rápida

### Paso previo: Crear entorno virtual

Se recomienda instalar el proyecto dentro de un entorno virtual, usando Python 3.11 (si tienes varias versiones de Python instaladas, usa `py -3.11` en vez de `python` para asegurar la versión correcta):

```bash
# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python3.11 -m venv .venv
source .venv/bin/activate
```

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

##  Verificar Instalación

### Probar Imports

```bash
python -c "from mingpt.model import GPT; from data.othello import Othello; print('✓ Imports funcionando correctamente')"
```

### Verificar GPU (si aplica)

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Si tienes GPU y todo quedó bien instalado, deberías ver la versión de PyTorch con el build de CUDA correspondiente (ej. `2.8.0+cu128`), `True`, y el nombre de tu GPU.

### Verificación realizada

Se probaron los siguientes pasos clave de esta guía (Opción 1 de instalación) en dos entornos distintos (venv separados, Python 3.11). No se probaron: Opciones 2/3 de instalación, instalaciones opcionales (`interpretability`, `dev`, `scienceplots`), datasets/checkpoints, notebooks, ni el entrenamiento de probes/SAE.

| Paso | Con GPU (RTX 5060, cu128) | Solo CPU |
|---|---|---|
| Creación de venv | ✅ | ✅ |
| `pip install -e ".[all]"` | ✅ | ✅ |
| PyTorch detecta el hardware correctamente | ✅ (`2.8.0+cu128`, `cuda.is_available()=True`) | ✅ (`2.12.1+cpu`, `cuda.is_available()=False`) |
| Imports del proyecto (`mingpt`, `data.othello`) | ✅ | ✅ |


## Documentación Adicional

- **README principal**: [README.md](README.md)
- **Paper original**: [arXiv:2210.13382](https://arxiv.org/abs/2210.13382)



