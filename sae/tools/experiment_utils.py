"""
Utilidades para gestionar rutas de experimentos locales de SAEs sobre OthelloGPT

Convención de carpetas:
    {base_path}/layer_{capa:02d}/sae_{variante}/sae_{tecnica}_{variante}/{juegos}games/
        plots/
        metrics/

donde base_path es la raíz de experimentos (ej. .../sae/experiments).

Convención de nombre de reporte:
    sae_{variante}_{tecnica}_l{capa}_{juegos}g_{tipo}.pdf

Ejemplos:
    Carpeta: .../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/
    Reporte: sae_standard_l1_l6_1000g_training.pdf

Uso típico desde un notebook de entrenamiento (sae/models/sae_standard/sae_l1_standar/):

    import sys, os
    sys.path.append(os.path.abspath('../../..'))
    from sae.tools.experiment_utils import (
        get_experiment_dir, get_plots_dir, get_metrics_dir, get_report_name
    )

    EXPERIMENT_BASE_PATH = os.path.abspath('../../../experiments')  # sae/experiments/

    exp_dir     = get_experiment_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000)
    plots_dir   = get_plots_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000)
    metrics_dir = get_metrics_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000)
    report_name = get_report_name('standard', 'l1', 6, 1000, 'training')

Uso típico desde un notebook de métricas (sae/metrics/):

    import sys, os
    sys.path.append(os.path.abspath('../..'))
    from sae.tools.experiment_utils import get_plots_dir, get_metrics_dir, get_report_name

    EXPERIMENT_BASE_PATH = os.path.abspath('../experiments')  # sae/experiments/
"""

from pathlib import Path


def get_experiment_dir(base_path, variante, tecnica, capa, juegos):
    """
    Construye el path base del experimento, creando las carpetas si no existen.

    Args:
        base_path: Raíz del proyecto (str o Path), se pasa desde cada notebook
        variante:  Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:   Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:      Número de capa del modelo (int)
        juegos:    Número de partidas de entrenamiento (int)

    Returns:
        Path completo del directorio base del experimento (Path)

    Ejemplo:
        >>> get_experiment_dir(".../sae/experiments", "standard", "l1", 6, 1000)
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games')
    """
    base_path = Path(base_path)
    exp_dir = (
        base_path
        / f"layer_{capa:02d}"
        / f"sae_{variante}"
        / f"sae_{tecnica}_{variante}"
        / f"{juegos}games"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_plots_dir(base_path, variante, tecnica, capa, juegos):
    """
    Retorna el path del subdirectorio plots/ del experimento, creándolo si no existe.

    Args:
        base_path: Raíz del proyecto (str o Path), se pasa desde cada notebook
        variante:  Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:   Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:      Número de capa del modelo (int)
        juegos:    Número de partidas de entrenamiento (int)

    Returns:
        Path del subdirectorio plots/ (Path)

    Ejemplo:
        >>> get_plots_dir(".../sae/experiments", "standard", "l1", 6, 1000)
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/plots')
    """
    plots_dir = get_experiment_dir(base_path, variante, tecnica, capa, juegos) / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def get_metrics_dir(base_path, variante, tecnica, capa, juegos):
    """
    Retorna el path del subdirectorio metrics/ del experimento, creándolo si no existe.

    Args:
        base_path: Raíz del proyecto (str o Path), se pasa desde cada notebook
        variante:  Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:   Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:      Número de capa del modelo (int)
        juegos:    Número de partidas de entrenamiento (int)

    Returns:
        Path del subdirectorio metrics/ (Path)

    Ejemplo:
        >>> get_metrics_dir(".../sae/experiments", "standard", "l1", 6, 1000)
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/metrics')
    """
    metrics_dir = get_experiment_dir(base_path, variante, tecnica, capa, juegos) / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    return metrics_dir


def get_report_name(variante, tecnica, capa, juegos, tipo):
    """
    Genera el nombre del archivo PDF de reporte según convención.

    Args:
        variante: Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:  Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:     Número de capa del modelo (int)
        juegos:   Número de partidas de entrenamiento (int)
        tipo:     Tipo de reporte: "training" o "metrics"

    Returns:
        Nombre del archivo PDF (str)

    Ejemplo:
        >>> get_report_name("standard", "l1", 6, 1000, "training")
        'sae_standard_l1_l6_1000g_training.pdf'
        >>> get_report_name("standard", "l1", 6, 1000, "metrics")
        'sae_standard_l1_l6_1000g_metrics.pdf'
    """
    return f"sae_{variante}_{tecnica}_l{capa}_{juegos}g_{tipo}.pdf"
