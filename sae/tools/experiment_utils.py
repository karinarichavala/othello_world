"""
Utilidades para gestionar rutas de experimentos locales de SAEs sobre OthelloGPT

Convención de carpetas (sin extras):
    {base_path}/layer_{capa:02d}/sae_{variante}/sae_{tecnica}_{variante}/{juegos}games/
        plots/
        metrics/

Convención de carpetas (con extras):
    {base_path}/layer_{capa:02d}/sae_{variante}/sae_{tecnica}_{variante}/{juegos}games/{extras_id}/
        plots/
        metrics/

donde base_path es la raíz de experimentos (ej. .../sae/experiments).

Convención de nombre de reporte (sin extras):
    sae_{variante}_{tecnica}_l{capa}_{juegos}g_{tipo}.pdf

Convención de nombre de reporte (con extras):
    sae_{variante}_{tecnica}_l{capa}_{juegos}g_{extras_id}_{tipo}.pdf

Ejemplos:
    Carpeta: .../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/
    Carpeta con extras: .../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/ex1/
    Reporte: sae_standard_l1_l6_1000g_training.pdf
    Reporte con extras: sae_standard_l1_l6_1000g_ex1_training.pdf

Uso típico desde un notebook de entrenamiento (sae/models/sae_standard/sae_l1_standar/):

    import sys, os
    sys.path.append(os.path.abspath('../../..'))
    from sae.tools.experiment_utils import (
        get_experiment_dir, get_plots_dir, get_metrics_dir, get_report_name
    )

    EXPERIMENT_BASE_PATH = os.path.abspath('../../../experiments')  # sae/experiments/

    # Sin extras
    exp_dir     = get_experiment_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000)
    plots_dir   = get_plots_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000)
    metrics_dir = get_metrics_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000)
    report_name = get_report_name('standard', 'l1', 6, 1000, 'training')

    # Con extras (extras_id viene de naming_utils.get_extras_id)
    exp_dir     = get_experiment_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000, extras_id='ex1')
    plots_dir   = get_plots_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000, extras_id='ex1')
    metrics_dir = get_metrics_dir(EXPERIMENT_BASE_PATH, 'standard', 'l1', 6, 1000, extras_id='ex1')
    report_name = get_report_name('standard', 'l1', 6, 1000, 'training', extras_id='ex1')
"""

from pathlib import Path


def get_experiment_dir(base_path, variante, tecnica, capa, juegos, extras_id=None):
    """
    Construye el path base del experimento, creando las carpetas si no existen.

    Args:
        base_path:  Raíz del proyecto (str o Path), se pasa desde cada notebook
        variante:   Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:    Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:       Número de capa del modelo (int)
        juegos:     Número de partidas de entrenamiento (int)
        extras_id:  Identificador de extras (ej. "ex1"), opcional

    Returns:
        Path completo del directorio base del experimento (Path)

    Ejemplos:
        >>> get_experiment_dir(".../sae/experiments", "standard", "l1", 6, 1000)
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games')
        >>> get_experiment_dir(".../sae/experiments", "standard", "l1", 6, 1000, extras_id="ex1")
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/ex1')
    """
    base_path = Path(base_path)
    exp_dir = (
        base_path
        / f"layer_{capa:02d}"
        / f"sae_{variante}"
        / f"sae_{tecnica}_{variante}"
        / f"{juegos}games"
    )
    if extras_id:
        exp_dir = exp_dir / extras_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_plots_dir(base_path, variante, tecnica, capa, juegos, extras_id=None):
    """
    Retorna el path del subdirectorio plots/ del experimento, creándolo si no existe.

    Args:
        base_path:  Raíz del proyecto (str o Path), se pasa desde cada notebook
        variante:   Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:    Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:       Número de capa del modelo (int)
        juegos:     Número de partidas de entrenamiento (int)
        extras_id:  Identificador de extras (ej. "ex1"), opcional

    Returns:
        Path del subdirectorio plots/ (Path)

    Ejemplos:
        >>> get_plots_dir(".../sae/experiments", "standard", "l1", 6, 1000)
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/plots')
        >>> get_plots_dir(".../sae/experiments", "standard", "l1", 6, 1000, extras_id="ex1")
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/ex1/plots')
    """
    plots_dir = get_experiment_dir(base_path, variante, tecnica, capa, juegos, extras_id) / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def get_metrics_dir(base_path, variante, tecnica, capa, juegos, extras_id=None):
    """
    Retorna el path del subdirectorio metrics/ del experimento, creándolo si no existe.

    Args:
        base_path:  Raíz del proyecto (str o Path), se pasa desde cada notebook
        variante:   Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:    Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:       Número de capa del modelo (int)
        juegos:     Número de partidas de entrenamiento (int)
        extras_id:  Identificador de extras (ej. "ex1"), opcional

    Returns:
        Path del subdirectorio metrics/ (Path)

    Ejemplos:
        >>> get_metrics_dir(".../sae/experiments", "standard", "l1", 6, 1000)
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/metrics')
        >>> get_metrics_dir(".../sae/experiments", "standard", "l1", 6, 1000, extras_id="ex1")
        Path('.../sae/experiments/layer_06/sae_standard/sae_l1_standard/1000games/ex1/metrics')
    """
    metrics_dir = get_experiment_dir(base_path, variante, tecnica, capa, juegos, extras_id) / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    return metrics_dir


def get_report_name(variante, tecnica, capa, juegos, tipo, extras_id=None):
    """
    Genera el nombre del archivo PDF de reporte según convención.

    Args:
        variante:   Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:    Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:       Número de capa del modelo (int)
        juegos:     Número de partidas de entrenamiento (int)
        tipo:       Tipo de reporte: "training", "coverage", "reconstruction", etc.
        extras_id:  Identificador de extras (ej. "ex1"), opcional

    Returns:
        Nombre del archivo PDF (str)

    Ejemplos:
        >>> get_report_name("standard", "l1", 6, 1000, "training")
        'sae_standard_l1_l6_1000g_training.pdf'
        >>> get_report_name("standard", "l1", 6, 1000, "training", extras_id="ex1")
        'sae_standard_l1_l6_1000g_ex1_training.pdf'
    """
    if extras_id:
        return f"sae_{variante}_{tecnica}_l{capa}_{juegos}g_{extras_id}_{tipo}.pdf"
    return f"sae_{variante}_{tecnica}_l{capa}_{juegos}g_{tipo}.pdf"
