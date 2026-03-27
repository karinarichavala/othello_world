"""
Utilidades para nombrar y guardar checkpoints de SAEs entrenados sobre OthelloGPT

Convención de carpetas:
    {base_path}/layer_{capa:02d}/models/sae_{variante}/sae_{tecnica}_{variante}/{juegos}games/

Convención de nombre de checkpoint (sin extras):
    sae_{variante}_{tecnica}_l{capa}_{juegos}g_{estado}.pt

Convención de nombre de checkpoint (con extras):
    sae_{variante}_{tecnica}_l{capa}_{juegos}g_{extras_id}_{estado}.pt

El registro de extras se guarda en:
    {checkpoint_dir}/extras.json

Ejemplos:
    Carpeta: layer_06/models/sae_standard/sae_l1_standard/1000games/
    Archivo: sae_standard_l1_l6_1000g_best.pt
    Archivo con extras: sae_standard_l1_l6_1000g_ex1_best.pt
"""

import json
import torch
from pathlib import Path


def get_model_name(variante, tecnica, capa, juegos, estado, extras_id=None):
    """
    Genera el nombre del archivo de checkpoint según convención.

    Args:
        variante:   Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica:    Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:       Número de capa del modelo (int)
        juegos:     Número de partidas de entrenamiento (int)
        estado:     "best" o "final"
        extras_id:  Identificador de extras (ej. "ex1"), opcional

    Returns:
        Nombre del archivo (str)

    Ejemplos:
        >>> get_model_name("standard", "l1", 6, 1000, "best")
        'sae_standard_l1_l6_1000g_best.pt'
        >>> get_model_name("standard", "l1", 6, 1000, "best", extras_id="ex1")
        'sae_standard_l1_l6_1000g_ex1_best.pt'
    """
    if extras_id:
        return f"sae_{variante}_{tecnica}_l{capa}_{juegos}g_{extras_id}_{estado}.pt"
    return f"sae_{variante}_{tecnica}_l{capa}_{juegos}g_{estado}.pt"


def get_checkpoint_dir(base_path, variante, tecnica, capa, juegos):
    """
    Construye el path completo del directorio y crea carpetas faltantes.

    Args:
        base_path: Ruta base donde guardar (str o Path)
        variante:  Arquitectura del SAE (ej. "standard")
        tecnica:   Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:      Número de capa del modelo (int)
        juegos:    Número de partidas de entrenamiento (int)

    Returns:
        Path completo del directorio (Path)

    Ejemplo:
        >>> get_checkpoint_dir("sae/models", "standard", "l1", 6, 1000)
        Path('sae/models/layer_06/models/sae_standard/sae_l1_standard/1000games')
    """
    base_path = Path(base_path)
    checkpoint_dir = (
        base_path
        / f"layer_{capa:02d}"
        / "models"
        / f"sae_{variante}"
        / f"sae_{tecnica}_{variante}"
        / f"{juegos}games"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_extras_id(checkpoint_dir, extras):
    """
    Busca o asigna un identificador ex{n} para el dict de hiperparámetros extras.

    El registro se persiste en extras.json dentro del checkpoint_dir.
    El contador reinicia por combinación variante_tecnica_capa_juegos (es decir,
    por checkpoint_dir), por lo que ex1 puede referirse a experimentos distintos
    en carpetas distintas.

    Args:
        checkpoint_dir: Path del directorio de checkpoints (str o Path)
        extras:         Dict con los hiperparámetros extra del experimento

    Returns:
        Identificador asignado (str), ej. "ex1", "ex2"

    Ejemplo:
        >>> get_extras_id(".../.../1000games", {"expansion": 32, "lambda": 0.02})
        'ex1'   # primera vez
        >>> get_extras_id(".../.../1000games", {"expansion": 16, "lambda": 0.02})
        'ex2'   # nuevos hiperparámetros → siguiente número
        >>> get_extras_id(".../.../1000games", {"expansion": 32, "lambda": 0.02})
        'ex1'   # mismos hiperparámetros → reutiliza
    """
    extras_path = Path(checkpoint_dir) / "extras.json"

    if extras_path.exists():
        with open(extras_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    # Reutilizar si los extras ya existen
    for ex_id, stored in registry.items():
        if stored == extras:
            return ex_id

    # Asignar el siguiente número
    next_n = len(registry) + 1
    ex_id = f"ex{next_n}"
    registry[ex_id] = extras

    with open(extras_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"✓ Nuevo extras registrado: {ex_id} → {extras}")
    return ex_id


def save_checkpoint(model, base_path, variante, tecnica, capa, juegos, estado,
                    config=None, history=None, optimizer=None, extras=None):
    """
    Guarda un checkpoint de SAE con metadata completa.

    Si se pasan extras, busca o asigna un identificador ex{n} en extras.json
    y lo incluye en el nombre del archivo.

    Args:
        model:      Instancia de SparseAutoencoder
        base_path:  Ruta base donde guardar (str o Path)
        variante:   Arquitectura del SAE (ej. "standard")
        tecnica:    Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa:       Número de capa del modelo (int)
        juegos:     Número de partidas de entrenamiento (int)
        estado:     "best" o "final"
        config:     Dict con configuración de entrenamiento (opcional)
        history:    Dict con métricas de entrenamiento (opcional)
        optimizer:  Estado del optimizador (opcional)
        extras:     Dict con hiperparámetros extras para identificar el experimento (opcional)

    Returns:
        Path completo del archivo guardado (Path)

    Ejemplos:
        >>> save_checkpoint(model, "sae/models", "standard", "l1", 6, 1000, "best")
        Path('sae/models/layer_06/models/sae_standard/sae_l1_standard/1000games/sae_standard_l1_l6_1000g_best.pt')
        >>> save_checkpoint(model, "sae/models", "standard", "l1", 6, 1000, "best",
        ...                 extras={"expansion": 32, "lambda": 0.02})
        Path('sae/models/layer_06/models/sae_standard/sae_l1_standard/1000games/sae_standard_l1_l6_1000g_ex1_best.pt')
    """
    checkpoint_dir = get_checkpoint_dir(base_path, variante, tecnica, capa, juegos)

    extras_id = None
    if extras is not None:
        extras_id = get_extras_id(checkpoint_dir, extras)

    filename = get_model_name(variante, tecnica, capa, juegos, estado, extras_id=extras_id)
    checkpoint_path = checkpoint_dir / filename

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config or {},
        'metadata': {
            'variante': variante,
            'tecnica': tecnica,
            'capa': capa,
            'juegos': juegos,
            'estado': estado,
            'extras_id': extras_id,
            'extras': extras,
        }
    }

    if history is not None:
        checkpoint['history'] = history

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint guardado: {checkpoint_path}")

    return checkpoint_path
