"""
Utilidades para nombrar y guardar checkpoints de SAEs entrenados sobre OthelloGPT

Convención de carpetas:
    {base_path}/layer_{capa:02d}/models/sae_{variante}/sae_{tecnica}_{variante}/

Convención de nombre de checkpoint:
    sae_{variante}_{tecnica}_l{capa}_{juegos}g_{estado}.pt

Ejemplos:
    Carpeta: layer_06/models/sae_standard/sae_l1_standard/
    Archivo: sae_standard_l1_l6_200g_best.pt
"""

import os
import torch
from pathlib import Path


def get_model_name(variante, tecnica, capa, juegos, estado):
    """
    Genera el nombre del archivo de checkpoint según convención.
    
    Args:
        variante: Arquitectura del SAE (ej. "standard", "topk", "gated")
        tecnica: Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa: Número de capa del modelo (int)
        juegos: Número de partidas de entrenamiento (int)
        estado: "best" o "final"
    
    Returns:
        Nombre del archivo (str)
    
    Ejemplo:
        >>> get_model_name("standard", "l1", 6, 200, "best")
        'sae_standard_l1_l6_200g_best.pt'
    """
    return f"sae_{variante}_{tecnica}_l{capa}_{juegos}g_{estado}.pt"


def get_checkpoint_dir(base_path, variante, tecnica, capa):
    """
    Construye el path completo del directorio y crea carpetas faltantes.
    
    Args:
        base_path: Ruta base donde guardar (str o Path)
        variante: Arquitectura del SAE (ej. "standard")
        tecnica: Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa: Número de capa del modelo (int)
    
    Returns:
        Path completo del directorio (Path)
    
    Ejemplo:
        >>> get_checkpoint_dir("sae/models", "standard", "l1", 6)
        Path('sae/models/layer_06/models/sae_standard/sae_l1_standard')
    """
    base_path = Path(base_path)
    layer_dir = f"layer_{capa:02d}"
    model_arch = f"sae_{variante}"
    model_tech = f"sae_{tecnica}_{variante}"
    
    checkpoint_dir = base_path / layer_dir / "models" / model_arch / model_tech
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    return checkpoint_dir


def save_checkpoint(model, base_path, variante, tecnica, capa, juegos, estado, 
                   config=None, history=None, optimizer=None):
    """
    Guarda un checkpoint de SAE con metadata completa.
    
    Args:
        model: Instancia de SparseAutoencoder
        base_path: Ruta base donde guardar (str o Path)
        variante: Arquitectura del SAE (ej. "standard")
        tecnica: Técnica de sparsity (ej. "l1", "l1-p-annealing")
        capa: Número de capa del modelo (int)
        juegos: Número de partidas de entrenamiento (int)
        estado: "best" o "final"
        config: Dict con configuración de entrenamiento (opcional)
        history: Dict con métricas de entrenamiento (opcional)
        optimizer: Estado del optimizador (opcional)
    
    Returns:
        Path completo del archivo guardado (Path)
    
    Ejemplo:
        >>> save_checkpoint(model, "sae/models", "standard", "l1", 6, 200, "best",
        ...                 config={'lr': 1e-4}, history={'loss': [0.1, 0.05]})
        Path('sae/models/layer_06/models/sae_standard/sae_l1_standard/sae_standard_l1_l6_200g_best.pt')
    """
    checkpoint_dir = get_checkpoint_dir(base_path, variante, tecnica, capa)
    filename = get_model_name(variante, tecnica, capa, juegos, estado)
    checkpoint_path = checkpoint_dir / filename
    
    # Construir checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config or {},
        'metadata': {
            'variante': variante,
            'tecnica': tecnica,
            'capa': capa,
            'juegos': juegos,
            'estado': estado,
        }
    }
    
    if history is not None:
        checkpoint['history'] = history
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Guardar
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint guardado: {checkpoint_path}")
    
    return checkpoint_path
