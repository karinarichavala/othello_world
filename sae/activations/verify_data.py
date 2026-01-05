"""
Script para verificar archivos .npy de activaciones extraídas
"""

import numpy as np
import sys
from pathlib import Path

def verify_activations(filepath):
    """
    Verifica el contenido de un archivo .npy de activaciones
    """
    if not Path(filepath).exists():
        print(f"❌ Archivo no encontrado: {filepath}")
        return
    
    print(f"\n📊 Verificando: {filepath}")
    print("=" * 70)
    
    # Cargar archivo
    activations = np.load(filepath)
    
    # Información básica
    print(f"\n✓ Archivo cargado exitosamente")
    print(f"  Shape: {activations.shape}")
    print(f"  Dtype: {activations.dtype}")
    print(f"  Tamaño en memoria: {activations.nbytes / 1024 / 1024:.2f} MB")
    
    # Estadísticas
    print(f"\n📈 Estadísticas:")
    print(f"  Min:  {activations.min():.6f}")
    print(f"  Max:  {activations.max():.6f}")
    print(f"  Mean: {activations.mean():.6f}")
    print(f"  Std:  {activations.std():.6f}")
    
    # Muestra de datos
    print(f"\n🔍 Primeras 3 activaciones (primeros 10 valores):")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    for i in range(min(3, len(activations))):
        values = activations[i][:10]
        values_str = ' '.join([f'{v:7.4f}' for v in values])
        print(f"  [{i}]: [{values_str}]")
    
    print("\n✓ Verificación completa\n")

if __name__ == "__main__":
    # Si se proporciona archivo como argumento, usar ese
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Por defecto, usar el más reciente
        filepath = "sae/activations/data/layer5_10games.npy"
    
    verify_activations(filepath)
