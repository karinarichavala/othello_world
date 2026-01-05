"""
Script principal para extraer activaciones del modelo Othello-GPT
"""

import os
import sys
import torch
import numpy as np

# Agregar el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from sae.activations.config import ExtractionConfig
from sae.activations.extractor import ActivationExtractor
from sae.activations.batch_processor import BatchProcessor
from sae.activations.tokenizer import OthelloTokenizer
from mingpt.model import GPT, GPTConfig
from data.othello import Othello


def load_gpt_model(checkpoint_path, device='cpu'):
    """
    Carga el modelo Othello-GPT desde un checkpoint
    
    Args:
        checkpoint_path: Ruta al checkpoint (.ckpt)
        device: 'cuda' o 'cpu'
    
    Returns:
        model: Modelo GPT cargado en modo evaluación
    """
    print(f"\n Cargando modelo GPT desde: {checkpoint_path}")
    
    # Configuración del modelo
    model_config = GPTConfig(
        vocab_size=61,
        block_size=59,
        n_layer=8,
        n_head=8,
        n_embd=512
    )
    
    # Crear modelo
    model = GPT(model_config)
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Modo evaluación
    model.eval()
    model.to(device)
    
    print(f" Modelo cargado en {device}")
    return model


def extract_from_memory(config, model):
    """
    Ruta rápida: Generar partidas y procesar en memoria
    Para num_games <= 1000
    """
    print("\n MODO RÁPIDO: Procesamiento en memoria")
    print(f" Generando {config.num_games} partidas sintéticas...")
    
    # Generar partidas
    othello_data = Othello(ood_num=config.num_games)
    print(f"✓ {len(othello_data.sequences)} partidas generadas")
    
    # Tokenizar secuencias
    print(f"🔤 Tokenizando secuencias...")
    tokenizer = OthelloTokenizer(othello_data)
    tokenized_sequences = tokenizer.tokenize_all()
    print(f"✓ {len(tokenized_sequences)} secuencias tokenizadas")
    
    # Crear extractor y extraer activaciones
    extractor = ActivationExtractor(model, config.target_layer, config.device)
    print(f"\n Extrayendo activaciones del Bloque {config.target_layer + 1}...")
    activations = extractor.extract_from_games(tokenized_sequences)
    
    return activations


def extract_from_file(config, model):
    """
    Ruta segura: Generar → Buscar archivo → Procesar por lotes
    Para num_games > 1000
    """
    print("\n MODO SEGURO: Usando archivo + procesamiento por lotes")
    
    # Crear carpeta si no existe
    os.makedirs("data/othello_synthetic", exist_ok=True)
    
    # Generar partidas (se guarda automáticamente)
    print(f" Generando {config.num_games} partidas sintéticas...")
    othello_data = Othello(ood_num=config.num_games)
    print(f"✓ {len(othello_data.sequences)} partidas generadas y guardadas")
    
    # Buscar el archivo más reciente
    batch_processor = BatchProcessor(None, config.batch_size_games)
    games_file = batch_processor.find_latest_games_file()
    
    if not games_file:
        raise FileNotFoundError(
            "No se encontró el archivo de partidas generado en data/othello_synthetic/. "
            "Verifica que la carpeta exista y tenga permisos de escritura."
        )
    
    print(f" Archivo encontrado: {games_file}")
    
    # Cargar partidas desde archivo
    games = batch_processor.load_games_from_file(games_file)
    
    # Tokenizar secuencias
    print(f" Tokenizando {len(games)} secuencias...")
    tokenizer = OthelloTokenizer(othello_data)
    tokenized_games = tokenizer.tokenize_all()
    print(f"✓ {len(tokenized_games)} secuencias tokenizadas")
    
    # Crear extractor y procesar por lotes
    extractor = ActivationExtractor(model, config.target_layer, config.device)
    batch_processor.extractor = extractor
    
    print(f"\n Extrayendo activaciones del Bloque {config.target_layer + 1}...")
    activations = batch_processor.process_in_batches(tokenized_games)
    
    return activations


def main():
    """Script principal"""
    
    print("=" * 70)
    print("EXTRACCIÓN DE ACTIVACIONES - Othello-GPT SAE")
    print("=" * 70)
    
    # Cargar configuración
    config = ExtractionConfig()
    print(f"\n📋 Configuración:")
    print(f"  - Número de partidas: {config.num_games}")
    print(f"  - Capa objetivo: Bloque {config.target_layer + 1} (índice {config.target_layer})")
    print(f"  - Device: {config.device}")
    print(f"  - Estrategia: {'MEMORIA' if config.num_games <= config.memory_threshold else 'ARCHIVO + LOTES'}")
    
    # Cargar modelo GPT
    model = load_gpt_model(config.checkpoint_path, config.device)
    
    # Decidir estrategia según número de partidas
    if config.num_games <= config.memory_threshold:
        activations = extract_from_memory(config, model)
    else:
        activations = extract_from_file(config, model)
    
    # Estadísticas
    print("\n Estadísticas de activaciones:")
    print(f"  - Shape: {activations.shape}")
    print(f"  - Total de tokens: {activations.shape[0]:,}")
    print(f"  - Dimensión: {activations.shape[1]}")
    print(f"  - Media: {activations.mean():.6f}")
    print(f"  - Desv. estándar: {activations.std():.6f}")
    print(f"  - Mínimo: {activations.min():.6f}")
    print(f"  - Máximo: {activations.max():.6f}")
    
    # Guardar activaciones
    output_path = config.get_activations_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, activations)
    
    print(f"\n Activaciones guardadas en: {output_path}")
    print(f" Tamaño del archivo: {os.path.getsize(output_path) / 1e6:.2f} MB")
    
    print("\n ¡Extracción completada exitosamente!")
    print("\n Próximo paso: Entrenar el SAE con estas activaciones")
    print(f"   python sae/training/run_training.py")


if __name__ == "__main__":
    main()
