"""
Procesamiento por lotes de partidas desde archivo
"""

import pickle
import numpy as np
import glob
import os
from tqdm import tqdm


class BatchProcessor:
    """
    Procesa partidas por lotes para no llenar la RAM
    """
    
    def __init__(self, extractor, batch_size=100):
        """
        Args:
            extractor: Instancia de ActivationExtractor
            batch_size: Número de partidas a procesar por lote
        """
        self.extractor = extractor
        self.batch_size = batch_size
    
    def find_latest_games_file(self):
        """
        Encuentra el archivo .pickle más reciente en data/othello_synthetic/
        
        Returns:
            str: Ruta al archivo más reciente, o None si no encuentra
        """
        pattern = "data/othello_synthetic/gen10e5_*.pickle"
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Retornar el más reciente por fecha de creación
        return max(files, key=os.path.getctime)
    
    def load_games_from_file(self, file_path):
        """
        Carga partidas desde un archivo pickle
        
        Args:
            file_path: Ruta al archivo .pickle
        
        Returns:
            list: Lista de partidas
        """
        print(f"📂 Cargando partidas desde: {file_path}")
        with open(file_path, 'rb') as f:
            games = pickle.load(f)
        print(f"✓ {len(games)} partidas cargadas")
        return games
    
    def process_in_batches(self, games):
        """
        Procesa partidas por lotes para evitar llenar la RAM
        
        Args:
            games: Lista de partidas
        
        Returns:
            np.ndarray: Array con todas las activaciones (N, hidden_dim)
        """
        all_activations = []
        num_batches = (len(games) + self.batch_size - 1) // self.batch_size
        
        print(f"\n🔄 Procesando {len(games)} partidas en {num_batches} lotes de {self.batch_size}")
        
        for i in range(0, len(games), self.batch_size):
            batch_games = games[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            # Extraer activaciones del lote
            desc = f"Lote {batch_num}/{num_batches}"
            batch_acts = self.extractor.extract_from_games(batch_games, desc=desc)
            
            if len(batch_acts) > 0:
                all_activations.append(batch_acts)
            
            # Liberar memoria
            del batch_acts
        
        # Concatenar todos los lotes
        if all_activations:
            print("\n📊 Concatenando activaciones...")
            final_activations = np.vstack(all_activations)
            return final_activations
        else:
            return np.array([])
