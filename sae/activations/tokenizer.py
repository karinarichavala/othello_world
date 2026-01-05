"""
Tokenizador para secuencias de Othello
"""

import torch
from mingpt.dataset import CharDataset


class OthelloTokenizer:
    """
    Convierte secuencias de movimientos raw a tokens del vocabulario del modelo
    """
    
    def __init__(self, othello_data):
        """
        Args:
            othello_data: Instancia de Othello con las secuencias generadas
        """
        # Crear dataset para obtener el mapeo stoi/itos
        self.dataset = CharDataset(othello_data)
        self.stoi = self.dataset.stoi
        self.itos = self.dataset.itos
        self.vocab_size = self.dataset.vocab_size
        self.max_len = self.dataset.max_len
    
    def tokenize_sequence(self, sequence):
        """
        Tokeniza una secuencia individual
        
        Args:
            sequence: Lista de movimientos (valores 0-63)
        
        Returns:
            torch.Tensor: Secuencia tokenizada (sin padding)
        """
        # Convertir movimientos a tokens usando stoi
        tokens = [self.stoi[move] for move in sequence]
        return torch.tensor(tokens, dtype=torch.long)
    
    def tokenize_batch(self, sequences):
        """
        Tokeniza un lote de secuencias
        
        Args:
            sequences: Lista de listas de movimientos
        
        Returns:
            list[torch.Tensor]: Lista de tensores tokenizados
        """
        return [self.tokenize_sequence(seq) for seq in sequences]
    
    def tokenize_all(self):
        """
        Tokeniza todas las secuencias del dataset
        
        Returns:
            list[torch.Tensor]: Lista de todas las secuencias tokenizadas
        """
        # Usar __getitem__ del CharDataset para obtener secuencias procesadas
        return [self.dataset[i][0] for i in range(len(self.dataset))]
    
    def detokenize_sequence(self, tokens):
        """
        Convierte tokens de vuelta a movimientos
        
        Args:
            tokens: torch.Tensor o lista de tokens
        
        Returns:
            list: Lista de movimientos originales
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        moves = []
        for token in tokens:
            move = self.itos[token]
            if move != -100:  # Ignorar padding
                moves.append(move)
        
        return moves
