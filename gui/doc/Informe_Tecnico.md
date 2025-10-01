# Informe Técnico - Modelo GPT para Othello

## 1. Arquitectura del Modelo

### Configuración GPT
- **Tipo**: Transformer Decoder (arquitectura GPT) <!-- `mingpt/model.py` clase GPT -->
- **Vocabulario**: 61 tokens (60 posiciones del tablero + "pass") <!-- `train_gpt_othello.ipynb` GPTConfig -->
- **Contexto máximo**: 59 movimientos <!-- `train_gpt_othello.ipynb` block_size -->
- **Capas Transformer**: 8 bloques <!-- `train_gpt_othello.ipynb` n_layer=8 -->
- **Cabezas de atención**: 8 por bloque (paralelismo) <!-- `train_gpt_othello.ipynb` n_head=8 -->
- **Dimensión embeddings**: 512D <!-- `train_gpt_othello.ipynb` n_embd=512 -->
- **Dimensión MLP**: 2048D (expansión 4x) <!-- `mingpt/model.py` clase MLP -->

### Detalles Arquitectónicos

```python
# Configuración extraída de train_gpt_othello.ipynb
# mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, 
#                   n_layer=8, n_head=8, n_embd=512)
GPTConfig(
    vocab_size=61,      # Tamaño del vocabulario
    block_size=59,      # Secuencia máxima
    n_layer=8,          # 8 capas Transformer
    n_head=8,           # 8 cabezas de atención
    n_embd=512          # 512 dimensiones
)
```

### Flujo de Datos

```
Input: [move1, move2, ..., moveN] (máx 59 movimientos)
         ↓
Token Embedding: 61 → 512D
         ↓
Positional Encoding: +512D
         ↓
8x Transformer Blocks:
  ┌─ LayerNorm
  ├─ Multi-Head Attention (8 cabezas x 64D)
  ├─ Residual Connection
  ├─ LayerNorm  
  ├─ MLP (512→2048→512)
  └─ Residual Connection
         ↓
LayerNorm Final
         ↓
Linear Head: 512 → 61 logits
         ↓
Softmax → Probabilidades para cada posición
```

### Atención Multi-Cabeza
- **8 cabezas en paralelo** por capa
- Cada cabeza: 64 dimensiones (512/8)
- **Atención causal**: Solo ve movimientos anteriores

## 2. Entrenamiento del Modelo

### Dataset
- **Fuente**: Partidas sintéticas de Othello generadas algorítmicamente <!-- `train_gpt_othello.ipynb` synthetic_or_championship = True -->
- **Formato**: Secuencias de movimientos tokenizados <!-- `mingpt/dataset.py` CharDataset -->
- **Preprocesamiento**: Conversión tablero 8x8 → tokens del modelo <!-- `data/othello.py` funciones de mapeo -->

### Mapeo Tablero → Tokens
- **Tablero original**: 64 casillas (8x8)
- **Modelo**: 61 tokens
- **Exclusiones**: Casillas centrales iniciales (D4, D5, E4, E5)
- **Token 0**: Movimiento "pass"
- **Tokens 1-60**: Posiciones válidas

### Función de Pérdida
- **Cross-entropy loss** para predicción del siguiente movimiento <!-- `mingpt/model.py` línea 260+ F.cross_entropy -->
- **Objetivo**: Maximizar probabilidad del movimiento real <!-- `mingpt/trainer.py` función run_epoch -->
- **Optimización**: AdamW optimizer <!-- `mingpt/model.py` línea 176+ torch.optim.AdamW -->

### Proceso de Entrenamiento <!-- `mingpt/trainer.py` función run_epoch -->
1. **Input**: Secuencia de N movimientos <!-- x = x.to(self.device) -->
2. **Target**: Movimiento N+1 <!-- y = y.to(self.device) -->
3. **Forward pass**: Generar probabilidades <!-- logits, loss = model(x, y) -->
4. **Loss**: Cross-entropy entre predicción y target <!-- loss.mean() -->
5. **Backprop**: Actualizar pesos <!-- loss.backward() -->

## 3. Pesos y Checkpoint

### Archivo de Modelo
- **Ubicación**: `ckpts/gpt_championship.ckpt` <!-- `train_probe_othello.py` load_state_dict -->

### Inicialización de Pesos
```python
# Extraído de mingpt/model.py función _init_weights
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

### Estructura de Parámetros
- **Token embeddings**: 61 × 512 = 31,232 params
- **Position embeddings**: 59 × 512 = 30,208 params
- **8 Transformer blocks**: ~25M params
- **Output head**: 512 × 61 = 31,232 params
- **Total aproximado**: ~25M parámetros

## 4. Características del Modelo Entrenado

### Capacidades Aprendidas
- **Reconocimiento de patrones**: Identifica configuraciones favorables
- **Estrategia posicional**: Prioriza esquinas y bordes estables
- **Planificación a futuro**: Considera consecuencias de movimientos
- **Evaluación de intercambios**: Calcula ganancias/pérdidas de fichas

### Comportamiento Observado
- **Probabilidades altas**: Movimientos estratégicamente sólidos
- **Probabilidades bajas**: Movimientos arriesgados o subóptimos
- **Distribución coherente**: Refleja comprensión del juego

## 5. Implementación Técnica


### Optimizaciones
- **Evaluación en CPU**: Inferencia rápida sin GPU <!-- `gui/model_handler.py` device configuración -->
- **Batch size 1**: Procesamiento de una partida a la vez <!-- `train_probe_othello.py` batch_size=1 -->
- **Cache de atención**: Reutilización para eficiencia <!-- `mingpt/model.py` implementación Transformer -->

## 6. Métricas del Modelo

### Parámetros por Componente
- **Embeddings**: ~61K parámetros <!-- `mingpt/model.py` tok_emb + pos_emb -->
- **8 Capas Transformer**: ~25M parámetros <!-- `mingpt/model.py` 8x Block calculado -->
- **Output Layer**: ~31K parámetros <!-- `mingpt/model.py` head Linear layer -->
- **Total**: ~25.1M parámetros <!-- Calculado basado en arquitectura GPTConfig -->
---
