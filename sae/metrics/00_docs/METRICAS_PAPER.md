# Métricas de Evaluación de SAE - NeurIPS 2024 Paper

Basado en: "Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models" (Karvonen et al., NeurIPS 2024)

---

## 🎯 DEFINICIONES FUNDAMENTALES

### Board State Property (BSP)

**Definición formal:** Una BSP es una función binaria `g: {game board} → {0, 1}`

**Para Othello:**
- **Gboard_state**: 8×8×2 = **128 BSPs**
  - 64 casillas × 2 colores (blanca, negra)
  - Ejemplo: "hay una ficha blanca en e4"
  - Ejemplo: "hay una ficha negra en d5"
  
**Nota importante del paper (página 4, nota al pie 3):**
> "We do not score empty squares"

Esto significa que aunque el tablero tiene 64 casillas, solo evaluamos las **128 BSPs de piezas** (blancas y negras), **NO las casillas vacías**.

---

## 📊 MÉTRICA 1: COVERAGE

### Definición

**Pregunta que responde:** ¿Qué porcentaje de BSPs el SAE puede detectar con alta precisión?

### Función clasificadora

Para cada feature `fi` del SAE y threshold `t ∈ [0,1]`:
```
ϕ_fi,t(x) = I[fi(x) > t · f_max_i]
```

Donde:
- `fi(x)`: activación de la feature i en el input x
- `f_max_i`: máximo valor de fi en todo el dataset
- `I[·]`: función indicadora (1 si cierto, 0 si falso)
- `t`: threshold como fracción del máximo

**Interpretación:** Binariza la activación de la feature en "activa" vs "inactiva"

### Fórmula de Coverage
```
Cov({fi}, G) = (1/|G|) × Σ_{g∈G} max_t max_fi F1(ϕ_fi,t; g)
```

**En palabras:**
1. Para cada BSP g:
   - Buscar la feature fi que mejor la clasifica
   - Buscar el threshold t óptimo
   - Calcular el F1-score máximo
2. Promediar todos los F1-scores máximos

**Tipo de promedio:** Coverage se calcula como **macro-average** sobre BSPs, 
dando peso igual a cada BSP independientemente de su frecuencia en el dataset.

### Implementación

**Thresholds a probar:** `t ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`

**Algoritmo:**
```python
coverage_scores = []

for bsp in bsps_ground_truth:  # 128 BSPs (sin vacías)
    best_f1 = 0
    
    for feature in sae_features:
        f_max = max(feature.activations)  # Máximo en dataset
        
        for threshold in [0, 0.1, 0.2, ..., 0.9]:
            # Binarizar feature
            predictions = (feature.activations > threshold * f_max)
            
            # Calcular F1
            f1 = f1_score(bsp_labels, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
    
    coverage_scores.append(best_f1)

coverage = mean(coverage_scores)
```

### Resultado esperado (Tabla 1, página 8)

**Para Othello:**
- SAE random: 0.27
- **SAE trained: 0.52** ← Objetivo
- Linear probe: 0.99 (upper bound)

---

## 📊 MÉTRICA 2: BOARD RECONSTRUCTION

### Definición

**Pregunta que responde:** ¿Podemos reconstruir el estado del tablero usando las features del SAE?

### Regla de reconstrucción

**Paso 1: Training set (Dtrain - 1000 partidas)**

Para cada feature fi del SAE:
1. Identificar todas las BSPs g para las cuales ϕ_fi,t tiene **precisión ≥ 0.95**
2. Guardar estas asociaciones (feature → BSPs de alta precisión)

**Paso 2: Test set (Dtest - 1000 partidas)**

Para predecir si una BSP g está activa en un tablero:
```
Pg({fi(x)}) = {
    1,  si ϕ_fi,t(x) = 1 para alguna fi que tenga 
        precisión ≥ 0.95 para g en Dtrain
    0,  en caso contrario
}
```

**En palabras:**
- Si alguna feature de "alta confianza" para g se activa → predecir g = True
- Si ninguna se activa → predecir g = False (casilla vacía por defecto)

### Fórmula de Reconstruction
```
Rec({xi}, Dtest) = (1/|Dtest|) × Σ_{x∈Dtest} F1(P({fi(x)}); b)
```

Donde:
- `P({fi(x)})`: predicción completa del tablero basada en features activas
- `b`: tablero real (ground truth)
- `F1(·)`: F1-score calculado **solo sobre casillas con piezas**

**Nota sobre thresholds:** Los thresholds `t` para cada feature se determinan 
en el training set (junto con la identificación de precision ≥ 0.95) y se aplican 
directamente en el test set. NO se re-optimizan en test.

### IMPORTANTE: NO puntuar casillas vacías

**Del paper (página 4, nota al pie 3):**
> "We do not score empty squares"

**Cálculo de F1:**
```python
# Para cada tablero
predicted_pieces = []  # Predicciones de piezas
actual_pieces = []     # Ground truth de piezas

for square in board:
    if actual_board[square] != EMPTY:  # Solo casillas con piezas
        predicted_pieces.append(prediction[square])
        actual_pieces.append(actual_board[square])

# Calcular F1 solo sobre piezas
f1 = f1_score(actual_pieces, predicted_pieces)
```

**Qué cuenta:**
- ✅ True Positive: Predijo pieza correctamente
- ✅ False Positive: Predijo pieza en casilla vacía (SÍ penaliza)
- ✅ False Negative: No detectó pieza que existe
- ❌ True Negative: Casilla vacía correctamente predicha (NO puntúa)

### Resultado esperado (Tabla 1, página 8)

**Para Othello:**
- SAE random: 0.08
- **SAE trained: 0.95** ← Objetivo
- Linear probe: 0.99 (upper bound)

---

## 🔧 DETALLES DE IMPLEMENTACIÓN

### Dataset splits

**Del paper (página 4):**
> "We use a consistent dataset of 1000 games as our training set Dtrain for identifying high-precision features across all Board State Properties (BSPs). An additional, separate set of 1000 games serves as our test set Dtest."

- **Training set:** 1000 partidas (identificar features de alta precisión)
- **Test set:** 1000 partidas (evaluar métricas finales)
- **NO MEZCLAR** entre train y test

### Extracción de activaciones

**Del paper (página 3):**
> "In this paper, we train SAEs on datasets consisting of activations extracted from the residual stream after the sixth layer"

- **Layer:** 6
- **Ubicación:** Post-MLP residual stream
- **Dimensión:** 512
- **Token:** Inmediatamente antes del movimiento de blancas

### Precisión para features de alta confianza

**Del paper (página 4):**
> "high precision (of at least 0.95)"

**Threshold fijo:** Precisión ≥ 0.95

---

## 📈 RESULTADOS COMPLETOS - TABLA 1 (Página 8)

### Chess

| Model | Coverage | Reconstruction |
|-------|----------|----------------|
| SAE random | 0.11 | 0.01 |
| SAE trained | 0.48 | 0.85 |
| Linear probe | 0.98 | 0.98 |

### Othello

| Model | Coverage | Reconstruction |
|-------|----------|----------------|
| SAE random | 0.27 | 0.08 |
| **SAE trained** | **0.52** | **0.95** |
| Linear probe | 0.99 | 0.99 |

---

## 💡 INTERPRETACIÓN DE MÉTRICAS

### Coverage = 0.52

**Significado:** El SAE puede detectar ~52% de las BSPs con alta precisión (F1 alto)

**Qué implica:**
- Para ~52% de las BSPs, existe alguna feature del SAE que actúa como buen clasificador
- Aún hay gap vs linear probes (0.99) → hay margen de mejora

### Reconstruction = 0.95

**Significado:** Usando las features del SAE, podemos reconstruir el tablero con F1=0.95

**Qué implica:**
- Las features capturan suficiente información para inferir dónde están las piezas
- Muy cercano al upper bound de linear probes (0.99)

---

## ⚠️ DIFERENCIAS CON NUESTRO SISTEMA

| Aspecto | Paper | Nuestro sistema |
|---------|-------|-----------------|
| BSPs totales | 128 (implícitas) | 198 (explícitas) |
| Estados por casilla | 2 (blanca, negra) | 3 (vacía, mía, oponente) |
| Vacías explícitas | No | Sí |
| **Coverage métrica principal** | 128 BSPs (solo piezas) | 128 BSPs (solo piezas) |
| **Reconstruction** | NO puntúa vacías | NO puntúa vacías |

**IMPORTANTE:** Aunque nuestro sistema tiene 198 BSPs, las métricas Coverage y Reconstruction se calculan IGUAL que el paper:
- **Coverage-pieces:** Solo 128 BSPs (sin vacías)
- **Reconstruction:** NO puntúa casillas vacías

---

## 📝 PSEUDOCÓDIGO COMPLETO

### Coverage
```python
def calculate_coverage(sae_features, ground_truth_bsps, dataset):
    """
    Args:
        sae_features: features del SAE con sus activaciones
        ground_truth_bsps: 128 BSPs de piezas (sin vacías)
        dataset: activaciones del modelo
    
    Returns:
        coverage_score: float entre 0 y 1
    """
    coverage_scores = []
    
    # Para cada BSP (128 BSPs de piezas)
    for bsp_id, bsp_labels in ground_truth_bsps.items():
        best_f1 = 0
        
        # Para cada feature del SAE
        for feature_id, feature_activations in sae_features.items():
            # Calcular f_max
            f_max = np.max(feature_activations)
            
            # Probar diferentes thresholds
            for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                # Binarizar feature
                predictions = (feature_activations > t * f_max).astype(int)
                
                # Calcular F1
                f1 = f1_score(bsp_labels, predictions)
                
                # Guardar mejor F1
                if f1 > best_f1:
                    best_f1 = f1
        
        coverage_scores.append(best_f1)
    
    # Promediar F1 scores
    coverage = np.mean(coverage_scores)
    
    return coverage
```

### Reconstruction
```python
def calculate_reconstruction(sae_features, ground_truth_bsps, train_set, test_set):
    """
    Args:
        sae_features: features del SAE
        ground_truth_bsps: 128 BSPs de piezas
        train_set: 1000 partidas para identificar features de alta precisión
        test_set: 1000 partidas para evaluar
    
    Returns:
        reconstruction_score: float entre 0 y 1
    """
    # PASO 1: Identificar features de alta precisión en train_set
    high_precision_features = {}  # bsp_id -> [list of (feature_id, threshold)]
    
    for bsp_id in ground_truth_bsps:
        high_precision_features[bsp_id] = []
        
        for feature_id, feature_activations in sae_features.items():
            f_max = np.max(feature_activations)
            
            for t in [0, 0.1, ..., 0.9]:
                predictions = (feature_activations > t * f_max).astype(int)
                
                # Calcular precisión
                precision = precision_score(ground_truth_bsps[bsp_id], predictions)
                
                # Si precisión ≥ 0.95, guardar
                if precision >= 0.95:
                    high_precision_features[bsp_id].append((feature_id, t))
    
    # PASO 2: Reconstruir tableros en test_set
    reconstruction_f1_scores = []
    
    for board in test_set:
        # Predecir BSPs usando regla de reconstrucción
        predicted_bsps = {}
        
        for bsp_id in ground_truth_bsps:
            # Verificar si alguna feature de alta precisión se activa
            activated = False
            
            for (feature_id, threshold) in high_precision_features[bsp_id]:
                feature_value = sae_features[feature_id][board]
                f_max = np.max(sae_features[feature_id])
                
                if feature_value > threshold * f_max:
                    activated = True
                    break
            
            predicted_bsps[bsp_id] = 1 if activated else 0
        
        # Calcular F1 SOLO sobre casillas con piezas
        actual_pieces = []
        predicted_pieces = []
        
        for bsp_id in ground_truth_bsps:
            if ground_truth_bsps[bsp_id][board] == 1:  # Si hay pieza real
                actual_pieces.append(1)
                predicted_pieces.append(predicted_bsps[bsp_id])
            elif predicted_bsps[bsp_id] == 1:  # Falso positivo (predijo pieza en vacía)
                actual_pieces.append(0)
                predicted_pieces.append(1)
            # Si ambos son 0 (vacía) → NO puntúa
        
        # F1 para este tablero
        f1 = f1_score(actual_pieces, predicted_pieces)
        reconstruction_f1_scores.append(f1)
    
    # Promediar F1 sobre test set
    reconstruction = np.mean(reconstruction_f1_scores)
    
    return reconstruction
```

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

### Para Coverage:

- [ ] Extraer activaciones del SAE para dataset
- [ ] Generar BSPs ground truth (128 BSPs, sin vacías)
- [ ] Para cada BSP, encontrar mejor (feature, threshold)
- [ ] Calcular F1 máximo para cada BSP
- [ ] Promediar F1 scores
- [ ] **Objetivo: ~0.52 para Othello**

### Para Reconstruction:

- [ ] Dividir dataset (1000 train, 1000 test)
- [ ] En train: identificar features con precisión ≥ 0.95
- [ ] En test: aplicar regla de reconstrucción
- [ ] Calcular F1 SOLO sobre casillas con piezas
- [ ] Promediar F1 sobre test set
- [ ] **Objetivo: ~0.95 para Othello**

---

**Última actualización:** Febrero 2026
**Fuente:** NeurIPS 2024, https://arxiv.org/abs/2408.00113