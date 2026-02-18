# Evaluación de Sparse Autoencoders para OthelloGPT usando Board State Properties

## 📋 Objetivo del proyecto

Evaluar la calidad de Sparse Autoencoders (SAE) entrenados en OthelloGPT mediante 
métricas objetivas basadas en Board State Properties (BSPs), siguiendo la metodología 
del paper:

**"Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models"**
- Autores: Adam Karvonen, Benjamin Wright, et al.
- Conferencia: NeurIPS 2024
- Paper: https://arxiv.org/abs/2408.00113
- Código: https://github.com/adamkarvonen/SAE_BoardGameEval

---

## 🎯 Resumen ejecutivo

Este proyecto implementa un sistema de evaluación para SAEs que permite medir 
cuantitativamente si el SAE captura conceptos interpretables del estado del tablero 
de Othello que el modelo ha aprendido internamente.

---

## 🧩 Sistema de Board State Properties (BSPs)

### Definición
Una BSP es una función binaria: `g: {game board} → {0, 1}`

### Nuestro sistema: 198 BSPs totales

#### 1. BSPs de casillas (192 BSPs)
Para cada una de las 64 casillas del tablero (A1-H8), tres estados mutuamente excluyentes:

- **BSP{casilla}0**: Casilla está vacía
- **BSP{casilla}1**: Casilla tiene ficha mía (del jugador actual)
- **BSP{casilla}2**: Casilla tiene ficha del oponente

**Ejemplo:**
- BSPA10 = True → A1 está vacía
- BSPA11 = False → A1 NO es mía
- BSPA12 = False → A1 NO es del oponente

**Propiedad importante:** Cada casilla tiene EXACTAMENTE 1 BSP activa (True)

**Total:** 64 casillas × 3 estados = 192 BSPs

#### 2. BSPs especiales (6 BSPs)

- **BSP_INICIAL**: Tablero en configuración inicial (4 fichas centrales)
- **BSP_TERMINADA**: Partida terminada (64 fichas o sin movimientos)
- **BSP_GANE**: Posición ganada (más fichas que oponente al final)
- **BSP_PERDI**: Posición perdida (menos fichas al final)
- **BSP_EMPATE**: Posición empatada (mismo número de fichas)
- **BSP_EN_CURSO**: Juego en curso (no terminado)

---

## 📊 Métricas de evaluación

**IMPORTANTE:** Coverage y Reconstruction son métricas **independientes** que miden aspectos diferentes:
- **Coverage** mide la detectabilidad de conceptos individuales
- **Reconstruction** mide el uso conjunto de features para reconstruir tableros completos

---

### 1. Coverage (Cobertura)

**Pregunta:** ¿Qué porcentaje de BSPs el SAE puede detectar con alta precisión?

**Dos variantes:**

#### A) Coverage-all (diagnóstico)
- Incluye: 192 BSPs de casillas (vacías + piezas)
- Uso: Sanity check interno, verificar que el SAE modela el tablero completo
- **⚠️ NO COMPARABLE CON EL PAPER** - Es solo diagnóstico interno
- **NO usar para comparar con Tabla 1** del paper

#### B) Coverage-pieces (MÉTRICA PRINCIPAL) ⭐
- Incluye: 128 BSPs (solo piezas, SIN vacías)
  * 64 BSPs de "casilla X es mía"
  * 64 BSPs de "casilla X es del oponente"
- **Esta es la métrica comparable con el paper**
- Evita inflación artificial por casillas vacías
- Refleja interpretabilidad real de piezas

**Fórmula:**
```
Coverage(SAE, G) = (1/|G|) × Σ_{g∈G} max_t max_fi F1(ϕ_fi,t; g)

Para cada BSP:
  1. Encontrar la feature del SAE que mejor la clasifica
  2. Encontrar el threshold óptimo
  3. Calcular F1-score máximo
  4. Promediar (macro-average) sobre todas las BSPs
```

**IMPORTANTE sobre el promedio:**
- Se usa **macro-average** sobre BSPs
- Cada BSP pesa igual, independientemente de su frecuencia en el dataset
- Esto evita que BSPs más comunes dominen la métrica

**Threshold:** Probar t ∈ {0, 0.1, 0.2, ..., 0.9} de la activación máxima de cada feature

---

### 2. Board Reconstruction (Reconstrucción)

**Pregunta:** ¿Podemos reconstruir el estado completo del tablero usando las features del SAE?

**Regla de reconstrucción:**

**PASO 1 - Training set (identificar features de alta precisión):**
1. Usar **solo el training set** (1000 partidas)
2. Para cada feature del SAE, identificar BSPs donde tiene **precision ≥ 0.95**
3. **⚠️ IMPORTANTE:** El umbral 0.95 se aplica a **PRECISION**, no a F1
4. Guardar estas asociaciones (feature → BSPs de alta precisión)

**PASO 2 - Test set (evaluar reconstrucción):**
1. Usar **solo el test set** (1000 partidas, diferentes del train)
2. Para cada tablero:
   - Si alguna feature de alta precisión se activa → predecir esa BSP
   - Si ninguna se activa → asumir casilla vacía (estado por defecto)
3. Calcular F1 **solo sobre casillas con piezas** (NO puntuar vacías)
4. Promediar F1 sobre todos los tableros del test set

**IMPORTANTE: NO puntuar casillas vacías** (siguiendo al paper)

**Razón:** "Vacía" es el estado por defecto. Lo interesante es reconstruir DÓNDE 
están las piezas.

**Cálculo de F1:**
- Solo sobre casillas con piezas (mías o del oponente)
- True Positive: Pieza correctamente predicha
- False Positive: Predice pieza en casilla vacía (SÍ penaliza)
- False Negative: No detecta pieza que existe
- True Negative de vacías: IGNORADOS (no puntúan)

**Fórmula:**
```
Reconstruction = (1/|Dtest|) × Σ_{x∈Dtest} F1(predicción_SAE, tablero_real)

Donde:
- Features de alta precisión se identifican EN TRAIN SET
- F1 se calcula EN TEST SET
- F1 se calcula solo sobre casillas con piezas
```

---

## 🔬 Diferencias con el paper

### Nuestro enfoque vs Paper:

| Aspecto | Paper | Nuestro enfoque |
|---------|-------|-----------------|
| BSPs de casillas | 128 (2 por casilla) | 192 (3 por casilla) |
| Estados por casilla | Blanca, Negra | Vacía, Mía, Oponente |
| Vacías explícitas | No (implícitas) | Sí (explícitas) |
| **Coverage principal** | Solo piezas (128) | Coverage-pieces (128) |
| **Coverage-all** | No existe | Solo diagnóstico (NO comparable) |
| Reconstruction | Sin puntuar vacías | Sin puntuar vacías |
| Promedio Coverage | Macro-average | Macro-average |

**Conclusión:** Nuestro sistema es MÁS COMPLETO (incluye vacías explícitas), 
pero las métricas **Coverage-pieces** y **Reconstruction** se implementan EXACTAMENTE 
igual que el paper para comparabilidad directa.

---

## 📁 Archivos del proyecto

### Ubicaciones clave:
```
othello_world/
├── ckpt/
│   └── gpt_championship.ckpt
├── sae/
│   ├── activations/
│   ├── metrics/
    ── CONTEXTO_BSP.md
    └── METRICAS_PAPER.md
│   ├── model/
│   ├── bsp_identifier.py           (Generador de BSPs - script limpio)
│   └── __init__.py
├── notebooks/
│   ├── board_state_properties.ipynb      (Ejemplos y documentación)
│   └── evaluar_sae_con_bsps.ipynb       (Evaluación - por crear)

```

---

## 📝 Descripción de archivos clave

### `sae/bsp_identifier.py`
**Propósito:** Script limpio de producción para generar BSPs ground truth

**Funciones principales:**
- `identificador(tablero, color_jugador)` → Genera las 198 BSPs para un tablero
- `imprimir_tablero(tablero)` → Visualización del tablero
- `imprimir_resumen(bsps)` → Resumen de BSPs activas
- `imprimir_bsps_activas_con_fichas(bsps)` → Solo piezas (sin vacías)
- `imprimir_todas_bsps_activas(bsps)` → Todas las BSPs incluyendo vacías
- `verificar_consistencia(bsps)` → Verifica propiedad one-hot
- `obtener_bsps_solo_piezas(bsps)` → Extrae 128 BSPs para Coverage-pieces ⭐
- `contar_bsps_activas(bsps)` → Conteos por categoría

**Uso:**
```python
from sae.bsp_identifier import identificador, obtener_bsps_solo_piezas

# Generar BSPs completas (198)
bsps = identificador(tablero, color_jugador=1)

# Extraer solo BSPs de piezas (128) para métrica Coverage-pieces
bsps_piezas = obtener_bsps_solo_piezas(bsps)
```

---

### `notebooks/board_state_properties.ipynb`
**Propósito:** Documentación viva y ejemplos interactivos del sistema de BSPs

**Contenido:**
- Explicación detallada del sistema de 198 BSPs
- Ejemplos visuales de tableros:
  * Tablero inicial (4 fichas centrales)
  * Tablero medio juego (múltiples piezas)
  * Tablero terminado (64 fichas)
- Visualizaciones interactivas con `imprimir_tablero()`
- Tests de consistencia (verificar one-hot encoding)
- Demostración de funciones auxiliares
- Documentación del diseño y decisiones tomadas

**Uso:** Referencia para entender cómo funcionan las BSPs y ver ejemplos prácticos

**Nota:** Este notebook NO se usa en el pipeline de evaluación. Es solo documentación.

---

### `notebooks/evaluar_sae_con_bsps.ipynb` (Por crear)
**Propósito:** Pipeline completo de evaluación del SAE

**Estructura esperada:**
1. Setup: Cargar modelo OthelloGPT, SAE, y dataset
2. Extracción de activaciones (layer 6, post-MLP)
3. Generación de BSPs ground truth con `identificador()`
4. Implementación de Coverage-pieces (macro-average sobre 128 BSPs)
5. Implementación de Board Reconstruction (train/test split)
6. Comparación con resultados del paper (Tabla 1)
7. Análisis y visualizaciones

---

## 🎮 Sobre el modelo OthelloGPT

### Arquitectura:
- 8-layer GPT
- 8 attention heads
- 512 dimensional hidden space
- Entrenado en 20M partidas sintéticas

### Características:
- Predice movimientos legales con alta precisión
- Tiene representación interna del estado del tablero
- Extraíble con linear probes
- Paper original: Li et al. 2021

### Activaciones a usar:
- **Layer 6, post-MLP residual stream**
- Por qué: Aquí los linear probes funcionan mejor
- Dimensión: 512

---

## 🤖 Sobre el SAE

### Archivo disponible:
- **Ubicación:** `saes/othello_layer_6_postMLP.pt`
- **Layer:** 6, post-MLP
- **Tamaño:** 0.8 MB

### Arquitectura SAE:
```
Input: x ∈ R^512 (activación del modelo)
Encoder: f(x) = ReLU(W_e(x - b_d) + b_e)
Decoder: x̂ = W_d f(x) + b_d

donde:
- f(x): feature activations (sparse)
- x̂: reconstrucción de x
- W_d: columnas normalizadas (L2 = 1)
```

---

## 📈 Resultados esperados (del paper)

### Para Othello (Tabla 1, pág. 8):

| Modelo | Coverage | Reconstruction |
|--------|----------|----------------|
| SAE random (baseline) | 0.27 | 0.08 |
| **SAE trained** | **0.52** | **0.95** |
| Linear probe (upper bound) | 0.99 | 0.99 |

**Interpretación:**
- Coverage 0.52: El SAE detecta ~52% de las BSPs con alta precisión (macro-average F1)
- Reconstruction 0.95: Puede reconstruir tablero con F1=0.95
- Aún hay gap vs linear probes (0.99)

**⚠️ IMPORTANTE:** Estos valores son para **Coverage-pieces** (128 BSPs, sin vacías).
**NO** comparar con Coverage-all.

---

## ⚙️ Detalles técnicos importantes

### 1. Representación "mine vs theirs"
El modelo aprende representación relativa al jugador actual:
- NO aprende "pieza blanca" vs "pieza negra" en absoluto
- SÍ aprende "mi pieza" vs "pieza del oponente"
- Por eso nuestras BSPs son "mía" y "oponente", no "blanca" y "negra"

### 2. Token de extracción
- Extraer activaciones del token **inmediatamente antes del movimiento de blancas**
- Así evitamos ambigüedad de perspectiva

### 3. Dataset splits
- **Training set:** 1000 partidas (para identificar features con **precision** ≥ 0.95)
- **Test set:** 1000 partidas (para evaluar métricas finales)
- **NO MEZCLAR** entre train y test
- Train se usa SOLO en Reconstruction (paso 1)
- Test se usa en Reconstruction (paso 2) y puede usarse en Coverage

### 4. Threshold para alta precisión
- **Precision** ≥ 0.95 para considerar feature como "alta precisión"
- **NO es F1 ≥ 0.95**, es **precision** específicamente
- Esto está en el paper (pág. 4)

### 5. Tipo de promedio
- Coverage usa **macro-average**: cada BSP pesa igual
- NO usar micro-average (se inflaría con BSPs frecuentes)

---

## 🚀 Plan de implementación

### Fase 1: Setup y pipeline básico
1. Cargar modelo OthelloGPT desde `ckpt/othello_synthetic_dataset.ckpt`
2. Cargar SAE desde `saes/othello_layer_6_postMLP.pt`
3. Cargar dataset de `data/othello_synthetic/train.txt` y `test.txt`
4. Verificar que todo funciona

### Fase 2: Pipeline de datos
1. Extraer activaciones del modelo (layer 6, post-MLP)
2. Pasar activaciones por el SAE
3. Para cada partida, generar ground truth con `identificador()`
4. Separar: 1000 train, 1000 test (NO mezclar)
5. Almacenar: (activaciones_sae, bsps_ground_truth)

### Fase 3: Calcular Coverage-pieces
1. Usar solo las 128 BSPs de piezas (sin vacías)
2. Para cada BSP:
   - Para cada feature del SAE:
     - Para cada threshold t ∈ {0, 0.1, ..., 0.9}:
       - Calcular F1-score
   - Guardar mejor F1 para esa BSP
3. **Macro-average:** Promediar F1 scores (cada BSP pesa igual)
4. Comparar con 0.52 del paper

### Fase 4: Calcular Reconstruction
1. **En training set (1000 partidas):**
   - Identificar features con **precision** ≥ 0.95 para cada BSP
   - Guardar asociaciones (feature → BSPs)

2. **En test set (1000 partidas):**
   - Para cada tablero:
     - Ver qué features se activan
     - Reconstruir tablero según regla
     - Calcular F1 solo sobre casillas con piezas
   - Promediar F1 sobre test set

3. Comparar con 0.95 del paper

### Fase 5: Análisis y visualización
1. Comparar Coverage-pieces y Reconstruction con paper
2. (Opcional) Calcular Coverage-all como diagnóstico interno
3. Analizar qué BSPs el SAE detecta mejor/peor
4. Visualizar features más interpretables
5. Generar gráficas para la tesis

---

## 📚 Referencias clave

### Paper principal:
- **Título:** Measuring Progress in Dictionary Learning for Language Model 
  Interpretability with Board Game Models
- **URL:** https://arxiv.org/abs/2408.00113
- **Código:** https://github.com/adamkarvonen/SAE_BoardGameEval

### Modelo OthelloGPT:
- **Paper:** Li et al. (2021) "Actually, Othello-GPT Has A Linear Emergent World Representation"
- **URL:** https://www.neelnanda.io/mechanistic-interpretability/othello
- **Análisis:** Nanda et al. "Actually, Othello-GPT Has A Linear Emergent World 
  Representation"


---

## ⚠️ Decisiones de diseño confirmadas

### 1. Incluir BSPs de casillas vacías
✅ **SÍ incluir en la definición** (sistema completo de 198 BSPs)
✅ **SÍ incluir en Coverage-all** (solo diagnóstico interno, NO comparable)
❌ **NO incluir en Coverage-pieces** (métrica principal comparable con paper)
❌ **NO puntuar en Reconstruction**

**Razón:** Sistema completo para diagnóstico + métricas comparables con el paper

### 2. Métrica principal
⭐ **Coverage-pieces (128 BSPs, sin vacías) es la métrica principal**
- Macro-average sobre BSPs
- Comparable con el paper (Tabla 1: 0.52)
- Evita inflación artificial
- Refleja interpretabilidad real

### 3. Coverage-all
⚠️ **Solo para diagnóstico interno**
- NO comparar numéricamente con el paper
- Sirve para verificar que el SAE modela el tablero completo
- NO usar como métrica de evaluación

### 4. Perspectiva del jugador
✅ Usar "mía" vs "oponente" (relativo al turno)
❌ NO usar "blanca" vs "negra" (absoluto)

**Razón:** El modelo aprende representación relativa al jugador

### 5. Precisión vs F1
✅ Reconstruction usa **precision** ≥ 0.95, NO F1
✅ Coverage usa **F1** para evaluar features
- No confundir ambos

---


## 💬 Notas finales

- Este archivo documenta TODAS las decisiones de diseño tomadas
- Usar junto con METRICAS_PAPER.md para detalles técnicos de implementación
- Para dudas conceptuales, referirse a la conversación original en claude.ai
- Este documento es la fuente de verdad del proyecto
- **Puntos clave técnicos:**
  * Coverage: macro-average, cada BSP pesa igual
  * Coverage-all: solo diagnóstico, NO comparable con paper
  * Reconstruction: precision ≥ 0.95 (no F1), identificada en train, evaluada en test
  * Coverage y Reconstruction: métricas independientes

---

**Última actualización:** Febrero 2026
**Autora:** Karina
**Curso:** Interactive Games / Tesis
**Institución:** Escuela Politécnica Nacional (EPN)