# SAE(Sparse Autoencoders) para OthelloGPT

Este módulo contiene el pipeline completo para entrenar y evaluar Sparse Autoencoders (SAEs) sobre las activaciones internas de Othello-GPT (`gpt_championship.ckpt`), con el objetivo de identificar features interpretables relacionadas con el estado del tablero.

## Estructura

```
sae/
├── activations/    # Extracción de activaciones del GPT
├── models/         # Implementaciones y notebooks de entrenamiento de cada variante de SAE
├── experiments/    # Resultados de entrenamiento (métricas, plots, reportes) por experimento
├── metrics/        # Preparación de datos y notebooks de evaluación (reconstruction, coverage)
└── tools/          # Utilidades compartidas (BSPs, naming de experimentos)
```

### `activations/`
Extrae y guarda las activaciones de una capa del GPT sobre partidas sintéticas de Othello.

- `config.py` — `ExtractionConfig`: capa objetivo, número de partidas, checkpoint, rutas.
- `extractor.py`, `batch_processor.py` — extracción de activaciones (con soporte batch para datasets grandes).
- `tokenizer.py`, `game_utils.py` — tokenización de partidas y utilidades de generación/guardado.
- `run_extraction.py` — script principal, carga el GPT y ejecuta la extracción end-to-end.
- `data/` — activaciones y partidas ya generadas (`games_*.txt`, `layer*_*games.npy`).

### `models/`
Cada variante de SAE vive en su propio subdirectorio con un notebook de entrenamiento:

- `sae_standard/sae_pannealing/` — SAE estándar con p-annealing.
- `sae_topk/sae_batch-topk/` — SAE con Batch TopK.
- `sae_matryoska/sae_matryoska_batch-topk/` — Matryoshka SAE (Batch TopK anidado).
- `sae.py` — `SparseAutoencoder` base (encoder/decoder simple con sparsity vía ReLU + L1), expuesto en `sae/__init__.py`.

### `experiments/`
Resultados versionados de entrenamientos, organizados por convención de carpetas (ver `tools/naming_utils.py` y `tools/experiment_utils.py`):

```
experiments/layer_{capa:02d}/sae_{variante}/sae_{tecnica}_{variante}/{juegos}games/[{extras_id}/]
    plots/      # curvas de entrenamiento
    metrics/    # training_metrics.json, coverage_absolute.npz, reconstruction_results.npz
```

### `metrics/`
Preparación de datos de evaluación y notebooks de métricas:

- `01_scripts_prep/` — genera board states y ground truth de BSPs (Board State Properties) a partir de partidas.
- `02_data/` — datos precomputados (`board_states_*.npz`, `bsp_ground_truth_*.npy`).
- `03_metrics_implementation/{standar,topk,matryoska}/` — notebooks de `reconstruction` y `coverage` por variante de SAE.

### `tools/`
- `bsp_identifier.py` — genera las 198 Board State Properties de un tablero (192 por casilla × 3 estados + 6 especiales).
- `experiment_utils.py` — helpers para construir rutas y nombres de reportes siguiendo la convención de `experiments/`.
- `naming_utils.py` — utilidades de naming compartidas.
