# Generar PDFs de Resultados

Ejecutar estos comandos desde la carpeta `03_metrics_implementation`:
```bash
quarto render coverage_gpu_optimized.ipynb --to pdf --output-dir ../04_results
```
```bash
quarto render reconstruction_gpu_optimized.ipynb --to pdf --output-dir ../04_results
```

Los PDFs se guardarán en la carpeta `04_results`.