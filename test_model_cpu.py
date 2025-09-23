import torch
import time
from mingpt.model import GPT, GPTConfig

# Configuración del modelo
model_config = GPTConfig(
    vocab_size=61,  # Tamaño del vocabulario
    block_size=59,  # Tamaño máximo de secuencia
    n_layer=8,      # Número de capas transformer
    n_head=8,       # Número de cabezas de atención
    n_embd=512      # Dimensión de embedding
)

# Cargar el modelo
model = GPT(model_config)
checkpoint_path = "ckpts/gpt_championship.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# Verificar si hay GPU disponible
print("¿GPU disponible?", torch.cuda.is_available())

# Crear una secuencia de prueba (59 movimientos)
input_sequence = torch.randint(1, 61, (1, 59), dtype=torch.long)  # Secuencia aleatoria

# Medir el tiempo de inferencia
start_time = time.time()
with torch.no_grad():
    logits, _ = model(input_sequence)
end_time = time.time()

# Mostrar resultados
print("Inferencia completada.")
print("Tiempo de inferencia en CPU: {:.2f} segundos".format(end_time - start_time))

# Obtener las probabilidades de la última posición
logits = logits[:, -1, :]
probs = torch.softmax(logits, dim=-1)
print("Probabilidades de salida:", probs)