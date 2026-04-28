import sys
import os
import numpy as np

# Asegurar que el path incluya la raíz de cross-sim
sys.path.append("..")

from cross_sim import Backprop, Parameters
from helpers.dataset_loaders import load_data_mnist

# ── 1. Arquitectura (Sincronizada con tu script) ─────────────────────────────
sizes = [(1,1,784), (1,1,300), (1,1,10)]
bp = Backprop(sizes, seed=0)

# ── 2. Parámetros (Copiados de tu configuración de entrenamiento) ────────────
params = Parameters()
params.algorithm_params.crossbar_type  = "BALANCED" # wtmodel="BALANCED"
params.algorithm_params.sim_type       = "NUMERIC"  # Para inferencia estándar

# El hardware físico siempre normalizado
params.xbar_params.weights.minimum     = -1.0 
params.xbar_params.weights.maximum     = 1.0

params.numeric_params.useGPU           = False # Cambiar a True si tienes CUDA configurado
params.weight_error_params.noise_model = "none"
params.weight_error_params.error_model = "none"

# Estos valores son críticos. Si usaste la LUT de EJEMPLO, 
# CrossSim calcula escalas basadas en el rango de esa LUT.
# Los valores abajo son los típicos que usa training_net.py internamente.
baseline_mat = np.array([0.218, 1.05])

# ── 3. Creación de Cores y Registro de Capas ─────────────────────────────────
layer_types = []
for k in range(len(sizes) - 1):
    params_k = params.copy()
    params_k.algorithm_params.weights.maximum = baseline_mat[k]
    params_k.algorithm_params.weights.minimum = -baseline_mat[k]
    
    # Usamos style="new_bias" porque learnbias=(True, True) en tu script
    bp.ncore(which=k+1, style="new_bias", params=params_k)
    layer_types.append("dense")

# Corrección del error de NoneType
bp.layerTypes = layer_types 

# ── 4. Activaciones (Sincronizadas con tu script) ────────────────────────────
# En tu script: activate = "SIGMOID", activate_output = "SOFTMAX"
bp.set_activations(layer=0, style="SIGMOID")
bp.set_activate_output(style="SOFTMAX")

# ── 5. Carga de Pesos ────────────────────────────────────────────────────────
# El nombre que genera tu script es: MLP_mnist_lookup_standard_run0.npz
model_path = "cross_sim_models/MLP_mnist_lookup_standard_run0.npz"

if os.path.exists(model_path):
    print(f"Cargando pesos desde: {model_path}")
    bp.read_weights_crossSim(model_path, verbose=True)
else:
    print(f"ERROR: No se encontró el archivo {model_path}")
    sys.exit()

# ── 6. Datos y Clasificación ─────────────────────────────────────────────────
print("Cargando datos de MNIST...")
(x_train, y_train), (x_test, y_test) = load_data_mnist(training=True)

# 1. Asegurar que x_test esté en rango [0, 1] y sea float32
if x_test.max() > 1.0:
    x_test = x_test.astype(np.float32) / 255.0
else:
    x_test = x_test.astype(np.float32)

# 2. Asegurar que y_test sea un vector columna (N, 1)
if len(y_test.shape) == 1:
    y_test = y_test.reshape(-1, 1)

# 3. Concatenar: x_test (N, 784) + y_test (N, 1) -> (N, 785)
data_test = np.hstack((x_test, y_test))

# 4. PASO CRÍTICO: 
# Si el código sigue fallando con scale, es un bug de tipos en CrossSim.
# Vamos a usar None en scale, que por defecto es 1.0 en la lógica interna.
try:
    # Si pasas scale=None, el código de CrossSim suele saltarse la validación 
    # que causa el error y usa 1.0 por defecto.
    bp.read_inputs(None, data=data_test, scale=None) 
except:
    # Si falla, intentamos la única cadena que el código SI reconoce sin error
    bp.read_inputs(None, data=data_test, scale="gauss")

print(f"\nDimensiones de entrada: {x_test.shape}")
print("Iniciando clasificación...")
count, frac = bp.classify(n=10000)

print("-" * 30)
print(f"Precisión: {frac*100:.2f}%")
print(f"Correctas: {count} / 10000")