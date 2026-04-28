import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)


def convert_crosssim_npz_to_h5(npz_file, h5_file):

    data = np.load(npz_file, allow_pickle=True)
    mats = data["mats"]

    layer1 = mats[0]
    layer2 = mats[1]

    # CrossSim -> Keras
    W1 = layer1[:, :-1].T
    b1 = layer1[:, -1]

    W2 = layer2[:, :-1].T
    b2 = layer2[:, -1]

    model = Sequential()
    model.add(layers.Dense(300, input_shape=(784,), activation="sigmoid"))
    model.add(layers.Dense(10, activation="softmax"))

    model.layers[0].set_weights([W1, b1])
    model.layers[1].set_weights([W2, b2])

    model.save(h5_file)
    print(" Guardado:", h5_file)


# ======================================
# LISTA DE LOS 3 ARCHIVOS
# ======================================

archivos = [
    "MLP_mnist_numeric_run0",
    "MLP_mnist_lookup_standard_run0",
    "MLP_mnist_lookup_multi_run0"
]

for nombre in archivos:

    entrada = f"./cross_sim_models/{nombre}.npz"
    salida  = f"./cross_sim_models/{nombre}.h5"

    if os.path.exists(entrada):
        convert_crosssim_npz_to_h5(entrada, salida)
    else:
        print(" No encontrado:", entrada)

print("\n Conversión completa.")