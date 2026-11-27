# Genera ventanas_prueba.h para el sketch de benchmark (versión multi-modelo)
# Requiere en la carpeta de Colab:
#   pics_todos_ventanas.npz            (de extraer_ventanas_pics.py)
#   modelo_float32.tflite, modelo_int8.tflite,
#   modelo_int8_poda50.tflite, modelo_int8_poda80.tflite   (de entrenar_y_comprimir.py)
#
# Qué contiene el .h:
#   - N ventanas REALES ya submuestreadas a 125 Hz (SIN normalizar: el ESP32 normaliza y
#     cuantiza, así se mide también ese costo)
#   - la etiqueta real de cada ventana
#   - para CADA modelo, la salida esperada: lo que da ese mismo modelo corriendo en Python
#     (sirve para comprobar que el ESP32 calcula lo mismo)

import os
import numpy as np
import tensorflow as tf
from scipy.signal import decimate

DATA = "pics_todos_ventanas.npz"
SALIDA = "ventanas_prueba.h"
FACTOR = 4            # 500 Hz -> 125 Hz (igual que en el entrenamiento)
N_POR_CLASE = 10      # 10 con evento + 10 normales = 20 ventanas (~100 KB de flash)
SEMILLA = 42

# (archivo del modelo, nombre del arreglo en el .h) - deben coincidir con el sketch
MODELOS = [
    ("modelo_float32.tflite", "esperado_float32"),
    ("modelo_int8.tflite", "esperado_int8"),
    ("modelo_int8_poda50.tflite", "esperado_int8_poda50"),
    ("modelo_int8_poda80.tflite", "esperado_int8_poda80"),
]

faltan = [m for m, _ in MODELOS if not os.path.exists(m)]
if faltan:
    raise SystemExit(f"Faltan estos archivos: {faltan}. Corre primero entrenar_y_comprimir.py.")

d = np.load(DATA)
X, y, g = d["X"], d["y"], d["grupos"]

# Elegir ventanas repartidas entre infantes distintos
rng = np.random.RandomState(SEMILLA)
idx = []
for clase in (1, 0):
    candidatos = np.where(y == clase)[0]
    idx.extend(rng.choice(candidatos, N_POR_CLASE, replace=False))
idx = np.array(idx)
rng.shuffle(idx)

Xs = decimate(X[idx], FACTOR, axis=1, zero_phase=True).astype("float32")   # sin normalizar
ys = y[idx]
n, largo = Xs.shape


def interprete(ruta):
    # Kernels de referencia (sin XNNPACK), lo más parecido a lo que corre en el chip
    try:
        it = tf.lite.Interpreter(
            model_path=ruta,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    except Exception:
        it = tf.lite.Interpreter(model_path=ruta)
    it.allocate_tensors()
    return it


def salida_esperada(ruta):
    it = interprete(ruta)
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    if ent["shape"][1] != largo:
        raise SystemExit(f"{ruta} espera {ent['shape'][1]} muestras y las ventanas tienen {largo}.")
    res = []
    for x in Xs:
        xn = (x - x.mean()) / (x.std() + 1e-8)
        if ent["dtype"] == np.int8:
            s, z = ent["quantization"]
            entrada = np.clip(np.round(xn / s + z), -128, 127).astype(np.int8)
        else:
            entrada = xn.astype(np.float32)
        it.set_tensor(ent["index"], entrada[None, :, None])
        it.invoke()
        out = float(it.get_tensor(sal["index"])[0, 0])
        if sal["dtype"] == np.int8:
            s, z = sal["quantization"]
            out = (out - z) * s
        res.append(out)
    return res


esperados = {nombre: salida_esperada(archivo) for archivo, nombre in MODELOS}

# Escribir el encabezado C++
with open(SALIDA, "w") as f:
    f.write("// Generado por generar_ventanas_prueba.py. No editar a mano.\n#pragma once\n")
    f.write(f"#define N_VENTANAS {n}\n#define LARGO_VENTANA {largo}\n\n")
    f.write("const float ventanas[N_VENTANAS][LARGO_VENTANA] = {\n")
    for x in Xs:
        f.write("  {" + ", ".join(f"{v:.7e}f" for v in x) + "},\n")
    f.write("};\n\n")
    f.write("const int etiquetas[N_VENTANAS] = {" + ", ".join(str(int(v)) for v in ys) + "};\n")
    for nombre, vals in esperados.items():
        f.write(f"const float {nombre}[N_VENTANAS] = {{" + ", ".join(f"{v:.6f}f" for v in vals) + "};\n")

print(f"Listo: {SALIDA} con {n} ventanas de {largo} muestras "
      f"({(ys == 1).sum()} con evento, {(ys == 0).sum()} normales) y {len(MODELOS)} modelos.")
print("Copia ventanas_prueba.h y modelos_esp32.h a la carpeta src/ del proyecto.")