# Entrenamiento, compresión y exportación - Bradicardia neonatal (PICS) con TinyML
# Entrada: pics_todos_ventanas.npz (generado por extraer_ventanas_pics.py)
# En Colab sube ese archivo (o móntalo desde Drive) y pega este script en una celda.
#
# Qué hace, en orden:
#   1) Prepara la señal (500 Hz -> 125 Hz, normaliza cada ventana)
#   2) Valida "dejando un infante afuera" (leave-one-infant-out), el estándar
#      defendible cuando hay solo 10 pacientes
#   3) En cada ronda compara: float32, float16, int8 completo, e int8 + poda
#   4) Imprime y guarda la tabla de resultados (resultados_cv.csv)
#   5) Entrena un modelo final con todos los infantes y exporta los .tflite y
#      un archivo .h listo para pegar en el proyecto del ESP32

import gzip
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import decimate
from sklearn.metrics import roc_auc_score

# ----------------------------- Parámetros ---------------------------------
DATA = "pics_todos_ventanas.npz"
FACTOR = 4          # 500 Hz -> 125 Hz (menos muestras = modelo y latencia menores)
EPOCHS = 40
BATCH = 32
SPARSITIES = [0.5, 0.8]   # fracciones de pesos a podar
FINETUNE_EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# --------------------------- 1) Datos --------------------------------------
d = np.load(DATA)
X, y, g = d["X"], d["y"], d["grupos"]
print(f"Datos: X={X.shape}, positivas={(y == 1).sum()}, normales={(y == 0).sum()}, "
      f"infantes={len(np.unique(g))}")

X = decimate(X, FACTOR, axis=1, zero_phase=True).astype("float32")
# Normalización por ventana: media 0, desviación 1. Ojo: en el ESP32 hay que
# replicar exactamente este paso (y el submuestreo) antes de cada inferencia.
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
X = X[..., None]
LEN = X.shape[1]
print(f"Ventana final: {LEN} muestras ({LEN / (500 / FACTOR):.0f} s a {500 // FACTOR} Hz)\n")


# --------------------------- 2) Modelo -------------------------------------
def build_model():
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input((LEN, 1)),
        L.Conv1D(8, 7, padding="same", activation="relu"),
        L.MaxPooling1D(4),
        L.Conv1D(16, 5, padding="same", activation="relu"),
        L.MaxPooling1D(4),
        L.Conv1D(16, 3, padding="same", activation="relu"),
        L.GlobalAveragePooling1D(),
        L.Dense(16, activation="relu"),
        L.Dense(1, activation="sigmoid"),
    ])


def compile_model(m):
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def train(Xtr, ytr, epochs=EPOCHS):
    m = compile_model(build_model())
    m.fit(Xtr, ytr, epochs=epochs, batch_size=BATCH, verbose=0, shuffle=True)
    return m


# --------------------------- 3) Conversión a TFLite -------------------------
def to_tflite(model, modo, Xcal=None):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if modo == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif modo == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        idx = np.random.RandomState(SEED).choice(len(Xcal), min(200, len(Xcal)), replace=False)
        conv.representative_dataset = lambda: ([Xcal[i:i + 1]] for i in idx)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        # IMPORTANTE para el ESP32: TensorFlow Lite Micro solo soporta UNA escala por tensor en
        # las capas Dense. Por defecto el conversor usa una escala por canal, y en el chip daría
        # resultados distintos a los de Python. Esta opción lo evita (las convoluciones siguen
        # por canal, que sí está soportado).
        if hasattr(conv, "_experimental_disable_per_channel_quantization_for_dense_layers"):
            conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv.convert()


def tflite_predict(modelo_bytes, Xte):
    it = tf.lite.Interpreter(model_content=modelo_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = []
    for x in Xte:
        x = x[None, ...]
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        it.invoke()
        p = it.get_tensor(out["index"])[0, 0]
        if out["dtype"] == np.int8:
            s, z = out["quantization"]
            p = (float(p) - z) * s
        preds.append(float(p))
    return np.array(preds)


def metricas(yte, p):
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (yte == 1)).sum()
    tn = ((pred == 0) & (yte == 0)).sum()
    fn = ((pred == 0) & (yte == 1)).sum()
    fp = ((pred == 1) & (yte == 0)).sum()
    sens = tp / max(tp + fn, 1)
    espec = tn / max(tn + fp, 1)
    auc = roc_auc_score(yte, p) if len(np.unique(yte)) == 2 else np.nan
    return sens, espec, auc


# --------------------------- 4) Poda por magnitud ---------------------------
def podar(model, sparsity):
    """Pone en cero el 'sparsity' de pesos de menor magnitud en cada kernel.
    Devuelve las máscaras para mantenerlas durante el ajuste fino."""
    masks = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Dense)):
            w, b = layer.get_weights()
            umbral = np.quantile(np.abs(w), sparsity)
            mask = (np.abs(w) > umbral).astype("float32")
            layer.set_weights([w * mask, b])
            masks[i] = mask
    return masks


class MantenerMascaras(tf.keras.callbacks.Callback):
    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def on_train_batch_end(self, batch, logs=None):
        for i, mask in self.masks.items():
            w, b = self.model.layers[i].get_weights()
            self.model.layers[i].set_weights([w * mask, b])


def podar_y_ajustar(model, Xtr, ytr, sparsity):
    clon = tf.keras.models.clone_model(model)
    clon.set_weights(model.get_weights())
    compile_model(clon)
    masks = podar(clon, sparsity)
    clon.fit(Xtr, ytr, epochs=FINETUNE_EPOCHS, batch_size=BATCH, verbose=0,
             callbacks=[MantenerMascaras(masks)])
    return clon


# --------------------------- 5) Validación por infante ----------------------
filas = []
infantes = np.unique(g)
for k in infantes:
    te, tr = g == k, g != k
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"Ronda: infante {k} fuera ({te.sum()} ventanas de prueba)")

    m = train(Xtr, ytr)
    variantes = {
        "float32": to_tflite(m, "float32"),
        "float16": to_tflite(m, "float16"),
        "int8": to_tflite(m, "int8", Xtr),
    }
    for s in SPARSITIES:
        mp = podar_y_ajustar(m, Xtr, ytr, s)
        variantes[f"int8+poda{int(s * 100)}"] = to_tflite(mp, "int8", Xtr)

    for nombre, blob in variantes.items():
        sens, espec, auc = metricas(yte, tflite_predict(blob, Xte))
        filas.append(dict(infante_fuera=k, config=nombre, sensibilidad=sens,
                          especificidad=espec, auc=auc,
                          tamano_kb=len(blob) / 1024,
                          tamano_gzip_kb=len(gzip.compress(blob)) / 1024))

df = pd.DataFrame(filas)
df.to_csv("resultados_cv.csv", index=False)

resumen = df.groupby("config")[["sensibilidad", "especificidad", "auc",
                                 "tamano_kb", "tamano_gzip_kb"]].agg(["mean", "std"]).round(3)
print("\n===== RESULTADOS (media y desviación entre infantes) =====")
print(resumen.to_string())
print("\nGuardado en resultados_cv.csv")

# --------------------------- 6) Modelo final y exportación -------------------
print("\nEntrenando modelo final con todos los infantes...")
final = train(X, y)
blobs = {
    "modelo_float32.tflite": to_tflite(final, "float32"),
    "modelo_float16.tflite": to_tflite(final, "float16"),
    "modelo_int8.tflite": to_tflite(final, "int8", X),
}
for nombre, blob in blobs.items():
    with open(nombre, "wb") as f:
        f.write(blob)
    print(f"  {nombre}: {len(blob) / 1024:.1f} KB")


def a_header(blob, nombre="g_model"):
    filas_hex = []
    for i in range(0, len(blob), 12):
        filas_hex.append("  " + ", ".join(f"0x{b:02x}" for b in blob[i:i + 12]))
    return (f"alignas(16) const unsigned char {nombre}[] = {{\n" + ",\n".join(filas_hex)
            + f"\n}};\nconst unsigned int {nombre}_len = {len(blob)};\n")


with open("modelo_int8.h", "w") as f:
    f.write(a_header(blobs["modelo_int8.tflite"]))
print("  modelo_int8.h (arreglo C para TensorFlow Lite Micro en el ESP32)")
print("\nDescarga estos archivos: modelo_*.tflite, modelo_int8.h y resultados_cv.csv")