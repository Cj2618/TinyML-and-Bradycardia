# Falsas alarmas por hora en registros continuos - Bradicardia neonatal (PICS)
# Requiere: pip install wfdb tensorflow scipy scikit-learn pandas matplotlib
# En Colab: quita el "#" de la línea siguiente y córrela una vez
# !pip install wfdb --quiet
#
# Entrada:  pics_todos_ventanas.npz (el mismo que usa entrenar_y_comprimir.py)
# Salidas:  continuo/probs_K.npz                    (probabilidades por bebé, permite reanudar)
#           continuo_resultado_global.csv           (la tabla principal)
#           continuo_resultado_por_infante.csv
#           fig_sensibilidad_vs_falsas_alarmas.png
#
# Qué hace, en orden:
#   1) Para cada bebé: entrena con los otros 9 (float32 e int8, como en el trabajo) y
#      recorre TODA la grabación del bebé que quedó fuera, ventana tras ventana.
#   2) Guarda las probabilidades. Es la parte lenta. Se hace una sola vez.
#   3) Con esas probabilidades calcula, para varios umbrales y reglas de persistencia:
#      sensibilidad por evento, retraso de detección y falsas alarmas por hora.
#      Esta parte tarda segundos, así que puedes cambiar umbrales y reglas sin recalcular.
#
# Si se interrumpe, vuelve a correrlo: salta los bebés que ya terminó.
# Prueba rápida: pon MAX_HORAS = 3 (usa solo las primeras 3 horas de cada bebé).

import glob
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import decimate, resample_poly

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Parámetros ---------------------------------
PN_DIR = "picsdb/1.0.0"
DIR_LOCAL = None          # solo para pruebas: carpeta con registros WFDB en el disco
NPZ = "pics_todos_ventanas.npz"
CARPETA = "continuo"
MAX_HORAS = None          # None = grabación completa; 3 = prueba rápida

FS_BASE = 500             # los registros se llevan a 500 Hz equivalentes, como en la extracción
FACTOR = 4                # 500 Hz -> 125 Hz, igual que en el entrenamiento
VENTANA_S = 10
PASO_S = 5                # cada cuántos segundos se evalúa una ventana nueva (2 = más fino y más lento)
LOTE = 512

EPOCHS = 40               # mismos valores que entrenar_y_comprimir.py
BATCH = 32
SEED = 42

UMBRALES = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995]   # int8 no pasa de 0.996 por su resolución
PERSISTENCIAS = [1, 3]    # 3 = alarma solo si 3 ventanas seguidas superan el umbral
DETECCION_S = 20          # un evento cuenta como detectado si suena la alarma hasta 20 s después de su inicio
BUFFER_S = 60             # "señal normal" = a más de 60 s de cualquier evento (igual que en el entrenamiento)
REFRACTARIO_S = 30        # alarmas separadas por menos de 30 s cuentan como una sola
CONFIGS = {"float32": "p_float32", "int8": "p_int8"}


# ------------------------------ Modelo -------------------------------------
def preparar(X):
    """Igual que en entrenar_y_comprimir.py: 500 -> 125 Hz y z-score por ventana."""
    X = decimate(X, FACTOR, axis=1, zero_phase=True).astype("float32")
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return X[..., None]


def construir_modelo(largo):
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input((largo, 1)),
        L.Conv1D(8, 7, padding="same", activation="relu"),
        L.MaxPooling1D(4),
        L.Conv1D(16, 5, padding="same", activation="relu"),
        L.MaxPooling1D(4),
        L.Conv1D(16, 3, padding="same", activation="relu"),
        L.GlobalAveragePooling1D(),
        L.Dense(16, activation="relu"),
        L.Dense(1, activation="sigmoid"),
    ])


def a_int8(modelo, Xcal):
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    idx = np.random.RandomState(SEED).choice(len(Xcal), min(200, len(Xcal)), replace=False)
    conv.representative_dataset = lambda: ([Xcal[i:i + 1]] for i in idx)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    if hasattr(conv, "_experimental_disable_per_channel_quantization_for_dense_layers"):
        conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv.convert()


def entrenar_fold(Xtr, ytr, k):
    tf.keras.utils.set_random_seed(SEED + int(k))
    m = construir_modelo(Xtr.shape[1])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=True)
    interp = tf.lite.Interpreter(model_content=a_int8(m, Xtr))
    interp.allocate_tensors()
    return m, interp


# ------------------------- Recorrido de una grabación -----------------------
def cargar_registro(nombre):
    ruta = os.path.join(DIR_LOCAL, nombre) if DIR_LOCAL else nombre
    kw = {} if DIR_LOCAL else {"pn_dir": PN_DIR}
    rec = wfdb.rdrecord(ruta, **kw)
    ann = wfdb.rdann(ruta, "atr", **kw)
    fs = int(rec.fs)
    senal = np.nan_to_num(rec.p_signal[:, 0]).astype(np.float32)
    del rec
    if fs != FS_BASE:                       # infantes 1 y 5 vienen a 250 Hz
        senal = resample_poly(senal, FS_BASE, fs).astype(np.float32)
    simbolos = np.array(ann.symbol)
    muestras = ann.sample[simbolos == "["] if (simbolos == "[").any() else ann.sample
    onsets = np.sort(muestras / fs)
    if MAX_HORAS:
        limite = MAX_HORAS * 3600
        senal = senal[: int(limite * FS_BASE)]
        onsets = onsets[onsets < limite - VENTANA_S]
    return senal, onsets


def probabilidades(senal, modelo, interp):
    """Probabilidad de cada ventana, con el mismo preproceso del entrenamiento (ventana por ventana)."""
    w, paso = VENTANA_S * FS_BASE, PASO_S * FS_BASE
    ventanas = sliding_window_view(senal, w)[::paso]
    n = len(ventanas)
    p32 = np.full(n, np.nan, np.float32)
    p8 = np.full(n, np.nan, np.float32)
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    for i0 in range(0, n, LOTE):
        i1 = min(n, i0 + LOTE)
        W = np.array(ventanas[i0:i1])
        valido = W.std(axis=1) > 1e-6       # ventanas planas (electrodo suelto) no se evalúan
        if not valido.any():
            continue
        X = preparar(W[valido])
        idx = i0 + np.where(valido)[0]
        p32[idx] = modelo(X, training=False).numpy()[:, 0]
        Xq = np.clip(np.round(X / s_in + z_in), -128, 127).astype(np.int8)
        for j in range(len(Xq)):
            interp.set_tensor(inp["index"], Xq[j:j + 1])
            interp.invoke()
            q = interp.get_tensor(out["index"])[0, 0]
            p8[idx[j]] = (float(q) - z_out) * s_out
    t_fin = (np.arange(n) * paso + w) / FS_BASE
    return t_fin, p32, p8


# ------------------------------ Análisis -----------------------------------
def zona_normal(t_fin, onsets):
    """True si la ventana está a más de BUFFER_S de cualquier inicio de evento."""
    if len(onsets) == 0:
        return np.ones(len(t_fin), bool)
    t_ini = t_fin - VENTANA_S
    i = np.searchsorted(onsets, t_ini - BUFFER_S, side="left")
    hay = i < len(onsets)
    cerca = np.zeros(len(t_fin), bool)
    cerca[hay] = onsets[i[hay]] <= t_fin[hay] + BUFFER_S
    return ~cerca


def disparos(p, umbral, k):
    """True donde suena la alarma: las últimas k ventanas superan el umbral."""
    with np.errstate(invalid="ignore"):
        a = p >= umbral
    if k == 1:
        return a
    cuenta = np.convolve(a.astype(int), np.ones(k, int), mode="full")[: len(a)]
    return cuenta >= k


def evaluar_infante(t_fin, p, onsets, umbral, k):
    trig = disparos(p, umbral, k)
    normal = zona_normal(t_fin, onsets) & ~np.isnan(p)
    horas = normal.sum() * PASO_S / 3600
    tt = t_fin[trig & normal]
    falsas = 0 if len(tt) == 0 else 1 + int((np.diff(tt) > REFRACTARIO_S).sum())
    detectados, retrasos = 0, []
    for o in onsets:
        a = np.searchsorted(t_fin, o, side="left")
        b = np.searchsorted(t_fin, o + DETECCION_S, side="right")
        hit = np.where(trig[a:b])[0]
        if len(hit):
            detectados += 1
            retrasos.append(float(t_fin[a + hit[0]] - o))
    return dict(eventos=len(onsets), detectados=detectados, retrasos=retrasos,
                falsas=falsas, horas=float(horas))


def analizar(archivos):
    datos = [np.load(a, allow_pickle=True) for a in archivos]
    filas = []
    for d in datos:
        k_inf = int(d["indice"])
        for conf, campo in CONFIGS.items():
            for k in PERSISTENCIAS:
                for u in UMBRALES:
                    r = evaluar_infante(d["t_fin"], d[campo], d["onsets"], u, k)
                    filas.append(dict(infante=str(d["nombre"]), grupo=k_inf, config=conf,
                                      persistencia=k, umbral=u, eventos=r["eventos"],
                                      detectados=r["detectados"],
                                      retraso_mediano_s=(np.median(r["retrasos"]) if r["retrasos"] else np.nan),
                                      falsas_alarmas=r["falsas"], horas_normales=r["horas"],
                                      falsas_por_hora=(r["falsas"] / r["horas"] if r["horas"] > 0 else np.nan),
                                      _retrasos=r["retrasos"]))
    por_inf = pd.DataFrame(filas)
    glob_filas = []
    for (conf, k, u), sub in por_inf.groupby(["config", "persistencia", "umbral"]):
        retrasos = [x for lst in sub["_retrasos"] for x in lst]
        horas = sub.horas_normales.sum()
        glob_filas.append(dict(
            config=conf, persistencia=k, umbral=u, eventos=int(sub.eventos.sum()),
            detectados=int(sub.detectados.sum()),
            sensibilidad=sub.detectados.sum() / max(sub.eventos.sum(), 1),
            retraso_mediano_s=(np.median(retrasos) if retrasos else np.nan),
            falsas_alarmas=int(sub.falsas_alarmas.sum()), horas_normales=horas,
            falsas_por_hora_global=sub.falsas_alarmas.sum() / horas if horas > 0 else np.nan,
            falsas_por_hora_mediana_infantes=sub.falsas_por_hora.median(),
            falsas_por_hora_peor_infante=sub.falsas_por_hora.max()))
    glob_df = pd.DataFrame(glob_filas).sort_values(["config", "persistencia", "umbral"])
    return glob_df, por_inf.drop(columns="_retrasos")


def figura(glob_df, salida):
    colores = {"float32": "#6b7684", "int8": "#0072B2"}
    estilos = {1: "-", 3: "--"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (conf, k), sub in glob_df.groupby(["config", "persistencia"]):
        sub = sub.sort_values("umbral")
        ax.plot(sub.falsas_por_hora_global, sub.sensibilidad, estilos.get(k, "-"), marker="o",
                color=colores.get(conf, "k"), label=f"{conf}, {k} ventana{'s' if k > 1 else ''} seguida{'s' if k > 1 else ''}")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Falsas alarmas por hora (en señal normal)")
    ax.set_ylabel("Eventos detectados (fracción)")
    ax.set_ylim(0, 1.02)
    ax.grid(color="#d8d2c4", lw=0.8)
    ax.axvline(1, color="#999999", lw=1, ls=":")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor="white")


# ------------------------------- Principal ----------------------------------
def lista_registros():
    if DIR_LOCAL:
        hea = sorted(glob.glob(os.path.join(DIR_LOCAL, "*.hea")))
        return sorted(os.path.splitext(os.path.basename(h))[0] for h in hea if "ecg" in h.lower())
    return sorted(r for r in wfdb.get_record_list(PN_DIR) if "ecg" in r.lower())


def principal():
    os.makedirs(CARPETA, exist_ok=True)
    d = np.load(NPZ)
    X, y, g = d["X"], d["y"], d["grupos"]
    X = preparar(X)
    nombres = lista_registros()
    grupos = np.unique(g)
    assert len(nombres) == len(grupos), f"{len(nombres)} registros y {len(grupos)} grupos: no coinciden"
    print(f"{len(nombres)} bebés, {len(X)} ventanas de entrenamiento. Paso de {PASO_S} s.\n")

    for k, nombre in enumerate(nombres):
        archivo = os.path.join(CARPETA, f"probs_{k}.npz")
        if os.path.exists(archivo):
            print(f"[{k}] {nombre}: ya estaba hecho")
            continue
        t0 = time.time()
        tr = g != k
        modelo, interp = entrenar_fold(X[tr], y[tr], k)
        t1 = time.time()
        senal, onsets = cargar_registro(nombre)
        t2 = time.time()
        t_fin, p32, p8 = probabilidades(senal, modelo, interp)
        t3 = time.time()
        np.savez_compressed(archivo, indice=k, nombre=nombre, t_fin=t_fin, p_float32=p32,
                            p_int8=p8, onsets=onsets, horas=len(senal) / FS_BASE / 3600)
        print(f"[{k}] {nombre}: {len(senal) / FS_BASE / 3600:.1f} h, {len(onsets)} eventos, "
              f"{len(t_fin)} ventanas, {np.isnan(p32).sum()} sin evaluar | "
              f"entrenar {t1 - t0:.0f}s, leer {t2 - t1:.0f}s, recorrer {t3 - t2:.0f}s")
        del senal

    archivos = sorted(glob.glob(os.path.join(CARPETA, "probs_*.npz")),
                      key=lambda a: int(os.path.basename(a)[6:-4]))
    glob_df, por_inf = analizar(archivos)
    glob_df.to_csv("continuo_resultado_global.csv", index=False)
    por_inf.to_csv("continuo_resultado_por_infante.csv", index=False)
    figura(glob_df, "fig_sensibilidad_vs_falsas_alarmas.png")

    pd.set_option("display.width", 200)
    print("\n===== RESULTADO GLOBAL (todos los bebés juntos) =====")
    cols = ["config", "persistencia", "umbral", "sensibilidad", "retraso_mediano_s",
            "falsas_por_hora_global", "falsas_por_hora_mediana_infantes", "falsas_por_hora_peor_infante"]
    print(glob_df[cols].round(3).to_string(index=False))
    print("\n===== Primer umbral con 1 falsa alarma por hora o menos (global) =====")
    for (conf, k), sub in glob_df.groupby(["config", "persistencia"]):
        ok = sub[sub.falsas_por_hora_global <= 1].sort_values("umbral")
        if len(ok):
            f = ok.iloc[0]
            print(f"{conf}, persistencia {k}: umbral {f.umbral}, detecta {f.sensibilidad:.2f} de los eventos, "
                  f"retraso mediano {f.retraso_mediano_s:.1f} s, {f.falsas_por_hora_global:.2f} falsas/h")
        else:
            print(f"{conf}, persistencia {k}: ningún umbral probado llega a 1 falsa alarma por hora")
    print("\nGuardado: continuo_resultado_global.csv, continuo_resultado_por_infante.csv, "
          "fig_sensibilidad_vs_falsas_alarmas.png")


if __name__ == "__main__":
    principal()