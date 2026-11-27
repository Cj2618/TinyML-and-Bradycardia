# Extracción de ventanas etiquetadas desde PICS (PhysioNet)
# Requiere: pip install wfdb numpy scipy
# En Colab, quita el "#" de la línea siguiente y córrela una vez:
# !pip install wfdb --quiet
#
# CORRECCIÓN IMPORTANTE: según la documentación de PICS, el ECG se grabó a 500 Hz EXCEPTO en
# los infantes 1 y 5, que se grabaron a 250 Hz (señal compuesta). La versión anterior asumía
# 500 Hz para todos: en esos dos infantes cada ventana abarcaba 20 s en vez de 10 s y la
# frecuencia cardíaca aparecía duplicada para el modelo. Ahora cada ventana se toma con la
# frecuencia real del registro (record.fs) y se reescala a 500 Hz equivalentes, de modo que
# el resto del flujo (submuestreo a 125 Hz, entrenamiento, ESP32) no cambia.

import numpy as np
import wfdb
from scipy.signal import resample_poly

PN_DIR = "picsdb/1.0.0"

# --- Parámetros a decidir con criterio, no por defecto ---
FS_SALIDA = 500          # Hz de salida: todas las ventanas quedan a esta frecuencia equivalente
WINDOW_SEC = 10          # duración de cada ventana de ECG que ve el modelo
LEAD_TIME_SEC = -10      # dónde termina la ventana respecto al inicio del evento.
                         # Valor NEGATIVO = termina DESPUÉS del inicio: con -10 la ventana
                         # abarca los primeros 10 s de la bradicardia (detección temprana).
BUFFER_SEC = 60          # zona de exclusión alrededor de cada evento para las
                         # ventanas "normales" (evita contaminar la clase negativa)


def a_fs_salida(ventana, fs):
    """Lleva una ventana de frecuencia fs a FS_SALIDA conservando la escala de tiempo."""
    if int(fs) == FS_SALIDA:
        return ventana
    return resample_poly(ventana, FS_SALIDA, int(fs)).astype(np.float32)


def extraer_desde_senal(senal, onsets, fs, n_negativas=None, semilla=None):
    """Núcleo de la extracción, sobre una señal ya cargada con frecuencia fs.

    Retorna X (n_ventanas, WINDOW_SEC * FS_SALIDA) e y (1 = evento, 0 = normal)."""
    rng = np.random.RandomState(semilla)
    fs = int(fs)
    win = int(WINDOW_SEC * fs)
    lead = int(LEAD_TIME_SEC * fs)
    buf = int(BUFFER_SEC * fs)
    n = len(senal)

    positivas, descartadas = [], 0
    for onset in onsets:
        fin = onset - lead
        ini = fin - win
        if ini < 0 or fin > n:
            descartadas += 1
            continue
        positivas.append(a_fs_salida(senal[ini:fin], fs))

    if n_negativas is None:
        n_negativas = len(positivas)  # balanceado 1:1 por defecto

    zonas = [(o - buf, o + buf) for o in onsets]
    negativas, intentos = [], 0
    max_intentos = max(n_negativas * 50, 1000)
    while len(negativas) < n_negativas and intentos < max_intentos:
        intentos += 1
        ini = rng.randint(0, max(n - win, 1))
        fin = ini + win
        if any(fin > iz and ini < fz for iz, fz in zonas):
            continue
        negativas.append(a_fs_salida(senal[ini:fin], fs))

    X = np.array(positivas + negativas, dtype=np.float32)
    y = np.array([1] * len(positivas) + [0] * len(negativas))
    return X, y, descartadas


def extraer_ventanas(record_name, semilla=None):
    """Descarga el registro completo de un infante (un solo fetch desde PhysioNet)."""
    ann = wfdb.rdann(record_name, "atr", pn_dir=PN_DIR)
    record = wfdb.rdrecord(record_name, pn_dir=PN_DIR)
    fs = int(record.fs)
    senal = record.p_signal[:, 0].astype(np.float32)
    X, y, descartadas = extraer_desde_senal(senal, ann.sample, fs, semilla=semilla)
    aviso = "" if fs == FS_SALIDA else f"  [fs={fs} Hz: reescalada a {FS_SALIDA} Hz equivalentes]"
    print(f"{record_name}: fs={fs} Hz, {len(ann.sample)} eventos totales, "
          f"{int((y == 1).sum())} ventanas con evento usables "
          f"({descartadas} descartadas por estar muy cerca de un borde del registro), "
          f"{int((y == 0).sum())} ventanas normales{aviso}")
    return X, y


if __name__ == "__main__":
    # Lista real de registros ECG en PhysioNet (uno por infante)
    records = wfdb.get_record_list(PN_DIR)
    ecg_records = sorted(r for r in records if "ecg" in r.lower())
    print(f"Registros ECG encontrados ({len(ecg_records)}): {ecg_records}\n")

    # Procesa TODOS los infantes y guarda un solo archivo combinado.
    # "grupos" dice a qué infante pertenece cada ventana (0 = infant10, k = infant k).
    todos_X, todos_y, grupos = [], [], []
    for i, rn in enumerate(ecg_records):
        try:
            X_i, y_i = extraer_ventanas(rn, semilla=42)
        except Exception as e:
            print(f"  !! {rn} falló y se omite: {e}")
            continue
        todos_X.append(X_i)
        todos_y.append(y_i)
        grupos.append(np.full(len(y_i), i))

    X_total = np.concatenate(todos_X)
    y_total = np.concatenate(todos_y)
    g_total = np.concatenate(grupos)

    print(f"\nForma final: X={X_total.shape}, y={y_total.shape}")
    print(f"Con evento: {(y_total == 1).sum()} | Normales: {(y_total == 0).sum()}")
    print(f"Infantes incluidos: {len(np.unique(g_total))}")

    np.savez("pics_todos_ventanas.npz", X=X_total, y=y_total, grupos=g_total)
    print("Guardado en pics_todos_ventanas.npz (descárgalo o cópialo a Drive)")