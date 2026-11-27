# Extracción de ventanas etiquetadas desde PICS (PhysioNet)
# Requiere: pip install wfdb numpy
# En Colab, quita el "#" de la línea siguiente y córrela una vez:
# !pip install wfdb --quiet

import numpy as np
import wfdb

PN_DIR = "picsdb/1.0.0"

# --- Parámetros a decidir con criterio, no por defecto ---
FS = 500                 # Hz. Confirma con record.fs al cargar un registro.
WINDOW_SEC = 10          # duración de cada ventana de ECG que ve el modelo
LEAD_TIME_SEC = 30       # cuánto antes del inicio del evento termina la ventana
                         # "pre-bradicardia". Ajusta esto con base en la literatura
                         # o en un criterio clínico de tiempo de reacción razonable.
BUFFER_SEC = 60          # zona de exclusión alrededor de cada evento para las
                         # ventanas "normales" (evita contaminar la clase negativa)

WINDOW_SAMPLES = WINDOW_SEC * FS
LEAD_SAMPLES = LEAD_TIME_SEC * FS
BUFFER_SAMPLES = BUFFER_SEC * FS


def extraer_ventanas(record_name, n_negativas=None, semilla=None):
    """Descarga el registro completo de un infante (streaming desde PhysioNet,
    un solo fetch) y devuelve ventanas de ECG etiquetadas.

    Retorna:
        X: array (n_ventanas, WINDOW_SAMPLES) con la señal de cada ventana
        y: array (n_ventanas,) con 1 = pre-bradicardia, 0 = normal
    """
    if semilla is not None:
        np.random.seed(semilla)

    # 1) Posiciones (en muestras) de cada inicio de bradicardia.
    #    Confirmado en la documentación de PICS: el archivo .atr contiene
    #    únicamente inicios de bradicardia, sin importar el símbolo usado.
    ann = wfdb.rdann(record_name, "atr", pn_dir=PN_DIR)
    onsets = ann.sample

    # 2) Señal completa del infante. Un solo fetch por infante es manejable
    #    en memoria (no necesitas los 1.61 GB completos del dataset a la vez,
    #    solo el registro de este infante).
    record = wfdb.rdrecord(record_name, pn_dir=PN_DIR)
    senal = record.p_signal[:, 0].astype(np.float32)  # canal ECG
    n_muestras = len(senal)

    # --- Ventanas positivas: terminan LEAD_TIME_SEC antes de cada evento ---
    positivas = []
    descartadas = 0
    for onset in onsets:
        fin = onset - LEAD_SAMPLES
        inicio = fin - WINDOW_SAMPLES
        if inicio < 0:
            descartadas += 1
            continue  # evento muy al comienzo del registro, no hay suficiente señal antes
        positivas.append(senal[inicio:fin])

    # --- Ventanas negativas: lejos de cualquier evento (fuera del buffer) ---
    if n_negativas is None:
        n_negativas = len(positivas)  # balanceado 1:1 por defecto

    zonas_excluidas = [(o - BUFFER_SAMPLES, o + BUFFER_SAMPLES) for o in onsets]

    negativas = []
    intentos = 0
    max_intentos = max(n_negativas * 50, 1000)
    while len(negativas) < n_negativas and intentos < max_intentos:
        intentos += 1
        inicio = np.random.randint(0, max(n_muestras - WINDOW_SAMPLES, 1))
        fin = inicio + WINDOW_SAMPLES
        if any(fin > iz and inicio < fz for iz, fz in zonas_excluidas):
            continue
        negativas.append(senal[inicio:fin])

    X = np.array(positivas + negativas)
    y = np.array([1] * len(positivas) + [0] * len(negativas))

    print(f"{record_name}: {len(onsets)} eventos totales, "
          f"{len(positivas)} ventanas pre-bradicardia usables "
          f"({descartadas} descartadas por estar muy al inicio del registro), "
          f"{len(negativas)} ventanas normales")

    return X, y


if __name__ == "__main__":
    # Prueba con un solo infante primero para confirmar que todo corre bien
    # antes de lanzarlo sobre los 10. El nombre se toma automáticamente de la
    # lista real de registros de PhysioNet (el primero que tenga "ecg").
    records = wfdb.get_record_list(PN_DIR)
    ecg_records = [r for r in records if "ecg" in r.lower()]
    print(f"Registros ECG encontrados: {ecg_records}")
    record_name = ecg_records[0] if ecg_records else records[0]
    print(f"Usando: {record_name}\n")

    X, y = extraer_ventanas(record_name, semilla=42)
    print(f"Forma final: X={X.shape}, y={y.shape}")

    # Guarda el resultado para no tener que repetir la descarga cada vez
    np.savez(f"{record_name}_ventanas.npz", X=X, y=y)
    print(f"Guardado en {record_name}_ventanas.npz")

    # Para procesar los 10 infantes, repite en un loop y concatena:
    # todos_X, todos_y = [], []
    # for rn in nombres_ecg_de_los_10_infantes:
    #     X_i, y_i = extraer_ventanas(rn, semilla=42)
    #     todos_X.append(X_i); todos_y.append(y_i)
    # X_total = np.concatenate(todos_X)
    # y_total = np.concatenate(todos_y)