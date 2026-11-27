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
LEAD_TIME_SEC = -10      # dónde termina la ventana respecto al inicio del evento.
                         # Valor NEGATIVO = termina DESPUÉS del inicio: con -10 la ventana
                         # abarca los primeros 10 s de la bradicardia (detección temprana).
                         # El diagnóstico mostró que 30 s ANTES del evento no hay señal
                         # detectable en la frecuencia cardíaca (AUC ~0.5).
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
        y: array (n_ventanas,) con 1 = evento de bradicardia, 0 = normal
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

    # --- Ventanas positivas: abarcan el inicio de cada evento (ver LEAD_TIME_SEC) ---
    positivas = []
    descartadas = 0
    for onset in onsets:
        fin = onset - LEAD_SAMPLES
        inicio = fin - WINDOW_SAMPLES
        if inicio < 0 or fin > n_muestras:
            descartadas += 1
            continue  # evento muy cerca del inicio o del final del registro
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
          f"{len(positivas)} ventanas con evento usables "
          f"({descartadas} descartadas por estar muy cerca de un borde del registro), "
          f"{len(negativas)} ventanas normales")

    return X, y


if __name__ == "__main__":
    # Lista real de registros ECG en PhysioNet (uno por infante)
    records = wfdb.get_record_list(PN_DIR)
    ecg_records = sorted(r for r in records if "ecg" in r.lower())
    print(f"Registros ECG encontrados ({len(ecg_records)}): {ecg_records}\n")

    # Procesa TODOS los infantes y guarda un solo archivo combinado.
    # "grupos" dice a qué infante pertenece cada ventana: lo necesitas para
    # separar entrenamiento y prueba POR INFANTE (no al azar), que es lo
    # defendible cuando solo hay 10 pacientes.
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