# Diagnóstico: ¿hay señal detectable en el ECG, y a qué distancia del evento?
# Requiere: pip install wfdb scipy scikit-learn pandas
# En Colab: quita el "#" de la línea siguiente y córrela una vez
# !pip install wfdb --quiet
#
# Idea: sin entrenar ninguna red, calcula la frecuencia cardíaca (FC) de cada
# ventana con un detector de picos simple, y mide con AUC qué tan bien separa
# "ventana cercana a una bradicardia" de "ventana normal". Si la FC baja con
# claridad cuando la ventana abarca el evento, tu alineación de anotaciones y
# tu pipeline están bien, y lo difícil es anticipar. Si ni siquiera ahí separa,
# hay un problema de alineación/datos que hay que arreglar antes de entrenar.

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import roc_auc_score

PN_DIR = "picsdb/1.0.0"
LARGOS_VENTANA = [10, 30]         # segundos
FIN_REL = [10, 0, -30, -60]       # fin de la ventana respecto al inicio del evento (s)
                                  #  +10 = la ventana abarca el evento (detección)
                                  #    0 = termina justo al inicio del evento
                                  #  -30 = termina 30 s ANTES del evento (anticipación)
BUFFER = 60                       # s de margen alrededor de eventos para ventanas normales
SEMILLA = 42


def features_fc(x, fs):
    """FC media, FC mínima y variabilidad de RR de una ventana de ECG."""
    b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
    y = np.abs(filtfilt(b, a, x))
    picos, _ = find_peaks(y, distance=int(0.25 * fs), height=0.5 * np.percentile(y, 99))
    rr = np.diff(picos) / fs
    rr = rr[(rr > 0.2) & (rr < 3.0)]
    if len(rr) < 3:
        return np.nan, np.nan, np.nan
    return 60.0 / rr.mean(), 60.0 / rr.max(), rr.std()


def analizar_registro(senal, onsets, fs, infante, rng):
    filas = []
    n = len(senal)
    zonas = [(o - BUFFER * fs, o + BUFFER * fs) for o in onsets]

    for L in LARGOS_VENTANA:
        w = int(L * fs)
        # Ventanas normales: una sola vez por largo de ventana
        negativas = []
        intentos = 0
        while len(negativas) < len(onsets) and intentos < len(onsets) * 50:
            intentos += 1
            ini = rng.randint(0, n - w)
            if any(ini + w > iz and ini < fz for iz, fz in zonas):
                continue
            negativas.append(ini)
        for ini in negativas:
            fm, fmin, sd = features_fc(senal[ini:ini + w], fs)
            for fr in FIN_REL:
                filas.append(dict(infante=infante, largo=L, fin_rel=fr, etiqueta=0,
                                  fc_media=fm, fc_min=fmin, rr_std=sd))
        # Ventanas positivas: una por evento y por distancia
        for o in onsets:
            for fr in FIN_REL:
                fin = int(o + fr * fs)
                ini = fin - w
                if ini < 0 or fin > n:
                    continue
                fm, fmin, sd = features_fc(senal[ini:fin], fs)
                filas.append(dict(infante=infante, largo=L, fin_rel=fr, etiqueta=1,
                                  fc_media=fm, fc_min=fmin, rr_std=sd))
    return filas


def resumen(df):
    salida = []
    for (L, fr), sub in df.groupby(["largo", "fin_rel"]):
        sub = sub.dropna()
        if sub.etiqueta.nunique() < 2:
            continue
        # Hipótesis: en bradicardia la FC es menor -> usamos -FC como puntaje
        auc_media = roc_auc_score(sub.etiqueta, -sub.fc_media)
        auc_min = roc_auc_score(sub.etiqueta, -sub.fc_min)
        salida.append(dict(largo_s=L, fin_ventana_vs_evento_s=fr,
                           n=len(sub), AUC_fc_media=round(auc_media, 3),
                           AUC_fc_min=round(auc_min, 3),
                           fc_media_pos=round(sub[sub.etiqueta == 1].fc_media.mean(), 1),
                           fc_media_neg=round(sub[sub.etiqueta == 0].fc_media.mean(), 1)))
    return pd.DataFrame(salida).sort_values(["largo_s", "fin_ventana_vs_evento_s"],
                                            ascending=[True, False])


if __name__ == "__main__":
    rng = np.random.RandomState(SEMILLA)
    records = wfdb.get_record_list(PN_DIR)
    ecg_records = sorted(r for r in records if "ecg" in r.lower())
    print(f"Registros ECG: {ecg_records}\n")

    todas = []
    for i, rn in enumerate(ecg_records):
        try:
            ann = wfdb.rdann(rn, "atr", pn_dir=PN_DIR)
            rec = wfdb.rdrecord(rn, pn_dir=PN_DIR)
            senal = rec.p_signal[:, 0].astype(np.float32)
            filas = analizar_registro(senal, ann.sample, rec.fs, i, rng)
            todas.extend(filas)
            print(f"{rn}: fs={rec.fs} Hz, {len(ann.sample)} eventos, listo")
        except Exception as e:
            print(f"  !! {rn} falló y se omite: {e}")

    df = pd.DataFrame(todas)
    res = resumen(df)
    print("\n===== AUC de la frecuencia cardíaca según dónde termina la ventana =====")
    print(res.to_string(index=False))
    res.to_csv("diagnostico_senal.csv", index=False)
    print("\nGuardado en diagnostico_senal.csv")