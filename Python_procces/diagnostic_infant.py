# Diagnóstico por infante sobre pics_todos_ventanas.npz (NO descarga nada, tarda segundos)
# Requiere: numpy, scipy, scikit-learn, pandas (ya vienen en Colab)
#
# Responde: en cada infante, ¿la frecuencia cardíaca (FC) separa por sí sola las
# ventanas con evento de las normales? Y, ¿la polaridad de la señal es distinta?
#  - Si la FC separa bien pero el modelo falla -> el problema es del modelo/generalización.
#  - Si la FC tampoco separa -> el problema está en los datos de ese infante.
# De paso, la columna AUC_fc_min es tu línea base de "regla simple" para el artículo.

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew
from sklearn.metrics import roc_auc_score

DATA = "pics_todos_ventanas.npz"
FS = 500  # Hz (la señal del .npz está a la frecuencia original, sin submuestrear)


def features_fc(x, fs=FS):
    """FC media y FC mínima de una ventana, con un detector de picos simple."""
    b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
    y = np.abs(filtfilt(b, a, x))
    picos, _ = find_peaks(y, distance=int(0.25 * fs), height=0.5 * np.percentile(y, 99))
    rr = np.diff(picos) / fs
    rr = rr[(rr > 0.2) & (rr < 3.0)]
    if len(rr) < 3:
        return np.nan, np.nan
    return 60.0 / rr.mean(), 60.0 / rr.max()


d = np.load(DATA)
X, y, g = d["X"], d["y"], d["grupos"]
print(f"Datos: X={X.shape}, infantes={len(np.unique(g))}\n")

fc = np.array([features_fc(x) for x in X])          # (n, 2)
sk = skew(X, axis=1)                                 # >0 picos R hacia arriba, <0 invertidos
df = pd.DataFrame(dict(infante=g, y=y, fc_media=fc[:, 0], fc_min=fc[:, 1], asim=sk))


def auc(sub, col):
    sub = sub.dropna(subset=[col])
    if sub.y.nunique() < 2:
        return np.nan
    return roc_auc_score(sub.y, -sub[col])   # menor FC -> más probable evento


filas = []
for k, sub in df.groupby("infante"):
    filas.append(dict(
        infante=k,
        n_evento=int((sub.y == 1).sum()), n_normal=int((sub.y == 0).sum()),
        AUC_fc_media=round(auc(sub, "fc_media"), 3),
        AUC_fc_min=round(auc(sub, "fc_min"), 3),
        fc_evento=round(sub[sub.y == 1].fc_media.mean(), 1),
        fc_normal=round(sub[sub.y == 0].fc_media.mean(), 1),
        asim_media=round(sub.asim.mean(), 2),
        pct_asim_negativa=round(100 * (sub.asim < 0).mean(), 0),
    ))

res = pd.DataFrame(filas)
print("===== Por infante (AUC > 0.5 = la FC baja en el evento) =====")
print(res.to_string(index=False))
print(f"\nTodos juntos: AUC_fc_media={auc(df, 'fc_media'):.3f} | AUC_fc_min={auc(df, 'fc_min'):.3f}")
res.to_csv("diagnostico_por_infante.csv", index=False)
print("Guardado en diagnostico_por_infante.csv")