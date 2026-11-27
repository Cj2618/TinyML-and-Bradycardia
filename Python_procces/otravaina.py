# Clasifica las alarmas de la regla de una corrida HIL (CSV *_ventanas.csv) contra las anotaciones de PICS.
# Responde: ¿los vetos de la CNN quitaron falsas alarmas o eventos reales? ¿qué eventos se perdieron y por qué?
# Usa las mismas definiciones del artículo: evento detectado = alarma en una ventana que termina entre
# el inicio del evento y 20 s después; "zona normal" = ventana a más de 60 s de cualquier inicio.
# Uso:
#   python clasificar_alarmas_hil.py hil_infant2_ecg_ventanas.csv infant2_ecg --minutos 120
#   (infant2_ecg = ruta sin extensión de la carpeta con .hea/.atr; si no la tienes, usa --pn picsdb/1.0.0)
import argparse
import numpy as np
import pandas as pd
import wfdb

ap = argparse.ArgumentParser()
ap.add_argument("ventanas")
ap.add_argument("registro", help="infant2_ecg (con .hea y .atr en la misma carpeta) o solo el nombre si usas --pn")
ap.add_argument("--pn", default=None, help="directorio de PhysioNet, p. ej. picsdb/1.0.0")
ap.add_argument("--inicio", type=float, default=0.0, help="minuto donde empezó el trozo (como en replay_hil)")
ap.add_argument("--minutos", type=float, default=0.0)
a = ap.parse_args()

kw = {"pn_dir": a.pn} if a.pn else {}
ann = wfdb.rdann(a.registro, "atr", **kw)
hdr = wfdb.rdheader(a.registro, **kw)
fs = float(hdr.fs)
sim = np.array(ann.symbol)
m = ann.sample[sim == "["] if (sim == "[").any() else ann.sample
onsets = np.sort(m / fs) - a.inicio * 60
d = pd.read_csv(a.ventanas)
d = d[d.t_fin_s.notna()]
dur = d.t_fin_s.max()
onsets = onsets[(onsets >= 0) & (onsets < dur - 10)]
print(f"{len(onsets)} eventos en el trozo analizado, {len(d)} ventanas\n")

alarma_regla = d.alarma_regla.values == 1
final = d.alarma_final.values == 1
t = d.t_fin_s.values
p = d.p.values

print("=== Eventos ===")
for o in onsets:
    en_rango = (t >= o) & (t <= o + 20)
    reg = en_rango & alarma_regla
    fin = en_rango & final
    if fin.any():
        estado = "DETECTADO (retraso %.1f s)" % (t[fin][0] - o)
    elif reg.any():
        estado = "PERDIDO POR VETO de la CNN (p = %s)" % ", ".join("%.2f" % x for x in p[reg])
    else:
        estado = "PERDIDO: la regla no alarmó"
    print(f"  inicio {o:8.1f} s -> {estado}")

print("\n=== Alarmas de la regla ===")
print("  t_fin(s)   p      final  clasificación")
for i in np.where(alarma_regla)[0]:
    dt = t[i] - onsets
    if ((dt >= 0) & (dt <= 20)).any():
        clas = "dentro de los 20 s de un evento (verdadera)"
    else:
        t_ini = t[i] - 10
        cerca = (onsets >= t_ini - 60) & (onsets <= t[i] + 60)
        if cerca.any():
            clas = "zona de 60 s alrededor de un evento (no cuenta como falsa)"
        else:
            previo = dt[dt > 0]
            clas = "FALSA" + (" (a %.0f s de un evento anterior)" % previo.min() if len(previo) else "")
    print(f"  {t[i]:8.0f}  {p[i]:5.3f}  {'sí' if final[i] else 'VETO'}   {clas}")