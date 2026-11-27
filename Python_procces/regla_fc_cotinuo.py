# Regla de frecuencia cardíaca por el MISMO protocolo continuo - Bradicardia neonatal (PICS)
# Requiere: pip install wfdb scipy pandas matplotlib numpy      (NO necesita tensorflow)
# En Colab: quita el "#" de la línea siguiente y córrela una vez
# !pip install wfdb --quiet
#
# Para qué sirve:
#   falsas_alarmas_continuo.py midió a la red neuronal sobre las 414 h de señal normal.
#   Este script mide a una regla de frecuencia cardíaca (sin red) con las MISMAS ventanas
#   (10 s cada 5 s), la misma zona normal, la misma ventana de detección (20 s) y el mismo
#   conteo de falsas alarmas (episodios unidos a menos de 30 s). Así las dos curvas
#   sensibilidad-vs-falsas-alarmas se pueden poner en la misma figura.
#
# Qué regla usa (la que cualquier monitor puede hacer):
#   - Latidos: se calcula el intervalo entre latidos (RR) dentro de la ventana de 10 s.
#   - "Racha m": alarma si hay m intervalos seguidos más largos que 60/FC_limite segundos.
#     Con m = 2 y FC_limite = 100 lpm es, casi palabra por palabra, la DEFINICIÓN con la que
#     PICS marcó los eventos (FC < 100 lpm durante al menos 2 latidos). Por eso es el
#     comparador más exigente posible.
#   - "FC media": alarma si la frecuencia media de la ventana baja de un límite (regla muy simple).
#   - Si pasa mucho tiempo sin latido, el intervalo "abierto" (tiempo desde el último latido)
#     también cuenta, como haría un monitor en tiempo real (GAP_ABIERTO = True).
#
# Dos fuentes de latidos, para separar "qué tan buena es la regla" de "qué tan bueno es el detector":
#   - regla_latidos_ref: latidos de referencia de PICS (archivo .qrsc, revisados a mano).
#                        Es un detector perfecto: el mejor caso posible para la regla.
#   - regla_latidos_det: latidos detectados aquí con un Pan-Tompkins simplificado sobre la señal.
#                        Es lo que haría el ESP32 sin ayuda: el caso realista.
#
# Salidas (carpeta regla_fc/ y archivos en la carpeta actual):
#   regla_fc/latidos_K.npz              (latidos y ventanas por bebé; permite reanudar)
#   regla_fc_resultado_global.csv       (tabla principal: reglas + red, mismo formato que el de la red)
#   regla_fc_resultado_por_infante.csv
#   regla_fc_calidad_detector.csv       (qué tan bien detecta latidos el detector propio)
#   fig_red_vs_regla_fc.png             (las dos curvas en la misma figura, con dos zonas normales)
#
# Si existe la carpeta continuo/ con los probs_K.npz que dejó falsas_alarmas_continuo.py, también
# recalcula la curva de la red aquí (segundos, sin reentrenar) y la compara con
# continuo_resultado_global.csv para confirmar que el conteo es idéntico.
#
# Zona normal con dos márgenes (BUFFERS_S):
#   60 s  = el del trabajo (señal a más de 60 s de cualquier inicio de evento).
#   180 s = PICS une en un solo evento las bradicardias que ocurren dentro de los 3 minutos
#           siguientes a una bradicardia. Entonces entre 60 s y 180 s después del inicio puede
#           haber bradicardia REAL que el conteo de 60 s cuenta como "falsa alarma".
#           Con 180 s esa contaminación desaparece.
#
# Si se interrumpe, vuelve a correrlo: salta los bebés que ya terminó.
# Prueba rápida: pon MAX_HORAS = 3 (usa solo las primeras 3 horas de cada bebé).

import glob
import os
import time
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import wfdb
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, decimate, find_peaks, resample_poly, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Parámetros ---------------------------------
PN_DIR = "picsdb/1.0.0"
DIR_LOCAL = None          # solo para pruebas: carpeta con registros WFDB en el disco
CARPETA = "regla_fc"
CARPETA_CNN = "continuo"  # donde quedaron los probs_K.npz de la red (opcional)
CSV_CNN = "continuo_resultado_global.csv"   # el que ya tienes (opcional, solo para verificar)
MAX_HORAS = None          # None = grabación completa; 3 = prueba rápida

# --- mismas constantes del protocolo de la red (NO cambiarlas si se quiere comparar) ---
FS_BASE = 500
VENTANA_S = 10
PASO_S = 5
DETECCION_S = 20          # evento detectado si suena la alarma hasta 20 s después de su inicio
REFRACTARIO_S = 30        # alarmas separadas por menos de 30 s cuentan como una sola
BUFFERS_S = [60, 180]     # zona normal: a más de 60 s (el del trabajo) y a más de 180 s (sensibilidad)
LOTE = 512

# --- de la regla ---
RACHAS = [1, 2, 3]                        # m latidos seguidos lentos (2 = definición de PICS)
FC_LIMITES = [100, 90, 80, 70, 60, 50]    # lpm: la racha cuenta si FC < límite (RR > 60/límite)
FC_MEDIA_LIMITES = [150, 140, 130, 120, 110, 100, 90]   # lpm: alarma si la FC media de la ventana < límite
GAP_ABIERTO = True        # contar el tiempo transcurrido desde el último latido como intervalo en curso

# --- de la red (solo para el re-cálculo opcional) ---
UMBRALES_CNN = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995]
PERSISTENCIAS_CNN = [1, 3]

# --- del detector propio de latidos ---
FS_DET = 250              # frecuencia de trabajo del detector
BANDA_HZ = (5.0, 25.0)
INTEGRACION_S = 0.08
REFRACTARIO_LATIDO_S = 0.2
FRACCION_UMBRAL = 0.25    # un pico cuenta como latido si su energía supera esta fracción de la de los QRS vecinos
CUANTIL_UMBRAL = 0.90
CHUNK_MIN = 20            # el detector procesa la grabación en trozos de 20 min (con 15 s de margen)
MARGEN_S = 15
NO_LATIDO = set('+~|x[]!"')   # símbolos de anotación en .qrsc que no son latidos

FUENTES = {"regla_latidos_ref": "ref", "regla_latidos_det": "det"}


# ------------------------- Lectura de un registro --------------------------
def _ruta(nombre):
    ruta = os.path.join(DIR_LOCAL, nombre) if DIR_LOCAL else nombre
    kw = {} if DIR_LOCAL else {"pn_dir": PN_DIR}
    return ruta, kw


def cargar_registro(nombre):
    """Igual que en falsas_alarmas_continuo.py: señal a 500 Hz equivalentes + inicios de evento (s)."""
    ruta, kw = _ruta(nombre)
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


def latidos_referencia(nombre):
    """Latidos de referencia de PICS (.qrsc, revisados a mano), en segundos."""
    ruta, kw = _ruta(nombre)
    fs = float(wfdb.rdheader(ruta, **kw).fs)
    ann = wfdb.rdann(ruta, "qrsc", **kw)
    simb = np.array(ann.symbol)
    t = np.sort(ann.sample[~np.isin(simb, list(NO_LATIDO))] / fs)
    if MAX_HORAS:
        t = t[t < MAX_HORAS * 3600]
    return t, Counter(simb.tolist())


def rejilla(senal):
    """Tiempos de fin de ventana y marca de ventanas planas (las mismas que la red no evalúa)."""
    w, paso = VENTANA_S * FS_BASE, PASO_S * FS_BASE
    ventanas = sliding_window_view(senal, w)[::paso]
    n = len(ventanas)
    plana = np.zeros(n, bool)
    for i0 in range(0, n, LOTE):
        i1 = min(n, i0 + LOTE)
        plana[i0:i1] = np.array(ventanas[i0:i1]).std(axis=1) <= 1e-6
    t_fin = (np.arange(n) * paso + w) / FS_BASE
    return t_fin, plana


# ------------------------- Detector propio de latidos -----------------------
_SOS = butter(2, BANDA_HZ, btype="band", fs=FS_DET, output="sos")


def _latidos_trozo(x500):
    """Pan-Tompkins simplificado sobre un trozo. Devuelve índices de pico a FS_DET."""
    xd = decimate(x500.astype(np.float64), FS_BASE // FS_DET, zero_phase=True)
    y = sosfiltfilt(_SOS, xd)
    e = np.diff(y, prepend=y[0]) ** 2
    e = uniform_filter1d(e, max(1, int(INTEGRACION_S * FS_DET)))
    pk, _ = find_peaks(e, distance=int(REFRACTARIO_LATIDO_S * FS_DET))
    if len(pk) == 0:
        return pk
    h = e[pk]
    bloque = 2 * FS_DET
    nb = len(xd) // bloque + 1
    umbral = np.full(nb, np.inf)
    for b in range(nb):
        c = b * bloque + bloque // 2
        lo = np.searchsorted(pk, c - 8 * FS_DET)
        hi = np.searchsorted(pk, c + 8 * FS_DET)
        if hi - lo >= 3:
            umbral[b] = FRACCION_UMBRAL * np.quantile(h[lo:hi], CUANTIL_UMBRAL)
    return pk[h >= umbral[pk // bloque]]


def detectar_latidos(senal500):
    """Latidos (s) detectados en toda la grabación, procesando por trozos."""
    n = len(senal500)
    paso = CHUNK_MIN * 60 * FS_BASE
    mar = MARGEN_S * FS_BASE
    salida = []
    for a in range(0, n, paso):
        b = min(n, a + paso)
        a0, b0 = max(0, a - mar), min(n, b + mar)
        if b0 - a0 < 2 * FS_BASE:
            continue
        pk = _latidos_trozo(senal500[a0:b0])
        t = (a0 + pk * (FS_BASE / FS_DET)) / FS_BASE
        salida.append(t[(t >= a / FS_BASE) & (t < b / FS_BASE)])
    return np.concatenate(salida) if salida else np.array([])


def calidad_detector(det, ref, tol=0.05):
    """Sensibilidad y valor predictivo del detector propio frente a los latidos de referencia."""
    if len(det) == 0 or len(ref) == 0:
        return dict(desfase_ms=np.nan, sensibilidad=np.nan, ppv=np.nan)

    def cercano(a, b):
        i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
        izq, der = b[i - 1], b[i]
        return np.where(np.abs(a - izq) <= np.abs(a - der), izq, der)

    d = det - cercano(det, ref)
    desfase = float(np.median(d[np.abs(d) < 0.15])) if (np.abs(d) < 0.15).any() else 0.0
    det2 = det - desfase                      # el detector marca un poco después del pico R: se compensa
    sens = float(np.mean(np.abs(ref - cercano(ref, det2)) <= tol))
    ppv = float(np.mean(np.abs(det2 - cercano(det2, ref)) <= tol))
    return dict(desfase_ms=desfase * 1000, sensibilidad=sens, ppv=ppv)


# ------------------------- Puntajes de la regla por ventana -----------------
def puntajes(latidos, t_fin, plana):
    """Para cada ventana devuelve, en segundos de RR (más alto = más bradicárdico):
       - racha[m]: el mayor valor x tal que hay m intervalos seguidos >= x  (alarma si x >= 60/FC_limite)
       - media:    intervalo medio de la ventana (FC media = 60/media)
       Ventanas planas -> NaN (igual que en la red)."""
    n = len(t_fin)
    racha = {m: np.zeros(n) for m in RACHAS}
    media = np.zeros(n)
    if len(latidos) == 0:
        for m in RACHAS:
            racha[m][:] = np.nan
        return racha, np.full(n, np.nan)
    t_ini = t_fin - VENTANA_S
    lo = np.searchsorted(latidos, t_ini, side="right")    # primer latido dentro de (t_ini, t_fin]
    hi = np.searchsorted(latidos, t_fin, side="right")
    rr_todos = np.diff(latidos)                           # rr_todos[k] = latidos[k+1] - latidos[k]
    for i in range(n):
        if plana[i]:
            for m in RACHAS:
                racha[m][i] = np.nan
            media[i] = np.nan
            continue
        a, b = lo[i], hi[i]
        rr = rr_todos[a:b - 1] if b - a >= 2 else np.empty(0)
        gap = t_fin[i] - (latidos[b - 1] if b > 0 else 0.0)     # tiempo desde el último latido
        local = np.append(rr, gap) if GAP_ABIERTO else rr
        media[i] = rr.mean() if len(rr) else gap
        for m in RACHAS:
            if len(local) >= m:
                racha[m][i] = local.max() if m == 1 else sliding_window_view(local, m).min(axis=1).max()
    return racha, media


# ------------------------------ Análisis -----------------------------------
# (idéntico a falsas_alarmas_continuo.py, salvo que el margen de la zona normal es un argumento)
def zona_normal(t_fin, onsets, buffer_s):
    """True si la ventana está a más de buffer_s de cualquier inicio de evento."""
    if len(onsets) == 0:
        return np.ones(len(t_fin), bool)
    t_ini = t_fin - VENTANA_S
    i = np.searchsorted(onsets, t_ini - buffer_s, side="left")
    hay = i < len(onsets)
    cerca = np.zeros(len(t_fin), bool)
    cerca[hay] = onsets[i[hay]] <= t_fin[hay] + buffer_s
    return ~cerca


def disparos(p, umbral, k):
    """True donde suena la alarma: las últimas k ventanas superan el umbral."""
    with np.errstate(invalid="ignore"):
        a = p >= umbral
    if k == 1:
        return a
    cuenta = np.convolve(a.astype(int), np.ones(k, int), mode="full")[: len(a)]
    return cuenta >= k


def evaluar_infante(t_fin, p, onsets, umbral, k, buffer_s):
    trig = disparos(p, umbral, k)
    normal = zona_normal(t_fin, onsets, buffer_s) & ~np.isnan(p)
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


def _fila(inf, grupo, fuente, regla, k, buffer_s, umbral, r):
    return dict(infante=inf, grupo=grupo, fuente=fuente, regla=regla, persistencia=k,
                buffer_s=buffer_s, umbral=umbral, eventos=r["eventos"], detectados=r["detectados"],
                retraso_mediano_s=(np.median(r["retrasos"]) if r["retrasos"] else np.nan),
                falsas_alarmas=r["falsas"], horas_normales=r["horas"],
                falsas_por_hora=(r["falsas"] / r["horas"] if r["horas"] > 0 else np.nan),
                _retrasos=r["retrasos"])


def filas_regla(d):
    """Todas las reglas de un bebé (a partir del npz de latidos)."""
    inf, grupo = str(d["nombre"]), int(d["indice"])
    t_fin, plana, onsets = d["t_fin"], d["plana"], d["onsets"]
    filas = []
    for fuente, clave in FUENTES.items():
        racha, media = puntajes(d["latidos_" + clave], t_fin, plana)
        for buf in BUFFERS_S:
            for m in RACHAS:
                for fc in FC_LIMITES:
                    r = evaluar_infante(t_fin, racha[m], onsets, 60.0 / fc, 1, buf)
                    filas.append(_fila(inf, grupo, fuente, f"racha{m}", 1, buf, fc, r))
            for fc in FC_MEDIA_LIMITES:
                r = evaluar_infante(t_fin, media, onsets, 60.0 / fc, 1, buf)
                filas.append(_fila(inf, grupo, fuente, "fc_media", 1, buf, fc, r))
    return filas


def filas_red(archivo):
    d = np.load(archivo, allow_pickle=True)
    inf, grupo = str(d["nombre"]), int(d["indice"])
    filas = []
    for fuente, campo in (("cnn_float32", "p_float32"), ("cnn_int8", "p_int8")):
        for buf in BUFFERS_S:
            for k in PERSISTENCIAS_CNN:
                for u in UMBRALES_CNN:
                    r = evaluar_infante(d["t_fin"], d[campo], d["onsets"], u, k, buf)
                    filas.append(_fila(inf, grupo, fuente, "red", k, buf, u, r))
    return filas


def agregar(por_inf):
    glob_filas = []
    claves = ["fuente", "regla", "persistencia", "buffer_s", "umbral"]
    for key, sub in por_inf.groupby(claves):
        retrasos = [x for lst in sub["_retrasos"] for x in lst]
        horas = sub.horas_normales.sum()
        glob_filas.append(dict(
            zip(claves, key),
            eventos=int(sub.eventos.sum()), detectados=int(sub.detectados.sum()),
            sensibilidad=sub.detectados.sum() / max(sub.eventos.sum(), 1),
            retraso_mediano_s=(np.median(retrasos) if retrasos else np.nan),
            falsas_alarmas=int(sub.falsas_alarmas.sum()), horas_normales=horas,
            falsas_por_hora_global=sub.falsas_alarmas.sum() / horas if horas > 0 else np.nan,
            falsas_por_hora_mediana_infantes=sub.falsas_por_hora.median(),
            falsas_por_hora_peor_infante=sub.falsas_por_hora.max()))
    return pd.DataFrame(glob_filas).sort_values(claves).reset_index(drop=True)


def verificar_contra_csv(glob_df):
    """La red recalculada aquí con zona normal de 60 s debe dar lo mismo que continuo_resultado_global.csv."""
    if not os.path.exists(CSV_CNN):
        return
    ref = pd.read_csv(CSV_CNN)
    mio = glob_df[(glob_df.regla == "red") & (glob_df.buffer_s == 60)]
    ref = ref.assign(fuente="cnn_" + ref.config)
    m = mio.merge(ref, on=["fuente", "persistencia", "umbral"], suffixes=("", "_csv"))
    if m.empty:
        print("[verificación] no encontré filas comparables con", CSV_CNN)
        return
    dif_s = (m.sensibilidad - m.sensibilidad_csv).abs().max()
    dif_f = (m.falsas_por_hora_global - m.falsas_por_hora_global_csv).abs().max()
    estado = "OK, el conteo es idéntico" if dif_s < 1e-9 and dif_f < 1e-9 else "REVISAR: no coincide"
    print(f"[verificación] red recalculada vs {CSV_CNN}: dif máx. sensibilidad {dif_s:.2e}, "
          f"dif máx. falsas/h {dif_f:.2e} -> {estado}")


# ------------------------------- Figura ------------------------------------
def figura(glob_df, salida):
    estilos = [
        ("cnn_float32", "red", 1, "Red CNN (float32)", "#0072B2", "-", "o"),
        ("regla_latidos_det", "racha2", 1, "Regla: 2 latidos lentos (detector propio)", "#D55E00", "-", "s"),
        ("regla_latidos_ref", "racha2", 1, "Regla: 2 latidos lentos (latidos de referencia)", "#009E73", "--", "^"),
        ("regla_latidos_det", "fc_media", 1, "Regla: FC media baja (detector propio)", "#7f7f7f", ":", "D"),
    ]
    fig, ejes = plt.subplots(1, len(BUFFERS_S), figsize=(5.8 * len(BUFFERS_S), 5.2), sharey=True)
    ejes = np.atleast_1d(ejes)
    for ax, buf in zip(ejes, BUFFERS_S):
        for fuente, regla, k, etiqueta, color, ls, mk in estilos:
            sub = glob_df[(glob_df.fuente == fuente) & (glob_df.regla == regla)
                          & (glob_df.persistencia == k) & (glob_df.buffer_s == buf)].sort_values("umbral")
            if sub.empty:
                continue
            ax.plot(sub.falsas_por_hora_global, sub.sensibilidad, ls, marker=mk, ms=5,
                    color=color, label=etiqueta)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("Falsas alarmas por hora (en señal normal)")
        ax.set_title(f"Zona normal: a más de {buf} s de cualquier evento"
                     + (" (la del trabajo)" if buf == 60 else " (excluye bradicardias unidas)"), fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#d8d2c4", lw=0.8)
        ax.axvline(1, color="#999999", lw=1, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    ejes[0].set_ylabel("Eventos detectados (fracción)")
    ejes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor="white")


# ------------------------------- Principal ----------------------------------
def lista_registros():
    if DIR_LOCAL:
        hea = sorted(glob.glob(os.path.join(DIR_LOCAL, "*.hea")))
        return sorted(os.path.splitext(os.path.basename(h))[0] for h in hea if "ecg" in h.lower())
    return sorted(r for r in wfdb.get_record_list(PN_DIR) if "ecg" in r.lower())


def procesar_bebes(nombres):
    os.makedirs(CARPETA, exist_ok=True)
    calidad = []
    for k, nombre in enumerate(nombres):
        archivo = os.path.join(CARPETA, f"latidos_{k}.npz")
        if os.path.exists(archivo):
            print(f"[{k}] {nombre}: ya estaba hecho")
            continue
        t0 = time.time()
        senal, onsets = cargar_registro(nombre)
        ref, simbolos = latidos_referencia(nombre)
        t1 = time.time()
        t_fin, plana = rejilla(senal)
        det = detectar_latidos(senal)
        t2 = time.time()
        horas = len(senal) / FS_BASE / 3600
        np.savez_compressed(archivo, indice=k, nombre=nombre, t_fin=t_fin, plana=plana, onsets=onsets,
                            latidos_ref=ref, latidos_det=det, horas=horas)
        q = calidad_detector(det, ref)
        fc_med = 60.0 / np.median(np.diff(ref)) if len(ref) > 2 else np.nan
        print(f"[{k}] {nombre}: {horas:.1f} h, {len(onsets)} eventos, {len(ref)} latidos de referencia "
              f"({dict(simbolos)}), {len(det)} detectados | FC mediana {fc_med:.0f} lpm | detector: "
              f"sens {q['sensibilidad']:.3f}, PPV {q['ppv']:.3f}, desfase {q['desfase_ms']:.0f} ms | "
              f"leer {t1 - t0:.0f}s, detectar {t2 - t1:.0f}s")
        del senal


def principal():
    nombres = lista_registros()
    print(f"{len(nombres)} bebés. Ventana {VENTANA_S} s cada {PASO_S} s. Zonas normales: {BUFFERS_S} s.\n")
    procesar_bebes(nombres)

    arch_lat = sorted(glob.glob(os.path.join(CARPETA, "latidos_*.npz")),
                      key=lambda a: int(os.path.basename(a)[8:-4]))
    filas, calidad = [], []
    for a in arch_lat:
        d = np.load(a, allow_pickle=True)
        filas += filas_regla(d)
        q = calidad_detector(d["latidos_det"], d["latidos_ref"])
        calidad.append(dict(infante=str(d["nombre"]), latidos_ref=len(d["latidos_ref"]),
                            latidos_det=len(d["latidos_det"]),
                            fc_mediana_ref_lpm=(60.0 / np.median(np.diff(d["latidos_ref"]))
                                                if len(d["latidos_ref"]) > 2 else np.nan), **q))
    pd.DataFrame(calidad).round(4).to_csv("regla_fc_calidad_detector.csv", index=False)

    arch_cnn = sorted(glob.glob(os.path.join(CARPETA_CNN, "probs_*.npz")),
                      key=lambda a: int(os.path.basename(a)[6:-4]))
    if arch_cnn:
        for a in arch_cnn:
            filas += filas_red(a)
        # las ventanas de la red y las de la regla deben ser las mismas
        for a_l, a_c in zip(arch_lat, arch_cnn):
            dl, dc = np.load(a_l, allow_pickle=True), np.load(a_c, allow_pickle=True)
            if (len(dl["t_fin"]) != len(dc["t_fin"]) or
                    not np.array_equal(np.isnan(dc["p_float32"]), dl["plana"])):
                print(f"[aviso] {dl['nombre']}: las ventanas de la red y de la regla NO coinciden "
                      "(¿MAX_HORAS distinto?). La comparación no sería válida.")
    else:
        print(f"[aviso] no hay {CARPETA_CNN}/probs_*.npz: solo se calcula la regla (sin la curva de la red).")

    por_inf = pd.DataFrame(filas)
    glob_df = agregar(por_inf)
    glob_df.to_csv("regla_fc_resultado_global.csv", index=False)
    por_inf.drop(columns="_retrasos").to_csv("regla_fc_resultado_por_infante.csv", index=False)
    if arch_cnn:
        verificar_contra_csv(glob_df)
    figura(glob_df, "fig_red_vs_regla_fc.png")

    pd.set_option("display.width", 220)
    cols = ["fuente", "regla", "umbral", "eventos", "detectados", "sensibilidad", "retraso_mediano_s",
            "falsas_por_hora_global", "falsas_por_hora_mediana_infantes", "falsas_por_hora_peor_infante"]
    for buf in BUFFERS_S:
        print(f"\n===== ZONA NORMAL A MÁS DE {buf} s =====")
        print("(umbral = FC en lpm para las reglas, probabilidad para la red)")
        sub = glob_df[(glob_df.buffer_s == buf) & (glob_df.persistencia == 1)
                      & (glob_df.regla.isin(["racha2", "fc_media", "red"]))
                      & (glob_df.fuente != "cnn_int8")]
        print(sub[cols].round(3).to_string(index=False))
        print(f"\n--- Mejor punto con 1 falsa alarma por hora o menos (global, zona normal {buf} s) ---")
        for (fuente, regla), s in glob_df[(glob_df.buffer_s == buf) & (glob_df.persistencia == 1)].groupby(
                ["fuente", "regla"]):
            ok = s[s.falsas_por_hora_global <= 1]
            if len(ok):
                f = ok.sort_values("sensibilidad", ascending=False).iloc[0]
                print(f"{fuente:20s} {regla:9s} umbral {f.umbral:>6}: detecta {f.sensibilidad:.2f}, "
                      f"retraso mediano {f.retraso_mediano_s:.1f} s, {f.falsas_por_hora_global:.2f} falsas/h")
            else:
                print(f"{fuente:20s} {regla:9s} ningún umbral probado llega a 1 falsa alarma por hora")
    print("\nGuardado: regla_fc_resultado_global.csv, regla_fc_resultado_por_infante.csv, "
          "regla_fc_calidad_detector.csv, fig_red_vs_regla_fc.png")


if __name__ == "__main__":
    principal()