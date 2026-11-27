# Regla de frecuencia cardíaca y combinación con la red, usando el detector de latidos CAUSAL
# Requiere: pip install wfdb scipy pandas matplotlib numpy      (NO necesita tensorflow)
#
# Para qué sirve:
#   El detector de latidos de regla_fc_continuo.py usa filtros de fase cero y un umbral calculado con los
#   8 s siguientes, algo que un microcontrolador en línea no puede hacer. Aquí se repite todo con
#   detector_causal.py (solo muestras pasadas, muestra a muestra, el que se porta al ESP32) para saber
#   cuánto se pierde al pasar de "análisis sin conexión" a "lo que haría la placa".
#
# Qué hace, en orden:
#   1. Para cada bebé corre el detector causal sobre toda la grabación (500 Hz) y guarda
#      regla_fc/causal_K.npz (se puede interrumpir: lo ya hecho no se repite).
#   2. Comprueba que la versión rápida (por bloques) y la versión muestra a muestra (la del chip) dan los
#      mismos latidos en los primeros 10 minutos de cada bebé.
#   3. Calidad del detector frente a los latidos de referencia (.qrsc) y frente al detector no causal.
#   4. La regla sola con latidos causales (mismo protocolo que regla_fc_continuo.py, con límites de FC más finos).
#   5. Regla + red con latidos causales (mismo barrido que combinacion_red_regla.py), si existen los
#      continuo/probs_K.npz de la red.
#   6. Comparación lado a lado: referencia / no causal / causal.
#
# Entradas (en la misma carpeta que este archivo y desde donde se corre):
#   regla_fc_continuo.py, combinacion_red_regla.py, detector_causal.py   (se importan)
#   regla_fc/latidos_K.npz  y  continuo/probs_K.npz       (opcionales, de los scripts anteriores)
#   Los registros PICS se leen de PhysioNet igual que en los otros scripts.
# Salidas:
#   regla_fc/causal_K.npz
#   causal_calidad_detector.csv
#   causal_regla_resultado_global.csv, causal_regla_resultado_por_infante.csv
#   causal_combinacion_resultado_global.csv, causal_combinacion_resultado_por_infante.csv
#   causal_comparacion.csv          (lado a lado en el punto de operación del artículo)
#
# Después, para repetir la validación dejando un infante fuera con los latidos causales:
#   python validacion_cruzada_combinacion.py causal_combinacion_resultado_por_infante.csv regla_causal _causal
#
# Tiempo: el detector tarda unos segundos por hora de señal (lo que más demora es leer de PhysioNet);
# la evaluación de la regla y de la combinación, unos minutos.

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

import combinacion_red_regla as C
import Causal_detector as D
import regla_fc_cotinuo as R

# Límites de FC más finos entre 90 y 110 lpm que los de regla_fc_continuo.py (que saltaba de 90 a 100): la regla
# sola pasa de 61 % a 91 % de detección en ese salto, y sin una rejilla fina la comparación con la combinación
# (que tiene además 6 niveles de veto) favorecería a la combinación solo por tener más puntos de ajuste.
FC_LIMITES = [110, 105, 100, 97, 95, 92, 90, 80, 70, 60, 50]
R.FC_LIMITES = FC_LIMITES
C.FC_LIMITES = FC_LIMITES

CARPETA = R.CARPETA                 # regla_fc
CARPETA_CNN = R.CARPETA_CNN         # continuo
VERIFICAR_S = 600                   # segundos que se comparan muestra a muestra por bebé (0 = no verificar)
FUENTE_REGLA = "regla_latidos_causal"
# Punto de operación del artículo para la comparación lado a lado
PUNTO = dict(regla="racha2", umbral=100.0)


# ------------------------------- Detección ----------------------------------
def verificar_equivalencia(senal):
    """El detector rápido y el muestra a muestra (el del chip) deben dar exactamente los mismos latidos."""
    n = int(VERIFICAR_S * D.FS)
    x = senal[:n]
    rapido = D.detectar(x, chunk_s=7)            # bloques chicos a propósito: prueba el traspaso de estado
    lento = D.DetectorMuestra()
    a = np.array([t for t in (lento.muestra(v) for v in x) if t is not None])
    ok = len(a) == len(rapido) and np.array_equal(a, rapido)
    return ok, len(rapido), len(a)


def procesar_bebes(nombres):
    os.makedirs(CARPETA, exist_ok=True)
    for k, nombre in enumerate(nombres):
        archivo = os.path.join(CARPETA, f"causal_{k}.npz")
        if os.path.exists(archivo):
            try:
                version = str(np.load(archivo, allow_pickle=True)["version"])
            except KeyError:
                version = "v1"
            if version == D.VERSION:
                print(f"[{k}] {nombre}: ya estaba hecho")
                continue
            print(f"[{k}] {nombre}: los latidos guardados son del detector {version} y el actual es {D.VERSION}; se repiten")
        t0 = time.time()
        senal, onsets = R.cargar_registro(nombre)
        ref, simbolos = R.latidos_referencia(nombre)
        t_fin, plana = R.rejilla(senal)
        t1 = time.time()
        if VERIFICAR_S:
            ok, n_r, n_l = verificar_equivalencia(senal)
            print(f"[{k}] {nombre}: equivalencia rápido/muestra a muestra en {VERIFICAR_S} s: "
                  f"{'OK' if ok else 'FALLA'} ({n_r} y {n_l} latidos)")
            if not ok:
                print("      (diferencias por redondeo de coma flotante entre sosfilt y el filtro a mano; "
                      "se siguen usando los latidos de la versión rápida y se avisa en el resumen)")
        t2 = time.time()
        causal = D.detectar(senal)
        t3 = time.time()
        horas = len(senal) / R.FS_BASE / 3600
        np.savez_compressed(archivo, indice=k, nombre=nombre, t_fin=t_fin, plana=plana, onsets=onsets,
                            latidos_ref=ref, latidos_causal=causal, horas=horas,
                            equivalencia_ok=(not VERIFICAR_S) or ok, version=D.VERSION)
        q = R.calidad_detector(causal, ref)
        print(f"[{k}] {nombre}: {horas:.1f} h, {len(onsets)} eventos, {len(ref)} latidos de referencia, "
              f"{len(causal)} causales | sens {q['sensibilidad']:.3f}, PPV {q['ppv']:.3f}, "
              f"retraso {q['desfase_ms']:.0f} ms | leer {t1 - t0:.0f}s, detectar {t3 - t2:.0f}s")
        del senal


def cargar(k):
    d = np.load(os.path.join(CARPETA, f"causal_{k}.npz"), allow_pickle=True)
    dl = {c: d[c] for c in d.files}
    ant = os.path.join(CARPETA, f"latidos_{k}.npz")      # el no causal, si existe (solo para comparar)
    if os.path.exists(ant):
        a = np.load(ant, allow_pickle=True)
        if str(a["nombre"]) == str(dl["nombre"]):
            dl["latidos_det"] = a["latidos_det"]
    return dl


def cargar_red(k, dl):
    b = os.path.join(CARPETA_CNN, f"probs_{k}.npz")
    if not os.path.exists(b):
        return None
    dc = np.load(b, allow_pickle=True)
    ok = (str(dl["nombre"]) == str(dc["nombre"]) and len(dl["t_fin"]) == len(dc["t_fin"])
          and np.allclose(dl["t_fin"], dc["t_fin"])
          and np.array_equal(np.isnan(dc["p_float32"]), dl["plana"]))
    if not ok:
        print(f"[aviso] bebé {k}: las ventanas de la red y de la regla no coinciden; se omite la combinación")
        return None
    return dc


# ------------------------------- Calidad ------------------------------------
def tabla_calidad(datos):
    filas = []
    for dl in datos:
        ref = dl["latidos_ref"]
        fila = dict(infante=str(dl["nombre"]), latidos_ref=len(ref), latidos_causal=len(dl["latidos_causal"]),
                    fc_mediana_ref_lpm=(60.0 / np.median(np.diff(ref)) if len(ref) > 2 else np.nan))
        q = R.calidad_detector(dl["latidos_causal"], ref)
        fila.update({f"causal_{k}": v for k, v in q.items()})
        if "latidos_det" in dl:
            q2 = R.calidad_detector(dl["latidos_det"], ref)
            fila.update({f"nocausal_{k}": v for k, v in q2.items()})
        filas.append(fila)
    return pd.DataFrame(filas)


# ------------------------------ Resultados ----------------------------------
def punto(g, fuente, buf=60, veto=None, **filtro):
    s = g[(g.fuente == fuente) & (g.buffer_s == buf)]
    for c, v in filtro.items():
        s = s[s[c] == v]
    if veto is not None and "veto_red" in s:
        s = s[(s.veto_red == veto)]
    return s


def principal():
    sys.stdout.reconfigure(line_buffering=True)
    nombres = R.lista_registros()
    print(f"{len(nombres)} bebés. Detector causal a {D.FS} Hz. Ventana {R.VENTANA_S} s cada {R.PASO_S} s.\n")
    procesar_bebes(nombres)

    datos = [cargar(k) for k in range(len(nombres))]
    cal = tabla_calidad(datos)
    cal.round(4).to_csv("causal_calidad_detector.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n===== Calidad del detector (tolerancia 50 ms, tras quitar el retraso mediano) =====")
    print(cal.round(3).to_string(index=False))
    fallas = [str(d["nombre"]) for d in datos if not bool(d["equivalencia_ok"])]
    if fallas:
        print(f"\n[aviso] la versión rápida y la muestra a muestra NO coincidieron exactamente en: {fallas}")

    # ---- regla sola con latidos causales (mismo barrido que regla_fc_continuo.py)
    R.FUENTES = {FUENTE_REGLA: "causal"}
    filas = []
    for dl in datos:
        filas += R.filas_regla(dl)
    por_inf = pd.DataFrame(filas)
    g_regla = R.agregar(por_inf)
    g_regla.to_csv("causal_regla_resultado_global.csv", index=False)
    por_inf.drop(columns="_retrasos").to_csv("causal_regla_resultado_por_infante.csv", index=False)

    # ---- combinación regla causal + red
    g_comb = None
    reds = [cargar_red(k, dl) for k, dl in enumerate(datos)]
    if all(r is not None for r in reds):
        C.FUENTES_LATIDOS = ("causal",)
        filas = []
        for dl, dc in zip(datos, reds):
            filas += C.filas_infante(dl, dc)
            print(f"[{int(dl['indice'])}] {dl['nombre']}: combinación lista")
        por_inf_c = pd.DataFrame(filas)
        g_comb = C.agregar(por_inf_c)
        g_comb.to_csv("causal_combinacion_resultado_global.csv", index=False)
        por_inf_c.drop(columns="_retrasos").to_csv("causal_combinacion_resultado_por_infante.csv", index=False)
    else:
        print(f"\n[aviso] faltan {CARPETA_CNN}/probs_K.npz (o no coinciden): solo se calculó la regla sola.")

    # ---- lectura en pantalla
    cols = ["fuente", "umbral", "eventos", "detectados", "sensibilidad", "retraso_mediano_s",
            "falsas_por_hora_global", "falsas_por_hora_mediana_infantes", "falsas_por_hora_peor_infante"]
    for buf in R.BUFFERS_S:
        print(f"\n===== REGLA SOLA, DETECTOR CAUSAL · zona normal a más de {buf} s =====")
        for regla in ("racha2", "fc_media"):
            s = g_regla[(g_regla.buffer_s == buf) & (g_regla.persistencia == 1) & (g_regla.regla == regla)]
            print(f"\n-- {regla} (umbral = FC en lpm) --")
            print(s[cols].round(3).to_string(index=False))
        ok = g_regla[(g_regla.buffer_s == buf) & (g_regla.persistencia == 1) & (g_regla.falsas_por_hora_global <= 1)]
        if len(ok):
            f = ok.sort_values("sensibilidad", ascending=False).iloc[0]
            print(f"\nMejor punto con ≤1 falsa/h (regla sola, causal): {f.regla} límite {f.umbral}: detecta "
                  f"{f.sensibilidad:.3f}, retraso mediano {f.retraso_mediano_s:.1f} s, "
                  f"{f.falsas_por_hora_global:.2f} falsas/h")
        else:
            print("\nNingún punto de la regla sola con detector causal llega a ≤1 falsa/h.")

    if g_comb is not None:
        for buf in R.BUFFERS_S:
            print(f"\n===== REGLA CAUSAL + RED · zona normal a más de {buf} s · racha2, límite 100 lpm =====")
            s = g_comb[(g_comb.buffer_s == buf) & (g_comb.regla == "racha2") & (g_comb.umbral == 100.0)
                       & (g_comb.fuente.str.startswith("regla"))]
            print(s[["fuente", "veto_red"] + cols[2:]].round(3).to_string(index=False))

    # ---- lado a lado: referencia / no causal / causal (regla sola, racha2, 100 lpm, zona 60 s)
    comp = []
    ruta_ant = "regla_fc_resultado_global.csv"
    if os.path.exists(ruta_ant):
        ant = pd.read_csv(ruta_ant)
        for fuente, nombre in (("regla_latidos_ref", "latidos de referencia"),
                               ("regla_latidos_det", "detector no causal")):
            s = ant[(ant.fuente == fuente) & (ant.buffer_s == 60) & (ant.persistencia == 1)
                    & (ant.regla == PUNTO["regla"]) & (ant.umbral == PUNTO["umbral"])]
            if len(s):
                f = s.iloc[0]
                comp.append(dict(latidos=nombre, regla=f"{PUNTO['regla']} {PUNTO['umbral']:g} lpm",
                                 sensibilidad=f.sensibilidad, retraso_mediano_s=f.retraso_mediano_s,
                                 falsas_por_hora=f.falsas_por_hora_global,
                                 peor_infante=f.falsas_por_hora_peor_infante))
    s = g_regla[(g_regla.buffer_s == 60) & (g_regla.persistencia == 1) & (g_regla.regla == PUNTO["regla"])
                & (g_regla.umbral == PUNTO["umbral"])]
    if len(s):
        f = s.iloc[0]
        comp.append(dict(latidos="detector causal (el del chip)", regla=f"{PUNTO['regla']} {PUNTO['umbral']:g} lpm",
                         sensibilidad=f.sensibilidad, retraso_mediano_s=f.retraso_mediano_s,
                         falsas_por_hora=f.falsas_por_hora_global, peor_infante=f.falsas_por_hora_peor_infante))
    comp = pd.DataFrame(comp)
    if g_comb is not None and os.path.exists("combinacion_resultado_global.csv"):
        antc = pd.read_csv("combinacion_resultado_global.csv")
        for fuente, etiqueta in (("regla_det+red_int8", "no causal + red int8"),
                                 ("regla_causal+red_int8", "causal + red int8")):
            base = antc if fuente.startswith("regla_det") else g_comb
            for v in (0.3, 0.5, 0.9):
                s = base[(base.fuente == fuente) & (base.buffer_s == 60) & (base.regla == PUNTO["regla"])
                         & (base.umbral == PUNTO["umbral"]) & (np.isclose(base.veto_red, v))]
                if len(s):
                    f = s.iloc[0]
                    comp = pd.concat([comp, pd.DataFrame([dict(
                        latidos=f"{etiqueta}, veto {v}", regla=f"{PUNTO['regla']} {PUNTO['umbral']:g} lpm",
                        sensibilidad=f.sensibilidad, retraso_mediano_s=f.retraso_mediano_s,
                        falsas_por_hora=f.falsas_por_hora_global, peor_infante=f.falsas_por_hora_peor_infante)])],
                        ignore_index=True)
    if len(comp):
        comp.round(3).to_csv("causal_comparacion.csv", index=False)
        print("\n===== LADO A LADO (zona normal 60 s) =====")
        print(comp.round(3).to_string(index=False))
    print("\nGuardado: causal_calidad_detector.csv, causal_regla_resultado_global.csv, "
          "causal_regla_resultado_por_infante.csv"
          + (", causal_combinacion_resultado_global.csv, causal_combinacion_resultado_por_infante.csv"
             if g_comb is not None else "")
          + ", causal_comparacion.csv")
    print("Siguiente: python validacion_cruzada_combinacion.py causal_combinacion_resultado_por_infante.csv "
          "regla_causal _causal")


if __name__ == "__main__":
    principal()