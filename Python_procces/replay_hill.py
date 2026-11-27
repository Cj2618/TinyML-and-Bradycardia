#!/usr/bin/env python3
"""Hardware-in-the-loop: reproduce una grabación de PICS hacia el ESP32 por el puerto serie y compara lo que decide
la placa con la referencia en Python.

El computador hace de convertidor analógico-digital: lee el registro (500 Hz), lo convierte a enteros de 16 bits
(opcionalmente a menos bits, como un ADC real) y lo manda muestra a muestra. La placa corre TODO el pipeline
(detector de latidos causal -> regla de frecuencia cardíaca -> red int8 solo si la regla alarma) y devuelve, por
cada ventana de 10 s, lo que decidió y cuánto tardó. Aquí se compara contra el mismo pipeline en Python.

Lo que NO incluye (hay que decirlo en el artículo): el acondicionamiento analógico (amplificador de instrumentación,
filtros, ruido de electrodos) ni el convertidor real. La señal entra ya digitalizada.

Ejemplos
  # tiempo real (500 muestras/s), 30 min del bebé 3, ventana de decisión 100 lpm / racha 2 / veto 0.5
  python replay_hil.py --puerto COM5 infant3_ecg --minutos 30 --tiempo-real
  # acelerado (lo más rápido que permita el cable): para reproducir las métricas del artículo con la placa
  python replay_hil.py --puerto /dev/ttyUSB0 infant3_ecg --metricas
  # con la red real de referencia en el computador (para comparar las probabilidades de la placa)
  python replay_hil.py --puerto COM5 infant3_ecg --modelo src/modelos_esp32.h
  # ADC de 12 bits
  python replay_hil.py --puerto COM5 infant3_ecg --bits 12
  # sin placa: contra la placa simulada (host_sim, ver README)
  python replay_hil.py --sim ./host_sim --sintetica dificil --minutos 10
  # mediciones de tiempo sueltas (para alinear con el medidor de energía)
  python replay_hil.py --puerto COM5 --bench inferencia --reps 200
  python replay_hil.py --puerto COM5 --bench reposo --reps 120        # 120 s con el reloj de 500 Hz y sin procesar

Salidas (carpeta --salida, por defecto hil/): hil_<nombre>_ventanas.csv, hil_<nombre>_resumen.csv (y _metricas.csv)
"""
import argparse
import csv
import os
import queue
import struct
import subprocess
import sys
import threading
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import decimate

import Causal_detector as D
import regla_fc_cotinuo as R

FS = D.FS
VENTANA, PASO = 10 * FS, 5 * FS
SYNC = 0xA5
T_CONFIG, T_MUESTRAS, T_FIN, T_BENCH, T_PING = 1, 2, 3, 4, 5
FRASE_PEND = 2500          # modo acelerado: muestras enviadas y aún no procesadas (la cola de la placa guarda 4096)
FIFO_MIN, FIFO_MAX = 200, 900   # modo tiempo real: la cola de la placa se mantiene entre estos valores


# ------------------------------------------------------------------------------------------------ protocolo
def trama(tipo, datos=b""):
    chk = tipo ^ (len(datos) & 0xFF) ^ (len(datos) >> 8)
    if datos:
        chk ^= int(np.bitwise_xor.reduce(np.frombuffer(datos, np.uint8)))
    return bytes([SYNC, tipo]) + struct.pack("<H", len(datos)) + datos + bytes([chk])


def trama_config(fc, racha, veto, modo_cnn, tiempo_real, latidos, cpu_mhz, eps):
    return trama(T_CONFIG, struct.pack("<fBfBBBHf", fc, racha, veto, modo_cnn, int(tiempo_real), int(latidos),
                                       cpu_mhz, eps))


class Estado:
    """Lo que la placa va diciendo (lo llena el hilo lector)."""

    def __init__(self):
        self.ventanas, self.latidos, self.errores, self.marcas = [], [], [], []
        self.ok, self.info, self.resumen, self.basura = set(), None, None, 0
        self.tramas = self.fifo = self.procesadas = 0
        self.t_ack = 0.0

    def linea(self, ln):
        c = ln.strip().split(",")
        try:
            k = c[0]
            if k == "W":
                self.ventanas.append(c[1:])
            elif k == "B":
                self.latidos.append(int(c[1]))
            elif k == "A":
                self.tramas, self.fifo, self.procesadas = int(c[1]), int(c[2]), int(c[3])
                self.t_ack = time.perf_counter()
            elif k == "S":
                self.resumen = c[1:]
            elif k == "OK":
                self.ok.add(int(c[1]))
            elif k == "H" and c[1] == "bradi_hil":
                self.info = c[2:]
            elif k == "E":
                self.errores.append(ln.strip())
            elif k == "M":
                self.marcas.append((c[1], float(c[2])))
            else:
                self.basura += 1
        except (ValueError, IndexError):
            self.basura += 1


class Enlace:
    """Puerto serie real (pyserial) o placa simulada por tuberías. Un hilo lee y reparte líneas."""

    def __init__(self, estado):
        self.estado = estado
        self.parar = False
        self.buf = b""
        self.hilo = threading.Thread(target=self._lector, daemon=True)

    def _lector(self):
        while not self.parar:
            try:
                datos = self.leer()
            except Exception:
                break
            if not datos:
                continue
            self.buf += datos
            while b"\n" in self.buf:
                ln, self.buf = self.buf.split(b"\n", 1)
                self.estado.linea(ln.decode("ascii", "replace"))

    def cerrar(self):
        self.parar = True


class EnlaceSerie(Enlace):
    def __init__(self, estado, puerto, baud, reset):
        super().__init__(estado)
        import serial
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout, self.s.write_timeout = puerto, baud, 0.1, 10
        if not reset:               # evita el reinicio automático de la placa al abrir el puerto
            self.s.dtr = False
            self.s.rts = False
        self.s.open()
        self.hilo.start()

    def leer(self):
        return self.s.read(max(1, self.s.in_waiting))

    def escribir(self, b):
        self.s.write(b)

    def cerrar(self):
        super().cerrar()
        self.s.close()


class EnlaceSim(Enlace):
    def __init__(self, estado, comando):
        super().__init__(estado)
        self.p = subprocess.Popen(comando, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.hilo.start()

    def leer(self):
        return os.read(self.p.stdout.fileno(), 65536)

    def escribir(self, b):
        self.p.stdin.write(b)

    def cerrar(self):
        super().cerrar()
        try:
            self.p.stdin.close()
            self.p.terminate()
            self.p.wait(2)
        except Exception:
            pass


def esperar(cond, timeout, enlace, que):
    t0 = time.perf_counter()
    while not cond():
        if time.perf_counter() - t0 > timeout:
            sys.exit(f"La placa no respondió a tiempo ({que}).")
        if enlace.estado.errores:
            sys.exit("La placa reportó un error: " + enlace.estado.errores[0])
        time.sleep(0.005)


def conectar(a, estado):
    enlace = EnlaceSim(estado, [a.sim] + (["--sin-red"] if a.sin_red else [])) if a.sim else \
        EnlaceSerie(estado, a.puerto, a.baud, a.reset)
    t0 = time.perf_counter()
    ping = 0.0
    while estado.info is None:
        if time.perf_counter() - t0 > 8:
            sys.exit("No llegó la línea de saludo de la placa (H,bradi_hil…). ¿Es el firmware correcto y el puerto/baudios bien?")
        if time.perf_counter() - ping > 1.0:
            enlace.escribir(trama(T_PING))
            ping = time.perf_counter()
        time.sleep(0.02)
    return enlace


# ------------------------------------------------------------------------------------------------ señal
def a_cuentas(x, bits):
    """Convierte la señal (mV) a cuentas enteras de un ADC de `bits` bits, centrado en 0 (acople AC) y con
    saturación: fondo de escala = 2 x el percentil 99.9 de |x - mediana|."""
    xc = x.astype(np.float64) - np.median(x)
    pico = float(np.percentile(np.abs(xc[:: max(1, len(xc) // 2_000_000)]), 99.9)) * 2.0
    if pico <= 0:
        pico = 1.0
    cmax = 2 ** (bits - 1) - 1
    q = np.clip(np.rint(xc / pico * cmax), -cmax - 1, cmax).astype(np.int16)
    return q, pico


def cargar(a):
    if a.sintetica:
        import test_paridad as T
        for nombre, s in T.senales_sinteticas(a.minutos or 10):
            if nombre == a.sintetica:
                return nombre, np.asarray(s, np.float32), np.array([])
        sys.exit("Señales sintéticas disponibles: dificil, dificil_sin_planos_exactos")
    if a.dir:
        R.DIR_LOCAL = a.dir
    nombre = a.registro
    s, onsets = R.cargar_registro(nombre)
    i0 = int(a.inicio * 60 * FS)
    i1 = len(s) if not a.minutos else min(len(s), i0 + int(a.minutos * 60 * FS))
    s = s[i0:i1]
    onsets = onsets - a.inicio * 60
    onsets = onsets[(onsets >= 0) & (onsets < len(s) / FS - R.VENTANA_S)]
    return nombre, s, onsets


# ------------------------------------------------------------------------------------------------ referencia
def red_mentira(z):
    """La misma función que host_sim.cpp (solo para probar sin placa)."""
    return float(1.0 / (1.0 + np.exp(-(np.mean(z.astype(np.float64) ** 4) - 15.0) / 5.0)))


def bytes_del_modelo(ruta):
    """Devuelve los bytes del modelo int8. `ruta` puede ser un .tflite o el modelos_esp32.h que se compila en el
    firmware (de ahí se saca el arreglo g_model_int8: es exactamente el modelo que corre la placa)."""
    import re
    if not os.path.isfile(ruta):
        sys.exit(f"No encuentro el archivo del modelo: '{ruta}'.\n"
                 "Usa --modelo con la ruta completa de modelos_esp32.h (el que copiaste a src/ del firmware) o la de "
                 "modelo_int8.tflite si lo descargaste de Colab. También puedes correr sin --modelo: se compara todo "
                 "menos las probabilidades de la red.")
    if ruta.lower().endswith(".h"):
        txt = open(ruta, encoding="utf-8", errors="replace").read()
        m = re.search(r"g_model_int8\s*\[\s*\]\s*=\s*\{(.*?)\}\s*;", txt, re.S)
        if not m:
            sys.exit(f"En '{ruta}' no encontré el arreglo g_model_int8.")
        datos = bytes(int(x, 16) for x in re.findall(r"0[xX]([0-9a-fA-F]{1,2})", m.group(1)))
        n = re.search(r"g_model_int8_len\s*=\s*(\d+)", txt)
        if n and int(n.group(1)) != len(datos):
            sys.exit(f"El arreglo g_model_int8 trae {len(datos)} bytes pero g_model_int8_len dice {n.group(1)}.")
        print(f"Modelo de referencia: g_model_int8 de {ruta} ({len(datos)} bytes)")
        return datos
    return open(ruta, "rb").read()


def red_tflite(ruta):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_content=bytes_del_modelo(ruta))
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]

    def f(z):
        if ent["dtype"] == np.int8:
            sc, zp = ent["quantization"]
            v = z.astype(np.float64) / sc
            q = np.clip(np.sign(v) * np.floor(np.abs(v) + 0.5) + zp, -128, 127).astype(np.int8)    # como roundf
        else:
            q = z.astype(np.float32)
        it.set_tensor(ent["index"], q.reshape(ent["shape"]))
        it.invoke()
        o = float(it.get_tensor(sal["index"]).ravel()[0])
        if sal["dtype"] == np.int8:
            sc, zp = sal["quantization"]
            o = (o - zp) * sc
        return o
    return f


def referencia(xq, a, red):
    """El mismo pipeline en Python (double): detector causal -> regla -> (red) sobre la señal ya cuantizada."""
    lat = D.detectar(xq.astype(np.float32))
    W = sliding_window_view(xq.astype(np.float32), VENTANA)[::PASO]
    n = len(W)
    plana = np.zeros(n, bool)
    for i0 in range(0, n, 512):
        plana[i0:i0 + 512] = np.array(W[i0:i0 + 512], dtype=np.float64).std(axis=1) <= a.eps_plana
    t_fin = (np.arange(n) * PASO + VENTANA) / FS
    racha, media = R.puntajes(lat, t_fin, plana)
    alarma = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        alarma = (racha[a.racha] >= 60.0 / a.fc) & ~plana
    p = np.full(n, np.nan)
    if red is not None and a.modo_cnn != 2:
        usa = ~plana if a.modo_cnn == 1 else alarma
        for i in np.where(usa)[0]:
            z = decimate(np.array(W[i], dtype=np.float64), 4, zero_phase=True)
            z = ((z - z.mean()) / (z.std() + 1e-8)).astype(np.float32)
            p[i] = red(z)
    return dict(lat=lat, t_fin=t_fin, plana=plana, racha=racha, media=media, alarma=alarma, p=p)


# ------------------------------------------------------------------------------------------------ transmisión
def transmitir(enlace, estado, cuentas, a):
    n = len(cuentas)
    datos = cuentas.astype("<i2").tobytes()
    i = 0
    t_prog = time.perf_counter()
    if a.tiempo_real:
        lead, marco = 250, 50
        t_ini = None
        acum = [0]                       # acum[k] = muestras enviadas al terminar la trama k (para fechar los acks)
        ultimo = time.perf_counter()
        while i < n:
            ahora = time.perf_counter()
            meta = lead + (0.0 if t_ini is None else (ahora - t_ini) * FS)
            # cola de la placa estimada: lo que dijo el último ack + lo enviado después - lo consumido desde entonces
            k = min(estado.tramas, len(acum) - 1)
            consumido = 0.0 if t_ini is None else max(0.0, (ahora - max(estado.t_ack, t_ini)) * FS)
            cola = max(0.0, estado.fifo + (i - acum[k]) - consumido) if estado.tramas else float(i)
            if (i < meta and cola < FIFO_MAX) or (t_ini is not None and cola < FIFO_MIN):
                j = min(n, i + marco)
                enlace.escribir(trama(T_MUESTRAS, datos[2 * i:2 * j]))
                if t_ini is None:
                    t_ini = time.perf_counter()
                i = j
                acum.append(i)
            else:
                time.sleep(0.003)
            if estado.errores:
                sys.exit("La placa reportó un error: " + estado.errores[0])
            if a.verbose and time.perf_counter() - ultimo > 10:
                ultimo = time.perf_counter()
                print(f"  {i / FS / 60:6.1f} min enviados | cola estimada de la placa {cola:.0f}", flush=True)
    else:
        marco = 500
        ultimo = time.perf_counter()
        while i < n:
            if i - estado.procesadas < FRASE_PEND:
                j = min(n, i + marco)
                enlace.escribir(trama(T_MUESTRAS, datos[2 * i:2 * j]))
                i = j
            else:
                time.sleep(0.0005)
            if estado.errores:
                sys.exit("La placa reportó un error: " + estado.errores[0])
            if a.verbose and time.perf_counter() - ultimo > 5:
                ultimo = time.perf_counter()
                print(f"  {i / FS / 60:6.1f} min enviados, {estado.procesadas / FS / 60:6.1f} min procesados", flush=True)
    enlace.escribir(trama(T_FIN))
    return time.perf_counter() - t_prog


# ------------------------------------------------------------------------------------------------ comparación
CAMPOS_V = ["indice", "plana", "racha", "media", "alarma_regla", "cnn", "p", "alarma_final", "us_prep", "us_inf",
            "push_max_us", "retraso"]


def tabla_placa(estado):
    v = estado.ventanas
    if not v:
        return None
    t = np.array([[float(x) for x in f] for f in v])
    return {k: t[:, j] for j, k in enumerate(CAMPOS_V)}


def comparar(estado, ref, a, veto):
    P = tabla_placa(estado)
    res = {}
    if P is None:
        print("La placa no devolvió ventanas.")
        return res, P
    idx = P["indice"].astype(int)
    n = min(len(idx), len(ref["t_fin"]))
    res["ventanas_placa"], res["ventanas_ref"] = len(idx), len(ref["t_fin"])
    ok_idx = np.array_equal(idx[:n], np.arange(n))
    pl_b, pl_r = P["plana"][:n].astype(bool), ref["plana"][:n]
    res["plana_distinta"] = int((pl_b != pl_r).sum())
    nf = ~pl_b & ~pl_r
    rr = ref["racha"][a.racha][:n]
    d = np.abs(P["racha"][:n][nf] - rr[nf]) if nf.any() else np.zeros(1)
    res["racha_dif_max_s"] = float(d.max())
    res["racha_dif_mayor_5ms"] = int((d > 0.005).sum())
    al_b, al_r = P["alarma_regla"][:n].astype(bool), ref["alarma"][:n]
    res["alarma_regla_distinta"] = int((al_b != al_r).sum())
    res["alarmas_regla_ref"], res["alarmas_regla_placa"] = int(al_r.sum()), int(al_b.sum())
    # latidos
    if estado.latidos or a.latidos:
        lb = np.array(estado.latidos, dtype=np.int64)
        lr = np.rint(ref["lat"] * FS).astype(np.int64)
        lr = lr[lr <= (n * PASO + VENTANA)]
        lb = lb[lb <= (n * PASO + VENTANA)]
        comunes = len(np.intersect1d(lb, lr))
        res["latidos_ref"], res["latidos_placa"], res["latidos_iguales"] = len(lr), len(lb), comunes
        res["latidos_solo_ref"], res["latidos_solo_placa"] = len(lr) - comunes, len(lb) - comunes
    # red
    if ref["p"] is not None and not np.all(np.isnan(ref["p"])):
        corrio = P["cnn"][:n].astype(bool)
        pr = ref["p"][:n]
        ambos = corrio & ~np.isnan(pr)
        res["ventanas_con_red_placa"], res["ventanas_con_red_ref"] = int(corrio.sum()), int((~np.isnan(pr)).sum())
        res["red_ventanas_distintas"] = int((corrio != ~np.isnan(pr)).sum())
        if ambos.any():
            dp = np.abs(P["p"][:n][ambos] - pr[ambos])
            res["red_dif_prob_max"], res["red_dif_prob_media"] = float(dp.max()), float(dp.mean())
            cerca = np.abs(pr[ambos] - veto) < 1e-3
            distinta = (P["p"][:n][ambos] >= veto) != (pr[ambos] >= veto)
            res["red_decision_distinta"] = int((distinta & ~cerca).sum())
    fin_b = P["alarma_final"][:n].astype(bool)
    if ref["p"] is not None and not np.all(np.isnan(ref["p"])):
        pr = ref["p"][:n]
        fin_r = ref["alarma"][:n] & ((pr >= veto) | np.isnan(pr)) if a.modo_cnn != 2 else ref["alarma"][:n]
        borde = ref["alarma"][:n] & (np.abs(pr - veto) < 1e-3)
        res["alarma_final_distinta"] = int(((fin_b != fin_r) & ~borde).sum())
        res["alarma_final_ref"] = int(fin_r.sum())
    res["alarma_final_placa"] = int(fin_b.sum())
    res["indices_en_orden"] = bool(ok_idx)
    return res, P


def estadistica_tiempos(estado, P, t_total_s, n_muestras):
    s = estado.resumen
    out = {}
    if s and s[0] != "bench" and len(s) >= 17:
        nombres = ["muestras", "ventanas", "latidos", "alarmas_regla", "inferencias", "alarmas_finales",
                   "push_media_us", "push_max_us", "prep_media_us", "prep_max_us", "inf_media_us", "inf_max_us",
                   "underruns", "overruns", "retraso_max_muestras", "heap_libre", "heap_minimo"]
        for k, v in zip(nombres, s):
            out[k] = float(v)
        # carga de CPU estimada = (tiempo en push + preproceso + inferencia) / tiempo de señal
        trabajo_us = out["push_media_us"] * out["muestras"] + out["prep_media_us"] * out["inferencias"] + \
            out["inf_media_us"] * out["inferencias"]
        out["cpu_ocupada_pct"] = 100.0 * trabajo_us / (n_muestras / FS * 1e6)
    return out


def metricas(ref_t_fin, onsets, alarma, plana, buffers=(60, 180)):
    filas = []
    p = np.where(plana, np.nan, alarma.astype(float))
    for b in buffers:
        r = R.evaluar_infante(ref_t_fin, p, onsets, 0.5, 1, b)
        filas.append(dict(buffer_s=b, eventos=r["eventos"], detectados=r["detectados"],
                          sensibilidad=(r["detectados"] / r["eventos"] if r["eventos"] else np.nan),
                          retraso_mediano_s=(float(np.median(r["retrasos"])) if r["retrasos"] else np.nan),
                          falsas_alarmas=r["falsas"], horas_normales=r["horas"],
                          falsas_por_hora=(r["falsas"] / r["horas"] if r["horas"] > 0 else np.nan)))
    return filas


def escribir_csv(ruta, filas):
    if not filas:
        return
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


# ------------------------------------------------------------------------------------------------ bench
def hacer_bench(a):
    estado = Estado()
    enlace = conectar(a, estado)
    kind = {"inferencia": 1, "preproceso": 2, "reposo": 3}[a.bench]
    reps = a.reps or {1: 100, 2: 20, 3: 60}[kind]
    print(f"Placa: {estado.info}")
    enlace.escribir(trama_config(a.fc, a.racha, a.veto, a.modo_cnn, False, False, a.cpu_mhz, a.eps_plana))
    esperar(lambda: 1 in estado.ok, 5, enlace, "CONFIG")
    enlace.escribir(trama(T_BENCH, struct.pack("<BI", kind, reps)))
    t0 = time.perf_counter()
    esperar(lambda: estado.resumen is not None, 60 + (reps if kind == 3 else 0) * 1.2 + 30, enlace, "BENCH")
    r = estado.resumen
    print(f"BENCH {a.bench}: {r[2]} repeticiones | media {float(r[3]) / 1000:.3f} ms | máx {float(r[4]) / 1000:.3f} ms | "
          f"heap libre {r[5]} | duración {time.perf_counter() - t0:.1f} s")
    print("Marcas de tiempo de la placa (ms desde el arranque):", estado.marcas)
    enlace.cerrar()


# ------------------------------------------------------------------------------------------------ principal
def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registro", nargs="?", help="registro de PICS, p. ej. infant3_ecg")
    ap.add_argument("--puerto")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--reset", action="store_true", help="dejar que abrir el puerto reinicie la placa")
    ap.add_argument("--sim", help="ruta del ejecutable host_sim (placa simulada; sin ESP32)")
    ap.add_argument("--sin-red", action="store_true", help="(con --sim) placa sin red")
    ap.add_argument("--dir", help="carpeta local con registros WFDB (por defecto se descargan de PhysioNet)")
    ap.add_argument("--sintetica", help="señal sintética de prueba (dificil | dificil_sin_planos_exactos)")
    ap.add_argument("--inicio", type=float, default=0.0, help="minuto donde empieza el trozo")
    ap.add_argument("--minutos", type=float, default=0.0, help="duración del trozo (0 = hasta el final)")
    ap.add_argument("--bits", type=int, default=16, help="bits del ADC simulado (16 por defecto; 12 = ESP32)")
    ap.add_argument("--fc", type=float, default=100.0, help="FC límite de la regla (lpm)")
    ap.add_argument("--racha", type=int, default=2)
    ap.add_argument("--veto", type=float, default=0.5, help="probabilidad mínima de la red para confirmar")
    ap.add_argument("--modo-cnn", type=int, default=0, choices=[0, 1, 2],
                    help="0 = la red solo si la regla alarma | 1 = en cada ventana | 2 = nunca")
    ap.add_argument("--eps-plana", type=float, default=0.3, help="desviación (en cuentas) bajo la cual la ventana es plana")
    ap.add_argument("--tiempo-real", action="store_true", help="500 muestras/s (si no, lo más rápido posible)")
    ap.add_argument("--cpu-mhz", type=int, default=0, help="0 = no cambiar | 80 | 160 | 240")
    ap.add_argument("--latidos", action="store_true", help="la placa reporta cada latido (compara con la referencia)")
    ap.add_argument("--modelo", help="modelos_esp32.h (el del firmware) o un .tflite int8, para calcular en el computador "
                                     "la probabilidad de referencia")
    ap.add_argument("--sin-referencia", action="store_true", help="no calcular la referencia en Python (grabaciones largas)")
    ap.add_argument("--metricas", action="store_true", help="sensibilidad y falsas alarmas con las decisiones de la placa")
    ap.add_argument("--salida", default="hil")
    ap.add_argument("--bench", choices=["inferencia", "preproceso", "reposo"])
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if not (a.puerto or a.sim):
        ap.error("indica --puerto o --sim")
    if a.bench:
        return hacer_bench(a)
    if not (a.registro or a.sintetica):
        ap.error("indica un registro (p. ej. infant3_ecg) o --sintetica")

    os.makedirs(a.salida, exist_ok=True)
    nombre, señal, onsets = cargar(a)
    xq, pico = a_cuentas(señal, a.bits)
    dur = len(xq) / FS
    print(f"{nombre}: {dur / 60:.1f} min, {len(onsets)} eventos | ADC simulado {a.bits} bits "
          f"(fondo de escala ±{pico / 2:.3g} x2, 1 cuenta = {pico / (2 ** (a.bits - 1) - 1):.3g} de la unidad original)")

    ref = None
    if not a.sin_referencia:
        red = None
        if a.modo_cnn != 2:
            red = red_tflite(a.modelo) if a.modelo else (red_mentira if (a.sim and not a.sin_red) else None)
        t0 = time.perf_counter()
        ref = referencia(xq, a, red)
        print(f"Referencia en Python lista ({time.perf_counter() - t0:.0f} s): {len(ref['lat'])} latidos, "
              f"{int(ref['alarma'].sum())} ventanas con alarma de la regla")

    estado = Estado()
    enlace = conectar(a, estado)
    print(f"Placa: {estado.info}   (cpu MHz, heap libre, heap mín, hay red, bytes modelo, arena usada)")
    enlace.escribir(trama_config(a.fc, a.racha, a.veto, a.modo_cnn, a.tiempo_real, a.latidos, a.cpu_mhz, a.eps_plana))
    esperar(lambda: 1 in estado.ok, 5, enlace, "CONFIG")
    t_envio = transmitir(enlace, estado, xq, a)
    espera = 120 + (len(xq) / FS if a.tiempo_real else 30 + len(xq) / FS / 20)
    esperar(lambda: estado.resumen is not None, espera, enlace, "resumen final")
    time.sleep(0.2)
    enlace.cerrar()
    print(f"Envío terminado en {t_envio:.1f} s ({dur / t_envio:.1f}x tiempo real)" if not a.tiempo_real else
          f"Envío terminado en {t_envio:.1f} s")
    if estado.errores:
        print("Errores de la placa:", *estado.errores[:5], sep="\n  ")

    veto = a.veto
    res, P = ({}, tabla_placa(estado))
    if ref is not None:
        res, P = comparar(estado, ref, a, veto)
    tim = estadistica_tiempos(estado, P, dur, len(xq))
    res.update(tim)
    if estado.resumen and tim and tim.get("muestras", 0) != len(xq):
        print(f"AVISO: la placa procesó {tim['muestras']:.0f} muestras y se enviaron {len(xq)}")

    print("\n=== Resultado ===")
    for k, v in res.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    base = os.path.join(a.salida, f"hil_{nombre}")
    if P is not None:
        n = len(P["indice"])
        filas = []
        for i in range(n):
            f = {k: P[k][i] for k in CAMPOS_V}
            f["t_fin_s"] = (i * PASO + VENTANA) / FS
            if ref is not None and i < len(ref["t_fin"]):
                f.update(ref_plana=int(ref["plana"][i]), ref_racha=ref["racha"][a.racha][i], ref_alarma=int(ref["alarma"][i]),
                         ref_p=ref["p"][i])
            filas.append(f)
        escribir_csv(base + "_ventanas.csv", filas)
    escribir_csv(base + "_resumen.csv", [{**{k: v for k, v in vars(a).items() if k in
                 ("fc", "racha", "veto", "modo_cnn", "bits", "tiempo_real", "cpu_mhz", "eps_plana")},
                 "registro": nombre, "minutos": dur / 60, **res}])
    if a.metricas and P is not None and len(onsets):
        n = len(P["indice"])
        t_fin = (np.arange(n) * PASO + VENTANA) / FS
        m = metricas(t_fin, onsets, P["alarma_final"].astype(bool), P["plana"].astype(bool))
        print("\n=== Métricas con las decisiones de la PLACA (alarma final) ===")
        for f in m:
            print("  zona normal > %ds: %d/%d eventos detectados (%.1f%%), %d falsas alarmas en %.2f h = %.3f por hora, "
                  "retraso mediano %.1f s" % (f["buffer_s"], f["detectados"], f["eventos"], 100 * f["sensibilidad"],
                                              f["falsas_alarmas"], f["horas_normales"], f["falsas_por_hora"],
                                              f["retraso_mediano_s"]))
        escribir_csv(base + "_metricas.csv", [{"registro": nombre, **f} for f in m])
        if ref is not None:
            fin_ref = ref["alarma"] & (np.nan_to_num(ref["p"], nan=1.0) >= veto) if a.modo_cnn != 2 else ref["alarma"]
            mr = metricas(ref["t_fin"], onsets, fin_ref, ref["plana"])
            print("  (referencia en Python: " + "; ".join("%d: %d/%d, %.3f FA/h" % (f["buffer_s"], f["detectados"], f["eventos"],
                  f["falsas_por_hora"]) for f in mr) + ")")
    print(f"\nArchivos en {a.salida}/")


if __name__ == "__main__":
    principal()