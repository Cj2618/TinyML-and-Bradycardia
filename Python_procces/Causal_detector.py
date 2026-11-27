# Detector de latidos CAUSAL (el que sí puede correr en el ESP32, muestra a muestra)
# Requiere: pip install numpy scipy
#
# Por qué existe:
#   El detector de regla_fc_continuo.py procesa la grabación completa con filtros de fase cero
#   (sosfiltfilt, decimate con zero_phase) y calcula el umbral con la señal de los 8 s SIGUIENTES.
#   Eso un microcontrolador en línea no lo puede hacer. Este detector usa solo muestras pasadas:
#
#       señal a 500 Hz -> pasabanda 5-25 Hz (Butterworth, causal) -> derivada -> al cuadrado
#       -> media móvil de 80 ms (causal) -> máximos locales -> decisión tipo Pan-Tompkins
#          con umbrales adaptativos (SPKI = nivel de latido, NPKI = nivel de ruido)
#
#   Decisión de cada máximo local de altura h, en la muestra n (tras el periodo de arranque):
#     - si hay menos de 0.2 s desde el último latido: se ignora (periodo refractario)
#     - umbral = NPKI + 0.25 (SPKI - NPKI); si han pasado más de 3 s sin latido se relaja a la mitad
#       y, mientras no aparece un latido, SPKI decae 5 % por cada máximo rechazado
#     - h > umbral : latido (SPKI = 0.125 min(h, 4 SPKI) + 0.875 SPKI)
#                    si no: ruido (NPKI = 0.125 h + 0.875 NPKI)
#     - h < 1e-5 SPKI : señal muerta (no ruido): se ignora sin tocar los niveles
#   No hay búsqueda hacia atrás (search-back): en una bradicardia real el intervalo largo es el dato
#   que la regla necesita, y rellenarlo con un pico dudoso lo taparía.
#
#   Arranque: se ignoran los primeros 2 s (transitorio del filtro) y los siguientes 4 s se usan solo
#   para fijar SPKI (mediana de las 3 alturas mayores) y NPKI (percentil 25 de los máximos). Solo ocurre
#   una vez por registro.
#
#   Tiempo de un latido = instante en que el detector lo conoce: (índice del máximo + 1) / 500 Hz.
#   La regla usa solo los latidos conocidos hasta el final de la ventana, igual que en línea.
#
# Hay dos implementaciones del MISMO algoritmo, y deben dar exactamente los mismos latidos:
#   - DetectorMuestra : una muestra a la vez (es lo que se porta a C++ para el ESP32; lento en Python)
#   - detectar()      : por bloques con numpy/scipy (rápida, la que se usa para evaluar las 450 h)
# Prueba: python detector_causal.py   (compara las dos sobre una señal sintética y mide la velocidad)

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

# ----------------------------- Parámetros ---------------------------------
FS = 500                      # Hz (la señal base del protocolo; el ADC del ESP32 tomaría 500 Hz)
BANDA_HZ = (5.0, 25.0)
ORDEN = 2                     # orden del prototipo pasabajas (el pasabanda queda de orden 4: 2 secciones)
INTEG_S = 0.08                # media móvil de 80 ms
REFRACTARIO_S = 0.2
ARRANQUE_S = 2.0              # se ignora el transitorio del filtro
APRENDER_S = 4.0              # se fijan los niveles iniciales
PESO = 0.125                  # peso de una muestra nueva en SPKI y NPKI
FRACCION = 0.25               # umbral = NPKI + FRACCION (SPKI - NPKI)
SIN_LATIDO_S = 3.0            # pasado este tiempo sin latido el umbral se relaja
RELAJO = 0.5
DECAE_SPKI = 0.95
PISO_REL = 1e-5              # un máximo con altura < 1e-5 x SPKI no es ruido sino polvo numérico (señal muerta: cable
                              # suelto, ADC saturado): se ignora del todo. Sin esto, la oscilación residual del filtro
                              # tras un tramo exactamente plano hace decaer SPKI y NPKI hasta 1e-30 en double, pero no
                              # en float32 (donde esa oscilación se redondea a 0): el mismo algoritmo daría latidos
                              # distintos en la prueba y en el chip, y falsos latidos al volver la señal.
VERSION = "v2"                # cambia cuando cambia el algoritmo (los .npz de latidos guardados llevan esta marca)
TOPE_SPKI = 4.0               # al actualizar SPKI, una altura se recorta a 4 veces SPKI (un artefacto enorme
                              # no puede disparar el umbral y dejar al detector ciego varios segundos)
CHUNK_S = 600                 # tamaño del bloque de la versión rápida (el resultado no depende de él)

SOS = butter(ORDEN, BANDA_HZ, btype="band", fs=FS, output="sos")
N_INT = int(round(INTEG_S * FS))
N_REF = int(round(REFRACTARIO_S * FS))
N_ARR = int(round(ARRANQUE_S * FS))
N_APR = int(round(APRENDER_S * FS))
N_SIN = int(round(SIN_LATIDO_S * FS))
B_MEDIA = np.full(N_INT, 1.0 / N_INT)


# ------------------------- Decisión (común a las dos versiones) -----------------------
class Decisor:
    """Recibe, en orden, los máximos locales (índice n, altura h) y dice si son latido."""

    def __init__(self):
        self.listo = False
        self.aprend = []
        self.spki = 0.0
        self.npki = 0.0
        self.n_ult = 0            # índice del último latido (o del fin del arranque)

    def candidato(self, n, h):
        if n < N_ARR:
            return False
        if not self.listo:
            if n < N_ARR + N_APR:
                self.aprend.append(h)
                return False
            a = np.asarray(self.aprend, dtype=np.float64)
            if len(a) >= 3:
                # SPKI = mediana de las 3 alturas más grandes (hay >= 2 latidos entre ellas aunque la FC sea
                # baja, y un artefacto aislado no la mueve); NPKI = percentil 25 de todos los máximos
                self.spki = float(np.median(np.sort(a)[-3:]))
                self.npki = float(np.quantile(a, 0.25))
            else:
                self.spki = float(max(a.max() if len(a) else 0.0, h))
                self.npki = 0.1 * self.spki
            self.n_ult = N_ARR + N_APR
            self.listo = True
        if h < PISO_REL * self.spki:
            return False
        if n - self.n_ult < N_REF:
            return False
        relajado = (n - self.n_ult) > N_SIN
        umbral = self.npki + FRACCION * (self.spki - self.npki)
        if relajado:
            umbral *= RELAJO
        if h > umbral:
            w = 2 * PESO if relajado else PESO
            self.spki = w * min(h, TOPE_SPKI * self.spki) + (1.0 - w) * self.spki
            self.n_ult = n
            return True
        self.npki = PESO * h + (1.0 - PESO) * self.npki
        if relajado:
            self.spki *= DECAE_SPKI
        return False


# ------------------------- Versión muestra a muestra -----------------------
class DetectorMuestra:
    """Una muestra por llamada. Es la referencia que se porta a C++."""

    def __init__(self):
        self.z = np.zeros((SOS.shape[0], 2))
        self.y_prev = 0.0
        self.anillo = np.zeros(N_INT)
        self.pos = 0
        self.m1 = 0.0             # m[n-1]
        self.m2 = 0.0             # m[n-2]
        self.n = 0
        self.dec = Decisor()

    def muestra(self, x):
        """Devuelve el tiempo (s) del latido si en esta muestra se confirma uno; si no, None."""
        v = float(x)
        for s in range(SOS.shape[0]):                       # biquads en forma traspuesta II
            b0, b1, b2, _, a1, a2 = SOS[s]
            out = b0 * v + self.z[s, 0]
            self.z[s, 0] = b1 * v - a1 * out + self.z[s, 1]
            self.z[s, 1] = b2 * v - a2 * out
            v = out
        d = v - self.y_prev
        self.y_prev = v
        self.anillo[self.pos] = d * d
        self.pos = (self.pos + 1) % N_INT
        m = self.anillo.sum() / N_INT                       # media de las últimas N_INT muestras
        latido = None
        # m[n-1] es máximo local si m[n-2] < m[n-1] >= m[n]
        if self.n >= 1 and self.m2 < self.m1 >= m:
            if self.dec.candidato(self.n - 1, self.m1):
                latido = self.n / FS                        # (índice del máximo + 1) / FS
        self.m2, self.m1 = self.m1, m
        self.n += 1
        return latido


# ------------------------- Versión por bloques (rápida) -----------------------
class DetectorBloques:
    def __init__(self):
        self.zi = np.zeros((SOS.shape[0], 2))
        self.y_prev = 0.0
        self.fir_zi = np.zeros(N_INT - 1)
        self.cola = np.zeros(2)       # m[n0-2], m[n0-1]
        self.n0 = 0
        self.dec = Decisor()

    def empujar(self, x):
        x = np.asarray(x, dtype=np.float64)
        y, self.zi = sosfilt(SOS, x, zi=self.zi)
        d = np.diff(y, prepend=self.y_prev)
        self.y_prev = y[-1]
        m, self.fir_zi = lfilter(B_MEDIA, [1.0], d * d, zi=self.fir_zi)
        mm = np.concatenate([self.cola, m])
        idx = np.where((mm[1:-1] > mm[:-2]) & (mm[1:-1] >= mm[2:]))[0] + 1
        n_glob = self.n0 - 2 + idx
        h = mm[idx]
        latidos = []
        cand = self.dec.candidato
        for n, hh in zip(n_glob.tolist(), h.tolist()):
            if n >= 0 and cand(n, hh):
                latidos.append((n + 1) / FS)
        self.cola = mm[-2:].copy()
        self.n0 += len(x)
        return latidos


def detectar(senal, chunk_s=CHUNK_S):
    """Latidos (s) de una grabación entera a FS Hz, procesada por bloques con estado."""
    det = DetectorBloques()
    paso = int(chunk_s * FS)
    salida = []
    for a in range(0, len(senal), paso):
        salida += det.empujar(senal[a:a + paso])
    return np.array(salida)


# ------------------------------- Prueba ----------------------------------------
def _senal_prueba(horas=0.05, seed=0):
    rng = np.random.RandomState(seed)
    n = int(horas * 3600 * FS)
    t = np.arange(n) / FS
    x = 0.3 * np.sin(2 * np.pi * 0.25 * t) + 0.05 * rng.randn(n)
    tl = 1.0
    k = np.arange(-100, 100) / FS
    qrs = np.exp(-(k / 0.012) ** 2) - 0.25 * np.exp(-((k - 0.02) / 0.015) ** 2)
    while tl < t[-1] - 1:
        i = int(tl * FS)
        x[i - 100:i + 100] += qrs * (1 + 0.1 * rng.randn())
        tl += 0.4 * (1 + 0.05 * rng.randn()) if rng.rand() > 0.02 else rng.uniform(0.8, 1.4)
    a = int(n * 0.5)
    x[a:a + 2 * FS] += 0.8 * rng.randn(2 * FS)           # una ráfaga de ruido
    return x


if __name__ == "__main__":
    import time
    x = _senal_prueba()
    t0 = time.time()
    det = DetectorMuestra()
    a = [t for t in (det.muestra(v) for v in x) if t is not None]
    t_m = time.time() - t0
    t0 = time.time()
    b = detectar(x, chunk_s=7)            # bloques chicos a propósito: prueba el traspaso de estado
    t_b = time.time() - t0
    c = detectar(x, chunk_s=600)
    print(f"{len(x) / FS:.0f} s de señal: muestra a muestra {len(a)} latidos ({t_m:.1f} s), "
          f"por bloques {len(b)} latidos ({t_b:.2f} s)")
    ok = len(a) == len(b) == len(c) and np.allclose(a, b, atol=0) and np.allclose(b, c, atol=0)
    print("Las tres versiones dan exactamente los mismos latidos:", ok)
    assert ok