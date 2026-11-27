# Combinación de la red con la regla de frecuencia cardíaca - mismo protocolo continuo (PICS)
# Requiere: pip install wfdb scipy pandas matplotlib numpy      (NO necesita tensorflow ni descargar nada)
#
# Pregunta que responde:
#   La regla de 2 latidos lentos (regla_fc_continuo.py) tuvo menos falsas alarmas que la red, pero
#   sus falsas alarmas vienen de latidos mal detectados y las de la red vienen de otra parte
#   (correlación entre bebés de -0.10). ¿Si la alarma exige que la REGLA y la RED coincidan en la
#   misma ventana, bajan las falsas alarmas sin perder demasiados eventos?
#
# Reglas de combinación (todas con las mismas ventanas, zona normal, ventana de detección de 20 s
# y conteo de falsas alarmas que los otros dos scripts):
#   - regla sola                : suena la regla (límite de FC en lpm).
#   - regla + red (veto = u)    : suena la regla Y la red da probabilidad >= u en esa misma ventana.
#                                 La red actúa como filtro: con u bajo (0.05-0.3) solo descarta las
#                                 ventanas en que la red está segura de que es normal; con u alto
#                                 exige que la red también vea el evento.
#   - red sola                  : como en falsas_alarmas_continuo.py (para dibujarla en la misma figura).
#
# Entradas (las que ya generaron los otros dos scripts, en la misma carpeta que este archivo):
#   regla_fc_continuo.py        (se importa: usa sus funciones y constantes, así el conteo es idéntico)
#   regla_fc/latidos_K.npz      (de regla_fc_continuo.py)
#   continuo/probs_K.npz        (de falsas_alarmas_continuo.py)
# Salidas:
#   combinacion_resultado_global.csv
#   combinacion_resultado_por_infante.csv
#   fig_combinacion.png
#
# Se usan los latidos del detector propio (el caso realista, el que correría el ESP32). Para ver
# también la combinación con los latidos de referencia pon FUENTES_LATIDOS = ("det", "ref").
# Tarda unos minutos (casi todo es recalcular los puntajes de la regla); no descarga nada.
#
# Recordatorio para interpretar: el barrido de umbrales y de vetos se hace sobre los mismos infantes
# de prueba, así que es exploratorio. Si una combinación sale bien, hay que confirmarla con un
# umbral elegido en otros infantes antes de afirmarla.

import glob
import os

import matplotlib
import numpy as np
import pandas as pd

import fc_continuo as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Parámetros ---------------------------------
CARPETA_REGLA = R.CARPETA          # regla_fc
CARPETA_CNN = R.CARPETA_CNN        # continuo
FUENTES_LATIDOS = ("det",)         # "det" = detector propio; agrega "ref" para latidos de referencia
RACHAS_COMB = [1, 2]               # 2 = la definición de PICS; 1 = un solo intervalo lento
FC_LIMITES = R.FC_LIMITES          # lpm
VETOS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]    # probabilidad mínima de la red para confirmar la alarma
MODELOS = {"int8": "p_int8", "float32": "p_float32"}   # int8 es el que correría en el chip
BUFFERS_S = R.BUFFERS_S            # [60, 180]
UMBRALES_RED = R.UMBRALES_CNN      # para dibujar la red sola


# ------------------------------ Cálculo -------------------------------------
def cargar_par(k):
    a = os.path.join(CARPETA_REGLA, f"latidos_{k}.npz")
    b = os.path.join(CARPETA_CNN, f"probs_{k}.npz")
    dl, dc = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    assert str(dl["nombre"]) == str(dc["nombre"]), f"bebé {k}: {dl['nombre']} != {dc['nombre']}"
    assert len(dl["t_fin"]) == len(dc["t_fin"]) and np.allclose(dl["t_fin"], dc["t_fin"]), \
        f"bebé {k}: las ventanas de la regla y de la red no coinciden"
    assert np.array_equal(np.isnan(dc["p_float32"]), dl["plana"]), \
        f"bebé {k}: las ventanas planas no coinciden (¿MAX_HORAS distinto?)"
    return dl, dc


def combinar(puntaje, p, u):
    """Puntaje de la regla donde la red confirma (p >= u); 0 donde no; NaN donde la regla no evalúa."""
    with np.errstate(invalid="ignore"):
        confirma = p >= u
    return np.where(np.isnan(puntaje), np.nan, np.where(confirma, puntaje, 0.0))


def fila(inf, grupo, fuente, regla, buf, veto, umbral, k, r):
    f = R._fila(inf, grupo, fuente, regla, k, buf, umbral, r)
    f["veto_red"] = veto
    return f


def filas_infante(dl, dc):
    inf, grupo = str(dl["nombre"]), int(dl["indice"])
    t_fin, plana, onsets = dl["t_fin"], dl["plana"], dl["onsets"]
    filas = []
    for clave in FUENTES_LATIDOS:
        racha, _ = R.puntajes(dl["latidos_" + clave], t_fin, plana)
        for buf in BUFFERS_S:
            for m in RACHAS_COMB:
                for fc in FC_LIMITES:
                    lim = 60.0 / fc
                    # regla sola
                    r = R.evaluar_infante(t_fin, racha[m], onsets, lim, 1, buf)
                    filas.append(fila(inf, grupo, f"regla_{clave}", f"racha{m}", buf, np.nan, fc, 1, r))
                    # regla + red
                    for mod, campo in MODELOS.items():
                        for u in VETOS:
                            s = combinar(racha[m], dc[campo], u)
                            r = R.evaluar_infante(t_fin, s, onsets, lim, 1, buf)
                            filas.append(fila(inf, grupo, f"regla_{clave}+red_{mod}", f"racha{m}",
                                              buf, u, fc, 1, r))
    # red sola (para la figura)
    for buf in BUFFERS_S:
        for mod, campo in MODELOS.items():
            for u in UMBRALES_RED:
                r = R.evaluar_infante(t_fin, dc[campo], onsets, u, 1, buf)
                filas.append(fila(inf, grupo, f"red_{mod}", "red", buf, np.nan, u, 1, r))
    return filas


def agregar(por_inf):
    por_inf = por_inf.copy()
    por_inf["veto_red"] = por_inf["veto_red"].fillna(-1)       # -1 = sin veto (para poder agrupar)
    claves = ["fuente", "regla", "persistencia", "buffer_s", "veto_red", "umbral"]
    filas = []
    for key, sub in por_inf.groupby(claves):
        retrasos = [x for lst in sub["_retrasos"] for x in lst]
        horas = sub.horas_normales.sum()
        filas.append(dict(
            zip(claves, key),
            eventos=int(sub.eventos.sum()), detectados=int(sub.detectados.sum()),
            sensibilidad=sub.detectados.sum() / max(sub.eventos.sum(), 1),
            retraso_mediano_s=(np.median(retrasos) if retrasos else np.nan),
            falsas_alarmas=int(sub.falsas_alarmas.sum()), horas_normales=horas,
            falsas_por_hora_global=sub.falsas_alarmas.sum() / horas if horas > 0 else np.nan,
            falsas_por_hora_mediana_infantes=sub.falsas_por_hora.median(),
            falsas_por_hora_peor_infante=sub.falsas_por_hora.max()))
    g = pd.DataFrame(filas).sort_values(claves).reset_index(drop=True)
    g["veto_red"] = g["veto_red"].where(g["veto_red"] >= 0)    # de vuelta a NaN
    return g


# ------------------------------- Figura ------------------------------------
def figura(g, salida, buf=60, modelo="int8"):
    paneles = [m for m in RACHAS_COMB]
    fig, ejes = plt.subplots(1, len(paneles), figsize=(5.8 * len(paneles), 5.4), sharey=True)
    ejes = np.atleast_1d(ejes)
    vetos_fig = [v for v in (0.1, 0.5, 0.9) if v in VETOS] or VETOS[:3]
    tonos = ["#e69f00", "#b8860b", "#7a5200"]
    for ax, m in zip(ejes, paneles):
        s = g[(g.fuente == f"red_{modelo}") & (g.buffer_s == buf)].sort_values("umbral")
        ax.plot(s.falsas_por_hora_global, s.sensibilidad, "-", marker="o", ms=5, lw=1.8,
                color="#0072B2", label=f"Red sola ({modelo}, umbral de 0.5 a 0.995)")
        s = g[(g.fuente == "regla_det") & (g.regla == f"racha{m}") & (g.buffer_s == buf)].sort_values("umbral")
        ax.plot(s.falsas_por_hora_global, s.sensibilidad, "-", marker="s", ms=5, lw=1.8,
                color="#D55E00", label="Regla sola (detector propio)")
        for v, c in zip(vetos_fig, tonos):
            s = g[(g.fuente == f"regla_det+red_{modelo}") & (g.regla == f"racha{m}")
                  & (g.buffer_s == buf) & (g.veto_red == v)].sort_values("umbral")
            ax.plot(s.falsas_por_hora_global, s.sensibilidad, "--", marker="^", ms=4.5, lw=1.5,
                    color=c, label=f"Regla + red (probabilidad de la red de {v} o más)")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(-0.01, 14)
        ax.set_xticks([0, 0.1, 1, 10])
        ax.set_xticklabels(["0", "0.1", "1", "10"])
        ax.set_ylim(0, 1.02)
        ax.grid(color="#e3dfd3", lw=0.8)
        ax.axvline(1, color="#999999", lw=1, ls=":")
        ax.set_title(("Regla de 2 latidos lentos" if m == 2 else f"Regla de {m} latido lento")
                     + f" (límite de {min(FC_LIMITES)} a {max(FC_LIMITES)} lpm)", fontsize=10.5)
        ax.set_xlabel("Falsas alarmas por hora de señal normal")
        ax.spines[["top", "right"]].set_visible(False)
    ejes[0].set_ylabel("Eventos detectados (fracción)")
    h, l = ejes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(salida, dpi=200, facecolor="white")


# ------------------------------- Principal ----------------------------------
def principal():
    n_reglas = len(glob.glob(os.path.join(CARPETA_REGLA, "latidos_*.npz")))
    n_red = len(glob.glob(os.path.join(CARPETA_CNN, "probs_*.npz")))
    assert n_reglas > 0 and n_red > 0, (
        f"Faltan archivos: {n_reglas} en {CARPETA_REGLA}/ y {n_red} en {CARPETA_CNN}/. "
        "Corre antes falsas_alarmas_continuo.py y regla_fc_continuo.py.")
    assert n_reglas == n_red, f"{n_reglas} bebés en {CARPETA_REGLA}/ y {n_red} en {CARPETA_CNN}/"
    filas = []
    for k in range(n_red):
        dl, dc = cargar_par(k)
        filas += filas_infante(dl, dc)
        print(f"[{k}] {dl['nombre']}: listo")
    por_inf = pd.DataFrame(filas)
    g = agregar(por_inf)
    g.to_csv("combinacion_resultado_global.csv", index=False)
    por_inf.drop(columns="_retrasos").to_csv("combinacion_resultado_por_infante.csv", index=False)
    figura(g, "fig_combinacion.png")

    pd.set_option("display.width", 220)
    cols = ["fuente", "veto_red", "umbral", "detectados", "sensibilidad", "retraso_mediano_s",
            "falsas_por_hora_global", "falsas_por_hora_mediana_infantes", "falsas_por_hora_peor_infante"]
    for buf in BUFFERS_S:
        print(f"\n===== ZONA NORMAL A MÁS DE {buf} s: regla de 2 latidos lentos con límite de 100 lpm =====")
        print("(veto_red = probabilidad mínima de la red para confirmar; vacío = regla sola)")
        s = g[(g.buffer_s == buf) & (g.regla == "racha2") & (g.umbral == 100.0)
              & (g.fuente.str.startswith("regla"))]
        print(s[cols].round(3).to_string(index=False))
        print(f"\n--- Mejor punto con 1 falsa alarma por hora o menos (zona normal {buf} s) ---")
        for (fuente, regla), sub in g[g.buffer_s == buf].groupby(["fuente", "regla"]):
            ok = sub[sub.falsas_por_hora_global <= 1]
            if len(ok):
                f = ok.sort_values("sensibilidad", ascending=False).iloc[0]
                v = "" if pd.isna(f.veto_red) else f" veto {f.veto_red}"
                print(f"{fuente:24s} {regla:7s}{v:11s} umbral {f.umbral:>6}: detecta {f.sensibilidad:.3f}, "
                      f"retraso mediano {f.retraso_mediano_s:.1f} s, {f.falsas_por_hora_global:.2f} falsas/h")
    print("\nGuardado: combinacion_resultado_global.csv, combinacion_resultado_por_infante.csv, "
          "fig_combinacion.png")


if __name__ == "__main__":
    principal()