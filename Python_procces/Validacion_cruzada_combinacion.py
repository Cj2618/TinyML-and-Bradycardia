"""
Validación cruzada de la combinación modelo + regla de frecuencia cardíaca.

Pregunta: ¿la mejora de la combinación (menos falsas alarmas a cambio de algo de sensibilidad) se mantiene
cuando el punto de operación (límite de la regla y probabilidad mínima de confirmación del modelo) se elige
SIN mirar al infante que se evalúa?

Cómo funciona (dejando un infante fuera):
  1. Para cada infante k, se toman los otros 9 y con ellos se elige el punto de operación según un criterio
     fijado de antemano (ver CRITERIOS).
  2. Ese punto se aplica al infante k, que no intervino en la elección.
  3. Se suman las alarmas y los eventos de los 10 infantes retenidos.
También se calcula la versión "sin validar" (el punto se elige con los 10 y se evalúa en los mismos 10), que es
la del artículo hasta ahora, para ver cuánto optimismo había.

Compara cuatro estrategias con la misma elección: modelo solo (umbral), regla sola (racha y límite), regla +
modelo (racha, límite y mínimo del modelo) y regla con o sin modelo (el criterio decide si usa el veto).

Entrada: combinacion_resultado_por_infante.csv (lo genera combinacion_red_regla.py). No hace falta volver a
procesar señales ni tener TensorFlow: solo pandas y numpy, y corre en segundos.

Uso:
  python validacion_cruzada_combinacion.py
        -> con los latidos del detector propio no causal (combinacion_resultado_por_infante.csv)
  python validacion_cruzada_combinacion.py causal_combinacion_resultado_por_infante.csv regla_causal _causal
        -> con los latidos del detector causal (el que corre en línea); los archivos de salida llevan el
           sufijo _causal, así que no pisan los anteriores

Salidas:
  validacion_cruzada_combinacion.csv   una fila por (zona, modelo, criterio, estrategia, validación)
  validacion_cruzada_elecciones.csv    punto elegido en cada pliegue, con su desempeño en entrenamiento y en el
                                       infante retenido

Nota: las probabilidades del modelo de cada infante vienen de un modelo entrenado sin él (validación por
paciente), pero ese modelo sí vio a los otros infantes, incluido el que aquí se retiene al elegir el punto. Es
una fuga leve; la validación sería totalmente limpia solo reentrenando dentro de cada pliegue.
El retraso se aproxima con el promedio, ponderado por eventos detectados, de las medianas de cada infante (el
archivo por infante no guarda los retrasos individuales). La sensibilidad y las falsas alarmas por hora sí son
exactas.
"""
import sys
import numpy as np
import pandas as pd

ARCHIVO = "combinacion_resultado_por_infante.csv"
BUFFERS_S = [60, 180]
MODELOS = ["int8", "float32"]
# (nombre, tipo, valor):  "fa" = máxima sensibilidad con a lo sumo `valor` falsas alarmas por hora
#                         "se" = mínimas falsas alarmas con al menos `valor` de sensibilidad
CRITERIOS = [("hasta 1 falsa alarma por hora", "fa", 1.0),
             ("hasta 0.5 falsas alarmas por hora", "fa", 0.5),
             ("al menos 90 % de detección", "se", 0.9),
             ("al menos 80 % de detección", "se", 0.8)]
CLAVE = ["fuente", "regla", "veto_red", "umbral"]
SUMAS = ["eventos", "detectados", "falsas_alarmas", "horas_normales"]


def estrategias(mod, regla="regla_det"):
    return {"Modelo solo": (f"red_{mod}",),
            "Regla sola": (regla,),
            "Regla + modelo": (f"{regla}+red_{mod}",),
            "Regla con o sin modelo": (regla, f"{regla}+red_{mod}")}


def tabla(d):
    """Sensibilidad y falsas por hora acumuladas de cada configuración, sobre los infantes de `d`."""
    g = d.groupby(CLAVE)[SUMAS].sum()
    g["sens"] = g.detectados / g.eventos.clip(lower=1)
    g["fa"] = g.falsas_alarmas / g.horas_normales.where(g.horas_normales > 0)
    return g.dropna(subset=["fa"])


def elegir(c, tipo, x):
    """Índice de la configuración elegida. Si el criterio no se puede cumplir, la más cercana."""
    if tipo == "fa":
        ok = c[c.fa <= x]
        if len(ok):
            return ok.sort_values(["sens", "fa"], ascending=[False, True]).index[0]
        return c.sort_values(["fa", "sens"], ascending=[True, False]).index[0]
    ok = c[c.sens >= x]
    if len(ok):
        return ok.sort_values(["fa", "sens"], ascending=[True, False]).index[0]
    return c.sort_values(["sens", "fa"], ascending=[False, True]).index[0]


def describir(cfg):
    fuente, regla, veto, umbral = cfg
    if fuente.startswith("red_"):
        return f"modelo con umbral {umbral:g}"
    s = f"{regla}, límite {umbral:g} lpm"
    return s + (f", modelo ≥ {veto:g}" if veto >= 0 else ", sin modelo")


def resumen(filas, nombre_cols):
    """Acumula las filas de los infantes retenidos en una sola fila de resultado."""
    f = pd.DataFrame(filas)
    ev, det = f.eventos.sum(), f.detectados.sum()
    horas = f.horas_normales.sum()
    con_retraso = f.dropna(subset=["retraso_mediano_s"])
    w = con_retraso.detectados
    return dict(nombre_cols, eventos=int(ev), detectados=int(det), sensibilidad=det / max(ev, 1),
                falsas_alarmas=int(f.falsas_alarmas.sum()), horas_normales=horas,
                falsas_por_hora=f.falsas_alarmas.sum() / horas,
                peor_infante=(f.falsas_alarmas / f.horas_normales.where(f.horas_normales > 0)).max(),
                retraso_aprox_s=(np.average(con_retraso.retraso_mediano_s, weights=w)
                                 if len(con_retraso) and w.sum() > 0 else np.nan))


def principal():
    ruta = sys.argv[1] if len(sys.argv) > 1 else ARCHIVO
    regla = sys.argv[2] if len(sys.argv) > 2 else "regla_det"      # nombre de la fuente de latidos en el CSV
    sufijo = sys.argv[3] if len(sys.argv) > 3 else ""
    d = pd.read_csv(ruta)
    d["veto_red"] = d["veto_red"].fillna(-1)
    d = d[d.persistencia == 1]
    infantes = sorted(d.infante.unique())
    print(f"{len(infantes)} infantes: {infantes}")
    assert len(infantes) >= 3, "Se necesitan al menos 3 infantes para dejar uno fuera."

    resultados, elecciones = [], []
    for buf in BUFFERS_S:
        db = d[d.buffer_s == buf]
        for mod in MODELOS:
            for estr, fuentes in estrategias(mod, regla).items():
                de = db[db.fuente.isin(fuentes)]
                if de.empty:
                    continue
                for crit, tipo, x in CRITERIOS:
                    base = dict(buffer_s=buf, modelo=mod, criterio=crit, estrategia=estr)
                    # --- validación cruzada: elegir sin el infante k y aplicar en k ---
                    filas = []
                    for k in infantes:
                        ent = tabla(de[de.infante != k])
                        cfg = elegir(ent, tipo, x)
                        fila = de[(de.infante == k)].set_index(CLAVE).loc[cfg]
                        filas.append(fila.to_dict())
                        elecciones.append(dict(
                            base, infante_retenido=k, elegido=describir(cfg),
                            fuente=cfg[0], regla=cfg[1], veto_red=cfg[2], umbral=cfg[3],
                            sens_entrenamiento=ent.loc[cfg, "sens"], falsas_h_entrenamiento=ent.loc[cfg, "fa"],
                            sens_retenido=fila.detectados / max(fila.eventos, 1),
                            falsas_h_retenido=(fila.falsas_alarmas / fila.horas_normales
                                               if fila.horas_normales > 0 else np.nan)))
                    resultados.append(resumen(filas, dict(base, validacion="cruzada")))
                    # --- sin validar: elegir y evaluar con los mismos infantes ---
                    tot = tabla(de)
                    cfg = elegir(tot, tipo, x)
                    filas = de.set_index(CLAVE).loc[[cfg]].reset_index().to_dict("records")
                    r = resumen(filas, dict(base, validacion="sin validar"))
                    r["elegido"] = describir(cfg)
                    resultados.append(r)

    res = pd.DataFrame(resultados)
    res.to_csv(f"validacion_cruzada_combinacion{sufijo}.csv", index=False)
    pd.DataFrame(elecciones).to_csv(f"validacion_cruzada_elecciones{sufijo}.csv", index=False)

    # ------------------------------ lectura en pantalla ------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    orden = ["Modelo solo", "Regla sola", "Regla + modelo", "Regla con o sin modelo"]
    for buf in BUFFERS_S:
        for mod in MODELOS:
            print(f"\n{'=' * 100}\nZONA NORMAL A MÁS DE {buf} s · modelo {mod}\n{'=' * 100}")
            for crit, _, _ in CRITERIOS:
                s = res[(res.buffer_s == buf) & (res.modelo == mod) & (res.criterio == crit)].copy()
                s["estrategia"] = pd.Categorical(s.estrategia, orden, ordered=True)
                s = s.sort_values(["estrategia", "validacion"])
                print(f"\n--- Criterio: {crit} ---")
                print(s[["estrategia", "validacion", "detectados", "eventos", "sensibilidad", "retraso_aprox_s",
                         "falsas_por_hora", "peor_infante"]].round(3).to_string(index=False))

    # qué elige la validación cruzada en la estrategia "con o sin modelo" (zona de 60 s, int8)
    e = pd.DataFrame(elecciones)
    e = e[(e.buffer_s == 60) & (e.modelo == "int8") & (e.estrategia == "Regla con o sin modelo")]
    print(f"\n{'=' * 100}\nPuntos elegidos en cada pliegue (regla con o sin modelo, zona 60 s, int8)\n{'=' * 100}")
    for crit, _, _ in CRITERIOS:
        print(f"\n--- {crit} ---")
        print(e[e.criterio == crit][["infante_retenido", "elegido", "sens_entrenamiento", "falsas_h_entrenamiento",
                                     "sens_retenido", "falsas_h_retenido"]].round(3).to_string(index=False))
    print(f"\nGuardado: validacion_cruzada_combinacion{sufijo}.csv, validacion_cruzada_elecciones{sufijo}.csv")


if __name__ == "__main__":
    principal()