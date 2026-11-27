# Exploración inicial del dataset PICS (PhysioNet) — pensado para correr en Google Colab
#
# Idea clave: NO estás descargando los 1.61 GB completos. wfdb trae directamente
# desde los servidores de PhysioNet solo el registro (o el fragmento) que pidas.
# El dataset completo se usa para ENTRENAR en Colab/tu PC; el ESP32 solo recibirá
# después el modelo ya entrenado y comprimido (unas pocas decenas de KB).
#
# Si corres esto en Colab, pega el bloque tal cual (el "!pip install" funciona
# como celda de Colab). Si lo corres en tu PC como script normal, instala antes
# con: pip install wfdb matplotlib

# !pip install wfdb --quiet

from collections import Counter

import wfdb
import matplotlib.pyplot as plt

# Directorio de PhysioNet para esta base de datos.
# Si 'picsdb/1.0.0' no funciona, prueba solo 'picsdb' (a veces PhysioNet acepta
# ambas formas según la versión de wfdb).
PN_DIR = "picsdb/1.0.0"

# 1) Descubrir los nombres reales de los registros disponibles (sin bajar nada pesado)
records = wfdb.get_record_list(PN_DIR)
print(f"Registros disponibles ({len(records)}):")
print(records)

# 2) Elegir un registro de ECG. Los nombres suelen tener sufijo _ecg / _resp
#    (ej. 'infant1_ecg', 'infant1_resp'); filtra los que digan 'ecg'.
ecg_records = [r for r in records if "ecg" in r.lower()]
record_name = ecg_records[0] if ecg_records else records[0]
print(f"\nUsando registro: {record_name}")

# 3) Traer SOLO un fragmento (streaming), por ejemplo los primeros 5 minutos
#    a 500 Hz: 500 * 60 * 5 muestras. Ajusta según lo que necesites explorar.
record = wfdb.rdrecord(record_name, pn_dir=PN_DIR, sampto=500 * 60 * 5)

print(f"Duración cargada: {record.sig_len / record.fs / 60:.1f} min")
print(f"Frecuencia de muestreo: {record.fs} Hz") 
print(f"Canal(es): {record.sig_name}")

# 4) Graficar el segmento para verlo con tus propios ojos
plt.figure(figsize=(12, 4))
plt.plot(record.p_signal[:, 0])
plt.title(f"ECG - {record_name} (primeros 5 min)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# 5) Intentar leer anotaciones de eventos (bradicardia, etc.), si el registro las trae.
#    La extensión puede no ser '.atr' — si falla, revisa el listado de archivos en
#    https://physionet.org/content/picsdb/1.0.0/ para ver qué extensiones existen.
try:
    ann = wfdb.rdann(record_name, "atr", pn_dir=PN_DIR)
    print(f"\nAnotaciones encontradas: {len(ann.sample)}")
    print(f"Tipos de evento: {set(ann.symbol)}")
    print(f"Total de anotaciones: {len(ann.sample)}")
    print(f"Símbolos únicos (ann.symbol): {Counter(ann.symbol)}")

    # La etiqueta real del evento suele estar aquí, no en 'symbol'
    notas = [n for n in ann.aux_note if n.strip() != ""]
    print(f"\nAux_note no vacíos ({len(notas)} de {len(ann.aux_note)}):")
    print(Counter(notas))
except Exception as e:
    print(f"\nNo se pudo leer con extensión 'atr' ({e}).")
    print("Revisa physionet.org/content/picsdb/1.0.0/ para confirmar la extensión correcta.")