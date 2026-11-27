// Versión para PlatformIO: es el mismo código de benchmark_esp32.ino con #include <Arduino.h> al inicio.
#include <Arduino.h>
// Benchmark TinyML en ESP32 (multi-modelo): latencia, memoria y paridad con Python.
// Compara en el mismo chip: float32, int8, int8+poda50 e int8+poda80.
//
// Librería: Chirale_TensorFlowLite 2.0.0 (TensorFlow Lite Micro para Arduino/ESP32)
//
// Archivos en src/ (PlatformIO) o en la carpeta del sketch (Arduino IDE):
//     modelos_esp32.h      <- lo genera entrenar_y_comprimir.py
//     ventanas_prueba.h    <- lo genera generar_ventanas_prueba.py
//
// Monitor Serie a 115200. Al terminar, escribe cualquier carácter para repetir.

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <esp_heap_caps.h>
#include <new>

#include "modelos_esp32.h"     // g_model_float32, g_model_int8, g_model_int8_poda50, g_model_int8_poda80
#include "ventanas_prueba.h"   // N_VENTANAS, LARGO_VENTANA, ventanas, etiquetas, esperado_*

// ----------------------- Estimación de energía (opcional) -----------------------
// Energía por ventana [uJ] = VOLTAJE_V * CORRIENTE_ACTIVA_mA * latencia_total_ms.
// Completa CORRIENTE_ACTIVA_mA con un valor MEDIDO o con el que vayas a citar del
// datasheet de tu módulo. En 0 no se calcula.
constexpr float VOLTAJE_V = 3.3f;
constexpr float CORRIENTE_ACTIVA_mA = 0.0f;

constexpr int N_REPS = 100;                      // repeticiones para la estadística de latencia
constexpr bool IMPRIMIR_VENTANAS = false;        // true = imprime una línea por ventana y modelo

// ----------------------- Modelos a medir -----------------------
struct Config {
  const char* nombre;
  const unsigned char* modelo;
  unsigned int bytes;
  const float* esperado;
};
const Config CONFIGS[] = {
  {"float32",     g_model_float32,     g_model_float32_len,     esperado_float32},
  {"int8",        g_model_int8,        g_model_int8_len,        esperado_int8},
  {"int8+poda50", g_model_int8_poda50, g_model_int8_poda50_len, esperado_int8_poda50},
  {"int8+poda80", g_model_int8_poda80, g_model_int8_poda80_len, esperado_int8_poda80},
};
constexpr int N_CONFIGS = sizeof(CONFIGS) / sizeof(CONFIGS[0]);

struct Resultado {
  bool ok = false;
  unsigned int arena_usada = 0;
  uint32_t init_us = 0;
  float pre_us = 0, inv_media_us = 0, inv_desv_us = 0;
  uint32_t inv_min_us = 0, inv_max_us = 0;
  float max_dif = 0;
  int dec_distintas = 0, aciertos = 0;
};
static Resultado resultados[N_CONFIGS];

// ----------------------- Memoria -----------------------
static uint8_t* arena = nullptr;
static size_t arena_size = 0;

// 7 operaciones que usan estos modelos
static tflite::MicroMutableOpResolver<8> resolver;
alignas(16) static uint8_t almacen_interprete[sizeof(tflite::MicroInterpreter)];

static uint32_t t_pre[N_VENTANAS];
static uint32_t t_rep[N_REPS];

// ----------------------- Preprocesamiento en el dispositivo -----------------------
// Normalización por ventana (media 0, desv. 1) y, si el modelo es int8, cuantización con la
// escala/zero-point del tensor de entrada. La ventana ya llega submuestreada a 125 Hz
// (ese paso no se mide aquí).
void preprocesar(const float* x, TfLiteTensor* entrada) {
  float suma = 0.0f;
  for (int i = 0; i < LARGO_VENTANA; i++) suma += x[i];
  const float media = suma / LARGO_VENTANA;

  float var = 0.0f;
  for (int i = 0; i < LARGO_VENTANA; i++) {
    const float d = x[i] - media;
    var += d * d;
  }
  const float desv = sqrtf(var / LARGO_VENTANA) + 1e-8f;

  if (entrada->type == kTfLiteFloat32) {
    for (int i = 0; i < LARGO_VENTANA; i++) entrada->data.f[i] = (x[i] - media) / desv;
  } else {
    const float inv_escala = 1.0f / entrada->params.scale;
    const int zp = entrada->params.zero_point;
    for (int i = 0; i < LARGO_VENTANA; i++) {
      int q = (int)roundf(((x[i] - media) / desv) * inv_escala) + zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      entrada->data.int8[i] = (int8_t)q;
    }
  }
}

float leer_probabilidad(const TfLiteTensor* salida) {
  if (salida->type == kTfLiteFloat32) return salida->data.f[0];
  return (salida->data.int8[0] - salida->params.zero_point) * salida->params.scale;
}

void estadisticos(const uint32_t* v, int n, float& media, float& desv, uint32_t& mn, uint32_t& mx) {
  double s = 0;
  mn = v[0];
  mx = v[0];
  for (int i = 0; i < n; i++) {
    s += v[i];
    if (v[i] < mn) mn = v[i];
    if (v[i] > mx) mx = v[i];
  }
  media = (float)(s / n);
  double q = 0;
  for (int i = 0; i < n; i++) q += ((double)v[i] - media) * ((double)v[i] - media);
  desv = (float)sqrt(q / n);
}

// ----------------------- Medición de UN modelo -----------------------
void medir_config(int k) {
  const Config& c = CONFIGS[k];
  Resultado& r = resultados[k];
  r = Resultado();

  Serial.printf("\n>>> %s (%u bytes)\n", c.nombre, c.bytes);

  const tflite::Model* modelo = tflite::GetModel(c.modelo);
  if (modelo->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("    ERROR: version de esquema distinta a la de la libreria.");
    return;
  }

  const uint32_t t0 = micros();
  tflite::MicroInterpreter* it =
      new (almacen_interprete) tflite::MicroInterpreter(modelo, resolver, arena, arena_size);
  if (it->AllocateTensors() != kTfLiteOk) {
    Serial.printf("    NO CABE o falla AllocateTensors con %u bytes de arena disponibles.\n",
                  (unsigned)arena_size);
    it->~MicroInterpreter();
    return;
  }
  r.init_us = micros() - t0;
  r.arena_usada = (unsigned)it->arena_used_bytes();

  TfLiteTensor* entrada = it->input(0);
  TfLiteTensor* salida = it->output(0);
  if (entrada->dims->data[1] != LARGO_VENTANA) {
    Serial.printf("    ERROR: el modelo espera %d muestras y ventanas_prueba.h trae %d.\n",
                  entrada->dims->data[1], LARGO_VENTANA);
    it->~MicroInterpreter();
    return;
  }

  // ---- Pasada por las ventanas: paridad con Python y latencia de preproceso ----
  for (int i = 0; i < N_VENTANAS; i++) {
    const uint32_t a = micros();
    preprocesar(ventanas[i], entrada);
    t_pre[i] = micros() - a;
    if (it->Invoke() != kTfLiteOk) {
      Serial.println("    ERROR: Invoke fallo.");
      it->~MicroInterpreter();
      return;
    }
    const float p = leer_probabilidad(salida);
    const float dif = fabsf(p - c.esperado[i]);
    if (dif > r.max_dif) r.max_dif = dif;
    if ((p >= 0.5f) != (c.esperado[i] >= 0.5f)) r.dec_distintas++;
    if ((p >= 0.5f) == (etiquetas[i] == 1)) r.aciertos++;
    if (IMPRIMIR_VENTANAS) {
      Serial.printf("    %d,%d,%.4f,%.4f\n", i, etiquetas[i], c.esperado[i], p);
    }
  }

  // ---- Repeticiones sobre una misma ventana para estabilizar la latencia ----
  preprocesar(ventanas[0], entrada);
  for (int rep = 0; rep < N_REPS; rep++) {
    const uint32_t a = micros();
    it->Invoke();
    t_rep[rep] = micros() - a;
  }
  float m_pre, d_pre;
  uint32_t mn_pre, mx_pre;
  estadisticos(t_rep, N_REPS, r.inv_media_us, r.inv_desv_us, r.inv_min_us, r.inv_max_us);
  estadisticos(t_pre, N_VENTANAS, m_pre, d_pre, mn_pre, mx_pre);
  r.pre_us = m_pre;
  r.ok = true;

  Serial.printf("    arena usada %u bytes | inferencia %.2f ms (desv %.2f) | preproceso %.2f ms\n",
                r.arena_usada, r.inv_media_us / 1000.0f, r.inv_desv_us / 1000.0f, r.pre_us / 1000.0f);
  Serial.printf("    paridad: dif max %.4f | decisiones distintas %d/%d\n",
                r.max_dif, r.dec_distintas, N_VENTANAS);

  it->~MicroInterpreter();
}

// ----------------------- Corrida completa -----------------------
void correr_todo() {
  Serial.println();
  Serial.println("============== BENCHMARK TinyML - ESP32 (multi-modelo) ==============");
  Serial.printf("CPU: %u MHz | Heap libre: %u | Mayor bloque libre: %u | Sketch en flash: %u bytes\n",
                (unsigned)ESP.getCpuFreqMHz(), (unsigned)ESP.getFreeHeap(),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                (unsigned)ESP.getSketchSize());
  Serial.printf("Arena reservada para el interprete: %u bytes (RAM interna)\n", (unsigned)arena_size);
  Serial.printf("Ventana: %d muestras = %.1f s | %d ventanas de prueba | %d repeticiones\n",
                LARGO_VENTANA, LARGO_VENTANA / 125.0f, N_VENTANAS, N_REPS);

  for (int k = 0; k < N_CONFIGS; k++) medir_config(k);

  // ---- Tabla resumen (CSV, lista para copiar) ----
  Serial.println("\n--- RESUMEN (CSV) ---");
  Serial.println("config,modelo_bytes,arena_usada_bytes,init_ms,preproceso_ms,inferencia_ms_media,inferencia_ms_desv,inferencia_ms_min,inferencia_ms_max,total_ms,margen_tiempo_real,max_dif_prob,decisiones_distintas,energia_uJ");
  for (int k = 0; k < N_CONFIGS; k++) {
    const Resultado& r = resultados[k];
    if (!r.ok) {
      Serial.printf("%s,%u,NO_MEDIDO\n", CONFIGS[k].nombre, CONFIGS[k].bytes);
      continue;
    }
    const float total_ms = (r.pre_us + r.inv_media_us) / 1000.0f;
    const float energia = (CORRIENTE_ACTIVA_mA > 0.0f) ? VOLTAJE_V * CORRIENTE_ACTIVA_mA * total_ms : 0.0f;
    Serial.printf("%s,%u,%u,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.4f,%d,%.1f\n",
                  CONFIGS[k].nombre, CONFIGS[k].bytes, r.arena_usada, r.init_us / 1000.0f,
                  r.pre_us / 1000.0f, r.inv_media_us / 1000.0f, r.inv_desv_us / 1000.0f,
                  r.inv_min_us / 1000.0f, r.inv_max_us / 1000.0f, total_ms,
                  (LARGO_VENTANA / 125.0f * 1000.0f) / total_ms, r.max_dif, r.dec_distintas, energia);
  }
  if (CORRIENTE_ACTIVA_mA <= 0.0f) {
    Serial.println("(energia_uJ = 0: define CORRIENTE_ACTIVA_mA en el sketch para estimarla)");
  }
  Serial.println("=====================================================================");
  Serial.println("Escribe cualquier caracter para repetir.");
}

bool iniciar() {
  resolver.AddExpandDims();
  resolver.AddConv2D();
  resolver.AddReshape();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddLogistic();

  // Reserva la mayor porción posible de RAM interna para el arena (dejando margen al sistema)
  const size_t bloque = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  const size_t margen = 16 * 1024;
  arena_size = (bloque > margen) ? (bloque - margen) : 0;
  if (arena_size > 200 * 1024) arena_size = 200 * 1024;
  arena = (uint8_t*)heap_caps_malloc(arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!arena) {
    Serial.println("ERROR: no se pudo reservar memoria para el arena.");
    return false;
  }
  return true;
}

static bool listo = false;

void setup() {
  Serial.begin(115200);
  delay(2000);   // tiempo para abrir el Monitor Serie
  listo = iniciar();
  if (listo) correr_todo();
}

void loop() {
  if (Serial.available()) {
    while (Serial.available()) Serial.read();
    if (listo) correr_todo();
  }
  delay(50);
}