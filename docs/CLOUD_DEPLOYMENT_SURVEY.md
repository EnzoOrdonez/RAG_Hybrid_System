# Nube como INFRAESTRUCTURA DE DESPLIEGUE del estudio SUS/Likert

**Estado: PROPUESTA. Cero gasto ejecutado. Requiere OK explícito de Enzo con el costo sobre la
mesa.** Fecha: 2026-08-04.

> **Este documento no reemplaza a `docs/CLOUD_EXPERIMENT_DESIGN.md`.** Aquel es la nube como
> **experimento de capacidad** (modelo mayor, ¿el techo de fidelidad es de cómputo?), y sigue en
> **NO-GO** tras exp18. Este es otro propósito y otra pregunta: **alojar el sistema ya medido para
> que los participantes puedan usarlo sin que la lentitud contamine sus respuestas.**

---

## 1. El problema, con su causa medida

Medido en la ruta de despliegue con `SURVEY_DEPLOY` y granite4.1:8b (n=12 por config,
`output/audit/survey_config_latency_2026-08-04.json`):

| config | retrieval p50 | **TTFT p50** | TTFT p90 | total p50 |
|---|---|---|---|---|
| k=5 | 5,1 s | **12,8 s** | 47,7 s | 179,8 s |
| k=10 | 5,0 s | **15,2 s** | 34,6 s | 132,7 s |

Un participante espera **13-15 s hasta la primera palabra** y **2-3 minutos hasta la respuesta
completa**. En un SUS/Likert eso no mide la calidad del sistema: mide la espera.

### La causa: el modelo no cabe

Del log de Ollama de la corrida de Tier A (`output/audit/tierA_run_2026-07-23_1218.ollama.err.log`):

```
print_info: file type = Q4_K - Medium
load_tensors: offloading 30 repeating layers to GPU
load_tensors: offloaded 30/41 layers to GPU            <- 11 capas se quedan en CPU
load_tensors:        CUDA0 model buffer size = 3402.09 MiB
llama_context: graph splits = 114 (with bs=512), 3 (with bs=1)
msg="vram-based default context" total_vram="6.0 GiB" default_num_ctx=4096
granite.context_length = 131072
```

**El prefill (bs=512) cruza la frontera CPU/GPU 114 veces; el decode (bs=1) solo 3.** El prefill
*es* el TTFT. Por eso el primer token tarda decenas de segundos mientras el resto fluye.

**No hace falta un modelo mayor ni una A100. Hace falta que el modelo quepa.** Con las 41 capas en
GPU no hay 114 saltos: hay 0.

Corolario documentado: el tope de **4096 tokens es derivado de la VRAM**, no del modelo. Granite
4.1 declara `context_length = 131072`. Las 74/194 queries que truncan a k=10 son artefacto de la
laptop de 6 GB.

---

## 2. Qué cambia y qué NO cambia

| | local (medido) | alojado (propuesto) |
|---|---|---|
| modelo | `granite4.1:8b` | **el mismo** |
| cuantización | Q4_K_M | **la misma**, mismo digest verificado |
| motor | Ollama | **el mismo** |
| ventana de contexto | 4096 | **4096, fijada explícitamente** |
| seed / temperatura | 42 / 0,0 | **iguales** |
| prompt / retrieval / rerank | `SURVEY_DEPLOY` | **idénticos** |
| **capas en GPU** | **30/41** | **41/41** ← **el único cambio** |

**Un solo cambio, deliberadamente.** Cambiar de modelo rompería la cadena de evidencia y el
argumento de "corre en hardware modesto". Cambiar de motor o de cuantización haría la equivalencia
ininterpretable: no se sabría a qué atribuir una diferencia.

**Por qué se fija la ventana en 4096 y no se deja crecer** (decisión de Enzo, 2026-08-04): con 4096
el estrato truncado se conserva igual que en exp18 (2/194 a k=5, 74/194 a k=10), así que **la
evidencia de fidelidad de exp18 transfiere tal cual**. Dejarla crecer a 32k eliminaría el truncado
y produciría un sistema genuinamente distinto del medido, obligando a re-medirlo entero y mezclando
dos cambios en un solo test de equivalencia.

---

## 3. GPU y presupuesto de VRAM

Derivado del log local (`CUDA0 model buffer = 3402 MiB` para 30/41 capas → ~4,65 GiB para 41/41;
KV cache 640 MiB por slot a 4096 cells, medido):

| componente | VRAM |
|---|---|
| granite 4.1 8B Q4_K_M, 41/41 capas | ~5,0 GiB |
| KV cache, 4096 × 4 slots concurrentes | ~2,5 GiB |
| bge-large-en-v1.5 (fp16) | ~0,7 GiB |
| ms-marco-MiniLM-L-12-v2 | ~0,15 GiB |
| nli-deberta-v3-small — la UI **verifica en vivo** (`chat_page.py:213` → `verify_answer`) | ~0,6 GiB |
| **total** | **~9 GiB** |

→ **16 GB bastan; 24 GB da margen cómodo** para más concurrencia o picos.

| GPU | VRAM | USD/h aprox. | nota |
|---|---|---|---|
| **RTX 4090** | 24 GB | 0,35-0,70 | **recomendada**: sobra para un 8B Q4 y es la más barata por VRAM útil |
| RTX A5000 | 24 GB | 0,26-0,45 | alternativa más barata, algo más lenta |
| L4 | 24 GB | 0,43-0,80 | eficiente, pensada para inferencia |
| A10G (AWS g5.xlarge) | 24 GB | ~1,00 | solo si ya hay cuenta AWS |
| L40S | 48 GB | 0,86-1,15 | **sobra**, no pagar esto |
| A100 80 GB | 80 GB | ~1,89 | el del otro documento; **aquí no hace falta** |

> **Los precios por hora son estimaciones de orden de magnitud y hay que verificarlos al
> reservar.** No son cotizaciones firmes.

**Proveedor recomendado: RunPod Secure Cloud**, por trazabilidad de la instancia (mismo criterio
que `CLOUD_EXPERIMENT_DESIGN.md` §8). Community Cloud es más barato y menos reproducible.

---

## 4. Coste del estudio

Modelo de exposición elegido: **bajo demanda, por sesiones agendadas** (decisión de Enzo). El pod
se levanta solo durante las sesiones.

| concepto | horas |
|---|---|
| setup + descarga (índices 190 MB · modelos ~1,4 GB · granite ~4,7 GB desde el CDN de Ollama) | ~1 |
| **validación de equivalencia** (§5): 194 generaciones + sondas 3× | ~2-3 |
| 20 participantes × 30 min | ~10 |
| margen y re-arranques | ~5 |
| **total** | **~15-20 h** |

**Costo esperado: USD 6-14.** **Techo sugerido: USD 30**, que cubre 2-3 arranques fallidos — el
modo de fallo real. Muy por debajo del techo de USD 50 del diseño de capacidad.

**Ahorro operativo:** hacer **snapshot de la imagen del pod** tras el primer setup, para que las
sesiones siguientes arranquen en minutos y no se vuelva a pagar la descarga.

**Lo que dispara el coste, y por eso no se elige:** dejar el pod encendido toda la ventana del
estudio son 168 h (7 días) = USD 59-118, o 336 h (14 días) = USD 118-235. Se paga sobre todo GPU
ociosa.

---

## 5. Validación de equivalencia — **la compuerta** (`exp21_hosted_equivalence`)

**Ningún participante toca el sistema alojado hasta que esto pase.** Si granite alojado no produce
lo mismo que granite local, la evidencia de la fase no describe lo que verían los participantes, y
hay que saberlo antes y no después.

IDs `exp21+`, banco separado, respetando la reserva de `CLOUD_EXPERIMENT_DESIGN.md` §2.
**Prohibido escribir** en `experiments/results/exp3..exp19`.

### Protocolo

1. **Verificar el digest del modelo** contra el local **antes de generar**. Mismos pesos o no hay
   experimento. Registrar el digest en el ledger.
2. **Brazo único:** granite alojado sobre los **contextos congelados** de exp18
   (`retrieval_ids.json::baseline_repro_ids`), las **194** queries, ruta de prompt canónica
   (`rgm.build_prompt`), temp 0, seed 42, `num_ctx=4096`, `num_parallel=1`.
   No se re-recupera nada: la recuperación es determinista y ya está medida; lo único que varía es
   dónde corren las capas.
3. **Sonda de determinismo 3×** antes de creerse ningún contraste. H5 demostró que el determinismo
   depende del entorno, y a granite le cambia la primera generación entre caché frío y caliente.
   Registrar en `probe_report`.
4. **Solo vuelven JSON de respuestas.** La **puntuación se queda en local** con los tres
   verificadores de siempre: que el instrumento cambiara de máquina sería un confound gratuito.
5. **Congelar y registrar el entorno**: versión de Ollama, driver CUDA, GPU exacta, imagen base,
   digest del modelo, con sha256. Al ledger.

### Criterio, pre-registrado antes de generar

- **Primaria: TOST de equivalencia** de la fidelidad por query, banda **±0,081** — la banda ya
  existente y ciega a este contraste — en **NLI-small + NLI-base + HHEM**.
- **Guarda de instrumento:** nivel HHEM del ancla en **0,40-0,55**. Fuera de rango se para y se
  diagnostica (un HHEM mal cargado puntuaba ~0,04 y "corría").
- **Secundarias:** divergencia de respuesta (jaccard 5-grama, jaccard de tokens, tasa de idénticas,
  reaparición de claims), claims genuinos por respuesta, clases de declinación y
  `asserts_nothing_rate`.

**Se espera que NO haya identidad bit a bit**, y eso no es el criterio: local corre 30/41 capas en
GPU y 11 en CPU, el alojado 41/41, así que el camino aritmético difiere. Por eso el criterio es
**equivalencia estadística declarada**, no igualdad.

### Regla dura (decisión de Enzo, pre-registrada)

> **Si no hay equivalencia → NO se despliega.** La encuesta corre local a k=5 asumiendo su latencia,
> y la no-equivalencia se reporta como **hallazgo**: el sistema es sensible al reparto GPU/CPU, lo
> que a su vez acota cuánto de la evidencia de la fase es propiedad del método y cuánto del hardware.
> Cadena de evidencia intacta, cero gasto adicional.

### Se reutiliza, no se reescribe

`scripts/compute_tierA_arm_stats.py` (contrastes pareados + BH) · `scripts/compute_exp18_diagnosis.py`
(TOST y divergencia) · `scripts/compute_exp16_guards.py` (guardas y clases) ·
`scripts/verify_summer_offline.py` (re-derivación offline).

---

## 6. Config de la encuesta: k=5 vs k=10 alojado (solo si §5 pasa)

Al fijar la ventana en 4096, el estrato truncado se conserva (74/194 a k=10, igual que exp18), así
que **la evidencia de fidelidad y cobertura de exp18 para k=10 transfiere tal cual**:

| | k=5 | k=10 |
|---|---|---|
| fidelidad (HHEM) | 0,4638 | 0,4835 — **plana**, n.s. |
| `answered` | 38,7 % | **67,0 %** |
| claims genuinos / respuesta | 10,6 | **15,0** |
| "no afirma nada" | 3,1 % | **0,5 %** |

Lo que **falta medir** es lo único que la nube cambia: **la latencia alojada**. Se mide con
`scripts/measure_survey_config_latency.py` apuntado al endpoint, con la misma estratificación
seeded por (tipo de routing × truncada-a-k10), y se entrega la tabla de trade-off completa.

**La elección final de config es de Enzo**, no de este documento.

---

## 7. Exposición y seguridad

- Endpoint público con **token de acceso**. Sin token, sin sesión.
- **`.env` no sale nunca de la máquina local.** Ningún secreto en la imagen del pod ni en variables
  de entorno del proveedor que queden persistidas.
- Del pod **solo vuelven JSON de resultados**. Si aparece un secreto en lo que vuelve: parar.
- `OLLAMA_KEEP_ALIVE` largo durante las sesiones, para que el modelo no se descargue a mitad de una
  encuesta y el participante pague un arranque en frío.
- El corpus es **documentación pública** de AWS/Azure/GCP: subirlo no plantea problema de privacidad.
- Los datos de los participantes (respuestas SUS/Likert) son lo único sensible; definir dónde se
  guardan **antes** de la primera sesión, no después.

---

## 8. Qué necesito de Enzo para ejecutar

1. **OK explícito** con el costo a la vista (**USD 6-14 esperado, techo sugerido USD 30**).
2. **Proveedor y credenciales** (recomendado RunPod Secure Cloud).
3. **Ventana de sesiones**: cuántos participantes y en qué horarios, para dimensionar las horas.
4. **Dónde se almacenan las respuestas** de los participantes.

Nada de esto se ejecuta sin el punto 1. El diseño no cuesta nada; la ejecución sí.
