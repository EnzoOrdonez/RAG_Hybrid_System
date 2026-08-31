# Diseño del experimento de nube — ¿el techo de fidelidad es de cómputo?

> **ARCHIVO HISTÓRICO — NO-GO registrado.** El diseño no se autorizó ni se ejecutó;
> se mantuvo cero gasto. No es un pendiente operativo. La propuesta original del
> 2026-07-30 se conserva debajo para trazabilidad.

**Estado original: PROPUESTA. Cero gasto ejecutado. Requería OK explícito de Enzo con el
costo sobre la mesa, y debía ir DESPUÉS de exp18** (§6).

---

## 1. La pregunta, y por qué la nube es la única forma de responderla

La fase de verano dejó el techo sin explicar: la fidelidad no se mueve con el arreglo del
contexto (Tier A, 0/4 en 3 verificadores) ni con el prompt (exp16, 0/2), y solo se mueve —poco—
con **qué evidencia entra** (Tier 3 1/12 bajo HHEM; exp17 +0,081 HHEM). Quedan dos explicaciones
que ningún experimento local ha podido separar:

- **(M) Método**: mejor selección de evidencia subiría la fidelidad, y aún no dimos con ella.
- **(C) Capacidad**: granite 4.1 8B no puede anclar mejor por más y mejor evidencia que reciba.

**exp18 separa M de C localmente y gratis** (brazo `oracle_evidence`: techo de selección con
oráculo independiente; brazo `evidence_swapped`: ¿el generador lee siquiera el contexto?). Por eso
va primero. Lo que exp18 **no** puede hacer es probar (C) directamente: haría falta un modelo que
no cabe en 6 GB.

### Lo que la nube NO compra (medido, no supuesto)

Un hallazgo de esta fase acota fuertemente el alcance del gasto. Proyección del tamaño de prompt
sobre el subset de 60 q (calibración `tokens ≈ 0,2228·chars + 261`, ajustada sobre los tokens
reales de exp12, n=192, R²=0,922):

| k (fragmentos) | p50 tok | p90 tok | supera 4096 |
|---|---|---|---|
| **5 (config actual)** | 1973 | 2999 | **0/60 (0 %)** |
| 10 | 3685 | 5737 | 24/60 (40 %) |
| 15 | 5397 | 8474 | 50/60 (83 %) |
| 20 | 7109 | 11212 | 55/60 (92 %) |

→ **A la configuración desplegada, la ventana de 4096 no ata en absoluto.** "Contexto completo sin
truncar" no es una palanca a k=5; solo empieza a serlo por encima de k≈7. (Consistente con exp12:
solo **2/194** prompts tocaron 4096.) Así que la nube compra **capacidad de modelo**, y compra
contexto **únicamente si exp18 muestra que más fragmentos ayudan** — cosa que localmente no se
puede testear limpio, porque a k=10 el 40 % ya viene truncado y "más evidencia" se confunde con
"evidencia cortada".

---

## 2. Qué correr (IDs nuevos `exp21+`, banco de pruebas separado)

Cuatro brazos. El tercero es el que responde la pregunta; los otros tres existen para que su
respuesta sea interpretable.

| # | Brazo | Motor | Modelo | Queries | Gen. | Qué aísla |
|---|---|---|---|---|---|---|
| A | Control de replicación | **Ollama** (idéntico a local) | granite4.1:8b | 60 | 60 | ¿el puerto funciona? Debe reproducir el nivel local dentro de la deriva H5 |
| B | Puente de motor | **vLLM** bf16 | granite4.1:8b | 60 | 60 | efecto del MOTOR, aislado del de capacidad |
| C | **Capacidad** | vLLM bf16 | modelo mayor (~32B) | 60 | 60 | **C vs B = capacidad pura**, mismo motor, mismos contextos |
| D | Confirmatorio exp17 | vLLM bf16 | granite + mayor | 25 × 2 brazos × 2 modelos | 100 | ¿la cobertura balanceada cruza significancia con más potencia? |
| E | *(condicional)* Contexto | vLLM bf16 | modelo mayor | 60 @ top-20 | 60 | **solo si exp18 muestra que top-10 ayuda** |

Total: **280 generaciones** (340 con E).

**Por qué B existe.** Es el brazo que casi todo el mundo se salta. Sin él, "modelo mayor en vLLM vs
granite en Ollama" confunde capacidad con motor de inferencia (kernels, atención, plantilla de chat,
manejo de seed). Con B, la comparación de capacidad es C−B a motor constante, y A−B mide el efecto
del motor por separado. Cuestan 60 generaciones y salvan la interpretación entera.

**bf16 sin cuantizar, no negociable en C.** Cuantizar confunde "capacidad" con "precisión numérica"
y convertiría un negativo en no-interpretable. Es la razón principal de pedir 80 GB de VRAM.

**Contextos congelados.** A, B y C usan **exactamente los mismos `retrieved_ids`** (los de exp11
híbrido firmado, vía el subset de 60 q). No se re-recupera nada en la nube: la recuperación es
determinista y ya está medida; lo único que varía es el generador.

**La puntuación NO se paga en la nube.** Vuelven solo los JSON de respuestas; NLI small/base y HHEM
se corren después en local con los scripts existentes (`run_exp15_ablation.py --pass N`,
`rescore_grounding_tierA.py --exp-dir`). Ahorra horas de GPU cara y mantiene el instrumento
idéntico al de toda la fase — que el verificador cambie de máquina sería un confound gratuito.

---

## 3. Costo

| Concepto | Estimación |
|---|---|
| GPU | **A100 80 GB** (bf16 de un 32B ≈ 64 GB + KV cache). L40S 48 GB **no alcanza** sin cuantizar |
| Precio de referencia | RunPod Secure Cloud ~USD 1,89/h · Lambda ~USD 1,29/h (A100 40 GB, insuficiente) · Vast.ai más barato, menos fiable |
| Descarga de pesos | 60-70 GB → 15-40 min según el enlace |
| Setup + sondas de determinismo | ~1 h |
| Generación (280) | ~1,5-2,5 h (≈10 s/respuesta a 40-60 tok/s, respuestas de ~500 tok) |
| **Reloj total** | **5-9 h** |
| **Costo esperado** | **USD 10-18** |
| **Techo sugerido** | **USD 50** (cubre 2-3 arranques fallidos, que es el modo de fallo real) |

El diagnóstico sale barato. Lo caro es el tiempo humano de montar un entorno nuevo y verificar que
no está mintiendo, no los créditos de GPU.

---

## 4. Protocolo de entorno (la parte que se puede hacer mal en silencio)

La nube rompe `HF_HUB_OFFLINE=1`, que es el guardarraíl bajo el que se produjo toda la evidencia.
Por tanto:

1. **Banco separado.** IDs `exp21+`, directorio propio. **Prohibido escribir** en
   `experiments/results/exp3..exp18`. Verificar al volver con
   `git diff --name-status nota3-evidencia-2026-06-11 -- experiments/results` (solo altas).
2. **Congelar y registrar** el entorno: versiones de torch / transformers / vLLM / Ollama, driver
   CUDA, GPU exacta, imagen base. Al ledger, junto al **sha256 de cada peso descargado**.
3. **Re-verificar el determinismo allí, no asumirlo.** H5 demostró que el determinismo depende del
   entorno: gemma y mistral no son deterministas a temp=0 ni en frío, y a granite le cambia la
   primera generación entre caché frío y caliente. Sonda 3× por brazo, registrada en `probe_report`,
   **antes** de creerse ningún contraste. Aviso concreto: **vLLM no es bit-determinista al variar el
   tamaño de batch** — fijar batch=1 para la sonda, o registrar la no-determinación y tratar los
   contrastes como pareados dentro de sesión (que es lo que ya hacemos).
4. **Seed 42, temp 0** en todo. `PYTHONHASHSEED=42`.
5. **Vuelven solo JSON de resultados.** Nada de `.env`, credenciales ni claves al repo. Si aparece
   un secreto: parar.
6. **A es la compuerta.** Si el control de replicación no reproduce el nivel local dentro de la
   deriva H5 conocida (+0,033, n.s., r=0,86 entre junio y julio), **nada de B/C/D es interpretable**
   y hay que arreglar el puerto antes de seguir gastando.

---

## 5. Cómo se lee el resultado

| Resultado | Lectura | Consecuencia |
|---|---|---|
| C ≈ B (modelo mayor no sube la fidelidad) | El techo **no** es capacidad | Fuerte: el cuello es método o instrumento. Refuerza la contribución local y manda el esfuerzo a exp19/exp20 y al gold |
| C > B claramente | El techo **es capacidad** | Hallazgo. **No** convierte la config de nube en la recomendada (§7) |
| A no reproduce el nivel local | El puerto está mal | Parar. No interpretar nada más |
| D cruza significancia | exp17 confirmado con potencia | El piloto pasa a resultado; sube de "Trabajo Futuro accionable" a hallazgo |
| D no cruza con n mayor | La cobertura es real pero pequeña | Honesto: se reporta el tamaño de efecto con su IC, no un p-valor |

---

## 6. Compuerta: nada de esto se lanza antes de exp18

exp18 corre local, cuesta ~3-4 h de GPU propia y **cero dinero**, y cambia qué vale la pena pagar:

- Si `evidence_swapped` ≈ `baseline` (el generador **ignora** el contexto) → el techo es de
  capacidad casi con seguridad, y la nube es la prioridad. Lanzar A-B-C-D.
- Si `oracle_evidence` >> `baseline` (hay margen de selección sin explotar) → el techo es de
  **método**; la nube baja de prioridad y el presupuesto se va a exp19/exp20. Correr como mucho D.
- Si `final_top_k_10` ayuda **en las queries no truncadas** → el brazo E (contexto) se justifica;
  si no ayuda, E se cae y se ahorran 60 generaciones.

---

## 7. La tensión de contribución (decisión de Enzo, no mía)

Parte del valor de la tesis es que el sistema **corre en 6 GB, reproducible en hardware modesto**.
Si C rompe el techo, eso es un **hallazgo** ("el cuello es capacidad, no método"), **no**
automáticamente la nueva configuración recomendada. Adoptar una config que solo existe en la nube
reencuadra la contribución y es una decisión explícita tuya. Lo que sí me toca es poner el
trade-off completo sobre la mesa —fidelidad ganada vs accesibilidad y reproducibilidad perdidas— y
no decidirlo por mi cuenta.

---

## 8. Qué necesito de ti para ejecutar

1. **OK explícito** con el costo a la vista, **después** de leer exp18.
2. **Proveedor y credenciales** (recomendado RunPod Secure Cloud por trazabilidad de la instancia;
   Vast.ai es más barato pero la reproducibilidad del entorno es peor).
3. **Techo de presupuesto** (sugerido USD 50).
4. **Elección del modelo mayor**: te llevaré 2-3 candidatos concretos con su VRAM en bf16 cuando
   exp18 haya decidido si esto se lanza.
