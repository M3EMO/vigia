# Informe técnico: señales del router MoE para detectar alucinaciones por token

Fecha: 2026-10-06. Estado: borrador con datos medidos en dos modelos (CPU y GPU). Procedimiento reproducible con
`vigia evidence` (ver §6).

## 1. Resumen

Vigía extrae señales que un modelo MoE ya calcula en cada token: los logits de salida (nivel F0)
y los logits del router de cada capa MoE (nivel F1). Un detector lineal calibrado convierte esas
señales en un P(error) por token.

En Granite 3.1 1B-A400M (MoE de 32 expertos, top-8) sobre 600 preguntas en español con respuesta
verificable, el detector F0+F1 logra **AUROC por token 0,906** [IC 95 %: 0,830–0,980]. La entropía
de salida, que es el baseline estándar, logra 0,601 [0,547–0,649]. La señal del router sola (F1)
logra 0,891 [0,808–0,969].

La ventaja se mantiene dentro de cada tipo de pregunta (§4.2), así que no proviene solo de
distinguir el tema. Una sonda lineal sobre el residual stream iguala al detector (0,906). Ese
resultado acota la tesis: el router no aporta información que el estado oculto no tenga, pero la
aporta con 1.271 features baratas en lugar de un vector denso por capa.

En gpt-oss-20b sobre GPU (§4.4), el detector logra AUROC 0,881 contra 0,643 de la entropía, y la latencia agregada es **+3,0 %** [2,97 %; 3,03 %]: la puerta 1 del plan se cumple en ese modelo.

## 2. Datos

| Fuente | Ítems | Etiqueta | Licencia |
|---|---|---|---|
| MKQA, español (Apple), muestra estratificada por tipo | 360 | Coincidencia exacta, F1 de tokens ≥ 0,5 o número/fecha con los alias gold | CC-BY 3.0 |
| Aritmética generada (sumas, restas y productos) | 240 | Resultado exacto | propio |

Las etiquetas son computables: ningún LLM juzga a otro. Cada token del canal final de una
respuesta incorrecta recibe la etiqueta 1; cada token de una respuesta correcta recibe 0. Esta
etiqueta por respuesta es más gruesa que una etiqueta por span: también marca como erróneo el
texto de relleno de una respuesta equivocada.

El modelo acierta el 16,2 % de las preguntas: 53,8 % de las sumas, 38,8 % de las restas, 2,5 % de
los productos y 5,2 % de las preguntas de entidades de MKQA.

## 3. Método

- **F0** (salida): entropía sobre el vocabulario, NLL del token emitido, margen top-1/top-2 y
  entropía top-20.
- **F1** (router, por capa MoE): entropía normalizada sobre los N expertos, entropía de los pesos
  top-k, masa top-k, márgenes z₁−z₂ y z_k−z_{k+1}, uso acumulado causal (Gini, HHI inverso,
  entropía), cambio de ruta respecto del token anterior, más media, máximo y desvío entre capas.
- **H** (residual stream): estado oculto de la capa ≈0,95·L, usado como baseline de sonda lineal.
- Detector: regresión logística con z-score y regularización elegida por validación cruzada
  agrupada; calibración de Platt; umbral por *conformal risk control* sobre la FNR por token.
- Split: 360/120/120 preguntas (train/calibración/test), agrupado por pregunta, sin fuga.
- Intervalos: bootstrap por cluster de pregunta, 2.000 réplicas, percentil 95 %.

## 4. Resultados

### 4.1 Conjunto completo (test: 1.035 tokens, 120 respuestas, prevalencia por token 92,1 %)

| Detector | AUROC por token | IC 95 % |
|---|---|---|
| Entropía de salida (baseline) | 0,601 | 0,547–0,649 |
| NLL del token (baseline) | 0,599 | 0,543–0,650 |
| F0 | 0,644 | 0,592–0,695 |
| **F1 (solo router)** | **0,891** | 0,808–0,969 |
| **F0+F1 (por defecto)** | **0,906** | 0,830–0,980 |
| Todos (F0+F1+H) | 0,926 | 0,865–0,985 |
| Sonda lineal sobre H (baseline) | 0,906 | 0,830–0,979 |

Calibración del detector por defecto: ECE 0,029 [0,016–0,057] y Brier 0,046. Con alpha = 0,1, la
FNR observada por token es 0,044 y la tasa de marcado es 0,911. Esa tasa alta refleja la
prevalencia del 92 %: el modelo se equivoca en casi todo.

### 4.2 Control de confusión por tipo de pregunta

La aritmética tiene una tasa de acierto mucho más alta que MKQA. Un detector podría aprender
"esto es aritmética" en lugar de "esto es un error". Para descartarlo, el detector se reentrena y
evalúa dentro de cada estrato (500 réplicas de bootstrap):

| Estrato | Prevalencia | Entropía de salida | F0 | F1 | F0+F1 | Sonda H |
|---|---|---|---|---|---|---|
| Aritmética (853 tokens) | 65,4 % | 0,779 | 0,809 | 0,838 | 0,883 | 0,880 |
| MKQA (4.384 tokens) | 97,3 % | 0,654 | 0,641 | 0,859 | 0,882 | 0,886 |

Dentro de MKQA, F1 supera a la entropía de salida por más de 0,2 de AUROC. Dentro de aritmética la
ventaja es menor y los intervalos se solapan. El estrato MKQA tiene muy pocas respuestas
correctas, así que su intervalo es ancho.

### 4.3 Comparación con InnerExpert

InnerExpert (arXiv 2608.17687) reporta AUROC por token 0,762 en OLMoE y 0,753 en Gemma, con
etiquetas de juez por span en inglés. Los números de este informe no son comparables de forma
directa: la etiqueta es por respuesta y la prevalencia es mucho más alta. La conclusión común es
que las señales del router superan a la entropía de salida.

### 4.4 gpt-oss-20b en GPU (Colab, Tesla T4)

Mismo dataset de 600 preguntas, con `reasoning_effort=low` y solo el canal `final` puntuado. El
modelo acierta el 54,3 %, así que las clases están más balanceadas que en Granite.

| Detector | AUROC por token | IC 95 % |
|---|---|---|
| Entropía de salida (baseline) | 0,643 | 0,565–0,702 |
| NLL del token (baseline) | 0,637 | 0,560–0,697 |
| F0 | 0,650 | 0,576–0,705 |
| **F1 (solo router)** | **0,875** | 0,778–0,947 |
| **F0+F1 (por defecto)** | **0,881** | 0,783–0,951 |
| Todos (F0+F1+H) | 0,903 | 0,796–0,970 |
| Sonda lineal sobre H (baseline) | 0,871 | 0,753–0,963 |

Control de confusión, con el detector reentrenado dentro de cada estrato:

| Estrato | Entropía de salida | F1 | F0+F1 |
|---|---|---|---|
| Aritmética | 0,625 | 0,969 | 0,983 |
| MKQA | 0,552 | 0,834 | 0,837 |
| Solo multiplicaciones | 0,605 | 0,941 | 0,976 |
| Solo preguntas de entidades | 0,555 | 0,898 | 0,898 |

La señal se sostiene dentro de un mismo tipo de operación y de pregunta. Los estratos chicos tienen
intervalos anchos (24 respuestas de test en multiplicaciones).

Latencia en GPU (A/B intercalado, 20 unidades, 5.100 latencias por token): la mediana pareada del
overhead de F0+F1 es **+3,0 %** [2,97 %; 3,03 %] y el del primer token es +0,6 %. Cumple la meta
de < 5 % del plan.

### 4.5 Generación de código (HumanEval, Granite 1B-A400M en CPU)

Las 164 tareas de HumanEval se etiquetan ejecutando sus tests en un proceso aislado: la solución
pasa o falla. Granite pasa el 35,4 % (58), falla el 61 % (100) y da error en 6.

| Medición | AUROC | Nota |
|---|---|---|
| Por token, etiqueta de la respuesta copiada a todos sus tokens | 0,41 [0,24; 0,60] | Sin señal |
| Por respuesta, entropía media del router (sin entrenar) | 0,74 [0,65; 0,82] | — |
| Por respuesta, solo el largo de la solución | 0,70 | Las soluciones que fallan son más largas |
| Por respuesta, largo + router (validación cruzada) | 0,735 | — |
| Por respuesta, largo + router + entropía de salida | 0,77 | — |

Dentro de tercios de largo parecido, la entropía del router da entre 0,61 y 0,76.

Lectura: en código, la etiqueta por token no sirve, porque una solución que falla tiene casi todo
el código bien y el error en pocas líneas. A nivel respuesta el router aporta, pero poco por encima
del largo. Con 164 tareas y un modelo de 1B, el resultado es débil; la corrida con gpt-oss-20b en
GPU decide si la señal sirve para código.

### 4.6 Generación de código con gpt-oss-20b (HumanEval + MBPP, 1.134 tareas, L4)

Se sumaron las 970 tareas de MBPP cuya solución de referencia pasa sus propios tests. gpt-oss-20b
(`reasoning_effort=low`) pasa el 79,5 %. Datos crudos: `docs/evidencia/gptoss20b_code_all-resultados.txt`.

| Medición | AUROC | Nota |
|---|---|---|
| Por token, detector F0+F1 (test, 222 tareas) | 0,675 [0,55; 0,79] | Ya no colapsa: con más datos el detector generaliza |
| Por token, entropía de salida | 0,547 [0,52; 0,58] | — |
| Por respuesta, entropía media del router (sin entrenar, 1.109) | 0,672 [0,63; 0,72] | — |
| Por respuesta, solo el largo | 0,59 | El largo pesa mucho menos que en Granite |
| Por respuesta, largo + router (validación cruzada) | 0,673 | — |
| Por respuesta, detector con las 243 señales del router + largo (CV) | **0,758** | Mejor resultado en código |
| Por respuesta, entropía de salida media sobre líneas de código | 0,555 | Hipótesis registrada: **refutada** |

Lectura:

- En gpt-oss, el router predice el fallo del código por encima del largo y de la entropía de salida.
- La hipótesis que salió de Granite (entropía de salida solo sobre líneas de código) no se replica.
  Se registró antes de ver estos datos, y por eso el resultado cuenta como refutación.
- El efecto de composición sí aparece en parte: la entropía del router sobre líneas de código baja
  de 0,672 a 0,614.
- Como alarma por respuesta con un umbral por token, el detector es débil: detecta el 85 % de las
  soluciones que fallan con 45 % de falsas alarmas (alpha 0,48, `pareto_code.txt`). Para código,
  el uso que corresponde es ordenar respuestas (mejor de N), no bloquear una sola.

## 5. Latencia

La meta del plan es < 5 % de latencia agregada. La medición limpia corrió en la misma laptop
(i5-1035G1, 4 hilos, bf16, CPU sin otros procesos pesados), con 3 prompts, 4 corridas
intercaladas, 24 tokens fijos (`ignore_eos`) y comparación A/B en el mismo proceso.

| Métrica | Sin hooks | F0+F1 | Diferencia |
|---|---|---|---|
| Latencia por token p50 (ms) | 280,6 | 303,2 | +8,0 % |
| TPOT medio (ms) | 331,6 | 359,8 | +8,5 % |
| Overhead pareado (mediana de cocientes) | — | — | +3,6 % [IC 95 %: −12,3 %; +36,4 %] |

El resultado no alcanza para afirmar ni rechazar la meta de 5 %. La estimación puntual cae entre
+3,6 % y +8,5 % según el estimador, y el intervalo es ancho porque la CPU de una laptop tiene
mucho ruido de latencia (p95 de 1,5 s por token). La medición que decide la puerta 1 corre en GPU,
donde el costo fijo de la reducción en Python pesa menos frente a un modelo más grande.

Una primera corrida del mismo benchmark se descartó: corrió en paralelo con otros procesos que
saturaban la CPU.

## 6. Reproducir

```bash
python -m vigia.cli evidence --model ibm-granite/granite-3.1-1b-a400m-instruct --dtype bfloat16 \
  --data evidence_v1.jsonl --out runs/evidence_v1 --max-tokens 24 --tiers F0,F1,H --n-boot 2000 --bench 4
```

La corrida es reanudable. En una laptop con CPU i5-1035G1 y 8 GB de RAM tarda unas 2 horas.

## 7. Límites y próximos pasos

1. La etiqueta es por respuesta; falta medir con etiquetas por span (juez validado o humanos).
2. gpt-oss-20b ya se midió en una T4; falta gpt-oss-120b y un servidor de producción (vLLM con plugin).
3. La sonda sobre H iguala a F1; falta medir si F1 transfiere mejor entre dominios o modelos.
4. La latencia en GPU cumple la meta con HF transformers; falta medirla en vLLM con batching.
