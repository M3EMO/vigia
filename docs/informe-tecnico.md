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

### 4.7 Mejor de N en código (gpt-oss-20b, 213 tareas de test × 5 muestras, temperatura 0,8)

El puntaje de Vigía se entrenó con las respuestas greedy de las 887 tareas de train+cal y eligió una
de cinco muestras por tarea de test. Los tests ocultos solo califican la elección. Datos crudos:
`docs/evidencia/gptoss20b_best_of_n.txt`.

| Política de elección | Pasan los tests |
|---|---|
| Azar (esperado) | 85,5 % |
| La más corta | 84,5 % |
| Menor entropía de salida en líneas de código | 85,4 % |
| Vigía (puntaje entrenado) | 86,9 % |
| Ejemplos visibles + azar | 89,1 % |
| Ejemplos visibles + Vigía | **89,7 %** |
| Techo: alguna de las 5 pasa | 93,0 % |

Diferencias con IC 95 % por bootstrap de tareas:

- Vigía contra la más corta: +2,3 puntos [0,0; +5,2], en el borde de la significancia.
- Ejemplos visibles + azar contra azar: +3,5 puntos [+2,0; +5,3], significativo.
- Ejemplos visibles + Vigía contra ejemplos visibles + azar: +0,6 puntos, no significativo.

Lectura: de los 7,5 puntos que separan el azar del techo, el chequeo gratis con los ejemplos del
enunciado recupera 3,5 y Vigía suma poco encima. En MBPP el ejemplo visible es además uno de los
tests ocultos, así que parte de esa ganancia es mecánica. Separado por fuente:

| Fuente | Azar | Vigía | Ejemplos + azar | Ejemplos + Vigía | Techo |
|---|---|---|---|---|---|
| HumanEval (n=39) | 92,3 % | 92,3 % | 93,5 % | 94,9 % | 100 % |
| MBPP (n=174) | 84,0 % | 85,6 % | 88,1 % | 88,5 % | 91,4 % |

En HumanEval, donde el ejemplo no es un test oculto, los ejemplos suman solo 1,2 puntos; con 39
tareas no se puede concluir más. En código con ejemplos ejecutables, el chequeo de ejemplos es la
acción principal y Vigía, un desempate.

### 4.8 Factual ampliado con gpt-oss-20b (2.600 preguntas, L4)

MKQA-es estratificado más aritmética (`data/evidence_v2.jsonl`). gpt-oss acierta el 45,7 %: casi
todo en aritmética, 30 % en entidades. Test: 478 preguntas, 5.320 tokens. Datos crudos:
`docs/evidencia/gptoss20b_v2-summary.txt` y `gptoss20b_v2-pareto.txt`.

| Medición | AUROC por token |
|---|---|
| Detector F0+F1 | 0,773 [0,66; 0,90] |
| Sonda sobre el residual (H) | 0,793 [0,68; 0,91] |
| Entropía de salida | 0,569 [0,53; 0,61] |

La aritmética no tiene errores en test, así que el AUROC global mezcla tipo de pregunta y error.
Dentro de cada tipo:

| Tipo | Tokens | AUROC detector | AUROC entropía |
|---|---|---|---|
| Fecha | 618 | 0,776 | 0,592 |
| Entidad | 3.912 | 0,689 | 0,544 |
| Número | 119 | 0,634 | 0,494 |
| Todo menos aritmética | 5.111 | **0,723** | 0,556 |

El router sigue ganando dentro de cada tipo, pero el número limpio es 0,72, no 0,77 ni el 0,88 de la
corrida de 600 preguntas. Frase corta (187 tokens) da 0,27 y número con unidad 0,97: muestras chicas.

Por respuesta, el alpha recomendado (0,26, elegido en calibración) detecta el 86 % de las respuestas
malas con 27 % de falsas alarmas; con alpha 0,42, 76 % y 16 %.

Por respuesta, sin aritmética (1.792 respuestas, 67 % fallan; `colab/factual_response.py`):

| Señal o detector | Global | Fecha | Entidad | Número |
|---|---|---|---|---|
| Entropía de salida, media | 0,646 | 0,566 | 0,668 | 0,431 |
| Entropía de salida, suma | 0,696 | 0,575 | 0,716 | 0,670 |
| Largo de la respuesta | 0,669 | 0,551 | 0,685 | 0,713 |
| CV: tipo + largo | 0,679 | 0,545 | 0,681 | 0,705 |
| CV: tipo + largo + salida (4 señales × media y máximo) | 0,684 | 0,563 | 0,685 | 0,601 |
| CV: tipo + largo + router | **0,811** | **0,759** | **0,813** | **0,815** |

Ningún armado de la entropía de salida (media, máximo, primer token, suma, NLL, margen) supera al
largo por más de 0,03, y sumada al largo no aporta. El router sí: +0,13 sobre tipo y largo, dentro de
cada tipo de pregunta. Es la medición más limpia del proyecto a favor de la hipótesis de partida.

Latencia en L4 (8 prompts × 5 corridas, 256 tokens fijos; `docs/evidencia/gptoss20b_bench_l4_sinH.json`):

| Nivel | p50 ms/token | Overhead (IC 95 %) |
|---|---|---|
| Sin hooks | 80,9 | — |
| F0 | 83,3 | +3,0 % [+2,7; +3,2] |
| F0+F1 | 87,1 | **+7,6 %** [+7,4; +7,9] |

El +7,6 % es del router (F1), no del nivel H. En la T4 el mismo F0+F1 dio +3,0 %: la L4 genera más
rápido y el costo fijo de los hooks en Python pesa más.

Medido en la CPU con las dimensiones de gpt-oss, el mayor costo por token era F0: copiar los
201.088 logits a la CPU y procesarlos en float64. Dos cambios (commit `fbf073a`): F0 se calcula en
el dispositivo de los logits y la CPU recibe 4 números; los pesos del router se pasan a float32 una
vez por contexto. Misma L4, mismo protocolo (`docs/evidencia/gptoss20b_bench_l4_f0gpu.json`):

| Nivel | Antes | Después |
|---|---|---|
| F0 | +3,0 % [+2,7; +3,2] | **+1,0 %** [+0,9; +1,1] |
| F0+F1 | +7,6 % [+7,4; +7,9] | **+4,7 %** [+4,6; +4,8] |

**La meta de 5 % se cumple en L4 con el backend HF**, con poco margen. La reducción del router en
numpy no explica el costo restante (§4.10).

Modo vLLM (solo IDs de expertos y top-20 logprobs, sin logits del router), medido offline sobre las
1.792 respuestas factuales sin aritmética: tipo + largo 0,679; + salida parcial y cambios de experto
0,742; router completo (modo HF) 0,811. vLLM conserva la mitad de la ganancia del router sin costo
de latencia por token.

### 4.9 Entropía semántica + router (351 preguntas de test sin aritmética, 5 muestras)

Cinco respuestas por pregunta (temperatura 0,8), agrupadas por significado: números iguales o
palabras que se solapan. La variante tolerante compara el primer número sin separadores (gpt-oss
escribe "560,000" y el parser rioplatense lo leía como 560), con 2 % de tolerancia y años exactos;
se corrigió al ver los clusters y se informan las dos. Datos: `docs/evidencia/gptoss20b_semantic_entropy.txt`.

| Señal | AUROC por respuesta | Generaciones |
|---|---|---|
| Largo | 0,660 | 1 |
| Router (entrenado en train+cal) | **0,809** [0,76; 0,86] | 1 |
| Entropía semántica, estricta | 0,747 | 5 |
| Entropía semántica, tolerante | 0,757 | 5 |
| Router + semántica tolerante | 0,833 [0,79; 0,88] | 6 |
| Cascada: semántica solo en el 50 % más dudoso | 0,823 | 3,5 en promedio |

Router + semántica tolerante contra router solo: +0,024 [IC 95 %: −0,009; +0,057], no
significativo. La entropía semántica, el método de referencia de la literatura, queda por debajo del
router con cinco veces más cómputo, y sumarla no mejora de forma demostrable.

Este detector (entrenado con 2.600 preguntas, con `recommended_alpha`) reemplaza al de 600 en
`detectors/gpt-oss-20b.json`. El artefacto anterior queda en `tests/fixtures/legacy_detectors/`.

### 4.10 Perfil del router y modo vLLM real (L4, 2026-10-13)

**Perfil del router en HF.** Seis prompts, 128 tokens fijos, tres corridas intercaladas (n = 18 por
variante). Datos: `docs/evidencia/gptoss20b_vllm_real.txt`.

| Variante | ms por token | Overhead |
|---|---|---|
| Sin hooks | 76,73 | — |
| F0+F1 (extractor actual) | 78,90 | +2,83 % |
| F0+F1 sin la reducción en CPU | 78,89 | +2,82 % |
| Logits que el router de gpt-oss ya calcula | inviable | — |

La reducción en CPU no cuesta nada medible: pasarla a la GPU no gana margen. El costo está en los
hooks. El hook sobre el módulo router no dispara en gpt-oss, porque el kernel fusionado no lo llama.
El +2,83 % no reemplaza al +4,7 % de §4.8: el protocolo cambia (prompts, 128 tokens fijos).

**Modo vLLM real.** vLLM 0.31 con `--enable-return-routed-experts`, a través del proxy de Vigía
(niveles F0 y U), sobre las 600 preguntas de §4.4 con `max_tokens` 256. La exactitud del modelo es
59,3 %. La aritmética acierta siempre, así que el AUROC global (0,947) mide sobre todo el tipo de
pregunta. La cifra comparable es la de MKQA sola, con el detector reentrenado dentro del estrato:

| MKQA sin aritmética | AUROC por token | IC 95 % |
|---|---|---|
| vLLM: F0 + uso de expertos | **0,784** | 0,639–0,916 |
| vLLM: solo F0 | 0,512 | 0,396–0,644 |
| vLLM: NLL del token (baseline) | 0,461 | 0,355–0,597 |
| HF: F0+F1 (§4.4, otra corrida y `max_tokens` 24) | 0,837 | — |

| Latencia por token p50 en vLLM | ms |
|---|---|
| Con `--enable-return-routed-experts` | 22,57 |
| Sin la opción | 22,61 |

**Concurrencia.** De 1 a 32 requests simultáneos con tareas de código (hasta 256 tokens, dos
rondas por nivel). "Directo" es vLLM sin logprobs ni expertos; "Vigía" es el proxy completo.

| Requests | Directo, servidor sin la opción | Directo, con la opción | Vigía | Tokens/s: sin opción → Vigía |
|---|---|---|---|---|
| 1 | 17,2 ms/token | 17,1 | 17,5 | 58 → 57 |
| 4 | 28,8 | 28,7 | 28,0 | 117 → 118 |
| 16 | 45,2 | 45,3 | 43,7 | 279 → 284 |
| 32 | 55,6 | 55,4 | 50,8 | 491 → 491 |

Con carga, la opción de expertos y el proxy tampoco cuestan: el throughput coincide en cada nivel.
Las diferencias de ms/token entran en el ruido de pocas rondas. `ignore_eos` no sirve en este modo:
con gpt-oss, vLLM 0.31 rompe su parser harmony en respuestas no streaming (HTTP 500).

Devolver los expertos no agrega latencia medible. En la misma L4, vLLM tarda unos 22,6 ms por token
y HF 76,7 ms, con protocolos distintos. El modo vLLM conserva la mayor parte de la señal del router dentro de MKQA, con un
intervalo ancho (114 preguntas de test). La medición destapó dos defectos del proxy, ya corregidos:
vLLM 0.31 entrega `routed_experts` como `.npy` en base64 con las filas del prompt, y los logprobs
traen el canal `analysis` de gpt-oss.

### 4.11 Réplica del protocolo de InnerExpert en OLMoE-1B-7B (L4, 2026-10-14)

Mismo modelo, mismos datos y misma partición que InnerExpert (arXiv 2608.17687): OLMoE-1B-7B-0924-
Instruct, dos respuestas por pregunta (sin y con evidencia), greedy, 65 tokens; entrenamiento con
RealtimeQA 2024-2025 (2.318 preguntas) y test fuera de distribución en RealtimeQA 2026 (enero a junio,
365 preguntas) y 200 preguntas de SQuAD, TruthfulQA, NQ-Open y FreshQA (`df.sample(n=200,
random_state=42)`, como su pipeline). Etiquetas por fragmento de un juez. Scripts:
`colab/innerexpert_prep.py` y `colab/innerexpert_eval.py`; datos: `docs/evidencia/olmoe_innerexpert_comparacion.txt`.

| Test (AUROC por token, IC 95 % por pregunta) | Vigía F0+F1 | Solo router (F1) | Entropía de salida |
|---|---|---|---|
| RealtimeQA 2026 | 0,748 [0,729; 0,765] | 0,740 | 0,595 |
| SQuAD | 0,650 [0,619; 0,682] | 0,632 | 0,603 |
| TruthfulQA | 0,618 [0,582; 0,649] | 0,605 | 0,565 |
| NQ-Open | 0,591 [0,558; 0,623] | 0,573 | 0,560 |
| FreshQA | 0,649 [0,619; 0,678] | 0,643 | 0,532 |
| **Promedio** | **0,651** | 0,639 | 0,571 |

InnerExpert reporta 0,762 de promedio con regresión logística en OLMoE, y 0,639 para la entropía.
Vigía queda 0,11 por debajo en el promedio y cerca en RealtimeQA (0,748). La ventaja sobre la entropía
se mantiene en los cinco conjuntos, pero es menor que en el paper (+0,08 contra +0,12).

La comparación no es exacta por cuatro diferencias:

1. Juez: DeepSeek-v4-flash con el prompt de Vigía, no GLM-5.1 con el suyo (la cuenta de GLM se quedó sin saldo).
2. Pesos en bf16; el paper cuantiza a 4 bits.
3. Vigía usa solo F0 y F1; InnerExpert suma desacuerdo entre expertos y señales del transformer, con
   más costo (la revisión del 2026-10-04 lo estima en +147 % solo para las señales MoE).
4. La evidencia se corta en 10.000 caracteres; el paper trunca el prompt por la derecha.

Rechazos del juez: 89 de 6.966 respuestas (1,3 %, por formato), sin fallas de transporte.

### 4.12 MoE fractal: ¿dividir y fusionar expertos reduce el olvido? (L4, 2026-10-15 a 2026-10-22)

Hipótesis: un MoE que divide expertos donde el router duda, repite datos viejos y fusiona expertos redundantes
olvida menos que un MoE fijo con los mismos parámetros y la misma repetición. Código: repo `mf-moe`; diseño y
decisiones: mapa `.scratch/mf-moe-olvido` (tickets 05 a 12).

**Fase 1 (MoE Qwen2 de ~90 M desde cero, 12 tareas aritméticas en fila, semillas 0-2 × nombre de tarea sí/no).**
Métrica primaria: BWT desde lo aprendido (exactitud final menos exactitud al terminar cada tarea, medida antes de
fusionar). Desvío del preregistro: el BWT clásico se mide después de fusionar y esconde el daño de la fusión; con
él, el fractal "gana" (+0,17) solo porque la fusión baja lo aprendido. Diferencia pareada contra fijo+repetición,
bootstrap IC 95 %:

| Variante (pares) | BWT desde lo aprendido | Exactitud media final |
|---|---|---|
| fijo+repetición (control, 6 corridas) | −0,674 | 30,5 % |
| división + fusión por salidas (5) | −0,081 [−0,119; −0,033], pierde | −12,4 pts [−19,7; −5,1], pierde |
| división + fusión alineada por permutación (5) | −0,096 [−0,162; −0,038], pierde | −12,1 pts [−20,9; −5,3], pierde |
| división + fusión con recuperación (3) | −0,013 [−0,035; +0,007], empate | −1,4 pts [−3,2; +0,2], empate |
| división sin fusión (6) | −0,011 [−0,055; +0,039], empate | −2,0 pts [−7,1; +3,3], empate |

Lectura: fusionar daña (25-35 puntos de exactitud en el momento de la fusión, aun con alineación al estilo
MC-SMoE); recuperar después de fusionar solo devuelve lo perdido; dividir por duda no mejora al MoE fijo. El ruido
entre corridas idénticas llega a 15 puntos de exactitud. Con 6 a 12 expertos no hubo presión de capacidad.

**Fase 2 (Granite-3.1-1B-A400M preentrenado, 2026-10-20 a 2026-10-22).** Pregunta: ¿la escala cambia el resultado? Capas 12-23
entrenables (fp32), de 32 a 48 expertos. Seis pilotos fijaron el protocolo:

1. Los dominios en inglés (matemática, código, biomedicina, derecho) no sirven: Granite ya los sabe y entrenarlos
   solo sube su pérdida, con cualquier lote y lr. Se reemplazan por idiomas fuera de Granite (FineWeb-2: finés,
   turco, indonesio y suajili): con 1,6 M tokens el finés baja 0,23 nats y al aprender turco olvida 0,62.
2. Un experto nuevo que el router elige desde el primer paso desplaza a otros y daña al modelo preentrenado
   (+0,5 nats aun como clon). Los nuevos nacen silenciosos (sesgo −1) y un balanceo solo para nacidos los recluta;
   el daño baja a +0,06-0,08.
3. El umbral de duda de la fase 1 (z 3,5) no se dispara en los idiomas: la división pasa a un calendario fijo y la
   duda decide qué experto se divide. Un brazo divide al azar con el mismo calendario para aislar el aporte de la duda.

4. Un diagnóstico sobre el control ubicó el olvido: devolver solo los expertos a su estado anterior recupera 96 %
   del olvido del finés; la atención sola, 69 %; el router solo, 6 %; las normas, 0 %. De ahí sale un sexto brazo,
   la consolidación por uso: al cerrar cada idioma (menos el último) congela los 8 expertos por capa que ese idioma
   más usó.

Diseño: 6 brazos × 3 semillas, 4 idiomas en fila (finés → turco → indonesio → suajili, ~10 M tokens cada uno,
152 pasos de 64 secuencias, lr 6e-5, 5 % de repetición de FineWeb-Edu). Métrica primaria: olvido medio en nats
(pérdida final menos pérdida al terminar cada idioma, promedio de los tres primeros; menos es mejor). Datos
idénticos en las 18 corridas (sha256). Diferencia pareada por semilla, `brazo − control` con el signo invertido
(positivo = el brazo olvida menos); IC 95 % por bootstrap, p de la t pareada y p corregido por Holm entre las 7
preguntas:

| Brazo (3 pares) | Olvido medio | Diferencia contra control | p (Holm) | Pérdida aprendida |
|---|---|---|---|---|
| control: 48 expertos desde el inicio | 0,729 | — | — | 2,401 |
| división (32 → 48 por duda) | 0,711 | +0,018 [−0,005; +0,051] | 0,41 (0,81), empate | 2,403 |
| división + padre congelado | 0,741 | −0,012 [−0,034; +0,016] | 0,51 (0,81), empate | 2,398 |
| azar + padre congelado | 0,767 | −0,038 [−0,082; +0,003] | 0,26 (0,78), empate | 2,403 |
| división + fusión alineada | 0,807 | −0,078 [−0,106; −0,049] | 0,042 (0,21), peor en las 3 semillas | 2,403 |
| **división + consolidación por uso** | **0,561** | **+0,169 [+0,145; +0,182]** | **0,005 (0,029), mejor** | 2,426 |

Lectura:
- La consolidación por uso es la única que gana y sobrevive a Holm: olvida 23 % menos que el control en las tres
  semillas (finés 0,75 contra 0,98 en la semilla 2). Paga un costo chico en plasticidad: aprende cada idioma
  0,025 nats peor (p 0,008). La pérdida general (FineWeb-Edu) no empeora (+0,008 a favor).
- Dividir, solo o congelando al padre, empata con el control: la escala no cambia el resultado de la fase 1.
- Fusionar empeora el olvido, igual que en la fase 1.
- La duda elige un poco mejor que el azar qué padre congelar (+0,026 en las tres semillas, p 0,13, Holm 0,53),
  pero congelar al padre no protege: lo que protege es congelar lo que el idioma viejo usó.
- Límites: un solo orden de idiomas; las semillas cambian el orden de los lotes y la repetición, no los datos, así
  que el ruido entre semillas (±0,01 en el control) puede subestimar la varianza real; con 3 pares, el IC del
  bootstrap es casi el rango de las diferencias, por eso se informa también la t con Holm.

**Etapa A (ticket 14, 2026-10-22 a 2026-10-24): ¿la consolidación necesita dividir?** Dos brazos más, con la misma
configuración y los mismos datos: la consolidación por uso sobre el control fijo (48 expertos desde el inicio, sin
división) y un control que congela al azar tantos expertos nuevos como el uso, en el mismo dominio y capa.
Holm entre las 11 preguntas:

| Comparación (3 pares) | Olvido medio | Diferencia | p (Holm) |
|---|---|---|---|
| consolidación por uso sin división contra control | 0,505 | +0,224 [+0,216; +0,234] | 0,0005 (0,006), mejor |
| consolidación al azar contra control | 0,615 | +0,114 [+0,096; +0,142] | 0,014 (0,12) |
| por uso contra al azar (sin división) | | +0,110 [+0,075; +0,129] | 0,025 (0,17), las 3 semillas a favor del uso |
| con división contra sin división (por uso) | 0,561 contra 0,505 | −0,056 [−0,089; −0,038] | 0,08 (0,40), las 3 semillas en contra de dividir |

Lectura:
- El mejor sistema plano es consolidar por uso sobre un MoE fijo: olvida 31 % menos que el control, el resultado
  más fuerte de la fase 2. Aprende cada idioma 0,022 nats peor (p 0,011); la pérdida general mejora (2,277 contra
  2,299).
- Dividir no le suma nada a la consolidación: con división olvida más en las tres semillas.
- Congelar al azar ya reduce el olvido 16 %: parte del efecto es reservar capacidad. Elegir por uso suma otro
  15 % sobre el azar, en las tres semillas; con 3 pares no sobrevive a Holm.
- Sin división, el ruteo viejo queda en los congelados: la parte de los tokens del finés que va a expertos
  congelados pasa de 62 % a 63 % (semilla 0).

**Etapa C (ticket 14, 2026-10-23 a 2026-10-25): ¿un árbol de expertos suma sobre la consolidación plana?** El árbol
nace de los 32 expertos de Granite; cada mitosis deja una hija vieja congelada y una hija nueva de bajo rango (rango
16), y una compuerta reparte los tokens entre las dos según la dirección de la duda. Todos los brazos consolidan por
uso. Controles: el mismo árbol con la hoja elegida al azar, un árbol de un solo nivel (las hijas no se dividen) y un
MoE plano que crece con expertos de bajo rango (los mismos parámetros nuevos que el árbol).

| Comparación | Olvido medio | Diferencia (3 pares) | p (Holm) |
|---|---|---|---|
| árbol contra consolidación plana (H7) | 0,506 contra 0,505 | −0,001 [−0,017; +0,011] | 0,94 (1,0), empate; aprende 0,024 peor |
| árbol contra plano de bajo rango (H7, V1) | 0,506 contra 0,524 | +0,018 [−0,010; +0,044] | 0,36 (1,0); aprende 0,013 mejor (p 0,036) |
| árbol libre contra árbol de un nivel (H10) | 0,506 contra 0,540 | +0,035 [+0,010; +0,082] | 0,28, las 3 semillas a favor de la recursión |
| duda contra azar para abrir rama (H9) | 0,495 contra 0,528 | +0,033 | solo semilla 0 (saldo) |

Forma y ruteo:
- El árbol casi no deja expertos muertos (4 % contra ~27 % del plano) y sus hojas se especializan 2,5 veces más por
  idioma.
- El árbol protege peor el ruteo viejo: los tokens del finés que van a expertos congelados bajan ~28 % (el plano,
  ~1 %). El balanceo de las compuertas reparte tokens viejos a las hijas nuevas.
- H8 refutada: la autosimilitud del ruteo del árbol es 0,025 y la del plano 0,052, en las 3 semillas.

H11, territorios (Granite capas 6, 12 y 18; OLMoE capas 4, 8 y 12; 12 dominios, 32 secuencias cada uno). La figura
de un territorio se mide con la dimensión de correlación, la extensión y el perfil de escalas, y se compara con
dos nulos: una nube gaussiana y una partición al azar de los mismos tokens en la misma cantidad de grupos.
- Los territorios tienen dimensión baja: ~13 en Granite y ~20 en OLMoE, la mitad del nulo gaussiano.
- Los territorios no se distinguen de una partición al azar: la dimensión difiere en menos de 2 y la extensión en
  menos de 3 %, en los niveles 2, 4, 8 y en los expertos solos.
- El perfil de escalas no se repite de un nivel a otro de forma consistente.
- H11 refutada: la variedad de baja dimensión es de los estados ocultos, no de los expertos; un experto es una
  rebanada de esa variedad, sin una figura propia.

Lectura (tabla de lectura del spec, fila "no, no"):
- La idea fractal pierde su base como regla de diseño: no hay autosimilitud en los datos (H11) y el árbol no le gana
  a la consolidación plana (H7).
- Dentro del árbol, la recursión ayuda un poco (H10, 3 de 3 semillas, sin significancia): si se usa un árbol, conviene
  dejarlo crecer.
- El sistema que queda es la consolidación por uso sobre un MoE fijo (−31 %, Holm 0,006). El árbol aporta
  especialización y pocos expertos muertos, pero cuesta protección del ruteo.
- Límites: 3 semillas, un solo orden de idiomas, H9 con una semilla, y solo se entrenan las capas 12 a 23 (la
  atención, con el 69 % del olvido del diagnóstico, queda para la etapa B).

### 4.13 Bloques elementales: ¿los expertos tienen piezas propias que se reusan? (CPU y L4, 2026-10-26 a 2026-11-03)

Pregunta: ¿un experto se distingue por su forma, y el conocimiento nuevo crece desde las piezas que ya existen?
Código: repo `mf-moe` (`mf_moe/blocks.py`, `colab/b9_bits.py`, `colab/b12_raices.py`, `colab/b12c_distincion.py`).
Decisiones: ticket 15 del mapa `.scratch/mf-moe-olvido`.

**Modelos grandes (Granite-3.1-1B-A400M; EMO y StdMoE de AllenAI como control positivo).**

| Prueba | Medida | Resultado |
|---|---|---|
| B1, pesos de neuronas (Granite) | autosimilitud, copias entre expertos, identidad de experto | sin árbol ni copias; la vecina está en el mismo experto 26-46 % (azar 3 %) |
| B1b, función de neuronas (Granite) | las mismas medidas sobre perfiles de activación | sin árbol ni copias; identidad más débil (9-22 %); pesos y función no correlacionan |
| B2, direcciones jerárquicas del router (Granite) | recall del top-8 con grupos de expertos | estructura sobre el azar (1,3-2,3 veces), pero ahorro casi nulo |
| Árbol o nube (Granite, EMO, StdMoE) | condición de 4 puntos de Gromov contra una gaussiana | pesos: nube (0,86-0,94); estados: algo arbóreos (0,57-0,83), lejos de un árbol (0,05-0,25) |
| Control EMO contra StdMoE | B1, B1b, B2 y árbol sobre dos MoE con los mismos datos | ningún medidor distingue al MoE modular; solo el uso de expertos por documento lo distingue |
| Uso por documento | expertos efectivos por documento sobre un nulo | EMO 0,56, StdMoE 0,72, Granite 0,88 (menor = más modular) |

Lectura: los medidores de forma (B1, B1b, B2 y árbol) son ciegos a la modularidad conocida de EMO. El medidor que sí
la ve, el uso por documento, ubica a Granite como el menos modular de los tres.

**MoE chico desde cero sobre tareas de bits con primitivas conocidas (16 expertos, top-2, d = 64, 4 capas).**
Cinco primitivas (not, xor, rot, par, andor) y sus pares. La etiqueta es computable, así que la especialización se
mide contra la verdad.

- B9b (3 semillas): la información mutua experto × primitiva llega a 0,19-0,30, contra 0,004-0,14 sin entrenar.
  Hay especialización parcial, no un experto por primitiva.
- B9b: aprender primitivas y después pares cuesta 1,4 a 4 veces menos pasos que aprender pares desde cero. Sin
  repetición, las primitivas caen a 0-1 %.
- B9b: los medidores de forma no ven la recursión aprendida en tareas recursivas. El control positivo falla.
- B10 (3 semillas): el orden del currículo no cambia la velocidad con un signo constante. Ningún orden compone
  pares no vistos (0-2,8 %). Esto replica el caso directo de Ramesh, Lubana et al. (ICML 2024).
- B12 y B12b (3 semillas): un experto hijo por par nace de la raíz (el experto de la primitiva), de un promedio, de
  padres ajenos o al azar. El origen de los pesos no cambia la velocidad con un signo constante. En B12 el "padre"
  era un experto genérico, compartido por todas las primitivas.

**B12c: dial de distinción (5 semillas, L4).** Hipótesis del usuario: las primitivas comparten el vocabulario 0/1 y
por eso no hay expertos propios. El dial p hace que cada primitiva escriba una fracción p de las posiciones con su
propio alfabeto. Con p = 1, par no aprende: falla en las 5 semillas de 20.000 pasos y en 4 de 5 diagnósticos de
12.000 pasos. El análisis usa p de 0 a 0,75.
IC 90 % bootstrap sobre semillas en la Fase A; IC 95 % en la Fase B.

| Medida | p = 0 | p = 0,75 | Pendiente o diferencia |
|---|---|---|---|
| IM experto × primitiva, capa 1, base | 0,106 | 0,360 | — |
| IM, capa 1, router congelado al azar | 0,117 | 0,192 | — |
| IM aprendida (base − azar), capa 1 | −0,011 | +0,168 | +0,178 [+0,101; +0,266] |
| IM aprendida, capa 2 | +0,009 | +0,099 | +0,090 [+0,061; +0,124] |
| IM aprendida condicionada a 0/1, capas 1 y 2 | ≈ 0 | +0,06 y +0,07 | pendiente con IC > 0 en las dos |
| AUC de pares en 1000 pasos, raíz − azar (Fase B) | +0,008 [−0,020; +0,029] | +0,021 [−0,000; +0,040] | dif. +0,013 [−0,022; +0,049] |
| AUC, raíz − padres ajenos (Fase B) | −0,021 [−0,080; +0,042] | +0,022 [+0,007; +0,036] | dif. +0,043 [−0,024; +0,094] |

Controles: el router congelado al azar entrena los mismos pasos que su base. Una base de 20.000 pasos sin corte
no crea IM aprendida con p = 0, así que el largo del entrenamiento no explica el efecto. Evaluar en bf16 o en fp32
da lo mismo.

Lectura:
- Con el mismo vocabulario, el router aprendido no separa primitivas mejor que un router al azar.
- La especialización aprendida crece con la distinción entre tareas (H-a se cumple).
- Una parte grande de la información mutua es mecánica: el router al azar da el 53-85 % con p alto. Toda medida
  de especialización necesita ese control.
- Heredar los pesos de un experto especializado casi no acelera al hijo: +2 puntos de exactitud media, con un IC
  que toca 0 contra azar (H-b no se cumple).
- Consecuencia para el MoE fractal: dividir un experto desde su padre no aporta una ventaja útil en este modelo.

**Novedad (auditoría con Lacuna, 2026-11-02).** El router como sensor de duda ya está publicado: "Routing Entropy:
A Hidden Self-Verifier for Free" (COLM 2026) reporta AUROC 0,655 con la entropía de ruteo sola en OLMoE. Este
informe lo cita como precedente. Vigía tiene que mostrar algo que ese trabajo no tiene. Los candidatos medidos acá
son la evaluación por token (§4.1), la sonda sobre el estado (§4.1) y la latencia en producción (§4.10). El nacimiento de un experto desde las raíces de sus partes no aparece en la literatura revisada, y
B12c lo responde con un efecto chico.

Límites: modelo de 16 expertos y tareas sintéticas; 5 semillas dan IC anchos; el ruteo forzado mide el origen de
los pesos, no si el router encuentra al hijo por sí solo.

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
4. La latencia en GPU cumple la meta con HF y vLLM, también con 32 requests simultáneos (§4.10).
5. El router como sensor de duda tiene precedente publicado (Routing Entropy, COLM 2026; §4.13).
