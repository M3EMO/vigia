# Informe técnico: señales del router MoE para detectar alucinaciones por token

Fecha: 2026-10-04. Estado: borrador con datos medidos. Procedimiento reproducible con
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

La latencia agregada en CPU queda entre +3,6 % y +8,5 % con un intervalo ancho (§5).

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
2. El modelo es chico y se equivoca en el 84 % de los casos; falta repetir en gpt-oss-20b con GPU.
3. La sonda sobre H iguala a F1; falta medir si F1 transfiere mejor entre dominios o modelos.
4. La latencia necesita una medición en GPU con más corridas; en CPU el ruido domina.
