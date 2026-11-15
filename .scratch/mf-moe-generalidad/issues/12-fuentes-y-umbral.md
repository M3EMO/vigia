# B30: ¿una voz contra fuentes independientes, con pesos o con umbral, reproduce la conducta del video?

Type: task
Status: el umbral reproduce el video y la puerta de pesos no (2026-11-14); integridad 1 falla por ruido de medida; B31 repite el umbral en el banco de B29; B32 lo repite con la sonda (2026-11-15)
Blocked by: 09, 10

## Question

La meta del video: un hecho seguro rechaza una voz y acepta muchas fuentes independientes. Un hecho dudoso cambia con una voz.
En B27b, con pesos desde 0,02, el 85 % de los hechos seguros adoptó lo nuevo con una voz. Esa dosis repetía la misma oración.
El usuario eligió probar fuentes independientes y un umbral de fuentes (respuestas 3 y 4 del ticket 10, 2026-11-14).
Tope: 2 unidades, compartido con el ticket 11. B29 gastó cerca de 0,32 u.

## Antecedentes (Lacuna, 2026-11-14)

- [Physics of Language Models 3.1](https://lacuna.tiptreesystems.com/work/physics-of-language-models-part-3-1-knowledge/version/art_ebc69237f6cb4ea9a55b04dfbfb627a7) (ICML 2024): un hecho se extrae bien solo si se entrenó con paráfrasis.
- [Injecting New Knowledge into LLMs via SFT](https://lacuna.tiptreesystems.com/work/injecting-new-knowledge-into-large-language-models/version/art_703560ef895d4f9e8c87b8d471e8d11b) (2024): el ajuste con paráfrasis de hechos generaliza mejor que el ajuste con una sola forma.
- Consecuencia para el diseño: la creencia se mide en frases que el ajuste no ve. Una voz repetida puede cambiar solo su frase.
- No encontré un trabajo que module la plasticidad de un hecho por su certeza y por el número de fuentes distintas.

## Prerregistro B30 (2026-11-14, antes de ver datos)

Diseño (`python -m colab.b30_fuentes`):

- Banco, conjuntos (tag «b25»), semillas, lotes y pesos fuertes de B27b: w = 0,02 + 1,96 · rango(duda) / (n − 1).
- Fuente «una»: cada hecho se entrena con una sola frase, la de B27 (T0).
- Fuente «fuentes»: cada exposición usa una de 10 frases distintas. La tasa de exposición por hecho es la misma.
- Puerta «peso»: el peso del hecho, siempre.
- Puerta «umbral»: el peso solo después de que el hecho vio tau frases distintas. Antes, el peso es 0.
- tau es 1 en la mitad dudosa del banco. En la mitad segura, tau sube con la certeza hasta 8.
- Cuatro brazos: una|peso, fuentes|peso, una|umbral, fuentes|umbral. Dosis 10 y 40.
- El brazo una|peso repite el oraculo_fuerte de B27b paso por paso. Es la integridad.
- Medida: el margen de cada hecho sobrescrito es la pérdida del valor nuevo menos la del viejo. Positivo es que resiste.
- El margen se mide en T0 y en dos frases que ningún brazo entrena: H1 «{h}'s {p} is» y H2 «Q: What is the {p} of {h}? A:».
- «Resiste en H» es el margen medio de H1 y H2 mayor que 0. Los tercios de certeza salen de la pérdida base del banco.

Reglas:

- Integridad 1: la subida de la ida de una|peso en la dosis 10 repite la de B27b con error medio < 0,01.
- Integridad 2: una frase H entra al análisis solo si la base gana contra los donantes en ≥ 60 % de los hechos.
- Si ninguna frase H entra, el resultado es «sin medida».
- C0, control positivo: en fuentes|peso, dosis 40, el tercio dudoso adopta lo nuevo en H en ≥ 80 %. Si falla, «sin medida».
- V1, una voz: con «una» en la dosis 10, el tercio seguro resiste en H en ≥ 50 %.
- V2, muchas fuentes: con «fuentes» en la dosis 40, el tercio seguro adopta en H en ≥ 80 %.
- V3, lo dudoso cambia con una voz: con «una» en la dosis 10, el tercio dudoso adopta en H en ≥ 80 %.
- Una puerta reproduce el video si V1, V2 y V3 pasan con esa puerta.
- Informativo: los mismos tercios en T0, la subida del resto del banco y la dosis en que se abre cada hecho seguro.

Qué decide cada resultado:

- La puerta «peso» reproduce el video: la conducta sale de la diversidad de fuentes, sin umbral. El paper lo suma.
- Solo la puerta «umbral» reproduce el video: hace falta contar fuentes. El paper lo presenta como mecanismo construido, con su costo.
- Ninguna reproduce el video: con este ajuste, la puerta solo modula el olvido. El paper lo dice.
- C0 falla: las frases H no miden la creencia nueva. El resultado es «sin medida».

Predicciones: integridad 1 0,95. C0 0,8. V1 con peso 0,35, con umbral 0,9. V2 con peso 0,6, con umbral 0,4. V3 0,5.
Video con peso 0,15. Video con umbral 0,2.
Costo: cerca de 0,45 u. Timer de 20 minutos, cerca de 0,63 u. Con B29, el peor caso suma cerca de 0,95 u del tope.

## Desvío durante la corrida (2026-11-14, 22:24)

- B30 corre más lento que lo estimado: 8 de 32 corridas en 10 minutos.
- Con el timer de 20 minutos, la corrida cortaba en la mitad de los conjuntos.
- Extendí el timer a 38 minutos, cerca de 1,2 u. Con B29, el peor caso suma cerca de 1,5 u, dentro del tope de 2.
- Las reglas y el análisis no cambian.

## Resultado B30 (2026-11-14, sesión b30fue, 22 minutos, cerca de 0,72 u)

Integridad 2: las dos frases H entran. La base gana en T0 0,903, en H1 0,845 y en H2 0,913.

Integridad 1 falla por la regla: una|peso difiere de B27b en 0,044 de error medio, contra un límite de 0,01.

- La pérdida base coincide exacta, la correlación es 0,998 y el sesgo es cerca de 0.
- El error no crece con la dosis: 0,044 en la dosis 10 y 0,046 en la 40. Si el ajuste divergiera, crecería.
- B27 repitió el ajuste de B25 entre VMs con error 0,0. El ajuste es determinista.
- Diagnóstico: B27b mide cada valor real en lotes con sus donantes, y B30 en lotes sin ellos. En bf16, eso cambia la pérdida cerca de 1 %.
- El error del prerregistro fue mío: la regla comparaba dos medidas con lotes distintos.
- Las reglas del video miden tasas que quedan cerca de 0 o de 1. Un ruido de 0,04 nats no las cambia.

Fracción de hechos sobrescritos que resisten en H (margen medio de H1 y H2 mayor que 0):

| Brazo | Dosis | Seguro | Medio | Dudoso |
|---|---|---|---|---|
| una, peso | 10 | 0,194 | 0,000 | 0,000 |
| una, peso | 40 | 0,000 | 0,000 | 0,000 |
| fuentes, peso | 10 | 0,224 | 0,000 | 0,000 |
| fuentes, peso | 40 | 0,000 | 0,000 | 0,000 |
| una, umbral | 10 | 1,000 | 0,373 | 0,000 |
| una, umbral | 40 | 0,985 | 0,373 | 0,000 |
| fuentes, umbral | 10 | 0,716 | 0,030 | 0,000 |
| fuentes, umbral | 40 | 0,015 | 0,000 | 0,000 |

- C0 pasa: en fuentes|peso, dosis 40, el tercio dudoso adopta lo nuevo en H en 100 %.
- Puerta «peso»: V1 falla (resiste 19 %), V2 pasa y V3 pasa. No reproduce el video.
- Puerta «umbral»: V1, V2 y V3 pasan. Reproduce el video.
- Según el prerregistro: hace falta contar fuentes. El paper presenta el umbral como mecanismo construido, con su costo.
- Brier: integridad 1 0,9025. C0 0,04. V1 peso 0,1225, umbral 0,01. V2 peso 0,16, umbral 0,36. V3 0,25 y 0,25.
- Brier del video: peso 0,0225, umbral 0,64.

Lectura:

- V1 con umbral pasa por construcción: un hecho seguro con una sola frase tiene peso 0.
- La parte no trivial es V2: después de abrirse, el hecho seguro todavía adopta lo nuevo en la dosis 40, en 98,5 %.
- Con una voz y umbral, el hecho seguro resiste aun en la dosis 40. Cerca de 26 repeticiones de la misma voz no lo convencen.
- Corrección (2026-11-15): decía «cien repeticiones». En la dosis 40 hay 640 ejemplos para 25 hechos, cerca de 26 por hecho.
- Las frases H siguen a T0 casi igual en todos los brazos. Una voz repetida cambia la creencia en frases nuevas.
- Eso contradice lo que esperaba de la literatura de paráfrasis. Granite ya trae la capacidad de extraer el hecho en otras frases.
- La diversidad sola no protege: fuentes|peso resiste 22 % en la dosis 10, casi como una|peso.

Costo del umbral, la subida media del resto del banco:

| Brazo | Dosis 10 | Dosis 40 |
|---|---|---|
| una, peso | 1,873 | 3,017 |
| una, umbral | 2,304 | 2,948 |
| fuentes, peso | 2,576 | 2,864 |
| fuentes, umbral | 2,615 | 3,109 |

- Con una voz, el umbral sube el daño al resto en la dosis 10: 2,30 contra 1,87.
- Lectura tentativa: con la mitad de los hechos en peso 0, AdamW concentra el paso en menos hechos.
- Análisis: el scratchpad de la sesión, `b30_analisis.py`. Salida en `b30_fuentes_gpu/analisis.out`.
- Gasto de los tickets 11 y 12: cerca de 1,04 u del tope de 2.
- Límite: un modelo, un banco, frases plantilla y fuentes que no son independientes en contenido, solo en forma.

## Prerregistro B31: el umbral en el banco de B29 (2026-11-15, antes de ver datos)

El usuario eligió esta réplica el 2026-11-15. B30 midió el umbral en un solo banco, el de B27.
La pregunta es si el umbral reproduce el video también en otros hechos.

Antecedente (Lacuna, 2026-11-15):

- [ClashEval](https://lacuna.tiptreesystems.com/work/clasheval-quantifying-the-tug-of-war/version/art_6e153639c2aa4cfea6b067e94869740f) (NeurIPS 2024) mide el conflicto entre el saber previo y la evidencia en el contexto.
- Ahí, un modelo con un saber previo más seguro adopta menos la evidencia del contexto.
- Ese trabajo mide el contexto, no los pesos. No encontré una réplica en los pesos.

Diseño (`python -m colab.b30_fuentes SALIDA colab/data/b29_banco.json b29 una:peso,una:umbral,fuentes:umbral`):

- Banco de B29: 256 hechos, filtrados solo por la ida. Conjuntos, lotes y semillas de B29 (tag «b29»), con 32 hechos por conjunto.
- Pesos fuertes, tau y frases de B30, sin cambios.
- Tres brazos: una|peso, una|umbral y fuentes|umbral. Dosis 10 y 40.
- una|peso es la línea de base del mismo banco, según la regla pareada del ticket 11.
- fuentes|peso no corre. El control positivo pasa al brazo fuentes|umbral, donde la mitad dudosa abre con tau 1.

Reglas:

- I0: la pérdida base de la ida repite la de B29 con error medio < 0,01. Las dos corridas miden con los mismos lotes.
- I2: una frase H entra solo si la base gana contra los donantes en ≥ 60 % de los hechos. Sin frases H, el resultado es «sin medida».
- C0: en fuentes|umbral, dosis 40, el tercio dudoso adopta lo nuevo en H en ≥ 80 %. Si falla, el resultado es «sin medida».
- R1: V1, V2 y V3 de B30 pasan con el umbral en este banco.
- P1, pareada: en la dosis 10, la fracción del tercio seguro que resiste en H sube con el umbral contra una|peso.
- P1 pasa si la diferencia por conjunto tiene IC90 sobre 0 y media ≥ 0,5.
- K1, pareada: en la dosis 10, la subida del resto del banco es mayor con una|umbral que con una|peso. Pasa con IC90 sobre 0.
- Informativo: los tercios en T0, la dosis 40 y la fracción de seguros abiertos en fuentes|umbral.

Qué decide cada resultado:

- R1 y P1 pasan: el umbral reproduce el video en dos bancos. El paper lo afirma fuera del banco de B27.
- R1 o P1 fallan: el resultado del umbral queda limitado al banco de B27, y el paper lo dice.
- K1 pasa: el costo del umbral se repite, y el paper lo presenta junto al mecanismo.
- K1 falla: el paper presenta el costo como propio del banco de B27.

Predicciones: I0 0,9. I2 0,85. C0 0,85. R1 0,6. P1 0,75. K1 0,6.
Costo: cerca de 20 minutos, cerca de 0,65 u. Timer de 28 minutos, cerca de 0,89 u. El peor caso suma cerca de 1,93 u del tope de 2.
Análisis: el scratchpad de la sesión, `b31_analisis.py`, escrito y probado sobre los datos de B30 antes de lanzar.

## Resultado B31 (2026-11-15, sesión b31umb, 15 minutos, cerca de 0,48 u)

Integridad: I0 pasa, con error 0,0 contra B29. I2 pasa: la base gana en T0 0,867, en H1 0,809 y en H2 0,840.

Fracción de hechos sobrescritos que resisten en H:

| Brazo | Dosis | Seguro | Medio | Dudoso |
|---|---|---|---|---|
| una, peso | 10 | 0,198 | 0,000 | 0,000 |
| una, peso | 40 | 0,000 | 0,000 | 0,000 |
| una, umbral | 10 | 0,953 | 0,353 | 0,000 |
| una, umbral | 40 | 0,977 | 0,376 | 0,000 |
| fuentes, umbral | 10 | 0,709 | 0,082 | 0,012 |
| fuentes, umbral | 40 | 0,023 | 0,000 | 0,000 |

- C0 pasa: en fuentes|umbral, dosis 40, el tercio dudoso adopta lo nuevo en H en 100 %.
- R1 pasa: V1 (resiste 95 %), V2 (adopta 97,7 %) y V3 (adopta 100 %).
- P1 pasa: el umbral sube la resistencia del tercio seguro en +0,753 [+0,680; +0,828] contra una|peso.
- K1 falla: la subida del resto crece +0,148 [−0,211; +0,497] con el umbral, y el IC90 cruza 0.
- Según el prerregistro: el umbral reproduce el video en dos bancos, y el paper lo afirma fuera del banco de B27.
- Según el prerregistro: el costo del umbral queda como propio del banco de B27.
- Brier: I0 0,01. I2 0,0225. C0 0,0225. R1 0,16. P1 0,0625. K1 0,36.

Lectura:

- Sin umbral, el banco de B29 se comporta igual que el de B27: el seguro resiste 0,198 contra 0,194.
- El banco de B29 es más frágil: con una voz y pesos, el resto sube 2,48 en la dosis 10, contra 1,87 en B27.
- En un banco más frágil, el costo extra del umbral queda dentro del ruido entre conjuntos.
- V1 y P1 salen casi por construcción, porque un hecho seguro con una sola frase tiene peso 0.
- Lo no trivial se repite: después de abrirse, el hecho seguro todavía adopta lo nuevo en 97,7 %.
- Análisis: el scratchpad de la sesión, `b31_analisis.py`. Salida en `b31_umbral_gpu/analisis.out`.
- Gasto de los tickets 11 y 12: cerca de 1,52 u del tope de 2. Quedan cerca de 0,48 u.

## Prerregistro B32: el umbral con la sonda (2026-11-15, antes de ver datos)

El usuario pidió seguir con los pasos propuestos el 2026-11-15. B30 y B31 calculan tau y los pesos con la pérdida real.
Un sistema real no tiene esa pérdida. B32 usa la sonda de B27c, que lee el estado al final de la pregunta.

Antecedente (Lacuna, 2026-11-15):

- [Ignorance Is Shared, Knowledge Is Private](https://lacuna.tiptreesystems.com/work/ignorance-is-shared-knowledge-is-private-efficiently-mapping) (2026) mapea los límites del conocimiento de un modelo.
- [Probe-Guided Training for LLM Tool Calls](https://lacuna.tiptreesystems.com/work/getting-the-parameters-right-a-difficulty-graded-benchmark) (2026) usa una sonda para guiar el ajuste.
- No encontré un trabajo que use una sonda de certeza para fijar un umbral de fuentes.

Diseño (`python -m colab.b30_fuentes SALIDA colab/data/b27_banco.json b25 una:umbral,fuentes:umbral colab/data/b25_candidatos.json`):

- Banco, conjuntos, lotes y semillas de B30 (tag «b25»). Dos brazos: una|umbral y fuentes|umbral. Dosis 10 y 40.
- Pesos y tau salen de la predicción cruzada de la sonda. Los tercios de certeza salen de la pérdida real, como en B30.
- La línea de base pareada es una|peso de B30, con los mismos conjuntos y lotes. El ajuste es determinista entre VMs.

Reglas:

- I0: la pérdida base repite la de B30 con error medio < 0,01.
- I3: la predicción de la sonda repite la de B27c con error medio < 0,01.
- I2: las frases H de B30 entran también aquí, con el mismo criterio de ≥ 60 %.
- C0: en fuentes|umbral, dosis 40, el tercio dudoso adopta lo nuevo en H en ≥ 80 %. Si falla, el resultado es «sin medida».
- R1: V1, V2 y V3 de B30 pasan con la sonda.
- S1, pareada: en la dosis 10, la resistencia del tercio seguro sube contra una|peso de B30. IC90 sobre 0 y media ≥ 0,5.
- Informativo: S2, la fracción del efecto del oráculo, (sonda − peso) / (oráculo − peso), con el umbral de B30 como oráculo.

Qué decide cada resultado:

- R1 y S1 pasan: el umbral funciona con una señal sin etiqueta. El paper lo presenta como mecanismo usable.
- R1 o S1 fallan: el umbral necesita una señal mejor que la sonda. El paper lo presenta como prueba de concepto con el oráculo.
- C0 falla: el resultado es «sin medida».

Predicciones: I0 0,95. I3 0,95. I2 0,95. C0 0,85. V1 0,8. V2 0,85. V3 0,5. R1 0,35. S1 0,75.
Lectura de V3: con Spearman 0,616, cerca de un quinto de los dudosos reales cae en la mitad segura de la sonda. Esos hechos resisten.
Costo: cerca de 11 minutos, cerca de 0,35 u. Timer de 15 minutos, cerca de 0,475 u. El peor caso llega a cerca de 2,0 u del tope.
El runner baja los resultados cada 2 minutos, así que un corte del timer guarda los conjuntos terminados.
Análisis: `colab/b32_analisis.py` en mf-moe, escrito antes de lanzar.

## Resultado B32 (2026-11-15, sesión b32son, 10,5 minutos, cerca de 0,33 u)

Integridad: I0, I2 e I3 pasan. La base y la sonda repiten B30 y B27c con error 0,0. La sonda sigue a la pérdida real con Spearman 0,616.

Fracción de hechos sobrescritos que resisten en H, con tercios por la pérdida real:

| Brazo | Dosis | Seguro | Medio | Dudoso |
|---|---|---|---|---|
| una, umbral con sonda | 10 | 0,791 | 0,403 | 0,197 |
| una, umbral con sonda | 40 | 0,761 | 0,403 | 0,182 |
| fuentes, umbral con sonda | 10 | 0,478 | 0,164 | 0,030 |
| fuentes, umbral con sonda | 40 | 0,015 | 0,000 | 0,000 |

- C0 pasa: en fuentes|umbral, dosis 40, el tercio dudoso adopta lo nuevo en H en 100 %.
- R1 pasa: V1 (resiste 79 %), V2 (adopta 98,5 %) y V3 (adopta 80,3 %).
- V3 pasa por 0,003 sobre el mínimo de 0,80. Un cambio de un solo hecho lo daría vuelta.
- S1 pasa: la resistencia del seguro sube +0,588 [+0,508; +0,657] contra una|peso de B30.
- S2: la sonda logra 0,736 del efecto del oráculo.
- Según el prerregistro: el umbral funciona con una señal sin etiqueta. El paper lo presenta como mecanismo usable.
- Brier: I0 0,0025. I3 0,0025. I2 0,0025. C0 0,0225. V1 0,04. V2 0,0225. V3 0,25. R1 0,4225. S1 0,0625.

Lectura:

- El error de la sonda tiene un costo visible: 15 de 66 dudosos reales caen en la mitad segura de la sonda.
- Esos 15 hechos adoptan lo nuevo solo en 13 %. Los otros 51 dudosos adoptan en 100 %.
- El costo del umbral es que un dudoso mal leído queda trabado con el valor viejo.
- La subida del resto en la dosis 10 da 2,08, entre una|peso (1,87) y el umbral con oráculo (2,30).
- Análisis: `colab/b32_analisis.py`. Salida en `b32_sonda_gpu/analisis.out`.
- Gasto de los tickets 11 y 12: cerca de 1,85 u del tope de 2. Quedan cerca de 0,15 u.

## Comments
