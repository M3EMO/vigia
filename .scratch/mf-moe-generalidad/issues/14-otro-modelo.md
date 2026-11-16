# B35 y B36: la puerta y el umbral en otro modelo

Type: task
Status: B35 corrido (2026-11-16): OLMo-2 pasa G1, G2, U1 y P1; B36 sin datos (ninguna H100 en 4 h)
Blocked by: 12

Pedido del usuario (2026-11-16): «Hace todo eso en colab». Este ticket cubre otro modelo y reemplaza la parte de tamaño del ticket 04.

Pregunta: ¿la puerta de certeza (B29) y el umbral de fuentes (B31) funcionan fuera de Granite 1B?

## Modelos

- B35: OLMo-2-0425-1B. Es denso, de otra familia, con datos de preentrenamiento abiertos. Corre en una L4.
- B36: Granite-3.1-3B-A800M. Es la misma familia con el triple de parámetros. Corre en una H100, porque sus capas entrenables no entran en 24 GB.
- FLAME-MoE no entra: el ticket 04 explica por qué.

## Diseño (`colab/b35_modelo.py`)

Una VM por modelo hace tres etapas con el mismo modelo cargado:
1. Banco: el filtro de B29 con este modelo. Son los mismos 1.200 hechos de UniEdit, filtrados por la ida contra los donantes.
2. Puerta: el código de B29, con los brazos oráculo, azar y sonda en la dosis 10.
3. Umbral: el código de B31, con los brazos una|peso, una|umbral y fuentes|umbral, en las dosis 10 y 40.

- Se entrena el mismo cuarto final de capas que en Granite 1B: desde la capa 6 de 24.
- La sonda usa las capas de B27c a la misma profundidad relativa. En OLMo-2 son las capas 5, 8, 11 y 13.
- Los conjuntos, los lotes y las semillas usan el tag «b29».
- El código es el de B29 y B31 sin cambios. Solo cambia la carga del modelo.

## Reglas (por modelo)

- B0 (banco): al menos 160 hechos pasan el filtro. Si no, el modelo no sabe bastante y la réplica se corta.
- H0 (frases H): las frases H sirven si el valor viejo gana contra los donantes en 60 % o más, como en B30.
- G1 (puerta, oráculo): la diferencia pareada de N2 contra el azar tiene IC90 bajo 0 y media de −0,15 o menos, en la dosis 10.
- G2 (puerta, sonda): la misma regla para la sonda.
- C0 (control positivo): en una|peso, el tercio dudoso adopta en 80 % o más en la dosis 10.
- U1 (umbral): pasa si V1, V2 y V3 pasan, con los cortes de B30:
  - V1: con una|umbral, el tercio seguro resiste en 50 % o más en la dosis 10;
  - V2: con fuentes|umbral, el tercio seguro adopta en 80 % o más en la dosis 40;
  - V3: con una|umbral, el tercio dudoso adopta en 80 % o más en la dosis 10.
- P1 (umbral contra pesos): la diferencia pareada de resistencia del seguro, una|umbral menos una|peso, tiene IC90 sobre 0 y media de 0,30 o más.

## Decisiones

- G1 y U1 pasan en OLMo-2: §4.15 vale en otra familia y en un modelo denso. El ítem 6 de §7 se achica.
- G1 y U1 pasan en Granite 3B: §4.15 vale con el triple de tamaño dentro de la familia.
- G1 falla en un modelo: la puerta depende del modelo. §4.15 nombra el modelo en cada afirmación.
- U1 falla y C0 pasa: el umbral no reproduce la conducta en ese modelo.
- B0 falla: el modelo no conoce bastantes hechos de UniEdit, y la réplica no mide nada.

## Predicciones (Brier al cierre)

- OLMo-2, G1 pasa: 0,70.
- OLMo-2, U1 pasa: 0,75.
- Granite 3B, G1 pasa: 0,75.
- Granite 3B, U1 pasa: 0,80.

## Costo y tope

- B35: unos 35 minutos de L4, 1,1 u. Tope por timer: 60 minutos.
- B36: unos 40 minutos de H100. La tasa de la H100 se lee con `colab usage` al arrancar. Tope por timer: 60 minutos.

## Puntos débiles conocidos

- El banco de cada modelo es distinto, porque cada modelo conoce hechos distintos. La comparación entre modelos usa la diferencia pareada dentro de cada banco.
- La sonda se entrena con los candidatos de B25, elegidos por Granite. Para OLMo-2 esos candidatos son hechos de UniEdit sin filtro.
- Granite 3B no cubre la pregunta original del ticket 04 sobre el gradiente y los mapas de §4.13.

## Resultados de B35, OLMo-2-0425-1B (2026-11-16, 18 minutos de L4, ~0,6 u)

Análisis: `.cache/runs/bloques/b35_olmo_gpu/analisis.out` en mf-moe.
- B0 pasa: 320 de 687 hechos ganan la ida, y el banco toma 256.
- H0: las frases H sirven (H1 0,816 y H2 0,832).
- El azar da N2 −0,023 [−0,117; +0,066]. OLMo-2 tampoco trae el nivel 2 de fábrica.
- G1 pasa: el oráculo contra el azar da −0,333 [−0,467; −0,199].
- G2 pasa: la sonda contra el azar da −0,253 [−0,329; −0,164]. La sonda tiene Spearman 0,612 y logra 0,76 del oráculo.
- C0 pasa: con una voz y pesos, el dudoso adopta en 100 % en la dosis 10.
- U1 pasa: V1 1,00, V2 1,00 y V3 1,00.
- P1 pasa: la resistencia del seguro sube +0,857 [+0,772; +0,936] contra los pesos.
- Con pesos, el seguro resiste una voz solo en 14 %, como en Granite.
- El daño al resto, en la dosis 10, es 1,31 con pesos, 1,36 con umbral y 1,93 con fuentes y umbral.
- Decisión según el prerregistro: §4.15 vale en otra familia y en un modelo denso. El ítem 6 de §7 se achica.

Brier de B35: G1 0,090 y U1 0,063.

## B36, Granite-3.1-3B-A800M (2026-11-16): sin datos

- `colab/cli/esperar_gpu.sh` pidió una H100 cada 5 minutos durante 4 horas, desde la 01:20.
- Colab respondió «Service Unavailable» en cada intento. El registro cierra con «sin H100 en 4 h», a las 05:20.
- La espera no gastó unidades.
- Una A100 de 40 GB no alcanza: los pesos en fp32 más Adam para las capas entrenables suman unos 43 GB.
- El ítem 6 de §7 sigue con «Falta otro tamaño».
