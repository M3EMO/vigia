# Fase 2: ¿el MoE fractal funciona a escala, con un MoE preentrenado?

Type: grilling
Status: resolved
Blocked by: 11

## Question

La fase 1 (MoE chico desde cero, 12 tareas de dígitos) muestra que la fusión ingenua destruye lo recién
aprendido y que la división sola casi empata con el control. Objeción del usuario (2026-10-17): el resultado
puede depender de la escala y del preentrenamiento, porque en un MoE preentrenado los expertos ya están
especializados. ¿Qué modelo, qué datos, qué condiciones y qué presupuesto prueban esa objeción en una L4?
Ninguna corrida entra a Colab hasta cerrar este ticket.

Research: nota del vault «mf-moe fase 2: probar la escala con un MoE preentrenado en una L4» (2026-10-17).

## Propuesta

- Modelo: Granite-3.1-1B-A400M-base (1,3 B totales, 0,4 B activos, 32 expertos × top-8 por capa, 24 capas,
  Apache-2.0, HF nativo `granitemoe`). Es el único MoE preentrenado que entra en una L4 con ajuste completo
  de pesos: AdamW de 8 bits, unos 8 a 13 GB. OLMoE-1B-7B solo entra con LoRA, y LoRA cambia el método.
- Datos: 4 dominios de texto en fila (matemática, código, biomédico, legal), unos 20 M de tokens por dominio;
  repetición de 5 % con texto general (FineWeb-Edu), que hace de dominio 0. Licencias por verificar.
- Olvido: loss held-out de 1 M de tokens por dominio después de cada dominio; olvido = loss final menos loss
  justo después de aprender el dominio. Es un oráculo computable, sin juez.
- Condiciones, todas con los mismos parámetros finales y top-8 fijo:
  - A, control: crece al tamaño final antes de empezar (32 → 48 expertos por clonado), con repetición.
  - B, división sola: 32 → 48 expertos con el disparador por duda, con repetición.
  - C, división + fusión alineada por permutación, con repetición.
  - D, opcional: división + congelar el padre (protege lo viejo, como Lifelong-MoE).
- Semillas: 3 por condición, pareadas con el mismo orden de datos.
- Precondición: un piloto de 1 h mide tokens por hora, VRAM y si el control olvida de forma medible. Si el
  control casi no olvida, las condiciones empatan por techo: se cambian dominios o tasa de aprendizaje.

## Costo estimado (sin medir)

| Parte | Horas de L4 | Unidades |
|---|---|---|
| Piloto | unas 8 | unas 12 |
| A, B y C con 3 semillas (9 corridas de ~6 h) | unas 54 | unas 83 |
| Retiros de VM (+15 %) | unas 9 | unas 14 |
| D con 3 semillas (opcional) | unas 18 | unas 28 |
| Total sin D / con D | 71 / 89 | 109 / 137 |

Presupuesto: la fase 1 todavía necesita unas 330 unidades de las ~420 disponibles dentro del techo de 550;
quedan unas 90 para la fase 2. Hace falta subir el techo, recortar la fase 1 o achicar la fase 2.

## Objeciones abiertas

- Cambiar de banco (dígitos → texto) y de escala a la vez mezcla dos explicaciones posibles.
- Ninguna evidencia publicada muestra que dividir con clon + ruido mejore la retención; lo que funciona agrega
  expertos y protege los viejos (Lifelong-MoE, LEGO). Drop-Upcycling sugiere reinicializar parte del clon.
- El throughput (~15-20 M tokens/h) y el layout de los tensores de expertos de Granite no están medidos.

## Decisiones abiertas para el usuario

1. ¿Modelo: Granite-1B con ajuste completo, u OLMoE con LoRA?
2. ¿Presupuesto: subir el techo, recortar la fase 1 o achicar la fase 2?
3. ¿Entra la condición D (congelar el padre)?

## Answer

Resuelto con el usuario el 2026-10-17.

- Modelo: Granite-3.1-1B-A400M-base con ajuste completo de pesos (AdamW de 8 bits).
- Presupuesto: sale de recortar las preguntas 5 y 6 de la fase 1 (fijo, Lifelong-MoE y sin repetición, 15
  corridas, unas 114 unidades); esas corridas quedan reservadas para una VM inexistente y la cola las saltea.
- La condición D (división con el padre congelado) entra.
- Antes de Colab: código del adaptador de Granite (división, fusión alineada, congelado) con tests en CPU
  sobre un modelo diminuto armado desde la config; datos y licencias verificados; piloto de 1 h que mida
  tokens por hora, VRAM y olvido del control.

## Comments

- 2026-10-17, datos elegidos (abiertos y sin registro previo): matemática `open-web-math/open-web-math` (ODC-By 1.0); código `common-pile/stackv2` (licencias abiertas); legal `common-pile/caselaw_access_project` (CC0 o dominio público); biomédico un subconjunto de abstracts de PubMed con CC0 (p. ej. `slinusc/PubMedAbstractsSubset`); repetición general `HuggingFaceFW/fineweb-edu` (ODC-By). Descartados: The Stack (pide aceptar términos con cuenta) y Pile of Law (licencia poco clara). El piloto verifica que los cinco se lean en streaming desde la VM.
- 2026-10-17, adaptador de Granite hecho (mf-moe fe77d60): router con buffers `bias` en cero y `load` de uso, y registro de la señal; división, fusión alineada y AdamW sirven sin cambios. Falta: datos en streaming, bucle por tokens con loss por dominio, condiciones A a D, retome, y el humo en CPU.
- 2026-10-17 (usuario), memoria: se entrenan las 12 capas superiores (expertos, router y atención) en fp32 con AdamW común; las 12 inferiores y los embeddings quedan congelados en bf16. Las capas entrenables crecen de 32 a 48 expertos. Cuenta: ajuste completo con AdamW fp32 y 48 expertos pide ~30 GB; con AdamW de 8 bits a 48, ~22,5 GB; a 40, ~19 GB, pero dividir, fusionar y alinear exigen reordenar momentos cuantizados por bloques. Con 12 capas, ~14-17 GB y el código ya probado sirve sin cambios.
