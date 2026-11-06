# Mapa: ¿lo medido en Granite y WordNet vale en otros modelos y datos?

Label: wayfinder:map · Creado: 2026-11-06

## Destination

Saber qué resultados del banco de conflicto dependen del modelo, de la ontología o del tamaño, antes de escribirlos en el
paper. Los resultados son: el gradiente de daño por jerarquía, factores planos contra árbol aprendido, y contención
contra consolidación. La salida es una tabla medida con una fila por resultado y una columna por modelo o banco, o el
registro de que un resultado solo vale en Granite con WordNet.

## Notes

- Viene de [Bloques elementales](../mf-moe-olvido/issues/15-bloques-elementales.md), B20: los IC ya son angostos, y falta variedad, no más conceptos.
- Pedido del usuario (2026-11-06): ampliar datasets, usar modelos con datos abiertos, transformarlos en las arquitecturas a medir.
- Corridas en Colab con `colab/cli/paralelo.sh` de mf-moe. La cuenta acepta 2 sesiones L4 a la vez.
- Regla de gasto del usuario: cada corrida responde por escrito qué dice la literatura y qué decisión cambia; control positivo claro antes de confirmar.
- Saldo de la cuenta: ~587 unidades (2026-11-06). El tope de 550 del otro mapa es de ese esfuerzo; falta confirmar si este esfuerzo tiene tope propio.
- Research al vault global, nunca al repo. Los tickets grilling los resuelve el usuario.
- Cada resultado nuevo se corrige por comparaciones múltiples (Holm) dentro de su ticket.

## Decisions so far

- Se abre el mapa y se empieza por contención contra consolidación en Granite (usuario, 2026-11-06).
- Orden propuesto: primero Granite, después otro modelo, otro banco, tamaños y conversión. La conversión queda última porque suma una variable.

## Not yet specified

- [B21: contención contra consolidación en Granite](issues/01-b21-contencion-consolidacion.md) — resuelto: la contención por tema no supera al azar; la consolidación por uso se sostiene.
- [B20 en OLMoE-1B-7B](issues/02-b20-olmoe.md) — ¿los mapas dependen del modelo? Falta medir si entra en una L4.
- [Banco con UniEdit](issues/03-banco-uniedit.md) — ¿los mapas dependen de WordNet? Falta licencia y conversión a hechos en conflicto.
- [Tamaños con FLAME-MoE](issues/04-tamanos-flame-moe.md) — ¿el tamaño cambia el resultado? Incluye 38M/100M, el tamaño del north-star.
- [Frecuencia en el corpus abierto](issues/05-frecuencia-corpus.md) — ¿lo visto en el entrenamiento explica el nivel por concepto?
- [Denso convertido a MoE](issues/06-denso-a-moe.md) — ¿una conversión sin reentrenar se comporta igual?
- [Qué entra al paper](issues/07-que-entra-al-paper.md) — grilling: lo decide el usuario.

## Out of scope

- Entrenar desde cero modelos de más de 100 M de parámetros: el piloto midió ~14,5 unidades por corrida a 80-100 M.
- Reentrenar conversiones con cientos de miles de millones de tokens (ExpertWeaver CPT).
- Cambios al producto Vigía.
