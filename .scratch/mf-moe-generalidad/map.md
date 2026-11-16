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
- [B20 en OLMoE-1B-7B](issues/02-b20-olmoe.md) — cerrado en esta forma: con 1 o 4 capas, B22 no reproduce el banco de B20, y OLMoE no entra en L4 con más.
- [Banco con UniEdit](issues/03-banco-uniedit.md) — cerrado: UniEdit sirve; B23 midió el gradiente por tipo y relación fuera de WordNet.
- [Tamaños con FLAME-MoE](issues/04-tamanos-flame-moe.md) — bloqueado: FLAME-MoE no carga en transformers y sus pickles figuran como inseguros; el tamaño de §4.15 pasa al ticket 14.
- [Frecuencia en el corpus abierto](issues/05-frecuencia-corpus.md) — cerrado: la frecuencia no entra como covariable (proxy DCLM).
- [Denso convertido a MoE](issues/06-denso-a-moe.md) — cerrado: descartado por costo; ninguna decisión lo necesita.
- [Qué entra al paper](issues/07-que-entra-al-paper.md) — resuelto: opción A, la estructura del olvido.
- [B24: olvido direccional](issues/08-olvido-direccional.md) — candidato 1: ¿sobrescribir la ida deja viva la vuelta?
- [B27: puerta de duda](issues/09-puerta-de-duda.md) — una sonda del estado, sin etiqueta, crea el nivel 2 con 0,83 del oráculo; el router de Vigía no sirve como señal.
- [B28: la puerta aprendida](issues/10-puerta-aprendida.md) — cerrado en Granite: entrenar al modelo para leerse reescribe lo que lee o lo vuelve menos legible, y releer en línea no hace durar la puerta.
- [B29: la puerta en otro banco](issues/11-replica-otro-banco.md) — G1 falla por la regla: el azar ya da N2 +0,11; la diferencia pareada contra el azar sí aparece; desde 2026-11-15, las réplicas usan la regla pareada.
- [B30: fuentes y umbral](issues/12-fuentes-y-umbral.md) — el umbral de fuentes reproduce el video en dos bancos (B30 y B31) y con la sonda (B32, 0,74 del oráculo); la puerta de pesos no. El costo del umbral no se repite.
- [B33 y B34: sonda mejor y contenido](issues/13-sonda-mejor-y-contenido.md) — más datos sin filtro empeoran la sonda; el ruido del mismo tamaño no se apaga en la dosis 160; el umbral cuenta fuentes sin etiquetas y trata diez paráfrasis como dos fuentes.
- [B35 y B36: otro modelo](issues/14-otro-modelo.md) — OLMo-2-1B repite la puerta (0,76 del oráculo con la sonda) y el umbral; Granite 3B quedó sin datos: ninguna H100 en 4 h.
- [B37: intuición de daño](issues/15-intuicion-de-dano.md) — repasar 20 hechos cualesquiera corta el daño cerca de 60 %; elegirlos por intuición o con el oráculo no suma.
- [FW1: premisa en FlyWire](issues/16-premisa-flywire.md) — la fuerza por par sube poco con el grado de los socios (Spearman 0,15 a 0,17); el club rico tiene aristas 1,4 veces más fuertes. El brazo de la mosca en B38 agrega capacidad.
- [B38: capacidad a la mosca](issues/17-capacidad-mosca.md) — la expansión de la mosca daña el resto casi 4 veces; EWC corta 61 % del daño y crea la mitad de la selectividad del oráculo en la dosis 10.
- [B39: EWC con la puerta](issues/18-ewc-con-puerta.md) — las selectividades no se suman (la sonda suma −0,064 sobre EWC); la combinación conserva la selectividad de la sonda y corta el daño 49 %.
- [B40: Fisher y certeza](issues/19-fisher-y-certeza.md) — prerregistrado: ¿el gradiente de sobrescribir un hecho seguro cae sobre pesos que EWC frena?

## Out of scope

- Entrenar desde cero modelos de más de 100 M de parámetros: el piloto midió ~14,5 unidades por corrida a 80-100 M.
- Reentrenar conversiones con cientos de miles de millones de tokens (ExpertWeaver CPT).
- Cambios al producto Vigía.
