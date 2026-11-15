# Borrador del §4.15 (2026-11-15): dos versiones para elegir

El usuario eligió la versión B el 2026-11-15. El texto final está en `docs/informe-tecnico.md`, §4.15.

Las dos versiones suman B28, B29, B30 y B31 al §4.15 actual. El usuario elige una, y después paso el texto al informe.
La versión A conserva el texto actual y agrega tres bloques. La versión B reescribe la sección alrededor de tres afirmaciones.

---

## Versión A: agregar al texto actual

(El texto actual del §4.15 queda igual hasta «Límites». Antes de «Límites» entran estos tres bloques.)

**Otro banco (B29, ~0,32 u).** La puerta corrió sobre 256 hechos nuevos de UniEdit, filtrados solo por la ida.

| Brazo | N2, dosis 10 | Diferencia pareada contra el azar |
|---|---|---|
| Azar | +0,114 [−0,019; +0,237] | — |
| Sonda | −0,081 [−0,224; +0,073] | −0,194 [−0,293; −0,093] |
| Oráculo | −0,166 [−0,288; −0,033] | −0,280 [−0,380; −0,186] |

- En este banco, Granite ya cambia más un hecho seguro que uno dudoso, sin ninguna puerta.
- La regla de B27 suponía un azar cerca de 0, y por eso falla aquí.
- Contra el azar del mismo banco, la puerta mueve N2 casi como en B27: −0,280 contra −0,366.
- Desde B29, las réplicas usan la diferencia pareada contra la línea de base del mismo banco.

**Una puerta aprendida o releída (B28, ~2,0 u).** Dos variantes buscaron una puerta que dure en dosis altas.

- Entrenar al modelo para leer su propia certeza reescribe los hechos que intenta leer.
- Con un KL fuerte que conserva los hechos, el modelo se vuelve menos legible: Δ −0,087 y −0,155.
- Releer la certeza en cada paso no hace durar la puerta: N2 +0,004 en la dosis 160.
- La relectura diluye la señal, porque el modelo que cambia ya no sabe lo que sabía.
- La lectura útil es la externa y fija: la sonda sobre el modelo de partida.

**Una voz contra muchas fuentes (B30, ~0,72 u, y B31, ~0,48 u).** La puerta de pesos no logra que un hecho seguro rechace una corrección.
B30 prueba si hace falta contar fuentes distintas. El umbral abre el peso de un hecho solo después de tau frases distintas.
tau vale 1 en la mitad dudosa del banco y sube hasta 8 en el hecho más seguro.
La creencia se mide en dos frases que ningún ajuste entrena.

| Brazo | Seguro resiste, dosis 10 | Seguro adopta, dosis 40 | Dudoso adopta, dosis 10 | Daño al resto, dosis 10 |
|---|---|---|---|---|
| Una voz, pesos | 0,19 | — | 1,00 | 1,87 |
| Una voz, umbral | 1,00 | — | 1,00 | 2,30 |
| Fuentes, umbral | — | 0,985 | — | 2,62 |

- Con el umbral, el hecho seguro rechaza 20 o más repeticiones de una voz y adopta diez fuentes distintas.
- El hecho dudoso cambia con una sola voz. Esa es la conducta buscada.
- Con pesos, la diversidad sola no protege: con fuentes distintas, el seguro resiste 0,22 en la dosis 10.
- En B30, el umbral sube el daño al resto del banco de 1,87 a 2,30, con una voz.
- B31 repite el umbral en el banco de B29: el seguro resiste una voz en 95 % y adopta diez fuentes en 97,7 %.
- Contra una voz con pesos en el mismo banco, la resistencia del seguro sube +0,75 [+0,68; +0,83].
- En B31, el costo extra no se repite: +0,15 [−0,21; +0,50]. Ese banco ya es más frágil sin umbral.
- El rechazo de una voz sale por construcción. Lo medido es que el hecho seguro todavía aprende cuando el umbral se abre.

(En «Límites», la primera viñeta pasa a decir: las fuentes de B30 difieren en la forma, no en el contenido.)

---

## Versión B: reescribir la sección

### 4.15 La certeza modula el olvido; rechazar una voz exige contar fuentes (L4 y CPU, 2026-11-13 a 2026-11-15)

Pregunta: ¿el modelo puede cambiar menos un hecho seguro que uno dudoso, y rechazar una sola corrección?
El banco, los conjuntos, los lotes y las dosis son los de B25, con 206 hechos reales. Solo cambia el peso de cada ejemplo.
La medida N2 es la de §4.14. Todos los diseños están preregistrados en los tickets 09 a 12 del mapa de generalidad.

Afirmación 1: una puerta de certeza crea la plasticidad selectiva.

(Aquí va la tabla actual de §4.15, sin cambios.)

- Sin puerta, Granite cambia igual un hecho seguro y uno dudoso (+0,003, B25).
- El efecto crece con la calidad de la señal. La sonda lee el estado al final de la pregunta, sin ver el valor.
- En otro banco (B29), la puerta mueve N2 contra el azar casi igual: −0,280 contra −0,366.
- Ese banco ya trae N2 positivo sin puerta (+0,114). Por eso las réplicas comparan contra la base del mismo banco.

Afirmación 2: la certeza útil es la que se lee desde afuera, sobre el modelo de partida.

- Entrenar al modelo para leerse reescribe los hechos que lee (B28 a B28d).
- Con un KL que conserva los hechos, el modelo se vuelve menos legible (B28f).
- Releer la certeza en cada paso no hace durar la puerta en la dosis 160 (B28e).
- Para la tesis del informe, es un efecto observador: medir la certeza desde adentro cambia lo medido.

Afirmación 3: rechazar una voz exige contar fuentes distintas.

(Aquí va la tabla de B30 de la versión A.)

- Con pesos desde 0,02, el tercio seguro igual adopta una sola voz en 81 % de los hechos, medido en las frases H de B30.
- Con el umbral, el seguro rechaza una voz, y adopta diez fuentes distintas en 98,5 %.
- El dudoso cambia con una sola voz en los dos casos.
- B31 repite la conducta en el banco de B29: el seguro resiste una voz en 95 % y adopta diez fuentes en 97,7 %.
- En B30, el umbral sube el daño al resto del banco de 1,87 a 2,30. En B31, esa suba no se repite.

Límites:
- Las fuentes de B30 difieren en la forma, no en el contenido.
- Con la sonda, el efecto de la puerta se apaga en la dosis 160. Con el oráculo se mantiene.
- Todo corre en un modelo. El tipo de hecho no se separó.
