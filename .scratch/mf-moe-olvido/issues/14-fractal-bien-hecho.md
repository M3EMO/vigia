# Fractal bien hecho: ¿qué parte de la consolidación es fractal, y cómo se arma el árbol que la aproveche?

Type: grilling
Status: open
Blocked by: 12 (semilla 2), 13 (árbol natural)

## Question

La fase 2 (ticket 12) muestra un solo brazo que gana: `division+consolidacion_uso` olvida 0,164 nats menos que
el control con 2 semillas (−0,182 y −0,145). La división sola empata, la fusión empeora y congelar al padre
elegido por la duda no ayuda. La síntesis de investigación (vault: "MoE fractal: por qué la idea tiene sentido
y cómo hacerla bien", 2026-10-21) resume el principio: "la duda asigna, el uso protege". ¿La ventaja viene de
la estructura fractal o solo de congelar a los expertos usados? ¿Y cómo se arma un fractal que la aproveche?

## Etapa A: controles de la consolidación (en la cola, 2026-10-22)

- `control+consolidacion_uso`: MoE fijo con 48 expertos desde el inicio, sin división, con la misma
  consolidación (8 más usados por capa al cerrar cada dominio menos el último). Pregunta: ¿hace falta dividir?
- `control+consolidacion_azar`: igual, pero congela al azar tantos expertos nuevos como la consolidación por
  uso en ese mismo dominio y capa. Pregunta: ¿importa elegir por uso, o alcanza con aislar capacidad?
- Por qué la cantidad se iguala así: los idiomas comparten sus expertos más usados; con 8 por dominio, la
  consolidación por uso termina con 10-16 congelados por capa, no 24.
- 3 semillas cada uno (6 corridas, ~36 unidades), mismos datos y configuración PRODUCTION; se parean con
  las corridas de la fase 2. Código: commit db4edd9 de mf-moe; el zip de la cola se rearmó con ese
  `phase2.py` (cambio aditivo, los brazos viejos no cambian). Saldo antes de lanzar: 207 unidades.
- Lectura:
  - si `division+consolidacion_uso` ≈ `control+consolidacion_uso`, la división plana no aporta nada y la
    ventaja es de la consolidación, una técnica conocida de aislamiento de parámetros;
  - si `control+consolidacion_uso` > `control+consolidacion_azar`, el uso elige bien qué proteger;
  - si empatan, alcanza con reservar capacidad y el criterio no importa.

## Etapa B: proteger la atención (spec, sin correr)

- El diagnóstico dice que la atención sola carga 69 % del olvido al devolverla. Ningún brazo la protege.
- Brazo propuesto: la consolidación ganadora de la etapa A más atención lenta (lr de la atención × 0,1
  desde el segundo dominio). Piloto de finés y turco (configuración del piloto 7) antes de las 3 semillas.

## Etapa C: el árbol (spec, depende de A y del ticket 13)

Los cinco principios de la síntesis, en orden de construcción:

1. Árbol desde el árbol natural: agrupar los 32 expertos de Granite por su perfil de ruteo (ticket 13 ya
   mide ese árbol); el router elige el grupo y después el experto dentro del grupo.
2. Mitosis guiada por datos, nunca por calendario: cada hoja mantiene dos sub-expertos latentes (la compuerta
   del prototipo `tree.py` ya parte los tokens) y se divide cuando las dos mitades piden gradientes distintos.
3. "La duda asigna, el uso protege": los tokens dudosos van a la hoja nueva; al cerrar un dominio se
   consolidan las hojas de más uso (el criterio que gane la etapa A).
4. Fusión solo como poda aceptada: dos hojas se fusionan si la pérdida de validación no sube; si sube, se
   deshace. Nunca promediar a ciegas (la fase 1 y la fase 2 muestran que promediar empeora).
5. Controles: el mismo árbol con mitosis al azar, y el MoE plano con la consolidación ganadora.

Predicción si la idea fractal es cierta: el árbol con mitosis guiada olvida menos que el plano con la misma
consolidación, o aprende cada idioma nuevo mejor con el mismo olvido.

## Preguntas abiertas para el usuario

1. Si la etapa A muestra que dividir no aporta nada sobre la consolidación, ¿se construye la etapa C igual
   (es la última prueba de la idea fractal) o se cierra la fase 2 con ese resultado?

## Comments
