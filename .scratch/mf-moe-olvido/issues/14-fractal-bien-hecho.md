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

(ninguna por ahora)

## Comments
- 2026-10-22 (usuario): si dividir no es la solución, ¿otro mecanismo podría "evolucionar" un fractal?
  Candidatos anotados (sin decidir; dependen de la etapa A):
  1. Consolidación recursiva: el experto consolidado queda congelado y aprende lo nuevo con hijos chicos
     adentro (adaptadores de bajo rango), que a su vez se consolidan. El fractal está en la protección.
  2. Asignación a cualquier profundidad (proceso del restaurante chino anidado): un dato nuevo baja por el
     árbol y abre rama donde deja de encajar. El árbol crece con los datos, no con un calendario.
  3. Árbol aprendido por gradiente con un prior jerárquico (árboles de decisión suaves); el ticket 13 ya
     muestra que los MoE preentrenados forman un árbol implícito sin ayuda.
  4. Búsqueda de orden cero sobre las reglas (idea DUST del 2026-10-14): evolucionar la regla, no los pesos.
  Requisito previo para cualquiera: una medida de "fractal" (misma estadística de ramificación y de
  especialización en cada nivel del árbol); sin esa medida no hay predicción que falsar.
- 2026-10-22 (usuario), decisiones: la etapa C se construye aunque dividir no sirva; se exploran los
  candidatos 1 + 2 juntos; el candidato 3 (árbol aprendido por gradiente) también interesa.
- 2026-10-22 (usuario): "¿habría que tener en cuenta la inferencia dispar y el colapso del enrutador?". Sí:
  entran a la spec como medidas obligatorias y reglas de corte (abajo).
- 2026-10-22, MEDIDA DE FRACTAL (`expert_tree.fractal`, mf-moe 5586ab3). Por nivel de la jerarquía de los
  datos (familia, idioma | familia, documento | idioma): información condicional experto-nivel normalizada,
  menos un nulo multinomial (cada hijo sortea sus tokens con el perfil del padre). Autosimilitud = exceso mínimo
  / exceso máximo: 1 = cada nivel se resuelve tanto como el anterior. Descartado: conteo de cajas sobre el
  dendrograma de expertos (r2 ~0,9 también en el nulo, no discrimina).
  - Árbol natural, dos niveles (sin documentos todavía): Granite exceso medio familia 0,076, idioma | familia
    0,031, autosimilitud por capa 0,23-0,52; OLMoE 0,119 y 0,068, autosimilitud 0,36-0,88. Lectura: los dos
    modelos resuelven el segundo nivel a la mitad del primero; hay estructura en dos escalas, más en OLMoE.
  - Falta el nivel documento | idioma: el análisis registra por dominio; hay que registrar por documento
    (inferencia, ~2 unidades).

## SPEC DE LA ETAPA C (2026-10-22, sin correr)

Un solo mecanismo con la misma regla en cada escala ("la duda asigna, el uso protege"), en Granite, sobre la
configuración PRODUCTION de la fase 2.

1. Hojas de bajo rango (candidato 1). Cada experto de Granite es la raíz de un subárbol. Una mitosis convierte
   una hoja en nodo interno con dos hijas: la vieja (los mismos pesos, congelada) y la nueva = vieja + U·V
   (rango 16, U en cero: la salida no cambia al nacer). Una nieta suma su propio U'·V' sobre su madre. Lo
   nuevo se aprende en deltas chicos y lo viejo queda intacto en cada nivel.
2. Asignación a cualquier profundidad (candidato 2). La compuerta de cada nodo parte sus tokens por la
   dirección de la duda (prototipo `tree.py` del ticket 13). La mitosis se abre en la hoja donde los tokens
   del dominio nuevo concentran la duda (z > 3 sobre las hojas de la capa), en cualquier profundidad; sin
   calendario. Tope: 16 mitosis por capa (igual que la fase 2).
3. Consolidación recursiva. Al cerrar un dominio se congelan las hojas que más usó (incluidas las de bajo
   rango); la próxima mitosis sobre una hoja congelada nace como nieta.
4. Fusión solo como poda aceptada: queda fuera del primer piloto.
5. Candidato 3, árbol aprendido (brazo aparte, después del piloto de 1 + 2): el router elige primero un grupo
   y después un experto del grupo; la pertenencia de los expertos a los grupos es una matriz suave que se
   aprende con un prior de entropía (se endurece con el tiempo). Al nacer, los grupos salen del árbol natural.

Medidas obligatorias en cada corrida (además del olvido):
- Colapso del enrutador: entropía normalizada del uso por capa, fracción de hojas muertas (menos del 1 % de
  su parte justa) y reparto de cada compuerta. Balanceo en dos niveles: sesgo por experto (como ahora) y por
  compuerta (solo nodos nacidos en la corrida). Corte del piloto: más del 25 % de hojas muertas, o una
  compuerta que manda más del 95 % a una hija en dos revisiones seguidas.
- Inferencia dispar (ruteo de entrenamiento distinto del de evaluación): coincidencia top-k (Jaccard) del
  mismo lote en modo train y en modo eval, y en fp32 contra bf16; se exige 0,99 o más. Hoy el sesgo de
  balanceo solo se actualiza en train y la compuerta usa argmax en los dos modos.
- Deriva del ruteo de lo viejo: coincidencia top-k de los tokens de evaluación de cada dominio terminado,
  antes y después de cada dominio nuevo. Si la consolidación protege expertos pero el router deja de mandarles
  los tokens viejos, la protección no sirve (el diagnóstico atribuyó 6 % del olvido al router solo).
- Fractal: `expert_tree.fractal` sobre el uso por hoja al final, con niveles familia / idioma / documento.

Brazos y pares:
- `arbol+consolidacion` (1 + 2 + 3 de arriba) y `arbol_azar+consolidacion` (mitosis en hojas al azar con la
  misma cantidad), 3 semillas, pareados con `control+consolidacion_uso` y `division+consolidacion_uso`.
- Predicción si la idea fractal es cierta: el árbol olvida menos que el plano con la misma consolidación, o
  aprende mejor cada idioma con el mismo olvido, y su autosimilitud supera a la del plano.
- Piloto antes: finés y turco, 1.600 secuencias (configuración del piloto 7), con las reglas de corte.
- Costo: piloto ~1,5 unidades; brazos ~36 unidades.
- 2026-10-22, CONSTRUIDO (mf-moe 85bb72e, c248fd8): `fractal_tree.py` (router en árbol sobre Granite, hojas de
  bajo rango rango 16, compuertas con balanceo propio, congelado de hija vieja y copias base, retome) y los
  brazos `arbol+consolidacion` y `arbol_azar+consolidacion`. 100 tests en CPU pasan, incluido el retome a mitad de
  dominio. Todo brazo nuevo reporta `ruteo`: entropía de uso, hojas muertas, fractal por documento, train contra
  eval, bf16 contra fp32 y deriva del ruteo de cada dominio terminado.
  - Decisión: el calendario dice cuándo (mismo presupuesto, 16 mitosis por capa) y la duda dice dónde. Un umbral
    absoluto nunca dispara: z > 3 en 0 de 192 revisiones de `division` s0, z > 2 en 9 %. El "cuándo" guiado por
    datos (desacuerdo de gradientes entre las dos mitades) queda pendiente.
  - Nota de capacidad: una hoja de bajo rango suma ~57 mil parámetros contra ~1,5 M de un experto entero; el árbol
    compite con menos parámetros entrenables que el control de 48 expertos. Es parte del diseño (hijos baratos).
  - Piloto 8 (`colab/pilot8_phase2.py`): `arbol+consolidacion` y `control+consolidacion_uso` en la configuración
    del piloto 7, con las reglas de corte. Corre en la primera VM que libera la fase 2 (unidad mfmoe-etapac).
  - La etapa A pasó a `colab/fase2b.txt` con el código nuevo (mide ruteo) y arranca al terminar fase 2 y piloto.
  - Los brazos del árbol se suman a fase2b.txt solo si el piloto pasa las reglas de corte.
- 2026-10-22 (usuario): ¿sirve "Fractal Generative Models" (Li, Sun, Fan, He 2025)? Nota en el vault. No
  reemplaza al transformer: lo anida (un autorregresivo genera parches con autorregresivos más chicos). Sirve como
  precedente de "misma regla en cada escala, módulos más chicos abajo"; no trata el olvido.
