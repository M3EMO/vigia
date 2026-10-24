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
- 2026-10-22, PILOTO 8 (configuración del piloto 7: finés y turco, 1.600 secuencias; los controles crecen con
  `grow_row` "clon", no silencioso, así que su L inicial general es 2,40 contra 2,14 del árbol).
  - El árbol dio NaN en las tres corridas del barrido (×1, ×3, ×10). Causa: el paso directo p / p.detach() de la
    compuerta; el sesgo de balanceo elegía una hija con p ~ 0 y el gradiente valía 1/p. Arreglo (mf-moe a2d8e65):
    escala 1 + p − p.detach() y sesgo de compuerta acotado a ±1; test de regresión con p = 0 en fp32. Ningún test
    lo vio antes porque el modelo diminuto no produce compuertas tan filosas. Se repite solo el árbol (piloto 8b),
    con g3 liberada de la etapa A.
  - Controles sanos. `control_bajo_rango+consolidacion` (V1) aprende 0,01-0,03 nats peor que
    `control+consolidacion_uso` (finés 3,004 contra 2,993; turco 2,892 contra 2,861) y olvida igual (0,26 contra
    0,25). El lr de los deltas ×10 casi no cambia al control de bajo rango (2,999 y 2,886).
  - R5 al límite: los tokens del finés que van a expertos congelados bajan de 64 % a 52 % al aprender turco (−19 %,
    la regla pide menos de 20 %); la deriva top-8 (Jaccard) es 0,61-0,64. El router aleja una parte del tráfico
    viejo de lo protegido: proteger también el ruteo es un candidato a mejora.
  - Inferencia dispar: train contra eval 1,00 en todos; bf16 contra fp32 0,990 en el control plano. La regla del
    piloto baja a 0,985 (el bf16 cambia ~1 % de los top-8 en cualquier brazo).
- 2026-10-22, ÁRBOL NATURAL POR DOCUMENTO Y TERRITORIOS (Granite y OLMoE, 12 dominios con estonio, 16 documentos
  por dominio; DeepSeek no se corrió).
  - Información por nivel, exceso sobre el nulo (familia, dominio | familia, documento | dominio): Granite 0,074,
    0,027, 0,008; OLMoE 0,113, 0,060, 0,025. Cada nivel resuelve entre un tercio y la mitad del anterior: una caída
    geométrica, más lenta en OLMoE (64 expertos). Autosimilitud (mínimo / máximo) 0,06-0,25 en Granite y 0,10-0,47
    en OLMoE.
  - Territorios: la dimensión de correlación es ~12-14 en Granite y ~18-22 en OLMoE, la mitad que el nulo
    gaussiano (24-44): la nube de estados tiene estructura no lineal fuerte. La dimensión casi no cambia entre los
    niveles del dendrograma (todo, 2, 4, 8 grupos, expertos solos).
  - Trampa detectada: una dimensión igual en cada nivel también sale de una partición al azar (cualquier submuestra
    conserva la dimensión). Se agregó la extensión y un nulo de partición al azar (mf-moe, commit de hoy). H11 queda
    sin decidir hasta repetir el análisis (~0,5 unidades) en una VM libre.
- 2026-10-22, PILOTO 8b (solo el árbol, con el arreglo de la compuerta; misma configuración): pasa todas las reglas
  de corte. 120 mitosis (10 por capa), sin NaN.
  - Pérdida aprendida: finés 2,971, turco 2,864 (×1); ×3 y ×10 empatan (2,971-2,973 y 2,865-2,866). Decisión
    (regla V8): ×1, el más simple. Olvido del finés 0,265-0,273.
  - Hojas muertas como máximo 23,8 % (regla: 25 %); compuerta más torcida 93 % (regla: 95 %); train contra eval
    1,00; bf16 contra fp32 0,994; 1.300 s por corrida (1,24 veces el piloto 7); 18,5 GB.
  - Ruteo de lo viejo mucho más estable que en los controles: deriva top-8 (Jaccard) del finés 0,87 contra 0,61-0,64,
    y los tokens del finés que van a congelados pasan de 66 % a 66 % (−1 %) contra −19 % en los controles. La
    partición de la compuerta deja los tokens viejos con la hija vieja congelada: el árbol protege también el ruteo.
  - Forma: crecimiento en cadena. La duda vuelve a elegir a la hija nueva y algunas raíces bajan hasta 7 niveles
    (p. ej. hojas por profundidad 31, 1, 0, 1, 4, 3, 2), mientras ~30 raíces nunca se dividen.
  - No se compara la pérdida con los controles del piloto 8: crecieron con clones no silenciosos (L general inicial
    2,40 contra 2,14). La comparación justa es la de la etapa C, con la configuración PRODUCTION.
  - Decisión: la etapa C entra a la cola fase2b (12 corridas, mf-moe 5b1b2f6, zip f43f3ab).
- 2026-10-22, etapa A, control+consolidacion_azar s0: olvido 0,625 (control 0,721; division+consolidacion_uso 0,539). Congelar al azar la misma cantidad también ayuda (−13 %), menos que por uso. Ruteo: deriva top-8 0,65-0,69; tokens viejos a congelados ~42-46 %, estable. control+consolidacion_uso s0 perdió su VM a las 19:02 UTC tras 1,5 h sin log (la PC se durmió con la tapa o a mano: systemd-inhibit ya cubre el sueño por inactividad); retoma desde 2:0.
- 2026-10-23, Drive para checkpoints (usuario autorizó; rclone, remoto mfmoe con scope drive.file en ~/.config/rclone/mfmoe.conf). Ida y vuelta probada desde la PC. El job copia cada checkpoint a mfmoe:mf-moe/fase2/<corrida> desde la VM y lo borra al terminar; el runner no sube el local si Drive tiene uno igual o más nuevo (mf-moe 23b726e, vigia 779b389). Rige para las corridas que arrancan desde ahora. Riesgo: rclone avisa que su client_id compartido de Google se retira durante 2026; si deja de andar, todo vuelve al camino viejo (bajada a la PC).
- 2026-10-23, la noche: dos retiros de VM sin checkpoint local (Colab sin capacidad y una subida a ~70 KB/s) hicieron
  fallar uso s0 y s1; se reinician de cero. Resultados nuevos (olvido medio):
  - control+consolidacion_uso s2: 0,508 (division+consolidacion_uso s2: 0,547; azar s2: 0,583; control s2: 0,725).
    Primera señal de H5: consolidar sin dividir olvida igual o menos. Falta s0 y s1.
  - control+consolidacion_azar: 0,625 / 0,636 / 0,583 (control 0,721 / 0,741 / 0,725): −13 a −20 %.
  - arbol+consolidacion s0 (T4, primera corrida con PRODUCTION): olvido 0,511 (division+consolidacion s0 0,539),
    aprendido 2,445 (peor: 2,429), general 2,238 (mejor: 2,285). Sin NaN, pero viola R1 y R2: hasta 31 % de hojas
    muertas en una capa y una compuerta al 100 %. Crece en cadena hasta 11 niveles. El ruteo viejo ya no queda
    protegido como en el piloto: los tokens del finés a congelados bajan de 76 % a 61 % (−19 %).
  - Decisión: la etapa C sigue. Las reglas de corte decidían el piloto; en las corridas largas se informan. H7 mide
    "el árbol tal como está construido", con la violación declarada.
- 2026-10-23, T1 (Drive) encontró un bug: el runner armaba la carpeta con $5 después de `shift 5`, así que todas
  las corridas escribían en mfmoe:mf-moe/fase2/ (la clave del checkpoint evitó cargar uno ajeno). Arreglado (vigia
  81b798a). El checkpoint 2:0 de control_bajo_rango s0, cortado a las 16:27 por un corte de cola mío, quedó
  rescatado en mfmoe:mf-moe/rescate/ y vuelve a su carpeta cuando la VM de arbol_un_nivel s0 purgue la carpeta común.
- 2026-10-23, T4 ARREGLADO (mf-moe 48056ac; pregunta del usuario: "¿no debería arreglarse?"). Dos causas:
  1. Crecimiento en cadena: la duda media elige siempre a la hija nueva, porque la compuerta le manda los tokens
     dudosos por construcción. Cada mitosis parte su tráfico a la mitad: hojas profundas casi sin tokens (muertas).
     Arreglo: el árbol elige la hoja con más duda total (suma sobre sus tokens), así divide donde hay más trabajo
     dudoso; la recursión aparece cuando una hija junta más duda que las raíces. Test con una hoja chica y dudosa.
  2. Compuerta saturada: el sesgo de balanceo de la compuerta está acotado a ±1 (arreglo del NaN), y con p = 0 o 1
     exactos no alcanza para dar vuelta la elección. Arreglo: logits de la compuerta acotados a ±4 (p ≥ 3e-4); se
     conserva el signo, así la mitosis no cambia la salida. Test con una compuerta saturada.
  - arbol s0 (versión con cadenas) pasa a `.cache/runs/fase2_obsoleto/`; arbol_un_nivel s0 (cortado a mitad) se
    repite con el código nuevo. Las otras corridas del árbol todavía no habían corrido.
  - Saldo: 96,5 unidades. Quedan ~13 corridas (~75 unidades): entra, con poco margen. Si falta, arbol_azar (H9)
    queda para el final.
- 2026-10-23, etapa A: control+consolidacion_uso s1 = 0,507 (finés 0,618, turco 0,554, indonesio 0,349); division+consolidacion_uso s1 0,596; azar s1 0,636; control s1 0,741. Con s1 y s2: sin división olvida menos que con división (0,507/0,508 contra 0,596/0,547) y el uso le gana al azar (contra 0,636/0,583). Aprendido igual (2,423/2,419 contra 2,423/2,426). Ruteo estable: tokens del finés a congelados 60 % → 62 %. Falta s0 (corre desde 20:34 UTC).
- 2026-10-24, ETAPA A COMPLETA. control+consolidacion_uso s0 = 0,499. Veredictos (3 pares, Holm entre 11 preguntas): sin división contra control +0,224 (p 0,0005, Holm 0,006); azar contra control +0,114 (p 0,014, Holm 0,12); uso contra azar +0,110 (p 0,025, Holm 0,17, las 3 a favor); con división contra sin división −0,056 (p 0,08, Holm 0,40, las 3 en contra de dividir). H5 refutada, H6 a favor. Paper §4.12 y spec actualizados.
- 2026-10-24, control_bajo_rango+consolidacion s0 (retomó 2:0 desde Drive): olvido 0,539 contra 0,499 de control+consolidacion_uso s0; aprendido 2,466 contra 2,428; general 2,207 contra 2,264. Con capacidad nueva de bajo rango olvida y aprende un poco peor, pero conserva mejor lo general. T1 verificado: uso s0 perdió la VM a las 22:20 y retomó desde Drive en 30 s.
- 2026-10-24, ARBOL s0 CON EL ARREGLO (48056ac+): olvido 0,495 (control+consolidacion_uso s0 0,499; control_bajo_rango
  s0 0,539; versión con cadenas 0,511); aprendido 2,448 (2,428 / 2,466); general 2,243 (2,264 / 2,207).
  - T4 verificado: hojas muertas como máximo 4 % (antes 31 %), entropía de uso ≥ 0,94, compuerta más torcida 56 %
    (antes 100 %), forma ancha y corta (p. ej. 21, 18, 7, 2 hojas por profundidad: hasta 3 niveles).
  - Nuevo problema, R5: la deriva del ruteo viejo empeora (Jaccard del finés 0,33 contra 0,61-0,62 en los planos) y
    los tokens del finés a congelados bajan de 82 % a 59 % (−28 %, la regla pide menos de 20 %). Causa probable: el
    balanceo de compuertas ahora lleva cada compuerta a 50/50 y reparte tokens viejos a las hijas nuevas. Hay una
    tensión entre evitar el colapso y proteger el ruteo.
  - Especialización: el exceso de información por idioma es 0,125 contra 0,049 del plano (las hojas se especializan
    2,5 veces más por idioma); por documento, igual (0,003). Autosimilitud ~0,02: H8 no se ve.
- 2026-10-24, arbol_azar+consolidacion s0: olvido 0,528 (árbol con duda 0,495); aprendido 2,447; hojas muertas 17 %; una compuerta con carga 100 % (hoja al azar con poco tráfico: el sesgo se mueve lento); profundidad 6. Primera señal de H9: la duda elige mejor dónde abrir rama. Nota: los controles planos también tienen ~27 % de expertos muertos (clones silenciosos que el balanceo nunca recluta).
- 2026-10-24, arbol_un_nivel+consolidacion s0 (H10): olvido 0,507 contra 0,495 del árbol libre; aprendido 2,449 contra 2,448; hojas muertas 4 %; forma 16 raíces sin dividir y 32 hojas en el nivel 1 (16 mitosis por capa, todas sobre expertos de Granite); tokens del finés a congelados 87 % → 65 % (−24 %). Con una semilla, la recursión le saca 0,012 al árbol de un nivel: dentro del ruido probable (±0,01-0,02). Semilla 0 de la etapa C completa.
