# 15. Bloques elementales de expertos y el mecanismo completo

Estado: abierto (2026-10-26). Pedido del usuario: una arquitectura donde se distingan los "bloques elementales" de
los expertos, se vea cómo se arman estructuras más complejas y se pueda distinguir a un experto por su forma.
Pidió además probar la regla fractal repetida y las direcciones jerárquicas, revisar papers de laboratorios y de
neurociencia, y pensar el crecimiento del tronco y las conexiones entre capas.

Punto de partida (ticket 14, circuito cerrado): H7, H8 y H11 refutadas; H10 a favor sin significancia. Un
experto no tiene una figura propia en su región de tokens. El mejor sistema es la consolidación por uso (−31 %).

Research en el vault: notas "prácticas de laboratorios", "Neural Networks Are Already Clean (Goodfire)" y
"Neurociencia para un MoE con aprendizaje continuo".

## Requisitos

| ID | Pregunta | Medida | Aceptación | Costo | Estado |
|---|---|---|---|---|---|
| B1 | ¿Las neuronas de los expertos forman familias que se repiten, en un árbol autosimilar? (Goodfire) | `mf_moe.blocks.composition` y `shared_atoms` sobre los pesos de Granite, capas 0, 6, 12, 18 y 23 | autosimilar: CV de las pendientes del radio < mitad del nulo gaussiano y forma estable; bloques compartidos: mezcla cerca de la barajada y vecinas con coseno > 0,7 en otro experto muy por encima del nulo | CPU | hecho: no autosimilar, sin átomos compartidos, expertos con firma propia |
| B1b | ¿Las neuronas se repiten por función? (Goodfire identifica neuronas por los tokens que las activan) | firma = perfil de activación de cada neurona sobre tokens fijos; mismas medidas que B1 | igual que B1; además, comparar pesos contra función | CPU | hecho: sin árbol ni copias; función no correlaciona con pesos; identidad de experto más débil |
| B2 | ¿El router admite direcciones jerárquicas? | `hierarchical_routing`: recall del top-8 con 2 niveles contra grupos al azar | recall con m grupos > p95 del nulo, con costo < 1 | CPU | hecho: estructura sobre el azar, sin ahorro útil |
| B3 | ¿Una sola regla genera los cambios de todos los niveles del árbol? | árbol con un generador compartido por dirección de hoja contra hijas con deltas propios | mismo olvido con menos parámetros nuevos | GPU | spec |
| B4 | ¿El tronco tiene que crecer? (usuario: un tronco débil frena el árbol) | "anillos": una LoRA de atención nueva por idioma, las viejas con lr bajo, sobre la consolidación por uso | aprende como el control sin perder el −31 % | GPU | spec |
| B5 | ¿La consolidación graduada le gana a la binaria? (Benna y Fusi; synaptic intelligence) | lr de cada experto ∝ 1/(1+Ω), con Ω acumulada | olvido ≤ consolidación por uso y aprendido mejor por > 0,01 | GPU | spec |
| B6 | ¿El olvido es de acceso? (engramas silenciosos, Ryan 2015) | modelo final con router y atención del cierre de cada idioma | la pérdida vieja vuelve > 50 % | GPU, solo evals, con checkpoints por dominio | spec |
| B7 | ¿Conexiones entre capas con una fórmula ayudan? (usuario) | enlaces x → x+2^j con compuerta en cero (patrón dilatado, autosimilar), entrenables | menos olvido o mejor aprendido sin perder la salida inicial | GPU | spec |
| B9 | ¿Con datos de bits de bloques conocidos, los expertos aprenden las primitivas y las reusan al componer? (usuario) | MoE chico desde cero en CPU; información mutua experto × primitiva; reuso en tareas compuestas; control positivo de autosimilitud con datos recursivos | MI muy sobre el nulo y reuso; las medidas de B1 y H11 detectan el fractal sembrado | CPU | spec (mf-moe-5tq.14, mf-moe-5tq.15) |
| B8 | Congelar por TF-IDF (Meta, 2510.15103) | uso del idioma / uso en inglés, sobre la consolidación por uso | olvido < consolidación por uso | GPU | spec |
| B13 | ¿El árbol de ruteo recupera el árbol de los datos? (mf-moe-5tq.27) | dendrograma del uso de expertos por lengua contra el árbol verdadero de 8 lenguas MRHM, por capa | cofenética sobre el nulo de hojas permutadas y exceso sobre router al azar con IC > 0 | GPU, ~1-2 unidades | plan |

## Decisiones

- 2026-10-26: B1 y B2 primero, en CPU (sin unidades). Las pruebas en GPU esperan sus resultados y el saldo
  (10,65 unidades, ~2 corridas).
- 2026-10-26: tareas atomizadas en beads de mf-moe (épica mf-moe-5tq, tareas .1 a .13) para trabajar en sesiones simultáneas. Las tareas que tocan phase2.py van en worktrees propios.
- 2026-10-26, B1 (pesos, Granite capas 0, 6, 12, 18 y 23; 8192 neuronas por capa; mf-moe 1dae7b2):
  - Autosimilitud: no. La pendiente del radio se acelera con el nivel, igual que en el nulo gaussiano (CV 0,90-1,08 contra 1,20-1,27; la regla pide menos de la mitad). La forma cae de ~100 a ~8-24 direcciones: no se repite.
  - Átomos compartidos: no. Casi ninguna neurona tiene una copia en otro experto (coseno > 0,7: 0 a 1 %; nulo 0 %).
  - Identidad de experto: sí. La neurona más parecida está en el mismo experto en 26-46 % de los casos; al azar sería 3 %. En los grupos más finos, la mezcla de expertos es 0,52-0,75 contra 0,94 con etiquetas barajadas.
  - Lectura: un experto se distingue por sus familias de neuronas propias, no por bloques repetidos. Falta B1b (función) para confirmar.
- 2026-10-26, B1b (función, Granite capas 0, 6, 12, 18 y 23; firma = activación silu(compuerta)·up de cada neurona sobre 2.346 tokens de 5 dominios, en todos los tokens y no solo en los ruteados; sin la posición 0 ni el 1 % de normas más altas; 8192 neuronas por capa; `colab/function_blocks_analysis.py`, `.cache/runs/bloques/b1b.json`; mf-moe 7efcf28):
  - Autosimilitud: no. CV de las pendientes 0,75-0,98 contra 0,95-1,12 del nulo gaussiano (la regla pide menos de la mitad). La forma cae de ~56-86 a ~8-11 direcciones, igual que en el nulo (7-11).
  - Átomos compartidos: casi no. Neuronas con coseno > 0,7 en otro experto: 0,1-3,2 % (nulo 0 %); la capa 0 tiene el máximo. Con todos los tokens daba 1,8-12,9 %: el exceso venía de picos compartidos en la posición 0 y en tokens de norma extrema (control en las capas 0, 12 y 23).
  - Identidad de experto: más débil que por pesos. La vecina funcional está en el mismo experto en 9-22 % de los casos (azar 3 %; por pesos 26-46 %). En los grupos más finos, la mezcla es 0,82-0,92 contra 0,94 barajada (por pesos 0,52-0,75).
  - Pesos contra función: sin relación. Sobre 20.000 pares al azar, la correlación entre el coseno por pesos y el coseno por función es -0,00 a 0,03 (Pearson y Spearman).
  - Lectura: B1 se confirma en lo central. No hay bloques repetidos ni árbol autosimilar, ni por pesos ni por función. Por función, los expertos se mezclan más: la firma propia de un experto está sobre todo en sus pesos. Cada firma es rala: pone 7-11 % de su peso en un solo token.
  - Límite: 2.346 tokens dan firmas cortas; una neurona rara puede no activarse en la muestra. La correlación sobre pares al azar mide la masa de pares lejanos, no las colas de pares muy parecidos.
- 2026-10-26, B2 (router de Granite, capas 0, 6, 12, 18 y 23; ~2.400 tokens por capa de 5 dominios; grupos Ward de 4 y 8 sobre las filas del router; 20 nulos con grupos al azar):
  - Estructura: sí, débil. Con un solo grupo (m=1), el recall del top-8 es 0,36-0,60 contra 0,21-0,39 del nulo, entre 1,3 y 2,3 veces. Supera al p95 del nulo en casi todas las capas con m ≤ 2.
  - Ahorro: casi nada. Para recuperar el 95 % del top-8 hace falta el 78-92 % del costo plano; en la capa 12 ningún corte llega al 95 % con costo < 1.
  - Lectura: el router de Granite tiene vecindarios de expertos, pero no un árbol que permita direcciones baratas. Con top-8 de 32, cada token usa un cuarto de los expertos y los reparte entre grupos.
  - Límite: los grupos salen de las filas del router, no de un entrenamiento jerárquico. B2 mide si la jerarquía ya existe, no si se puede aprender. La validación con EMO contra un MoE estándar (mf-moe-5tq.20) dice si este medidor distingue un router modular.
- 2026-10-30, ¿mapa árbol o nube? (idea del usuario: el conocimiento crece con raíces, no por interpolación entre nubes). Medida nueva `mf_moe.blocks.tree_likeness`: condición de 4 puntos de Gromov relativa a cada cuarteto, contra una gaussiana de igual espectro (mf-moe 7b6cf99). En datos sintéticos con árbol, el árbol da entre 4 y 20 veces menos que el nulo.
  - Pesos de neuronas de Granite (firmas de B1, capas 0, 6, 12, 18 y 23): real/nulo 0,93-0,94 en todas las capas. Los pesos forman una nube, no un árbol.
  - Falta lo central: los estados ocultos por token, que son el "mapa de ideas" (mf-moe-5tq.22). HypLoRA (arXiv 2410.04010) reporta embeddings de tokens con forma de árbol.
  - Tareas nuevas sobre las tareas de bits (B9a hecho, mf-moe 9d9571e): currículo con raíces contra azar (mf-moe-5tq.23) y experto nuevo con raíz contra interpolado (mf-moe-5tq.24).
- 2026-10-31, validación con EMO contra un MoE estándar (mf-moe-5tq.20 y .22; allenai/Emo_1b14b_1T y StdMoE_1b14b_1T: misma arquitectura y mismos datos, 127 expertos ruteados más 1 compartido, top-8; capas 0, 4, 8, 12 y 15; 32 documentos de 256 tokens por dominio, 5 dominios; B1 con 128 neuronas al azar por experto, 16.256 por capa; B1b sobre 2.524 tokens; B2 sobre 8.192 tokens; `colab/emo_analysis.py`, `.cache/runs/emo/*.json`; mf-moe rama mf-moe-5tq.20, sin commit):
  - Control positivo: sí. EMO usa menos expertos por documento: 19-37 expertos efectivos contra 33-69 de StdMoE, en las 16 capas. La etiqueta conocida se ve en el uso. La pérdida LM es 3,01 contra 2,95 nats/token: el forward propio funciona.
  - B1 (pesos): no distingue. En las capas 4-15 los dos modelos dan lo mismo. CV de las pendientes 1,10-1,21 contra 1,11-1,17 (nulo 1,25-1,28; capa 15 de StdMoE 0,83). Vecina en otro experto 72-77 % contra 72-75 % (nulo 99 %). Mezcla fina 0,88-0,89 contra 0,85-0,89 (barajada 0,98).
  - B1 en la capa 0: los modelos difieren, pero al revés de la etiqueta. StdMoE muestra más identidad de experto: la vecina está en otro experto en 10 % de los casos, contra 48 % en EMO.
  - B1b (función): no distingue. En las capas 4-15, CV 0,84-0,97 contra 0,82-0,98; vecina en otro experto 95-98 % en los dos.
  - B2 (router): no distingue. Con G=4 y m=1, el recall es 0,41-0,71 en EMO y 0,36-0,69 en StdMoE. Los dos superan al p95 del nulo en casi todas las capas. EMO gana en las capas 0 y 15 y pierde en las capas 8 y 12. El costo para recall ≥ 0,95 con G=8 es 0,57-0,95 contra 0,68-0,88.
  - Huella: no distingue en el sentido correcto. La JSD entre dominios es 0,43 contra 0,33 bits, pero la JSD dentro del dominio es 0,053 contra 0,022. El cociente entre/dentro da 8,2 en EMO y 15,0 en StdMoE: premia al estándar. EMO elige un grupo por documento, no por dominio, y por eso dos mitades del mismo dominio difieren más.
  - Árbol (mf-moe-5tq.22): no distingue. Estados ocultos por token, real/nulo: 0,83, 0,71, 0,71, 0,68 y 0,59 en EMO; 0,64, 0,72, 0,73, 0,65 y 0,57 en StdMoE. En las capas 4-15 la diferencia es ≤ 0,03. Los estados son más arbóreos que la gaussiana, pero lejos del árbol sintético (0,05-0,25). Pesos: 0,86-0,89 en los dos (Granite 0,93-0,94).
  - Veredicto: nuestros medidores no distinguen un MoE modular de uno estándar. Solo el uso de expertos por documento lo distingue. Los veredictos de B1 y B2 sobre Granite pierden peso: "sin estructura" puede ser ceguera del medidor y no ausencia de estructura.
  - Lectura: B1 y B1b miden la estructura de las neuronas dentro y entre expertos. B2 agrupa las filas del router con Ward. La modularidad de EMO vive en qué expertos usa cada documento, y ningún medidor mira eso.
  - Siguiente: medir el uso de expertos por documento contra por dominio en Granite. Es barato: un solo forward, sin B1.
  - Límite: una semilla y una muestra por modelo, sin IC entre modelos. B1 usa 128 neuronas por experto; Granite usó todas. La capa 0 de StdMoE es anómala (CV 2,49, árbol de pesos 0,10, división por cero en participation_ratio) y queda sin explicar. Corrida en CPU: 43 min por modelo, pico de 9 GB.
- 2026-10-31, uso por documento en Granite (control positivo de modularidad, tras mf-moe-5tq.20):
  - Método: los mismos 160 textos que EMO (32 por dominio, 256 tokens). Medida: expertos efectivos por documento sobre los del dominio, promedio de capas. Nulo: documentos cuyos tokens eligen k expertos distintos al azar según el uso del dominio. El nulo corrige la dependencia de k y E.
  - Resultado, real/nulo (menor = más modular): EMO 0,56; StdMoE 0,72; Granite 0,88.
  - Veredicto: Granite no concentra cada documento en un grupo de expertos. Es menos modular por documento que el MoE estándar.
  - Lectura: el "sin estructura" de B1 y B2 en Granite queda sostenido por un medidor que sí ve modularidad. La ceguera de B1 y B2 sigue en pie para otros modelos.
  - Límite: una semilla, una muestra y sin IC. Granite tiene 32 expertos y top-8: el rango posible es más angosto que con 127 y top-7.
  - Script: `mf-moe/colab/granite_doc_usage.py`; resultado en `mf-moe/.cache/runs/bloques/granite_doc_usage.json`.
- 2026-10-31, B9b semilla 0 (mf-moe-5tq.15; `mf-moe/colab/b9_bits.py`, `.cache/runs/bloques/b9_s0.json`). Semillas 1 y 2 en curso.
  - Diseño: MoE de la fase 1 (16 expertos, top-2, d = 64, 4 capas), tareas con nombre. Ruteo medido solo en la respuesta. Línea base: el mismo modelo sin entrenar.
  - Recorte: sin rev. Con rev, las 6 tareas que la llevan seguían en cero a los 10.000 pasos; sola se aprende en 3000. Quedan 5 primitivas y 10 pares balanceados.
  - Curvas: todas las etapas se traban y después saltan de golpe (grokking). Conjunto: 70 % de media del paso 3000 al 8000, todo dominado en el 9500.
  - P1, información mutua experto × primitiva (sobre H), capas 0-3: 0,01, 0,19, 0,30 y 0,18. Sin entrenar: 0,02, 0,03, 0,14 y 0,14. Barajado: < 0,001. Veredicto: especialización parcial, no un experto por primitiva.
  - P2, composición (0,5 = azar), capas 0-3: 0,77, 1,00, 1,00 y 0,83. Sin entrenar: 0,87, 0,77, 0,77 y 0,80. Veredicto: en las capas 1 y 2, «a+b» rutea como la mezcla de «a» y «b». Límite: 3 pares ajenos por par.
  - P3, en fila: primitivas en 1500 pasos y pares en 2500, contra 16.500 desde cero (4 veces menos en total). Sin repetición, las primitivas caen a 0-1 %. Veredicto: reuso fuerte y olvido total de las raíces como tareas sueltas.
  - P4, control positivo: rec:p tarda 14.500 pasos y las planas 2500. Árbol de estados real/nulo, capas 0-3: recursivas 0,68, 0,93, 0,88 y 0,77; planas 0,35, 0,90, 0,93 y 0,92. CV de pendientes: 0,87-0,64 contra 0,82-0,75. Veredicto: no separa con un signo constante. El control positivo falla: o las medidas no ven la recursión, o el modelo no la guarda como estructura autosimilar.
  - Límite: una semilla; el paso del salto cambia con la semilla, y el factor 4 de P3 depende de él.
- 2026-11-01, B9b con 3 semillas (0, 1 y 2; `.cache/runs/bloques/b9_s{0,1,2}.json`). Cierra mf-moe-5tq.15.
  - P1, información mutua, capas 1 y 2: entrenado 0,19-0,25 y 0,27-0,30; sin entrenar 0,004-0,04 y 0,06-0,14. Las 3 semillas van en la misma dirección. Veredicto firme: especialización parcial (techo ≈ 0,3), no un experto por primitiva.
  - P2, composición, capas 1 y 2: entrenado 0,90-1,00 en las 3 semillas; sin entrenar 0,57-0,87 y 0,77-0,97. Veredicto: «a+b» rutea como la mezcla de sus partes. La línea base ya es alta: buena parte viene de la entrada.
  - P3: camino en fila (primitivas + pares) 4000, 7000 y 3500 pasos; desde cero 16.500, más de 20.000 (sin dominar) y 5000. El camino en fila gana en las 3 semillas, por 1,4 a 4 veces o más. Olvido de las primitivas sin repetición: 0-1 % en las 3. Veredicto firme: reuso y olvido total.
  - P4: las recursivas tardan 3-6 veces más que las planas (4500-14.500 contra 1000-2500). Árbol de estados real/nulo, capa 0: recursivas 0,63-0,80, planas 0,35-0,38; es decir, menos arbóreas. CV de pendientes: la diferencia con su nulo es -0,02 a -0,07 en los dos modelos, y el nulo sigue al real. Veredicto firme: los medidores no ven la recursión aprendida.
  - Lectura de P4: el control positivo no separa "medidor ciego" de "el modelo no guarda la recursión como estructura". Además, el modelo recursivo entrena más pasos, y el CV baja con el entrenamiento (1,02 sin entrenar).
  - Semilla 2: el modelo conjunto no dominó en 20.000 pasos; P1 y P2 se midieron igual.
  - Siguiente: una sonda que busque la escala de la recursión en los estados (¿el modelo sabe en qué nivel está cada bit?). Si la encuentra, el medidor es el ciego.
- 2026-11-03, B10 con 3 semillas (mf-moe-5tq.23; `mf-moe/colab/b10_curriculum.py`, `.cache/runs/bloques/b10_s{0,1,2}.json` y `b10_s0_reves.json`). Idea del usuario: el conocimiento nuevo necesita raíces.
  - Diseño: tareas de B9b (5 primitivas, 10 pares). Tres órdenes: raíces (primitivas, después pares), azar (todo junto), revés (pares, después primitivas). 10 % de repetición. Generalización: los 10 pares ordenados no entrenados.
  - Pasos totales, semillas 0-2: raíces 4000, 13.500 y 3000; azar 9500, 4500 y 20.000 (sin dominar); revés 2500, 8000 y 6000.
  - Retención media de lo entrenado: raíces 0,975-0,989; azar 0,922-1,00; revés 0,809-0,968.
  - Pares no vistos: 0-2,8 % en los 9 entrenamientos, incluidos los 2 pares que equivalen como función a tareas entrenadas.
  - Veredicto sobre la velocidad: el orden no gana con un signo constante. El factor 2,4 de raíces sobre azar en la semilla 0 no se repite; en la semilla 1 azar gana por 3 veces. El salto tipo grokking domina la varianza.
  - Veredicto sobre el olvido: revés retiene menos en las 3 semillas. Lo aprendido primero se olvida al pasar a la otra etapa.
  - Veredicto sobre la generalización: ningún orden compone pares nuevos. El modelo lee el nombre del par como una etiqueta entera.
  - Ya conocido: Ramesh, Lubana et al., ICML 2024 (arXiv 2311.12997) muestran que la composición directa no generaliza, que los pasos intermedios sí, y que el sesgo de orden rompe composiciones. B10 replica su caso directo. B10b (cobertura × pasos) se canceló antes de correr por la misma razón (mf-moe-5tq.26).
  - Siguiente: B12 (mf-moe-5tq.24), experto hijo con raíz contra interpolado, al azar y de padres ajenos. Hueco confirmado con Lacuna (vault, 2026-11-02).
- 2026-11-02, B12 con 3 semillas (mf-moe-5tq.24; `mf-moe/colab/b12_raices.py`, `.cache/runs/bloques/b12_s{0,1,2}.json`). Idea del usuario: el experto nuevo necesita raíces.
  - Diseño: etapa 1, primitivas con 16 expertos. Etapa 2: nace un hijo por par en cada capa (26 expertos), ruteo forzado por el nombre del par, hijo mudo para las primitivas. Brazos: raíz (copia del experto de f), interp (promedio de f y g), azar, padres_azar (dos primitivas ajenas) y fijo (26 expertos desde el principio).
  - Pasos de pares, semillas 0-2: raíz 2000, 2500 y 1500; interp 2000, 3500 y 1500; azar 2000, 2000 y 2000; padres_azar 2000, 4500 y 1500; fijo 3500, 1500 y 4000. Resolución: 500 pasos.
  - Retención de primitivas: raíz 0,978-1,00; interp 0,988-0,998; azar 0,993-1,00; padres_azar 0,944-1,00; fijo 0,895-1,00.
  - Pares sin forzar el ruteo: 0,2-7,9 %. Los pares dependen por completo de su hijo. Pares no vistos: 0-1,4 %.
  - Falla de diseño: el «experto de f» era el más elegido por f, y es el mismo para casi todas las primitivas. Semilla 2: las 5 primitivas comparten el top en las 4 capas (25-50 % del ruteo), así que raíz = interp = padres_azar por construcción (curvas idénticas). Semillas 0 y 1: capas 0 y 3 compartidas; capas 1 y 2 en parte distintas. Chequeo: `.cache/runs/bloques/b12_tops.txt`.
  - Veredicto: el origen de los pesos del hijo no cambia la velocidad ni la retención con un signo constante. Azar es el brazo más estable (2000 en las 3). Crecer no gana a fijo con un signo constante.
  - Veredicto sobre la hipótesis de raíces: no se probó. El «padre» elegido es un experto genérico, no la raíz de f. Coincide con B9b P1 (información mutua ≤ 0,3): no hay un experto por primitiva.
  - Nacimiento: las primitivas caen a 0,775, 0,907 y 0,928 al nacer los hijos, igual en todos los brazos que crecen. Las filas nuevas diluyen la softmax del router.
  - Siguiente: B12b, padre elegido por uso diferencial (uso en f menos uso medio en las otras primitivas), con la especialización medida antes de crecer. Si ningún experto se especializa, la pregunta de raíces no tiene sujeto en este modelo.
- 2026-11-02, B12b con 3 semillas (mf-moe-5tq.28; `mf-moe/colab/b12_raices.py` en 01e1bcc, `.cache/runs/bloques/b12b_s{0,1,2}.json`). Padre = experto de mayor uso diferencial.
  - Padres: en las capas 1 y 2 hay 4-5 padres distintos para 5 primitivas en las 3 semillas. En la capa 0 se repiten. Uso diferencial: 0,02-0,25.
  - Pasos de pares, semillas 0-2: raíz 2000, 3500 y 1500; interp 3000, 3500 y 2500; azar 2500, 6000 y 1500; padres_azar 2500, 3500 y 1500; fijo 3500, 1500 y 4000.
  - Media de pares a los 1000 pasos: raíz 0,96, 0,86 y 0,95; azar 0,85, 0,83 y 0,92; padres_azar 0,88, 0,81 y 0,98; interp 0,94, 0,81 y 0,90.
  - Raíz supera a azar a los 1000 pasos en las 3 semillas (+0,11, +0,03 y +0,03). Contra padres_azar gana en 2 de 3. Interp es el brazo que crece más lento al principio.
  - Ruido: el brazo azar con la misma semilla tardó 2000 contra 2500 pasos (semilla 0) y 2000 contra 6000 (semilla 1) entre B12 y B12b. Solo cambió el estado del generador al azar. Los pasos hasta dominar no sirven para comparar brazos con 3 semillas.
  - Veredicto: señal débil a favor de la raíz al principio, sin un signo constante contra padres ajenos. La especialización es parcial (uso diferencial ≤ 0,25). Pasa a B12c (mf-moe-5tq.29): dial de distinción, AUC temprana y 5 semillas pareadas.
- 2026-11-02, B12c Fase A con 5 semillas (mf-moe-5tq.29; `mf-moe/colab/b12c_distincion.py` en c54da6d, GPU L4, `.cache/runs/bloques/b12c_a_gpu/b12c_a_s{0..4}.json`). Hipótesis del usuario: las primitivas de B12 se parecen demasiado para que los expertos se especialicen.
  - Dial p: cada primitiva escribe una fracción p de las posiciones con su propio alfabeto (not AB, xor CD, rot EF, par GH, andor IJ). Con p = 0 todas usan 0/1.
  - Medidas por capa: información mutua (IM) experto × primitiva, IM condicionada a los tokens 0/1 compartidos, y especificidad de ablación. IC90 bootstrap sobre 5 semillas.
  - Controles con los mismos pasos que su base: router congelado al azar, sin balanceo, y base larga de 20.000 pasos sin corte.
  - p = 1 es degenerado: par no aprende en 20.000 pasos en las 5 semillas de GPU. En CPU, la semilla 1 tampoco aprende en 12.000 pasos, y la semilla 0 sí aprende en 4500. Evaluar en bf16 o en fp32 da lo mismo. El análisis usa p de 0 a 0,75.
  - Gran parte de la IM es mecánica. Un router congelado al azar da el 53-85 % de la IM de la base con p = 0,75 y 1, porque el alfabeto ya separa los estados.
  - Exceso aprendido (base menos router al azar) con p = 0: entre -0,011 y +0,024 en las 4 capas. Con el mismo vocabulario, el router aprendido no separa primitivas más que uno al azar.
  - Exceso aprendido de IM, de p = 0 a 0,75: capa 1 +0,178 [0,101, 0,266]; capa 2 +0,090 [0,061, 0,124]. La pendiente tiene IC > 0 en las 4 capas.
  - Exceso de IM condicionada a 0/1: pendiente capa 1 +0,074 [0,006, 0,143], capa 2 +0,088 [0,033, 0,153]. En las capas 0 y 3, el IC cruza 0.
  - Ablación: con p ≥ 0,25, sacar un experto daña sobre todo a una primitiva en las capas 0, 2 y 3 (especificidad de 0,10 a 0,4-0,8). La capa 1 no sube. El router al azar queda cerca de 0.
  - Largo del entrenamiento: con p = 0, 20.000 pasos no crean IM aprendida (capa 1: -0,001). Con p = 0,75 suma de +0,04 a +0,06. El largo no explica el efecto.
  - Padres distintos (mayor uso diferencial): 4,4-5 de 5 en las capas 1 y 2 ya con p = 0. En las capas 1-2 el padre es distinto aunque la IM aprendida sea nula.
  - Veredicto H-a: se cumple. La especialización aprendida crece con la distinción en las capas 1 y 2, con IC > 0 y sobre el nulo y el router al azar. Con p = 0, la especialización aprendida es nula. La hipótesis del usuario explica por qué B12 y B12b no encontraron raíces.
  - Siguiente: Fase B con p ∈ {0; 0,75} (p = 1 queda afuera por par). Regla H-b: la AUC temprana pareada raíz − azar y raíz − padres_azar tiene IC > 0 con p = 0,75 y es mayor que con p = 0.
- 2026-11-03, B12c Fase B con 5 semillas pareadas (mf-moe-5tq.29; `mf-moe/colab/b12_raices.py`, GPU L4, `.cache/runs/bloques/b12c_b_gpu/b12c_b_p{0.0,0.75}_s{0..4}.json`). p = 0,75 reemplaza a p = 1 porque par no aprende con p = 1.
  - Medida preregistrada: AUC de la exactitud media de los pares en los primeros 1000 pasos, con evaluación cada 100. IC95 bootstrap sobre semillas.
  - Padres con p = 0,75: uso diferencial de 0,13 a 0,41, contra ≤ 0,25 en B12b.
  - AUC media con p = 0: raíz 0,677, raíz_g 0,698, interp 0,659, padres_azar 0,698, azar 0,669.
  - AUC media con p = 0,75: raíz 0,800, raíz_g 0,804, interp 0,788, padres_azar 0,778, azar 0,778.
  - Con p = 0: raíz − azar +0,008 [-0,020, +0,029]; raíz − padres_azar -0,021 [-0,080, +0,042].
  - Con p = 0,75: raíz − azar +0,021 [-0,000, +0,040]; raíz − padres_azar +0,022 [+0,007, +0,036]. Probabilidad de mejora: 0,8 en los dos.
  - Diferencia entre p = 0,75 y p = 0: raíz − padres_azar +0,043 [-0,024, +0,094]; raíz − azar +0,013 [-0,022, +0,049].
  - Raíz de f y raíz de g dan lo mismo (-0,005 [-0,016, +0,009]).
  - Retención de primitivas: 0,98-1,00 en todos los brazos. Pares no vistos: 0-0,4 %.
  - Veredicto H-b: no se cumple. Con p = 0,75 la raíz gana a padres ajenos con IC > 0, pero contra azar el IC toca 0, y ninguna diferencia entre p = 0,75 y p = 0 tiene IC > 0.
  - Tamaño: la ventaja de la raíz es de 2 puntos de exactitud media en los primeros 1000 pasos. La dirección coincide con la hipótesis, pero el efecto es chico.
  - Lectura del plan: «A sube y B no». Hay expertos especializados cuando las tareas se distinguen, pero heredar sus pesos casi no acelera.
  - Decisión: no se agregan semillas ni p = 0,5. Un efecto de 2 puntos no cambia el diseño del MF-MoE. La pregunta de raíces se cierra con este resultado.
- 2026-11-03, B13, plan antes de gastar (mf-moe-5tq.27). Investigación: vault «árbol de ruteo y jerarquía de datos» y pasada de Lacuna del 2026-11-03.
  - Corrección: Doshi et al. (arXiv 2610.07168) usan dos lenguas MRHM, no una filogenia. B13 la generaliza a 8 lenguas.
  - Ya se sabe: en un denso, la similitud entre lenguas crece con la cercanía y pica en las capas medias (Doshi). No se corre un denso.
  - Ya se sabe: el router lineal hereda la geometría del estado oculto (Myth, 2604.09780). La predicción es empate con k-means.
  - Lacuna 2026-11-03: BRANCH-MoE (2610.06725) impone un árbol y no lo compara con datos. Softmax Routing Dynamics (2610.39442) estudia colapso. Ninguno hace B13.
  - Datos: MRHM con L = 4, s = 2, m = 4, v = 16 (código de d-doshi/platonic, BSD-3). Dos lenguas a distancia d del árbol de lenguas difieren en las reglas de los d niveles de abajo. El nivel de arriba es común.
  - Vocabulario terminal compartido en la base, para que el léxico no delate la rama.
  - Modelo: el de B12c (d = 64, 4 capas, 16 expertos, top-2, balanceo por sesgo), 10.000 pasos con checkpoints en 1000, 3000 y 10.000.
  - Condiciones por semilla: base, router al azar, sin balanceo y vocabulario por rama (control positivo). Sin entrenar: modelo en inicialización y k-means de 16 sobre los estados propios.
  - Medidas por capa: correlación cofenética con test de Mantel, acuerdo de tripletas y Robinson-Foulds. Matriz capa × nivel de IM experto × ancestro.
  - Regla H1: cofenética sobre el p95 del nulo permutado en ≥ 4 de 5 semillas, y base − router al azar con IC90 > 0.
  - Regla H1 estricta: ruteo − k-means con IC90 > 0. Un empate dice que el ruteo hereda la geometría y que H2 puede dividir por k-means.
  - Corte: si el vocabulario por rama no recupera el árbol, la medida está rota y se frena.
  - Paso 0, CPU: generador y tests; piloto de 1 semilla, base, 3000 pasos (~22 min a 430 ms/paso). Si no aprende, bajar a L = 3 antes de GPU.
  - Costo GPU: 4 × 10.000 pasos por semilla, 5 semillas en paralelo en una L4 (~27 ms/paso por trabajo, medido en B12c) ≈ 30-40 min ≈ 1 unidad. Con 20.000 pasos, ≈ 2 unidades. En CPU serían ~24 h.
  - tree_likeness no entra en B13: la medida es contra un árbol conocido, no la δ de Gromov.
