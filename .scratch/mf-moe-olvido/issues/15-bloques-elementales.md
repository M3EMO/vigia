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
