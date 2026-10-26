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
| B1b | ¿Las neuronas se repiten por función? (Goodfire identifica neuronas por los tokens que las activan) | firma = perfil de activación de cada neurona sobre tokens fijos; mismas medidas que B1 | igual que B1; además, comparar pesos contra función | CPU | spec |
| B2 | ¿El router admite direcciones jerárquicas? | `hierarchical_routing`: recall del top-8 con 2 niveles contra grupos al azar | recall con m grupos > p95 del nulo, con costo < 1 | CPU | código listo |
| B3 | ¿Una sola regla genera los cambios de todos los niveles del árbol? | árbol con un generador compartido por dirección de hoja contra hijas con deltas propios | mismo olvido con menos parámetros nuevos | GPU | spec |
| B4 | ¿El tronco tiene que crecer? (usuario: un tronco débil frena el árbol) | "anillos": una LoRA de atención nueva por idioma, las viejas con lr bajo, sobre la consolidación por uso | aprende como el control sin perder el −31 % | GPU | spec |
| B5 | ¿La consolidación graduada le gana a la binaria? (Benna y Fusi; synaptic intelligence) | lr de cada experto ∝ 1/(1+Ω), con Ω acumulada | olvido ≤ consolidación por uso y aprendido mejor por > 0,01 | GPU | spec |
| B6 | ¿El olvido es de acceso? (engramas silenciosos, Ryan 2015) | modelo final con router y atención del cierre de cada idioma | la pérdida vieja vuelve > 50 % | GPU, solo evals, con checkpoints por dominio | spec |
| B7 | ¿Conexiones entre capas con una fórmula ayudan? (usuario) | enlaces x → x+2^j con compuerta en cero (patrón dilatado, autosimilar), entrenables | menos olvido o mejor aprendido sin perder la salida inicial | GPU | spec |
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
