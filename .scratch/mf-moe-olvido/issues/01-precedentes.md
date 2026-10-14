# Cómo crecen y se fusionan expertos en los trabajos publicados

Type: research
Status: resolved

## Question

¿Cómo implementan Lifelong-MoE, SEMA, HC-SMoE y MC-SMoE el crecimiento, la división y la fusión de expertos? Hace falta su disparador, su criterio de fusión, sus baselines, sus métricas de olvido (por ejemplo BWT), su escala y si publican código y con qué licencia. Resultado: nota en el vault con puntero acá.

## Answer

- Ningún precedente dispara el crecimiento con la entropía del router. Usan calendario manual (Lifelong-MoE), z-score del error de reconstrucción (SEMA) o tokens sin experto (DynMoE).
- El disparador por duda del router es la contribución nueva y necesita un comparador: z-score estilo SEMA o pérdida por experto, a igual cantidad de divisiones.
- Criterio de fusión único: similitud de la salida media del experto (HC-SMoE). Operador: promedio ponderado por frecuencia. El router queda intacto y redirige filas.
- Los clones de un padre arrancan alineados; hay que medir si hace falta alinear neuronas (MC-SMoE) antes de fusionar clones divergidos.
- Simetría de clones: ningún trabajo la mide. Inicializar la fila del router del clon con la media de los tokens de alta duda (idea de DynMoE) y medir la divergencia en los primeros pasos.
- Control: el fijo arranca con los parámetros finales del fractal (como el 10 fijo contra 4 → 7 → 10 de Lifelong-MoE) y con top-k constante para igualar FLOPs.
- La repetición pesa mucho (Thérien et al.: 30-40 %); fijo+repetición es el baseline fuerte.
- Ningún trabajo reporta BWT: MF-MoE calcula la matriz R[i,j] y deriva exactitud media, BWT y forgetting, con 3 semillas.
- Nota completa: `~/.claude/orchestration/vault/research/mf-moe-precedentes-crecimiento-fusion-2026-10-14.md`

## Comments

- 2026-10-14, verificación del orquestador: el resumen del agente atribuía a Thérien et al. (arXiv 2503.05029) que el menor olvido del MoE viene de tener más parámetros totales. El abstract no lo dice: reporta robustez del MoE a cambios de distribución ligada al balanceo del router, sin explicación causal. Esa atribución queda [no verificado] hasta leer el cuerpo. El control con parámetros iguales sigue justificado por la objeción 5 de `docs/spec/mf-moe-revision.md`.
