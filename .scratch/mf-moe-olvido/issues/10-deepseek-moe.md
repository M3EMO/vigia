# Qué enseñan los papers de DeepSeek sobre expertos finos, compartidos y balanceo

Type: research
Status: resolved

## Question

¿Qué aportan DeepSeekMoE (expertos finos y expertos compartidos), DeepSeek-V2 y DeepSeek-V3 (balanceo sin pérdida auxiliar, gating sigmoide) y trabajos posteriores de DeepSeek al diseño del MoE fractal de 80-100 M? En particular: si los expertos compartidos sirven de ancla contra el olvido, si los expertos finos equivalen a una división hecha de antemano, cómo interactúa el sesgo de balanceo con dividir o fusionar expertos, y cómo miden la especialización de expertos. Resultado: nota en el vault con puntero acá.

## Answer

- Ningún trabajo de DeepSeek mide olvido ni entrena tareas en fila; toda conexión con el olvido es inferencia.
- Los expertos finos son una división fija hecha antes de entrenar. El control fijo usa la granularidad final del fractal: mismo tamaño de experto, mismo número y mismos activados.
- La división por neuronas (padre de dimensión h → dos hijos de h/2) preserva la función y los parámetros. Queda como alternativa al clon para probar.
- El experto compartido no tiene evidencia de ancla. Recibe el gradiente de todas las tareas. Va 1 compartido en todas las condiciones, nunca se divide ni se fusiona, y se mide su deriva.
- Balanceo: sesgo por experto sin gradiente, u = 0,001, carga contada sobre una ventana de varias tareas y no por lote. DeepSeek-V3 muestra que el alcance del balanceo decide la especialización.
- El sesgo no rompe la simetría del clon. La fila del router del clon rompe la simetría. Un sesgo mal inicializado tarda Δ/u pasos en corregirse.
- Gating softmax en fase 1, para que la entropía conserve su sentido. La entropía se calcula sobre el puntaje crudo, sin el sesgo.
- Huella: carga relativa tarea × experto × capa (figura 9 de DeepSeek-V3) más JSD, con enmascarado de expertos top como control causal y MaxVio como confusor.
- Punto de partida [inferencia]: d = 512, 8 capas, 1 compartido + 12 ruteados de 0,25 FFN, 1 + 3 activados, ≈ 90 M.
- Nota completa: `~/.claude/orchestration/vault/research/mf-moe-papers-deepseek-2026-10-14.md`

## Comments

- 2026-10-14, verificación del orquestador: DeepSeek-V4 (arXiv 2606.19348) y Engram (arXiv 2601.07372) existen y son de DeepSeek-AI. El abstract de V4 no nombra su gating; el detalle Sqrt(Softplus) viene del PDF según el agente. La nota se escribió por path directo: el MOC del vault no se regeneró.
