# Plan técnico MF-MoE (insumo del usuario, 2026-10-04)

> Texto provisto por el usuario. La revisión crítica y su mapeo al producto viven en
> `docs/spec/mf-moe-revision.md`.

## 1. Fundamento teórico: morfología y conocimiento arquitectónico

La premisa central es que el conocimiento no reside solo en los valores de los pesos (datos), sino
en la topología cambiante del grafo (estructura). Bajo entrenamiento sectorial continuo, la
supermatriz fractal sufre una deformación geométrica con huellas estructurales por dominio:

- **Legal (árbol profundo):** mitosis verticales profundas y estrechas en coordenadas específicas
  (Sintaxis + Normativa) para capturar excepciones taxonómicas.
- **Finanzas (red plana dinámica):** latido rápido de mitosis y fusiones entre las celdas de Lógica
  Matemática y Heurística, sin profundidad extrema.
- **Creativo (topología esponjosa de alta entropía):** se inhibe la contracción; los micro-expertos
  mantienen alta dispersión de opiniones en el comité de enrutamiento.

## 2. Algoritmo del "latido" fractal

### A. Expansión (mitosis dinámica)

Si `(1/T) · Σ_{t=1..T} H(R_t) > θ_mitosis` → mitosis. `H(R_t)` es la entropía de los logits del
enrutador local para el token t; T es la ventana de tokens procesados por ese experto. El nodo se
clona en K sub-nodos con ruido `ε ~ N(0, 1e-5)` para romper la simetría.

### B. Contracción (fusión de pesos)

Si `CosineSimilarity(∇W_i, ∇W_j) > θ_merging` → `W_nuevo = α·W_i + (1−α)·W_j`, con α basado en la
confianza histórica de cada experto. Se libera la memoria del nodo redundante.

## 3. Protocolo experimental (Fase 1: 4 meses, 1 GPU de 80 GB)

| Mes | Fase | Método | Métrica de éxito |
|---|---|---|---|
| 1 | Señal primaria | Forward hooks en routers de capas intermedias y profundas de gpt-oss-20b o Mixtral-8x7B; dataset balanceado de respuestas verdaderas y alucinaciones (diseño de Fonseca, Rodrigues y Romano) | AUROC > 0,76 con entropía del router + dispersión inter-expertos; latencia extra < 5 % |
| 2 | Contracción | Flujo de un dominio estrecho; matriz de covarianza de uso; fusión de pares sobre `θ_merging = 0,92` | −15 % de memoria en módulos afectados; perplejidad general ±2 % |
| 3 | Mitosis | Un experto bajo ráfaga de datos difíciles (Rust asíncrono); clon con ruido ε; micro-enrutador lineal entre los dos clones | Convergencia de la pérdida 25 % más rápida que el experto monolítico |
| 4 | Consenso | KL entre distribuciones de tokens de los micro-expertos activos; `Confianza_t = 1 − KL(comité local)` | Informe técnico público y repo abierto; marcado en rojo de fragmentos con error fáctico en español |

## 4. Entorno de ejecución

1. **Virtualización de expertos en VRAM:** pool estático de matrices pre-asignadas; la mitosis
   cambia punteros indexados en lugar de crear `nn.Module` en caliente.
2. **Agrupamiento estratificado:** los tokens que descienden a niveles profundos se pausan, se
   agrupan con gathers de Triton y se ejecutan en un bloque contiguo.
