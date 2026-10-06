# Ideas del usuario y su destino

Cada idea tiene su texto original, su revisión crítica y su estado en el producto.

| Idea | Texto original | Revisión | Estado |
|---|---|---|---|
| Señal primaria: hooks en el router, AUROC y latencia (MF-MoE fase 1) | `docs/mf-moe-plan.md` | `docs/spec/mf-moe-revision.md` | **En Vigía y medida**: AUROC 0,89 en Granite; gpt-oss en curso |
| Consenso de expertos por KL (MF-MoE fase 4) | `docs/mf-moe-plan.md` | `docs/spec/mf-moe-revision.md` (objeción 1 y 2) | **En Vigía**, corregida a JSD normalizada (`e_jsd`, nivel F2, opt-in) |
| Contracción por fusión de pesos (fase 2) | `docs/mf-moe-plan.md` | `docs/spec/mf-moe-revision.md` | Investigación; necesita GPU de 80 GB y fijar el criterio |
| Mitosis de expertos (fase 3) | `docs/mf-moe-plan.md` | `docs/spec/mf-moe-revision.md` | Investigación; primer paso barato: MoE diminuto en CPU con control de parámetros iguales |
| Huellas topológicas por sector | `docs/mf-moe-plan.md` | `docs/spec/mf-moe-revision.md` (objeción 6) | Investigación; Vigía ya guarda el uso capa × experto por traza |
| Deriva del espacio vectorial y anclaje | `docs/ideas/deriva-procrustes.md` | `docs/spec/mf-moe-revision.md` (sección de deriva) | **En Vigía** como control: `router_sha256` en el fingerprint y monitor PSI |
| Alineación de Procrustes | `docs/ideas/deriva-procrustes.md` | ídem | Útil como diagnóstico (Procrustes/CKA), no como corrección |
| Congelar la salida ante baja confianza | `docs/ideas/mf-moe-v2-os-harness.md` | `docs/spec/mf-moe-v2-revision.md` | **En Vigía**: acciones `abstain`, `rag_retry`, `human_review` |
| OS Harness (API, UI Automation, frame-buffer) | `docs/ideas/mf-moe-v2-os-harness.md` | `docs/spec/mf-moe-v2-revision.md` | **Fase 2 del roadmap**, después de cerrar Vigía |
| Agente con acceso a nivel kernel | `docs/ideas/mf-moe-v2-os-harness.md` | `docs/spec/mf-moe-v2-revision.md` (riesgo 2) | **Fase 3 del roadmap**; revisar riesgos al empezar |
| Bits dinámicos por profundidad | `docs/ideas/mf-moe-v2-os-harness.md` | `docs/spec/mf-moe-v2-revision.md` | Investigación; ternario exige entrenar desde cero |
| Contexto de 1M tokens en O(1) | `docs/ideas/mf-moe-v2-os-harness.md` | `docs/spec/mf-moe-v2-revision.md` | Investigación; la afirmación O(1) necesita medición |
| Telemetría de gradientes de clientes | `docs/ideas/mf-moe-v2-os-harness.md` | `docs/spec/mf-moe-v2-revision.md` (riesgo 1) | **Rechazada**: un gradiente filtra datos del cliente |
