---
name: vigia-proyecto
description: "Vigía (capa de confianza por token para MoE autoalojados): ruta, estado y orden de fases decidido por el usuario"
metadata:
  node_type: memory
  type: project
  originSessionId: 983d4a16-7fab-466c-9791-cbed900c30cc
  modified: 2026-10-12T16:23:30.084Z
---

Vigía vive en `C:\Users\map12\Desktop\Proyectos\vigia` (git, master). Sale del PDF "Plan de proyecto: de la investigación MoE a un emprendimiento" (2026-10-04). Núcleo `vigia/` Apache 2.0 + plataforma `vigia_ee/` (panel + API multi-tenant). Specs en `docs/spec/`, informe en `docs/informe-tecnico.md`, defectos en `docs/verificacion/defectos.md`. Research en el vault: notas `vigia-*-2026-10-04.md`.

Medido 2026-10-05: en Granite 3.1 1B-A400M (CPU), AUROC por token F1 (router) 0,891 vs entropía de salida 0,601; se sostiene dentro de cada estrato. La sonda sobre el estado oculto iguala al detector.

Medido 2026-10-08 en gpt-oss-20b (L4, informe §4.6–4.8): factual 2.600 preguntas, detector 0,773 global pero 0,723 dentro de tipo (aritmética infla); entropía 0,56. Código 1.134 tareas: detector por respuesta 0,758; la hipótesis de Granite (entropía solo en líneas de código) quedó refutada. Mejor de N: los ejemplos visibles del enunciado pesan más que Vigía (Vigía = desempate). Latencia F0+F1 en L4: +7,6 % → +4,7 % (2026-10-12, commit fbf073a: F0 en GPU, pesos del router en fp32 una vez); meta 5 % cumplida con poco margen. Modo vLLM (solo IDs): AUROC 0,742 vs 0,811 del router completo. Entropía semántica (5 muestras, 2026-10-09): 0,757 contra router 0,809; sumada no mejora de forma significativa (informe §4.9). Granite y gpt-oss se comportan distinto: calibrar por modelo.

Orden de fases decidido por el usuario (2026-10-06): 1) terminar Vigía listo para producción; 2) OS Harness (agente que opera apps por API/UI Automation/captura de pantalla); 3) agente con acceso a nivel kernel. MF-MoE (mitosis/fusión/bits dinámicos) queda como investigación.

**Why:** el usuario quiere el producto vendible primero y el agente después.
**How to apply:** no mezclar features del harness en Vigía hasta cerrarlo. Al llegar a la fase 3, repetir el aviso de riesgo (driver firmado, EDR de clientes regulados); la telemetría de gradientes quedó rechazada por privacidad (ver `docs/spec/mf-moe-v2-revision.md`). La laptop tiene 8 GB y sin GPU: gpt-oss-20b no corre local.
