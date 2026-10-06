# Revisión de la propuesta MF-MoE v2 (OS Harness, bits dinámicos, contexto extendido)

Insumo del usuario, 2026-10-06. Esta revisión separa lo que tiene evidencia, lo que entra en Vigía
y lo que cambia el producto. Revisión previa: `docs/spec/mf-moe-revision.md`.

## Veredicto por pilar

| Pilar | Qué propone | Estado de la evidencia | Destino |
|---|---|---|---|
| A. Supermatriz fractal | Mitosis y fusión offline por dominio | Sin medición; objeciones abiertas en la revisión previa | Investigación |
| B. OS Harness | Agente que opera Excel, Power BI y editores por API, UI Automation o captura del frame-buffer | Producto distinto: un agente de uso de computadora | Fuera de Vigía; decisión de producto pendiente |
| C. Bits dinámicos | Superficie ternaria (1,58 bits) y hojas en FP16/FP32 por token | Pesos ternarios exigen entrenar desde cero (familia BitNet); cuantizar a ternario un modelo existente degrada mucho la calidad | Investigación |
| D1. Contexto de 1M tokens en O(1) | Indexar el historial en las celdas y atender solo el micro-contexto | La atención exacta no es O(1); lo propuesto es recuperación o atención dispersa, con costo de recuperación | Investigación; la afirmación O(1) necesita medición |
| D2. Confianza por KL entre expertos | Divergencia cruzada en una pasada | La señal del router está medida en Vigía: AUROC 0,891 en Granite; la KL sin cota ya se corrigió a JSD | **Ya en Vigía** (F1 y `e_jsd`) |
| D3. Congelar la salida si la confianza cae | Forzar razonamiento o un entorno seguro | Equivale a las acciones de política de Vigía | **Ya en Vigía** (`abstain`, `rag_retry`, `human_review`) |
| Ventaja 4. Telemetría de gradientes | Recolectar "huellas" de gradientes de los clientes | Choca con la promesa de privacidad; ver abajo | **Rechazado** para Vigía |

## Riesgos que bloquean partes de la propuesta

1. **La telemetría de gradientes filtra datos.** Un gradiente permite reconstruir entradas de
   entrenamiento (ataques de inversión de gradientes, por ejemplo "Deep Leakage from Gradients",
   Zhu et al., NeurIPS 2019). Recolectarlos contradice el argumento de venta de Vigía: los datos
   no salen del cliente. Además, la Ley 25.326 art. 12 restringe transferencias internacionales.
   Un despliegue de solo inferencia tampoco calcula gradientes.
2. **La inyección de entrada a nivel kernel es un riesgo de seguridad.** Un driver de kernel
   necesita firma, dispara los EDR de los clientes regulados y amplía la superficie de ataque.
   Windows ya ofrece `SendInput` y UI Automation en modo usuario.
3. **"Equivalente a un modelo Frontier en una GPU de 80 GB" no tiene respaldo.** El plan del PDF
   pide no afirmar sin datos; esta frase no entra en materiales de venta ni en el informe.

## Qué sí aporta al producto

- El caso de uso del agente con OS Harness es un cliente natural de Vigía: un agente que actúa
  en Excel necesita saber cuándo dudar antes de hacer clic. Vigía puede exponer la acción
  `abstain` como señal de "no actuar" para un agente externo, sin construir el agente.
- La promoción de precisión por token sugiere una feature de investigación: correlacionar la
  confianza de Vigía con la sensibilidad del token a la cuantización. Queda como pregunta abierta.

## Decisión del usuario (2026-10-06)

El orden de trabajo es: 1) terminar Vigía listo para producción; 2) construir el OS Harness;
3) construir el agente con acceso a nivel kernel. Los riesgos de la fase 3 se revisan al empezarla.
