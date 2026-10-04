# Revisión crítica de MF-MoE y mapeo al producto

Fuente: `docs/mf-moe-plan.md`. Esta revisión separa lo que entra en Vigía de lo que queda como
investigación. Cada objeción nombra el dato o el experimento que falta.

## Mapeo

| Fase MF-MoE | Destino | Implementación en Vigía |
|---|---|---|
| 1. Señal primaria | Producto, P0 | Pre-hooks en bloques MoE, F1, detector, AUROC y overhead (AT-M11, AT-M12) |
| 2. Contracción | Investigación | Fuera del producto; protocolo corregido abajo |
| 3. Mitosis | Investigación | Fuera del producto; protocolo corregido abajo |
| 4. Consenso | Producto, P1 | Feature `e_jsd` con JSD normalizada (opt-in, nivel F2) |
| §4 Pool en VRAM / Triton | Investigación | Solo aplica si existen las fases 2 y 3 |

## Objeciones

1. **`Confianza = 1 − KL` no está acotada.** KL vive en `[0, ∞)` y no es simétrica, así que la
   confianza puede dar negativa y depende del orden. Vigía usa la divergencia Jensen-Shannon
   generalizada de k distribuciones ponderadas por el router, dividida por `log k`. El resultado
   cae en `[0, 1]`. Además, Vigía no muestra `1 − JSD` como confianza: la JSD entra al detector y la
   salida es un P(error) calibrado.
2. **Las "distribuciones de tokens de cada experto" no existen en el modelo.** Un experto devuelve
   un vector oculto, no una distribución sobre el vocabulario. Vigía proyecta la contribución de
   cada experto elegido por la norma final y la cabeza de salida (*logit lens*). Esa proyección
   cuesta `k × |V| × d` por token, así que solo corre en la última capa MoE y es opt-in.
3. **La entropía del router no mide saturación del experto.** `H(R_t)` mide la duda del router
   entre expertos para el token t. Un experto saturado con un router seguro da entropía baja. El
   disparador de mitosis necesita una señal del propio experto, por ejemplo la pérdida o la norma
   del gradiente de los tokens que recibe.
4. **El criterio de fusión cambia entre secciones.** §2.B usa coseno de gradientes. §3 Fase 2 usa
   la covarianza de uso. Son criterios distintos: dos expertos pueden co-activarse sin ser
   redundantes. El protocolo debe fijar uno y comparar contra la fusión por similitud de pesos o
   de salidas (HC-SMoE, MC-SMoE).
5. **El control de la Fase 3 no iguala parámetros.** Dos clones más un micro-router tienen el
   doble de parámetros que el experto monolítico. Una convergencia 25 % más rápida puede venir
   solo de la capacidad extra. El control justo es un experto monolítico con el mismo número de
   parámetros o con el mismo presupuesto de FLOPs.
6. **Las huellas topológicas por sector no son medibles como están.** "Árbol profundo",
   "red plana" y "esponjosa" no tienen métrica. Una versión testeable mide, por dominio, la
   distribución de uso de expertos por capa y su distancia entre dominios (JSD de uso).
   Vigía ya registra el uso capa × experto en cada traza, así que ese dato sale gratis del panel.
7. **El ruido `N(0, 1e-5)` puede no romper la simetría.** Con optimizadores adaptativos el ruido
   chico diverge, pero con un micro-router recién inicializado los dos clones reciben el mismo
   gradiente al inicio. Hay que medir la divergencia de los clones en los primeros pasos.
8. **"Precisión milimétrica en rojo" choca con la evidencia de UX.** Un estudio con n=192 muestra
   que la incertidumbre por token aumenta el acuerdo del usuario con la IA. Vigía usa una rampa de
   un solo tono (no rojo-verde), tres codificaciones por fragmento y una acción obligatoria.

## Protocolo mínimo para retomar las fases 2 y 3

- Hardware: 1 GPU de 80 GB, como pide el plan. Esta laptop no alcanza (8 GB, sin GPU).
- Fase 2: fijar el criterio de fusión, medir memoria y perplejidad en dominio y fuera de dominio,
  comparar contra cuantización (el plan del PDF ya marca este riesgo).
- Fase 3: control con parámetros iguales, 3 semillas, curva de pérdida con IC.
- Dato de partida: el uso capa × experto por dominio que Vigía ya guarda.
