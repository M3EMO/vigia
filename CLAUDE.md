# Vigía

North-star: Vigía marca en tiempo real los tokens dudosos de un modelo MoE autoalojado con señales
del router en una sola pasada. Entrega un P(error) calibrado por token, fragmentos y una acción por
respuesta. Meta del plan (docs/plan-original.txt): informe técnico, librería abierta y 3 pilotos al 2027-10-04.

- Arquitectura, flujo y autoría por módulo: `docs/spec/arquitectura.md`.
- Features canónicas: `docs/spec/matriz-features.md`. Aceptación: `docs/spec/aceptacion.md`.
- Contratos compartidos: `vigia/contracts.py` y `vigia_ee/schema.sql` (cambian solo por el orquestador).
- Research: vault global, notas `vigia-*` (índice en `docs/research/README.md`). Nunca research local.
- Convenciones: @docs/claude/convenciones.md
- Gotchas (leer bajo demanda): `docs/claude/gotchas.md`.
