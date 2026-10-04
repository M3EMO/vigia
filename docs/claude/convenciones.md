# Convenciones

- Python 3.12, venv en `.venv` (uv). Correr con `.venv/Scripts/python.exe -m pytest`.
- `vigia/` es núcleo Apache 2.0 y nunca importa `vigia_ee`. `vigia_ee/` es la plataforma.
- `vigia/contracts.py` solo usa stdlib. `vigia/features.py` es puro (sin estado).
- Tests en `tests/` plano, `test_<modulo>.py`. Los tests por defecto corren en CPU en segundos con
  modelos diminutos creados desde config (sin red). `@pytest.mark.slow` usa Granite real.
- Sin pickle para artefactos: JSON con hash SHA-256.
- Texto de UI y docs en español rioplatense técnico. Código e identificadores en inglés.
- Números en UI con locale es-AR; fechas guardadas en UTC.
- Toda consulta a datos de cliente filtra por `org_id`. Recursos de otro tenant → 404.
