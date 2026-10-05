# Matriz de cumplimiento

> **Esto no es asesoramiento legal.** Esta matriz describe qué controles técnicos trae Vigía y
> cuáles quedan a cargo de quien lo despliega. No afirma que un despliegue cumpla una norma. Cada
> organización debe validar su caso con un profesional matriculado en la jurisdicción que
> corresponda. Algunas referencias vienen de fuentes secundarias y están marcadas.

## 1. Supuestos

- Vigía es software autoalojado. El cliente lo instala en su infraestructura y es el responsable
  del tratamiento (Ley 25.326) o el *controller* / *covered entity* (GDPR, HIPAA).
- El proveedor de Vigía no accede a los datos salvo que un contrato de soporte lo prevea. Ese
  acceso puede convertir al proveedor en encargado (GDPR art. 28) o en *business associate*
  (HIPAA). Esa calificación requiere opinión legal.
- Vigía no envía datos fuera del nodo por defecto. Las salidas posibles son tres, todas
  configuradas por el cliente:
  - Webhooks: llevan ids, acciones y scores; nunca texto de prompts ni respuestas.
  - Re-consulta RAG (`rag_url`): envía una consulta armada con la pregunta y la respuesta, con los
    fragmentos dudosos enmascarados.
  - Juez LLM de etiquetado: solo existe en la CLI del núcleo (`vigia label`, `vigia evidence`) y
    exige `--judge-url` explícito. La calibración de la plataforma no llama a ningún juez.

Leyenda de la columna "Vigía": **Sí** = control implementado y testeado; **Parcial** = ayuda,
pero no alcanza solo; **No** = responsabilidad del cliente.

## 2. Argentina: Ley 25.326 de Protección de Datos Personales

| Artículo | Requisito (resumen) | Vigía | Cubierto por Vigía | Responsabilidad del cliente |
|---|---|---|---|---|
| Art. 9 — Seguridad | Medidas técnicas y organizativas que garanticen seguridad y confidencialidad | Parcial | Control de acceso por roles, aislamiento por tenant, contraseñas scrypt, keys con HMAC, CSRF, CSP, auditoría con cadena de hashes, logs sin texto de clientes, redacción de PII al guardar ([seguridad](spec/seguridad.md)) | Cifrado de disco y backups, TLS, red, parches del sistema, gestión de usuarios, política de seguridad escrita |
| Art. 12 — Transferencia internacional | Prohibida hacia países sin protección adecuada, salvo excepciones | Parcial | Sin salida de datos por defecto; imagen sin red (`HF_HUB_OFFLINE=1`); webhooks sin texto; allowlist para RAG y webhooks; ADR A-16 (juez sin modelo por defecto) | Decidir y documentar cada destino externo (RAG, juez, receptor de webhooks); evaluar el país de destino; contratos con cláusulas |
| Art. 14 — Derecho de acceso | Responder al titular dentro de 10 días corridos | Parcial | Export JSON completo del tenant (panel → Datos) y lectura de cada traza; el export queda en auditoría | Identificar los datos de un titular dentro del export; responder en plazo; validar identidad del solicitante |
| Art. 16 — Rectificación, actualización y supresión | Suprimir o rectificar dentro de 5 días hábiles | Parcial | Borrado de una traza a pedido, purga automática por `retention_days`, borrado del tenant; cada borrado queda en auditoría | Localizar las trazas del titular; re-aplicar supresiones tras un restore (los backups conservan datos); rectificación (Vigía no edita trazas) |

## 3. Argentina: AAIP Resolución 47/2018 (medidas de seguridad recomendadas)

El resumen de las 8 categorías del Anexo I sale de la nota de research; no se verificó cada medida.

| Categoría | Vigía | Cubierto por Vigía | Responsabilidad del cliente |
|---|---|---|---|
| Recolección de datos | Parcial | `store_text` por tenant (`none`, `redacted` por defecto, `full`); redacción de DNI, CUIT/CUIL, CBU, tarjeta, email, teléfono e IP | Decidir el modo según la finalidad; informar a los titulares |
| Control de acceso | Sí | RBAC deny-by-default, 404 entre tenants, sesiones con timeout, bloqueo de cuenta, keys con scopes, vencimiento y revocación inmediata | Alta y baja de usuarios; revisión periódica de permisos |
| Control de cambios | Parcial | Políticas y detectores versionados; cada cambio queda en auditoría; migraciones atómicas | Procedimiento de cambios; pruebas antes de actualizar |
| Respaldo y recuperación | Parcial | Backup consistente con la API de SQLite y verificación de integridad y de la cadena de auditoría ([operación §6](operacion.md#6-backups-y-restore)) | Programar, cifrar, guardar fuera del nodo y probar restores |
| Gestión de vulnerabilidades | Parcial | Verificación adversarial con tests de regresión ([defectos](verificacion/defectos.md)); imagen mínima no-root | Escaneo de dependencias e imagen; aplicar actualizaciones |
| Destrucción de la información | Parcial | Purga por retención, supresión puntual, borrado de tenant | `VACUUM` si exige borrado físico; destrucción de backups y discos |
| Incidentes de seguridad | Parcial | Auditoría exportable por rango; logs JSON con `request_id`; métricas de fallos de auth | Plan de respuesta y notificación |
| Entornos de desarrollo | Parcial | Modelos diminutos y datos sintéticos para pruebas, sin datos reales | No copiar bases de producción a desarrollo |

## 4. Estados Unidos: HIPAA, 45 CFR 164.312 (salvaguardas técnicas)

| Estándar | Tipo | Vigía | Cubierto por Vigía | Responsabilidad del cliente |
|---|---|---|---|---|
| (a)(1) Control de acceso — identificación única de usuario | Required | Sí | Cada usuario tiene su cuenta y su email único; cada key tiene su id | Prohibir cuentas compartidas |
| (a)(2)(ii) Acceso de emergencia | Required | No | — | Procedimiento de acceso de emergencia (por ejemplo, un owner de resguardo) |
| (a)(2)(iii) Cierre automático de sesión | Addressable | Sí | 30 minutos de inactividad y 8 horas absolutas, en servidor | Bloqueo de pantalla en los equipos |
| (a)(2)(iv) Cifrado y descifrado | Addressable | No | — | Cifrado del disco y de los backups |
| (b) Controles de auditoría | Standard | Sí | Auditoría append-only con cadena de hashes por org; export por rango | Revisar la auditoría; conservarla fuera del nodo |
| (c)(1) Integridad | Standard | Parcial | Triggers append-only; cadena de hashes; detectores con hash SHA-256 | Controles de integridad del almacenamiento |
| (d) Autenticación de persona o entidad | Standard | Parcial | Contraseña de 15+ caracteres, keys HMAC; sin MFA (pendiente) | MFA en el proxy o SSO delante, si se exige |
| (e)(1) Seguridad en la transmisión | Standard | Parcial | Cookie `Secure`, HSTS; webhooks con https obligatorio | Terminación TLS delante de Vigía; red interna hacia vLLM |

La desidentificación por Safe Harbor (164.514) exige quitar 18 identificadores. La redacción de
Vigía no cubre nombres ni fechas: no alcanza para Safe Harbor. Con PHI, usá `store_text=none`.

Según fuentes secundarias, la regla que haría obligatorios el cifrado y el MFA no estaba vigente a
octubre de 2026 [no verificado en el Federal Register].

## 5. Unión Europea: GDPR

| Artículo | Requisito (resumen) | Vigía | Cubierto por Vigía | Responsabilidad del cliente |
|---|---|---|---|---|
| Art. 28 — Encargado del tratamiento | El encargado trata solo bajo instrucciones, garantiza confidencialidad, aplica art. 32, asiste con derechos, borra o devuelve al terminar, permite auditorías | Parcial | Autoalojado: el proveedor no recibe datos por defecto; export y borrado del tenant para el offboarding; auditoría exportable | Firmar un DPA si el soporte accede a datos; listar subencargados (RAG, juez, receptores de webhooks) |
| Art. 32 — Seguridad del tratamiento | Seudonimización y cifrado, confidencialidad, integridad, disponibilidad y resiliencia, restauración, pruebas periódicas | Parcial | Redacción de PII, RBAC, aislamiento, auditoría, health checks, backup verificable, tests de seguridad automatizados | Cifrado, alta disponibilidad, restores probados, evaluación de riesgos documentada |
| Art. 33 — Notificación de brechas (referencia) | El responsable notifica en 72 horas | Parcial | Auditoría y logs para reconstruir el incidente | Detectar, evaluar y notificar |

## 6. Unión Europea: AI Act, artículo 50 (transparencia)

El art. 50 obliga a informar a las personas que interactúan con un sistema de IA y a marcar el
contenido sintético. Rige desde el 2 de agosto de 2026. Fuentes secundarias indican un plazo hasta
el 2 de diciembre de 2026 para el marcado de sistemas lanzados antes [no verificado en el Diario
Oficial].

| Obligación | Vigía | Cubierto por Vigía | Responsabilidad del cliente |
|---|---|---|---|
| Informar que el usuario interactúa con IA | No | Vigía no es la interfaz con el usuario final | Mostrar el aviso en su aplicación |
| Marcar contenido sintético en formato legible por máquina | Parcial | Cada respuesta lleva `trace_id`, `calibration_id`, acción y scores en el campo `vigia`; la traza queda guardada | Incluir el marcado en el contenido que publica |
| Transparencia sobre la confiabilidad | Parcial | P(error) calibrado, fragmentos marcados, `score: null` sin calibración, reporte con AUROC e intervalos | Explicar al usuario que la ausencia de marca no prueba que la respuesta sea correcta |

Vigía no es por sí mismo un sistema de alto riesgo del Anexo III. Un cliente que lo integra en
scoring crediticio, seguros, beneficios públicos o justicia puede estar alcanzado. En ese caso,
el reporte de evaluación y el [informe técnico](informe-tecnico.md) sirven como insumo de su
documentación técnica, con los límites que el informe declara.

## 7. Qué entrega el proyecto para una auditoría

- Este documento y [`docs/spec/seguridad.md`](spec/seguridad.md) con cada control y su test.
- [`docs/operacion.md`](operacion.md): backups, retención, rotación y troubleshooting.
- Export de la auditoría por rango desde el panel (`/audit/export`) con su verificación.
- Export JSON del tenant (`/data/export`).
- Reporte de evaluación del detector (`report.html`) y el informe técnico.

Pendiente: un modelo de DPA para el plan con soporte y un texto de aviso de uso de IA para que el
cliente adapte.
