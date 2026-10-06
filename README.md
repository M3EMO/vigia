# Vigía

Vigía marca en tiempo real los tokens dudosos de un modelo MoE (*mixture of experts*) autoalojado.
Vigía entrega un P(error) calibrado por token, los fragmentos dudosos y una acción por respuesta.
Vigía no llama a otro modelo: usa señales que el modelo ya calcula en la misma pasada.

## El problema

Un LLM afirma cosas falsas con el mismo tono que las verdaderas. La entropía de salida es la
señal barata habitual, pero separa mal los errores. Los detectores más fuertes cuestan una o más
pasadas extra. Un sector regulado (finanzas, salud, legal, gobierno) necesita una señal barata,
local y auditable.

## Cómo funciona

Vigía engancha el bloque MoE de cada capa y reduce cada paso de generación a un vector de features.

- **F0 (salida):** entropía sobre el vocabulario, NLL del token emitido, margen top-1/top-2 y
  entropía top-20.
- **F1 (router):** entropía y márgenes de los logits del router, masa top-k, uso acumulado de
  expertos y cambio de ruta, por capa y agregados entre capas.

Un detector lineal (regresión logística con z-score y Platt) convierte esas features en
P(error). El umbral sale de *conformal risk control* sobre la tasa de falsos negativos por token.
El usuario elige esa tasa con el parámetro alpha. Sin detector calibrado, Vigía devuelve
`p_error: null` y no resalta nada.

## Resultado medido

El informe completo está en [`docs/informe-tecnico.md`](docs/informe-tecnico.md). Es un borrador
con datos medidos.

| Detector (Granite 3.1 1B-A400M, 600 preguntas en español) | AUROC por token | IC 95 % |
|---|---|---|
| Entropía de salida (baseline) | 0,601 | 0,547–0,649 |
| F1 (solo router) | 0,891 | 0,808–0,969 |
| F0+F1 (por defecto) | 0,906 | 0,830–0,980 |
| Sonda lineal sobre el estado oculto (baseline) | 0,906 | 0,830–0,979 |

Límites de este resultado, sin suavizar:

- El modelo es chico y se equivoca en el 84 % de las preguntas. La prevalencia de error por token
  en test es 92 %.
- La etiqueta es por respuesta, no por fragmento. Un token de relleno de una respuesta errónea
  cuenta como error.
- La sonda lineal sobre el estado oculto iguala al detector. El router no aporta información
  nueva; la aporta con features más baratas.
- El test tiene 120 respuestas. Los intervalos son anchos.
- La latencia agregada en Granite (CPU) quedó entre +3,6 % y +8,5 %, con un intervalo ancho por
  el ruido de la laptop. La meta del plan es < 5 % y todavía no está demostrada.
- gpt-oss-20b en GPU T4: F0+F1 0,881 contra 0,643 de la entropía, con +3,0 % de latencia (ver informe §4.4).

## Instalación

Vigía necesita Python 3.11 o superior (el desarrollo usa 3.12). Los extras eligen qué se instala.

| Extra | Instala | Para qué |
|---|---|---|
| (ninguno) | numpy | Detector y evaluación sobre features ya extraídas |
| `hf` | torch, transformers ≥ 5, safetensors, accelerate | Extraer señales de un modelo de Hugging Face |
| `serve` | fastapi, uvicorn, httpx | `vigia serve` y el modo vLLM |
| `ee` | jinja2, python-multipart | Plataforma multi-tenant con panel (licencia comercial) |
| `dev` | pytest, ruff, scikit-learn, playwright, openai, axe | Tests y e2e |

Con uv (torch solo CPU):

```bash
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[hf,serve]"          # núcleo abierto
uv pip install -e ".[hf,serve,ee]"       # núcleo + plataforma
```

Con pip:

```bash
python -m venv .venv && . .venv/bin/activate    # en Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[hf,serve]"
```

La instalación editable (`-e`) evita un problema abierto: el empaquetado todavía no declara los
archivos que no son Python de `vigia_ee/` (esquema SQL, plantillas y estáticos).

Comprobá la instalación:

```bash
vigia models        # lista las arquitecturas soportadas y los modelos diminutos
```

## Quickstart de la librería

### Con el modelo diminuto (sin red, segundos de CPU)

`tiny:mixtral` es un MoE de 2 capas y 4 expertos con pesos aleatorios fijos. Sirve para probar el
circuito completo. Su texto no tiene sentido.

Este script arma 40 respuestas dadas. La respuesta greedy del modelo cuenta como correcta y un
texto al azar cuenta como erróneo:

```python
# demo_data.py
import json
import random

from vigia.cli import make_tiny_backend

backend = make_tiny_backend("mixtral")
rng = random.Random(0)
with open("demo.jsonl", "w", encoding="utf-8") as fh:
    for i in range(40):
        messages = [{"role": "user", "content": f"Pregunta de prueba {i}"}]
        if i % 2 == 0:
            stream = backend.stream(messages, max_tokens=10)
            list(stream)
            response, label = stream.text, 0
        else:
            response, label = "".join(rng.choice("abcdefghij ") for _ in range(10)), 1
        item = {"id": f"d{i}", "messages": messages, "response": response, "label": label}
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
```

```bash
python demo_data.py
vigia extract   --model tiny:mixtral --data demo.jsonl --out run/   # rows.jsonl + features.npz
vigia calibrate --features run/                                      # detector.json + split.json
vigia evaluate  --features run/ --n-boot 200                         # report.json + report.html
```

`report.html` trae AUROC con IC por bootstrap de pregunta, calibración y los baselines. El formato
de cada item del JSONL está en el docstring de `vigia/datasets.py`.

### Con Granite 3.1 1B-A400M (CPU, ~3 GB de RAM)

El experimento del informe usa preguntas de MKQA en español (Apple, CC-BY 3.0). Vigía espera
`.cache/data/mkqa_es.jsonl` con una línea `{id, question, answers, type}` por pregunta. Este
script convierte el `mkqa.jsonl.gz` original:

```python
import gzip
import json

TYPES = {"entity", "date", "number", "number_with_unit", "short_phrase"}
with (
    gzip.open("mkqa.jsonl.gz", "rt", encoding="utf-8") as src,
    open(".cache/data/mkqa_es.jsonl", "w", encoding="utf-8") as dst,
):
    for line in src:
        row = json.loads(line)
        first = row["answers"]["es"][0]
        extra = first.get("aliases") or []
        extra = [extra] if isinstance(extra, str) else extra
        answers = list(dict.fromkeys(a for a in [first.get("text"), *extra] if a))
        if first["type"] in TYPES and answers:
            item = {
                "id": str(row["example_id"]),
                "question": row["queries"]["es"],
                "answers": answers,
                "type": first["type"],
            }
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
```

```bash
vigia evidence --model ibm-granite/granite-3.1-1b-a400m-instruct --dtype bfloat16 \
  --mkqa 300 --out runs/granite --max-tokens 24 --tiers F0,F1,H --n-boot 2000
```

`vigia evidence` hace extract, calibrate y evaluate en una corrida. Es reanudable: el mismo comando
continúa donde quedó. En una laptop i5 con 8 GB tarda unas 2 horas. La primera corrida descarga
el modelo de Hugging Face.

También existen `vigia label` (re-etiqueta con reglas computables o con un juez
OpenAI-compatible; la key va en `VIGIA_JUDGE_API_KEY`) y `vigia bench` (A/B de latencia con hooks
prendidos y apagados).

## Quickstart del servidor single-tenant (`vigia serve`)

`vigia serve` es parte del núcleo abierto. Sirve una API compatible con OpenAI para un solo
cliente. Las trazas viven en memoria (últimas 1000).

```bash
export VIGIA_SERVER_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
vigia serve --model tiny:mixtral --detector run/detector.json \
  --api-key-env VIGIA_SERVER_KEY --port 8000
```

- Sin `--api-key-env`, el servidor solo escucha en loopback y rechaza requests de navegador de
  otro sitio.
- La key debe tener al menos 16 caracteres.
- `--policy politica.json` carga una política (campos de `vigia.contracts.Policy`).

## Quickstart de la plataforma (`vigia_ee`)

La plataforma agrega organizaciones, usuarios, API keys, panel web, cola de revisión, calibración
gestionada, auditoría, retención y webhooks. Corre en un solo proceso que hospeda el modelo.

```bash
python -c "import secrets; print('k1:' + secrets.token_urlsafe(48))" > peppers.txt
printf '%s\n' 'una frase larga que no contenga el email' > owner_password.txt
chmod 600 peppers.txt owner_password.txt

export VIGIA_MODEL=tiny:mixtral              # o un id/ruta HF; ver docs/operacion.md
export VIGIA_DATA_DIR=./var
export VIGIA_PEPPERS_FILE=./peppers.txt
export VIGIA_BASE_DETECTOR=./run/detector.json   # opcional: detector inicial de cada org nueva

VIGIA_OWNER_PASSWORD_FILE=./owner_password.txt \
  python -m vigia_ee.app init --org "Demo SA" --email owner@demo.com.ar
# Organización creada: <ORG_ID>

python -m vigia_ee.app create-key --org-id <ORG_ID> --name demo --scopes ingest,read
# imprime el token vg_live_... una sola vez

python -m vigia_ee.app serve --port 8000     # panel en http://127.0.0.1:8000/login
```

La contraseña del owner necesita entre 15 y 128 caracteres y no puede contener el usuario del
email. `vigia-platform` es el mismo comando que `python -m vigia_ee.app`.

### Cliente con el SDK de OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="vg_live_...")
resp = client.chat.completions.create(
    model="tiny:mixtral",
    messages=[{"role": "user", "content": "¿En qué año llegó el hombre a la Luna?"}],
    max_tokens=64,
    extra_body={"vigia": {}},  # sin esto, la respuesta es OpenAI pura
)
vigia = resp.choices[0].model_extra["vigia"]
print(vigia["action"])  # pass | annotate | human_review | rag_retry | abstain
print(vigia["response_p_error"], vigia["calibration_id"], vigia["mode"])
for span in vigia["spans"]:
    print(span)  # fragmentos marcados: offsets y score
for tok in vigia["tokens"]:
    print(tok["text"], tok["p_error"], tok["flag"])
```

- `extra_body={"vigia": {"alpha": 0.1, "tiers": ["F0", "F1"]}}` cambia alpha y niveles por request.
- El streaming SSE lleva los tokens de cada chunk y el chunk final lleva fragmentos y acción.
- Un cliente que descarta campos extra (LangChain) lee la traza con
  `GET /v1/vigia/traces/{trace_id}`. Esa ruta pide una key con scope `read`.

## Modelos soportados

| `config.model_type` | Familia |
|---|---|
| `gpt_oss` | gpt-oss 20b y 120b (formato harmony; solo se puntúa el canal `final`) |
| `mixtral` | Mixtral |
| `qwen2_moe`, `qwen3_moe` | Qwen MoE (el experto compartido queda fuera de F1) |
| `olmoe` | OLMoE |
| `granitemoe` | Granite MoE (medido en el informe) |
| `deepseek_v3` | DeepSeek V3 (router sigmoide por grupos) |
| `phimoe` | Phi MoE |

Un modelo denso o un `model_type` sin adaptador falla con un error que lista las arquitecturas.
Solo Granite y los modelos diminutos tienen corridas reales; el resto tiene tests sobre configs
diminutas de cada arquitectura.

## Arquitectura en 10 líneas

1. Un `forward_pre_hook` sobre cada bloque MoE recalcula los logits del router en float32 (ADR A-01).
2. El extractor reduce cada paso a escalares por token; nunca guarda tensores por experto.
3. El detector es un JSON con hash SHA-256, sin pickle y sin scikit-learn en runtime (A-04).
4. El umbral τ(alpha) sale de *conformal risk control* sobre la FNR por token (A-05).
5. `vigia.spans` agrupa subtokens en palabras y palabras marcadas en fragmentos.
6. `vigia.policy` elige la acción: abstain > rag_retry > human_review > annotate > pass.
7. `vigia.api` sirve la API OpenAI con el campo `vigia` por choice y por chunk (A-09).
8. La plataforma es un proceso: modelo, API, panel, worker de jobs y SQLite con WAL (A-07, A-10).
9. Con vLLM, Vigía actúa como proxy en modo reducido: IDs de expertos y logprobs top-20 (A-08).
10. `vigia/` nunca importa `vigia_ee/`; un test lo verifica (A-13).

Más detalle: [arquitectura](docs/spec/arquitectura.md), [features](docs/spec/matriz-features.md),
[aceptación](docs/spec/aceptacion.md), [API por módulo](docs/spec/api-modulos.md),
[seguridad](docs/spec/seguridad.md), [operación](docs/operacion.md) y
[cumplimiento](docs/cumplimiento.md).

## Licencias

- `vigia/` (núcleo: extractor, detector, evaluación, CLI, `vigia serve`) usa
  [Apache License 2.0](LICENSE).
- `vigia_ee/` (plataforma: panel, multi-tenant, calibración gestionada) usa una licencia comercial.
  Ver [`vigia_ee/LICENSE`](vigia_ee/LICENSE). Su uso requiere un contrato.
- Cada modelo tiene su propia licencia. Revisala antes de usarlo.

## Estado y límites

- Estado: versión 0.1.0, sin pilotos en producción. Los tests de aceptación corren en CPU
  ([`docs/spec/aceptacion.md`](docs/spec/aceptacion.md)). Los defectos abiertos están en
  [`docs/verificacion/defectos.md`](docs/verificacion/defectos.md).
- gpt-oss no se probó en GPU. Su soporte se probó sobre una config diminuta con formato harmony.
- El modo vLLM es reducido: sin logits del router no hay F1, solo F0 parcial y uso de expertos (U).
  El streaming llega al final del request porque vLLM entrega los expertos al terminar.
- La plataforma con backend HF corre solo en CPU y genera de a una respuesta por vez. Las demás
  esperan en cola hasta 120 s; después reciben 503.
- La latencia agregada no tiene medición limpia en un modelo real (ver arriba).
- SQLite limita la plataforma a un nodo. El rate limiting vive en memoria del proceso.
- La redacción de PII cubre formatos argentinos y de contacto. No cubre nombres ni fechas.
- La ausencia de marca no prueba que una respuesta sea correcta.
