"""Soporte de los tests del panel: Runtime real con un backend falso rápido, usuarios y sesiones.

El backend falso emite tokens con `out_nll` y expertos fijos; el detector a mano los convierte en
P(error) = sigmoid(2 * out_nll - 2), así "18" y "20" quedan marcados con tau = 0,5.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vigia.contracts import Capabilities, ModelInfo, Tier, TokenEvent
from vigia.detector import Detector
from vigia_ee.auth import PepperRing, install_auth, register_user
from vigia_ee.panel import install_panel
from vigia_ee.runtime import Runtime, Settings
from vigia_ee.security import install_security

PEPPERS = PepperRing.parse("k1:" + "p" * 40)
SCRYPT_N = 2**10  # mínimo que acepta auth._parse_hash; el costo real se prueba en test_auth
PW = "una frase larga y propia para entrar al panel"
INFO = ModelInfo(
    model_id="tiny-panel",
    model_type="mixtral",
    n_layers=2,
    moe_layers=(0, 1),
    n_experts=4,
    top_k=2,
    vocab_size=100,
    gating="softmax_topk",
)
ANSWER = ["Belgrano", " nació", " en", " 18", "20", " en", " Rosario", "."]
NLL = {"18": 3.0, "20": 1.5}  # resto: 0,5 -> p = 0,27
CSRF_RE = re.compile(r'name="csrf_token" value="([0-9a-f]{64})"')


class PanelStream:
    def __init__(self, pieces: list[str], max_tokens: int) -> None:
        self.pieces = pieces[:max_tokens]
        self.finish_reason = None
        self.text = ""
        self.prompt_tokens = 3
        self.completion_tokens = 0
        self.timings = {"prefill_s": 0.0, "decode_s": 0.0, "total_s": 0.0}
        self._it = self._gen()

    def _gen(self) -> Iterator[TokenEvent]:
        pos = 0
        for idx, piece in enumerate(self.pieces):
            nll = NLL.get(piece.strip(), 0.5)
            yield TokenEvent(
                idx,
                10 + idx,
                piece,
                pos,
                pos + len(piece),
                {"out_nll": nll, "out_entropy": nll / 2},
                expert_ids=[[idx % 4, (idx + 1) % 4], [(idx + 2) % 4, 3]],
            )
            pos += len(piece)
            self.text += piece
            self.completion_tokens += 1
        self.finish_reason = "stop" if len(self.pieces) == len(ANSWER) else "length"

    def __iter__(self) -> Iterator[TokenEvent]:
        return self

    def __next__(self) -> TokenEvent:
        return next(self._it)

    def close(self) -> None:
        self._it.close()


class PanelBackend:
    def __init__(self, pieces: list[str] | None = None) -> None:
        self.pieces = pieces or ANSWER

    def model_info(self) -> ModelInfo:
        return INFO

    def capabilities(self) -> Capabilities:
        return Capabilities(tiers=(Tier.F0, Tier.F1), streaming=True)

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        return 3

    def stream(self, messages, *, max_tokens, temperature=0.0, top_p=1.0, seed=None, stop=None, tiers=()):
        return PanelStream(self.pieces, max_tokens)


def make_detector(*, min_alpha: float = 0.1) -> Detector:
    return Detector.build(
        model=INFO.fingerprint(),
        feature_names=["out_nll"],
        mean=[0.0],
        std=[1.0],
        weights=[2.0],
        bias=-2.0,
        platt={"a": 1.0, "b": 0.0},
        response_calibrator={"a": 1.0, "b": 0.0},
        thresholds=[
            {"alpha": min_alpha, "tau": 0.5, "expected_flag_rate": 0.25, "expected_precision": 0.6},
            {"alpha": 0.3, "tau": 0.7, "expected_flag_rate": 0.1, "expected_precision": 0.8},
        ],
        default_alpha=min_alpha,
        metrics={},
        training={
            "dataset_sha256": "0" * 64,
            "n_tokens": 100,
            "n_responses": 20,
            "label_source": "computable",
            "judge": None,
            "created_at": "2026-10-04T00:00:00+00:00",
        },
    )


def make_runtime(tmp_path, **settings_kw: Any) -> Runtime:
    kw: dict[str, Any] = {
        "data_dir": tmp_path,
        "model": "tiny:mixtral",
        "secure_cookies": False,
        "peppers": PEPPERS,
    }
    kw.update(settings_kw)
    runtime = Runtime(Settings(**kw), backend=PanelBackend())
    runtime.auth.scrypt_n = SCRYPT_N
    runtime.auth.login_ip_limiter = None
    return runtime


def make_app(runtime: Runtime) -> FastAPI:
    """La misma composición de seguridad que `vigia_ee.app`, sin worker ni métricas."""
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    install_security(app, max_body_bytes=runtime.settings.max_upload_bytes + 65_536)
    install_auth(app, runtime.auth)
    install_panel(app, runtime)
    return app


def activate_detector(runtime: Runtime, org_id: str, detector: Detector | None = None) -> Detector:
    det = detector or make_detector()
    repo = runtime.repo(org_id)
    repo.insert_detector(det.id, det.to_json())
    repo.activate_detector(det.id)
    runtime.invalidate_detector(org_id)
    return det


def add_user(runtime: Runtime, org_id: str, role: str, email: str | None = None) -> str:
    email = email or f"{role}-{org_id[:6]}@ejemplo.com.ar"
    register_user(runtime.auth, org_id, email, PW, role)
    return email


def login(app: FastAPI, email: str, password: str = PW) -> TestClient:
    client = TestClient(app, base_url="https://testserver")
    resp = client.post(
        "/login", data={"email": email, "password": password, "next": "/"}, follow_redirects=False
    )
    assert resp.status_code == 303, resp.text
    return client


def csrf_of(client: TestClient, path: str = "/account") -> str:
    match = CSRF_RE.search(client.get(path).text)
    assert match, "la página no tiene token CSRF"
    return match.group(1)


def principal_for(runtime: Runtime, email: str):
    from vigia_ee.auth import Principal
    from vigia_ee.db import lookup_user_by_email

    user = lookup_user_by_email(runtime.db, email)
    return Principal(org_id=user["org_id"], role=user["role"], user_id=user["id"])


def play(runtime: Runtime, email: str, prompt: str = "¿Dónde nació Belgrano?") -> dict[str, Any]:
    """Genera por el playground del runtime y devuelve el evento `done`."""
    events = list(
        runtime.playground_stream(
            principal_for(runtime, email), [{"role": "user", "content": prompt}], max_tokens=16
        )
    )
    done = [e for e in events if e["type"] == "done"]
    assert done, events
    return done[0]
