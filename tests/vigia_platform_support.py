"""Helpers compartidos por los tests de composición de la plataforma (runtime, calibración, app).

No es un conftest: cada test lo importa explícitamente. El backend diminuto se crea una vez por
proceso y lo comparten todos los Runtime de los tests (carga rápida, sin red, sin checkpoints).
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

from vigia.contracts import ModelInfo
from vigia.detector import Detector
from vigia_ee.auth import PepperRing, Principal, register_user
from vigia_ee.runtime import Runtime, Settings

PEPPERS = PepperRing.parse("k1:" + "p" * 40)
PASSWORD = "una frase de prueba bastante larga 2026"  # noqa: S105 - contraseña de usuarios de prueba
FAST_SCRYPT_N = 2**10


@cache
def shared_backend(arch: str = "mixtral") -> Any:
    from vigia.cli import make_tiny_backend

    return make_tiny_backend(arch)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = {"data_dir": tmp_path, "model": "tiny:mixtral", "peppers": PEPPERS}
    kwargs.update(overrides)
    return Settings(**kwargs)


def make_runtime(tmp_path: Path, *, backend: Any = None, **overrides: Any) -> Runtime:
    runtime = Runtime(make_settings(tmp_path, **overrides), backend=backend or shared_backend())
    runtime.auth.scrypt_n = FAST_SCRYPT_N  # scrypt real pero barato: los tests no miden costo de hash
    return runtime


def add_user(runtime: Runtime, org_id: str, email: str, role: str) -> Principal:
    user_id = register_user(runtime.auth, org_id, email, PASSWORD, role)
    return Principal(org_id=org_id, role=role, user_id=user_id)


def constant_detector(
    info: ModelInfo,
    p: float = 0.9,
    *,
    alphas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3),
    taus: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5),
    model: dict[str, Any] | None = None,
) -> Detector:
    """Detector válido que devuelve `p` en cada token (peso 0, sesgo logit(p))."""
    return Detector.build(
        model=model or info.fingerprint(),
        feature_names=["out_entropy"],
        mean=[0.0],
        std=[1.0],
        weights=[0.0],
        bias=math.log(p / (1 - p)),
        platt={"a": 1.0, "b": 0.0},
        response_calibrator={"a": 1.0, "b": 0.0},
        thresholds=[
            {"alpha": a, "tau": t, "expected_flag_rate": 0.5, "expected_precision": None}
            for a, t in zip(alphas, taus, strict=True)
        ],
        default_alpha=0.1 if min(alphas) <= 0.1 else min(alphas),
        metrics={},
        training={
            "dataset_sha256": "0" * 64,
            "n_tokens": 100,
            "n_responses": 10,
            "label_source": "computable",
            "judge": None,
            "created_at": "2026-10-04T00:00:00+00:00",
        },
    )


def install_detector(runtime: Runtime, org_id: str, detector: Detector) -> None:
    repo = runtime.repo(org_id)
    repo.insert_detector(detector.id, detector.to_json())
    repo.activate_detector(detector.id)
    runtime.invalidate_detector(org_id)


def learnable_jsonl(n: int = 40, *, prefix: str = "q") -> bytes:
    """Dataset `response` + `label` separable para el modelo diminuto (texto común contra símbolos)."""
    lines = []
    for i in range(n):
        response = "la casa es azul y grande" if i % 2 == 0 else "9 7 3 # @ ! 8 2 % & 5 0"
        lines.append(
            json.dumps(
                {"id": f"{prefix}{i}", "question": f"pregunta {i}", "response": response, "label": i % 2}
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
