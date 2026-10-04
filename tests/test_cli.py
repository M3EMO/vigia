"""CLI de Vigía: extract -> calibrate -> evaluate sobre el modelo diminuto (AT-A10, P-16)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import torch
from test_evidence import synthetic_items

from vigia import cli, labels
from vigia.cli import JUDGE_KEY_ENV, main, make_tiny_backend
from vigia.datasets import write_jsonl
from vigia.detector import Detector
from vigia.evidence import FeatureSet
from vigia.harmony import is_scorable

ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - el test verifica que sin API key se rechace


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "items.jsonl"
    write_jsonl(path, synthetic_items(make_tiny_backend("mixtral")))
    return path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_at_a10_extract_calibrate_evaluate(dataset, tmp_path, capsys):
    """AT-A10: cada comando sale con código 0 y deja sus artefactos."""
    run_dir = tmp_path / "run"
    code, out, err = run(
        capsys,
        "extract",
        "--model",
        "tiny:mixtral",
        "--data",
        str(dataset),
        "--out",
        str(run_dir),
        "--max-tokens",
        "8",
    )
    assert code == 0, err
    assert (run_dir / "rows.jsonl").is_file() and (run_dir / "features.npz").is_file()
    assert "30 respuestas" in out and "[30/30]" in err
    fs = FeatureSet.load(run_dir / "features.npz")
    assert fs.model_info.model_id == "tiny:mixtral" and any(n.startswith("h_") for n in fs.names)

    code, out, err = run(capsys, "calibrate", "--features", str(run_dir / "features.npz"))
    assert code == 0, err
    det = Detector.load(run_dir / "detector.json")
    assert (run_dir / "split.json").is_file() and det.id in out
    assert not any(n.startswith("h_") for n in det.feature_names)

    code, out, err = run(capsys, "evaluate", "--features", str(run_dir), "--n-boot", "100")
    assert code == 0, err
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["detector_id"] == det.id and (run_dir / "report.html").is_file()
    assert report["detector"]["token"]["auroc"]["ci"][0] is not None
    assert "AUROC por token" in out


def test_label_recomputes_and_judge_key_only_from_env(dataset, tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "run"
    assert (
        run(
            capsys,
            "extract",
            "--model",
            "tiny:mixtral",
            "--data",
            str(dataset),
            "--out",
            str(run_dir),
            "--max-tokens",
            "4",
            "--limit",
            "4",
            "--quiet",
            "--tiers",
            "F0,F1",
        )[0]
        == 0
    )
    code, out, _ = run(capsys, "label", "--run", str(run_dir))
    assert code == 0 and json.loads(out)["label_source"] == "computable"

    monkeypatch.delenv(JUDGE_KEY_ENV, raising=False)
    code, _, err = run(
        capsys,
        "label",
        "--run",
        str(run_dir),
        "--judge-url",
        "http://127.0.0.1:9/v1",
        "--judge-model",
        "juez",
    )
    assert code == 1 and JUDGE_KEY_ENV in err
    code, _, err = run(capsys, "label", "--run", str(run_dir), "--judge-url", "http://127.0.0.1:9/v1")
    assert code == 1 and "--judge-model" in err

    secret = "sk-juez-de-prueba-123"  # noqa: S105 - clave falsa de prueba
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        content = json.dumps({"label": 0, "hallucinated_spans": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    real = labels.OpenAICompatibleJudge

    def judge_with_mock(base_url, api_key, model, **kw):
        return real(base_url, api_key, model, client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr("vigia.labels.OpenAICompatibleJudge", judge_with_mock)
    monkeypatch.setenv(JUDGE_KEY_ENV, secret)
    code, out, err = run(
        capsys,
        "label",
        "--run",
        str(run_dir),
        "--judge-url",
        "http://127.0.0.1:9/v1",
        "--judge-model",
        "juez",
    )
    assert code == 0, err
    report = json.loads(out)
    assert report["label_source"] == "judge" and report["judge"]["model"] == "juez"
    assert len(seen) == 4
    assert all(r.headers["Authorization"] == f"Bearer {secret}" and secret not in str(r.url) for r in seen)
    assert secret not in out and secret not in err


def test_evaluate_requires_split_with_detector(dataset, tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert (
        run(
            capsys,
            "extract",
            "--model",
            "tiny:mixtral",
            "--data",
            str(dataset),
            "--out",
            str(run_dir),
            "--max-tokens",
            "6",
            "--quiet",
            "--tiers",
            "F0,F1",
        )[0]
        == 0
    )
    assert run(capsys, "calibrate", "--features", str(run_dir))[0] == 0
    (run_dir / "split.json").unlink()
    code, _, err = run(capsys, "evaluate", "--features", str(run_dir))
    assert code == 1 and "split" in err


def test_evidence_command_end_to_end(dataset, tmp_path, capsys):
    run_dir = tmp_path / "ev"
    code, out, err = run(
        capsys,
        "evidence",
        "--model",
        "tiny:mixtral",
        "--data",
        str(dataset),
        "--out",
        str(run_dir),
        "--max-tokens",
        "8",
        "--n-boot",
        "100",
        "--quiet",
    )
    assert code == 0, err
    assert (run_dir / "summary.txt").read_text(encoding="utf-8") in out
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["evidence"]["n_rows"] == 30


def test_extract_from_mkqa_sample(tmp_path, capsys):
    mkqa = tmp_path / "mkqa_es.jsonl"
    rows = [
        {"id": str(i), "question": f"pregunta {i}", "answers": [str(i)], "type": t}
        for i, t in enumerate(["number", "date", "entity", "entity"])
    ]
    mkqa.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    code, out, err = run(
        capsys,
        "extract",
        "--model",
        "tiny:mixtral",
        "--mkqa",
        "3",
        "--mkqa-path",
        str(mkqa),
        "--out",
        str(tmp_path / "run"),
        "--max-tokens",
        "3",
        "--quiet",
    )
    assert code == 0, err
    assert "3 respuestas" in out


def test_errors_exit_with_code_and_message(dataset, tmp_path, capsys):
    code, _, err = run(capsys, "calibrate", "--features", str(tmp_path / "no.npz"))
    assert code == 1 and err.startswith("error: ") and "no existe" in err
    code, _, err = run(
        capsys, "extract", "--model", "tiny:llama", "--data", str(dataset), "--out", str(tmp_path)
    )
    assert code == 1 and "desconocido" in err
    code, _, err = run(
        capsys, "extract", "--model", "tiny:mixtral", "--data", "x.jsonl", "--out", str(tmp_path)
    )
    assert code == 1 and "no existe" in err
    code, _, err = run(capsys, "extract", "--model", "tiny:mixtral", "--out", str(tmp_path))
    assert code == 1 and "--data o --mkqa" in err
    code, _, err = run(
        capsys, "extract", "--model", "tiny:mixtral", "--data", "x", "--out", "o", "--tiers", "F9"
    )
    assert code == 1 and "nivel inválido" in err
    with pytest.raises(SystemExit) as exc:
        main(["no-existe"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--features", "x", "--alpha", "1.5"])
    assert exc.value.code == 2


def test_serve_refuses_network_without_key(capsys, monkeypatch):
    code, _, err = run(capsys, "serve", "--model", "tiny:mixtral", "--host", ALL_INTERFACES)
    assert code == 1 and "127.0.0.1" in err
    monkeypatch.delenv("VIGIA_TEST_SERVER_KEY", raising=False)
    code, _, err = run(capsys, "serve", "--model", "tiny:mixtral", "--api-key-env", "VIGIA_TEST_SERVER_KEY")
    assert code == 1 and "VIGIA_TEST_SERVER_KEY" in err


def test_serve_builds_app_and_calls_uvicorn(capsys, monkeypatch):
    import uvicorn

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
    monkeypatch.setenv("VIGIA_TEST_SERVER_KEY", "clave-servidor-1234567890")
    code, _, err = run(
        capsys,
        "serve",
        "--model",
        "tiny:mixtral",
        "--host",
        ALL_INTERFACES,
        "--port",
        "8123",
        "--api-key-env",
        "VIGIA_TEST_SERVER_KEY",
    )
    assert code == 0, err
    assert captured["host"] == ALL_INTERFACES and captured["port"] == 8123
    assert captured["server_header"] is False and captured["app"].openapi_url is None


def test_models_lists_architectures(capsys):
    code, out, _ = run(capsys, "models")
    assert code == 0
    for name in ("mixtral", "gpt_oss", "granitemoe", "tiny:mixtral", "tiny:gpt_oss"):
        assert name in out


def test_entrypoint_runs_as_module():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(  # noqa: S603 - comando fijo con el intérprete del venv
        [sys.executable, "-m", "vigia.cli", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("extract", "label", "calibrate", "evaluate", "bench", "evidence", "serve", "models"):
        assert name in proc.stdout
    assert "torch" not in proc.stderr  # --help no carga el modelo


def test_tiny_backends_are_deterministic_and_offline():
    a, b = make_tiny_backend("mixtral"), make_tiny_backend("mixtral")
    for pa, pb in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(pa, pb)
    state = torch.random.get_rng_state()
    make_tiny_backend("mixtral", seed=5)
    assert torch.equal(state, torch.random.get_rng_state())  # no toca el RNG global
    g = make_tiny_backend("gpt_oss")
    assert g.model_info().model_type == "gpt_oss" and g.model_info().model_id == "tiny:gpt_oss"
    events = list(g.stream([{"role": "user", "content": "¿Qué es Vigía?"}], max_tokens=5))
    assert events and all(e.channel in ("final", "control", "none", "header", "analysis") for e in events)
    assert is_scorable(events[0])  # la plantilla abre el canal final
    with pytest.raises(cli.CLIError, match="opciones"):
        make_tiny_backend("llama")


def test_bench_command_writes_json(dataset, tmp_path, capsys):
    """P-09 por CLI: el benchmark reporta p50/p95/p99 de base y con Vigía (módulo vigia.bench)."""
    out = tmp_path / "bench.json"
    code, _, err = run(
        capsys,
        "bench",
        "--model",
        "tiny:mixtral",
        "--data",
        str(dataset),
        "--n-prompts",
        "2",
        "--tiers",
        "F0,F1",
        "--max-tokens",
        "3",
        "--warmup",
        "0",
        "--runs",
        "1",
        "--out",
        str(out),
    )
    assert code == 0, err
    result = json.loads(out.read_text(encoding="utf-8"))
    assert {"p50", "p95", "p99"} <= set(result["baseline"]) and {"p50", "p95", "p99"} <= set(result["vigia"])


def test_serve_rag_allowed_hosts_enables_policy_retriever(capsys, monkeypatch, tmp_path):
    """vigia serve acepta una política con rag_retry pero nunca re-consulta: --rag-allowed-hosts
    habilita un retriever local y, sin la allowlist, la política se rechaza al arrancar."""
    import uvicorn

    import vigia.serve as serve_mod

    built = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(
        serve_mod, "HttpRetriever", lambda url, *, allowed_hosts=(): built.append((url, tuple(allowed_hosts)))
    )
    pol = tmp_path / "pol.json"
    pol.write_text(
        json.dumps({"rag_retry_at": 0.8, "rag_url": "http://rag.local:9000/rag"}), encoding="utf-8"
    )
    code, _, err = run(capsys, "serve", "--model", "tiny:mixtral", "--policy", str(pol))
    assert code == 1 and "rag_url" in err and built == []
    code, _, err = run(
        capsys,
        "serve",
        "--model",
        "tiny:mixtral",
        "--policy",
        str(pol),
        "--rag-allowed-hosts",
        "rag.local, otro",
    )
    assert code == 0, err
    assert built == [("http://rag.local:9000/rag", ("rag.local", "otro"))]
