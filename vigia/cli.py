"""CLI de Vigía (P-16): `vigia extract | label | calibrate | evaluate | bench | evidence | serve | models`.

Flujo típico sobre una carpeta de corrida:

    vigia extract   --model tiny:mixtral --data items.jsonl --out run/   # rows.jsonl + features.npz
    vigia label     --run run/ [--judge-url URL --judge-model M]          # re-etiqueta (opcional)
    vigia calibrate --features run/features.npz                           # detector.json + split.json
    vigia evaluate  --features run/features.npz                           # report.json + report.html
    vigia evidence  --model ibm-granite/... --mkqa 300 --out run/         # todo junto, reanudable
    vigia evidence  --model ... --humaneval 164 --allow-code-exec --out run/  # código (ver riesgo)

`--model` acepta un id de Hugging Face, una ruta local o `tiny:mixtral` / `tiny:gpt_oss` (modelo
diminuto aleatorio con tokenizer byte-level, para demos y pruebas sin red). La API key del juez se
lee solo de la variable de entorno `VIGIA_JUDGE_API_KEY`.

Los items de código (`--humaneval` o `tests` + `entry_point` en `--data`) se etiquetan ejecutando el
código que genera el modelo. Eso no es un sandbox: hace falta `--allow-code-exec` y conviene correrlo
solo en una VM descartable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vigia.contracts import Tier, VigiaError

TINY_PREFIX = "tiny:"
TINY_ARCHS = ("mixtral", "gpt_oss")
JUDGE_KEY_ENV = "VIGIA_JUDGE_API_KEY"
DTYPES = ("auto", "float32", "bfloat16", "float16")
CODE_EXEC_HELP = (
    "permite ejecutar el código que genera el modelo para etiquetar items de código. RIESGO: no es "
    "un sandbox; el código corre con tus permisos (archivos, red, procesos). Usalo solo en una VM "
    "descartable"
)
_TINY_GPT_OSS_TEMPLATE = (
    "{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant<|channel|>final<|message|>{% endif %}"
)

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


class CLIError(VigiaError):
    """Error de uso o de configuración que la CLI informa sin traza."""


# --- Backends -------------------------------------------------------------------------------------


def _byte_tokenizer(specials: Sequence[str]) -> Any:
    """Tokenizer byte-level sin merges: un token por byte UTF-8 más los especiales."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {ch: i for i, ch in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk)
    tok.add_special_tokens({"additional_special_tokens": list(specials)})
    return tok


def make_tiny_backend(arch: str = "mixtral", *, seed: int = 0) -> Any:
    """Backend HF con un MoE diminuto aleatorio (2 capas, 4 expertos, top-2) y tokenizer byte-level.

    Es determinista para una `seed` dada: el mismo `tiny:<arch>` en otro proceso tiene los mismos pesos,
    así que un detector calibrado sobre él sirve en `vigia serve`. No descarga nada.
    `gpt_oss` usa el formato harmony y su plantilla abre directamente el canal `final`.
    """
    if arch not in TINY_ARCHS:
        raise CLIError(f"modelo diminuto desconocido {arch!r}; opciones: {', '.join(TINY_ARCHS)}")
    import torch
    import transformers as tf
    from transformers import AutoModelForCausalLM

    from vigia.backends.hf import HFBackend
    from vigia.harmony import RETURN, SPECIAL_TOKENS

    specials = list(SPECIAL_TOKENS) if arch == "gpt_oss" else ["<|eos|>"]
    tokenizer = _byte_tokenizer(specials)
    eos = tokenizer.convert_tokens_to_ids(RETURN if arch == "gpt_oss" else "<|eos|>")
    common = dict(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=1024,
        eos_token_id=eos,
        bos_token_id=None,
        pad_token_id=None,
    )
    if arch == "gpt_oss":
        tokenizer.chat_template = _TINY_GPT_OSS_TEMPLATE
        cfg = tf.GptOssConfig(**common, head_dim=8)
    else:
        cfg = tf.MixtralConfig(**common)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg).eval()
    model.generation_config.eos_token_id = eos
    return HFBackend(model, tokenizer, model_id=f"{TINY_PREFIX}{arch}")


def load_backend(
    model: str, *, device: str = "cpu", dtype: str = "auto", reasoning_effort: str | None = None
) -> Any:
    """`tiny:<arch>` o un checkpoint de Hugging Face (id o ruta local) con el backend HF."""
    if model.startswith(TINY_PREFIX):
        return make_tiny_backend(model[len(TINY_PREFIX) :])
    if dtype not in DTYPES:
        raise CLIError(f"dtype inválido {dtype!r}; opciones: {', '.join(DTYPES)}")
    import torch

    from vigia.backends.hf import HFBackend

    torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else None
    return HFBackend(model, device=device, dtype=torch_dtype, chat_template_kwargs=kwargs)


# --- Utilidades de argumentos ---------------------------------------------------------------------


def parse_tiers(text: str) -> tuple[Tier, ...]:
    """'F0,F1,H' -> (Tier.F0, Tier.F1, Tier.H)."""
    parts = [p.strip().upper() for p in text.split(",") if p.strip()]
    if not parts:
        raise CLIError("hace falta al menos un nivel (ej. F0,F1)")
    try:
        return tuple(dict.fromkeys(Tier(p) for p in parts))
    except ValueError as exc:
        raise CLIError(f"nivel inválido en {text!r}; opciones: {', '.join(t.value for t in Tier)}") from exc


def _parse_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(p) for p in text.split(",") if p.strip())
    except ValueError as exc:
        raise CLIError(f"lista de números inválida: {text!r}") from exc
    if not values:
        raise CLIError("la lista de alphas está vacía")
    return values


def _load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    from vigia.datasets import load_mkqa_es, read_jsonl, stratified_sample

    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else None
    if args.data:
        items = read_jsonl(args.data)
        if types:
            items = [it for it in items if it.get("type") in types]
        if args.limit:
            items = items[: args.limit]
        if not items:
            raise CLIError("el filtro dejó el dataset vacío")
        return items
    if args.mkqa:
        return load_mkqa_es(args.mkqa_path, n=args.mkqa, seed=args.seed, types=types)
    if args.humaneval:
        if not Path(args.humaneval_path).is_file():
            raise CLIError(
                f"no existe {args.humaneval_path}: convertí HumanEval con "
                'python -c "from vigia.datasets import prepare_humaneval; prepare_humaneval()"'
            )
        return stratified_sample(read_jsonl(args.humaneval_path), args.humaneval, seed=args.seed)
    raise CLIError("indicá el dataset con --data, --mkqa o --humaneval")


def _make_judge(args: argparse.Namespace) -> Any:
    if not args.judge_url and not args.judge_model:
        return None
    if not (args.judge_url and args.judge_model):
        raise CLIError("el juez necesita --judge-url y --judge-model juntos")
    api_key = os.environ.get(JUDGE_KEY_ENV, "")
    if not api_key:
        raise CLIError(f"falta la API key del juez en la variable de entorno {JUDGE_KEY_ENV}")
    from vigia.labels import OpenAICompatibleJudge

    return OpenAICompatibleJudge(args.judge_url, api_key, args.judge_model, timeout=args.judge_timeout)


def _features_path(value: str) -> Path:
    from vigia.evidence import FEATURES_FILE

    path = Path(value)
    return path / FEATURES_FILE if path.is_dir() else path


def _progress(done: int, total: int, item_id: str) -> None:
    print(f"[{done}/{total}] {item_id}", file=sys.stderr, flush=True)


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# --- Subcomandos ----------------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    from vigia.evidence import FEATURES_FILE, ROWS_FILE, build_features, extract, require_code_permission

    tiers = parse_tiers(args.tiers)
    items = _load_items(args)
    require_code_permission(items, args.allow_code_exec)  # antes de cargar el modelo
    backend = load_backend(
        args.model, device=args.device, dtype=args.dtype, reasoning_effort=args.reasoning_effort
    )
    rows = extract(
        backend,
        items,
        args.out,
        tiers=tiers,
        max_tokens=args.max_tokens,
        seed=args.seed,
        progress_cb=None if args.quiet else _progress,
        allow_code_exec=args.allow_code_exec,
    )
    fs = build_features(args.out, rows)
    skipped = sum(1 for r in rows if r.get("skipped"))
    print(
        f"{len(rows)} respuestas ({skipped} salteadas), {len(fs.labels)} tokens, "
        f"{fs.labeled_count()} etiquetados, {len(fs.names)} features."
    )
    print(f"Artefactos: {Path(args.out) / ROWS_FILE}, {Path(args.out) / FEATURES_FILE}")
    return EXIT_OK


def cmd_label(args: argparse.Namespace) -> int:
    from vigia.evidence import relabel

    report = relabel(args.run, judge=_make_judge(args), allow_code_exec=args.allow_code_exec)
    _print_json(report)
    return EXIT_OK


def cmd_calibrate(args: argparse.Namespace) -> int:
    from vigia.calibrate import DEFAULT_ALPHAS
    from vigia.contracts import feature_tier
    from vigia.evidence import FeatureSet, calibrate_features, save_calibration

    path = _features_path(args.features)
    fs = FeatureSet.load(path)
    tiers = set(parse_tiers(args.tiers))
    names = [n for n in fs.names if feature_tier(n) in tiers]
    if not names:
        raise CLIError(f"la matriz no tiene features de los niveles {args.tiers}")
    detector, split_report = calibrate_features(
        fs,
        seed=args.seed,
        alphas=_parse_floats(args.alphas) if args.alphas else DEFAULT_ALPHAS,
        default_alpha=args.default_alpha,
        feature_names=names,
    )
    det_path, split_path = save_calibration(
        Path(args.out) if args.out else path.parent, detector, split_report
    )
    test = detector.artifact["metrics"].get("test") or {}
    print(
        f"Detector {detector.id}: {len(detector.feature_names)} features, C = {split_report['c_selected']}."
    )
    if test.get("auroc") is not None:
        print(f"AUROC por token en test (sin IC; usá evaluate): {test['auroc']:.3f}")
    print(f"Artefactos: {det_path}, {split_path}")
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    from vigia.detector import Detector
    from vigia.evidence import (
        DETECTOR_FILE,
        SPLIT_FILE,
        FeatureSet,
        evaluate_features,
        summary_text,
        write_report,
    )

    path = _features_path(args.features)
    fs = FeatureSet.load(path)
    det_path = Path(args.detector) if args.detector else path.parent / DETECTOR_FILE
    split_path = Path(args.split) if args.split else path.parent / SPLIT_FILE
    detector = Detector.load(det_path) if det_path.is_file() else None
    if args.detector and detector is None:
        raise CLIError(f"no existe el detector {det_path}")
    split = None
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))["groups"]
    elif detector is not None:
        raise CLIError(
            f"falta {split_path}: sin el split de calibración, "
            "el test puede incluir preguntas de entrenamiento"
        )
    report = evaluate_features(
        fs, detector=detector, split=split, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha
    )
    paths = write_report(Path(args.out) if args.out else path.parent, report)
    print(summary_text(report), end="")
    print(f"Artefactos: {paths['json']}, {paths['html']}")
    return EXIT_OK


def _bench_prompts(items: Sequence[dict[str, Any]], n: int) -> list[list[dict[str, str]]]:
    return [it["messages"] for it in items[:n]]


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        from vigia.bench import run_overhead_bench
    except ImportError as exc:
        raise CLIError(f"el módulo de benchmark no está disponible: {exc}") from exc
    items = _load_items(args)
    backend = load_backend(
        args.model, device=args.device, dtype=args.dtype, reasoning_effort=args.reasoning_effort
    )
    result = run_overhead_bench(
        backend,
        _bench_prompts(items, args.n_prompts),
        tiers_list=[parse_tiers(t) for t in args.tiers],
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Artefacto: {args.out}")
    else:
        print(text)
    return EXIT_OK


def cmd_evidence(args: argparse.Namespace) -> int:
    from vigia.evidence import REPORT_HTML, REPORT_JSON, require_code_permission, run_evidence, summary_text

    tiers = parse_tiers(args.tiers)
    judge = _make_judge(args)
    items = _load_items(args)
    require_code_permission(items, args.allow_code_exec)  # antes de cargar el modelo
    backend = load_backend(
        args.model, device=args.device, dtype=args.dtype, reasoning_effort=args.reasoning_effort
    )
    report = run_evidence(
        backend,
        items,
        out_dir=args.out,
        max_tokens=args.max_tokens,
        tiers=tiers,
        seed=args.seed,
        judge=judge,
        n_boot=args.n_boot,
        alpha=args.alpha,
        bench_prompts=_bench_prompts(items, args.bench) if args.bench else None,
        bench_runs=args.bench_runs,
        progress_cb=None if args.quiet else _progress,
        allow_code_exec=args.allow_code_exec,
    )
    print(summary_text(report), end="")
    print(f"Artefactos: {Path(args.out) / REPORT_JSON}, {Path(args.out) / REPORT_HTML}")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    from vigia.serve import create_app, is_loopback

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise CLIError(f"la variable de entorno {args.api_key_env} está vacía o no existe")
    elif not is_loopback(args.host):
        raise CLIError("sin API key el servidor solo escucha en 127.0.0.1; pasá --api-key-env")
    detector = None
    if args.detector:
        from vigia.detector import Detector

        detector = Detector.load(args.detector)
    rag_hosts = tuple(h.strip() for h in (args.rag_allowed_hosts or "").split(",") if h.strip())
    policy = None
    if args.policy:
        from vigia.policy import policy_from_dict

        raw = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        policy = policy_from_dict(raw, rag_allowed_hosts=rag_hosts)
    backend = load_backend(
        args.model, device=args.device, dtype=args.dtype, reasoning_effort=args.reasoning_effort
    )
    app = create_app(backend, detector=detector, policy=policy, api_key=api_key, rag_allowed_hosts=rag_hosts)
    import uvicorn

    print(
        f"Vigía sirviendo {backend.model_info().model_id} en http://{args.host}:{args.port}", file=sys.stderr
    )
    uvicorn.run(app, host=args.host, port=args.port, server_header=False, log_level="info")
    return EXIT_OK


def cmd_models(args: argparse.Namespace) -> int:
    from vigia.adapters import supported_model_types

    print("Arquitecturas MoE soportadas (config.model_type):")
    for name in supported_model_types():
        print(f"  {name}")
    print("Modelos diminutos sin red (--model):")
    for arch in TINY_ARCHS:
        print(f"  {TINY_PREFIX}{arch}")
    return EXIT_OK


# --- Parser ---------------------------------------------------------------------------------------


def _positive(value: str) -> int:
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"se esperaba un entero: {value!r}") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("debe ser >= 1")
    return n


def _non_negative(value: str) -> int:
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"se esperaba un entero: {value!r}") from exc
    if n < 0:
        raise argparse.ArgumentTypeError("debe ser >= 0")
    return n


def _alpha(value: str) -> float:
    try:
        a = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"se esperaba un número: {value!r}") from exc
    if not 0 < a < 1:
        raise argparse.ArgumentTypeError("alpha debe estar en (0, 1)")
    return a


def _add_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="id de HF, ruta local, tiny:mixtral o tiny:gpt_oss")
    p.add_argument("--device", default="cpu", help="dispositivo de torch (por defecto cpu)")
    p.add_argument("--dtype", default="auto", choices=DTYPES, help="dtype de los pesos")
    p.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help="esfuerzo de razonamiento para la plantilla de chat (gpt-oss); otros modelos lo ignoran",
    )


def _add_data(p: argparse.ArgumentParser) -> None:
    from vigia.datasets import DEFAULT_HUMANEVAL_PATH, DEFAULT_MKQA_PATH

    g = p.add_mutually_exclusive_group()
    g.add_argument("--data", help="JSONL con items (formato de la plataforma)")
    g.add_argument("--mkqa", type=_positive, metavar="N", help="muestra estratificada de N items de MKQA-es")
    g.add_argument(
        "--humaneval",
        type=_positive,
        metavar="N",
        help="muestra de N problemas de HumanEval (código; etiquetar exige --allow-code-exec)",
    )
    p.add_argument("--mkqa-path", default=str(DEFAULT_MKQA_PATH), help="JSONL de MKQA-es convertido")
    p.add_argument(
        "--humaneval-path", default=str(DEFAULT_HUMANEVAL_PATH), help="JSONL de HumanEval convertido"
    )
    p.add_argument("--types", help="filtra por tipo de pregunta (lista separada por comas)")
    p.add_argument("--limit", type=_positive, help="usa solo los primeros N items de --data")


def _add_judge(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-url", help=f"endpoint OpenAI-compatible del juez (API key en {JUDGE_KEY_ENV})")
    p.add_argument("--judge-model", help="modelo del juez (sin valor por defecto)")
    p.add_argument("--judge-timeout", type=float, default=120.0, help="timeout por llamada, en segundos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vigia", description="Vigía: confianza por token para LLM MoE.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="comando")

    p = sub.add_parser("extract", help="genera o fuerza respuestas y guarda features.npz + rows.jsonl")
    _add_model(p)
    _add_data(p)
    p.add_argument("--out", required=True, help="carpeta de la corrida (reanudable)")
    p.add_argument("--tiers", default="F0,F1,H", help="niveles de features (por defecto F0,F1,H)")
    p.add_argument("--max-tokens", type=_positive, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="sin progreso por item")
    p.add_argument("--allow-code-exec", action="store_true", help=CODE_EXEC_HELP)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("label", help="re-etiqueta una corrida con reglas computables o con un juez")
    p.add_argument("--run", required=True, help="carpeta de la corrida (rows.jsonl)")
    _add_judge(p)
    p.add_argument("--allow-code-exec", action="store_true", help=CODE_EXEC_HELP)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("calibrate", help="features.npz -> detector.json + split.json")
    p.add_argument("--features", required=True, help="features.npz o la carpeta de la corrida")
    p.add_argument("--out", help="carpeta de salida (por defecto la de features.npz)")
    p.add_argument("--tiers", default="F0,F1", help="niveles de features del detector (por defecto F0,F1)")
    p.add_argument(
        "--alphas",
        default=None,
        help="alphas de CRC separados por coma (por defecto la grilla 0.02 a 0.50 cada 0.02)",
    )
    p.add_argument(
        "--default-alpha",
        type=_alpha,
        default=None,
        help="alpha por defecto del detector (por defecto, el recomendado en calibración)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("evaluate", help="-> report.json + report.html con IC por bootstrap de pregunta")
    p.add_argument("--features", required=True, help="features.npz o la carpeta de la corrida")
    p.add_argument("--detector", help="detector.json (por defecto el de la carpeta, si existe)")
    p.add_argument("--split", help="split.json de calibrate (por defecto el de la carpeta)")
    p.add_argument("--out", help="carpeta de salida (por defecto la de features.npz)")
    p.add_argument("--n-boot", type=_positive, default=2000)
    p.add_argument("--alpha", type=_alpha)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("bench", help="benchmark de overhead con hooks on/off")
    _add_model(p)
    _add_data(p)
    p.add_argument("--n-prompts", type=_positive, default=8)
    p.add_argument("--tiers", nargs="+", default=["F0", "F0,F1"], help="un set de niveles por argumento")
    p.add_argument("--max-tokens", type=_positive, default=24)
    p.add_argument("--warmup", type=_non_negative, default=1)
    p.add_argument("--runs", type=_positive, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="archivo JSON de salida (por defecto, stdout)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("evidence", help="experimento completo y reanudable (extract, calibrate, evaluate)")
    _add_model(p)
    _add_data(p)
    _add_judge(p)
    p.add_argument("--out", required=True, help="carpeta de la corrida (reanudable)")
    p.add_argument("--tiers", default="F0,F1,H")
    p.add_argument("--max-tokens", type=_positive, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=_positive, default=2000)
    p.add_argument("--alpha", type=_alpha)
    p.add_argument("--bench", type=_non_negative, default=0, metavar="N", help="prompts para el benchmark")
    p.add_argument("--bench-runs", type=_positive, default=3)
    p.add_argument("--quiet", action="store_true", help="sin progreso por item")
    p.add_argument("--allow-code-exec", action="store_true", help=CODE_EXEC_HELP)
    p.set_defaults(func=cmd_evidence)

    p = sub.add_parser("serve", help="servidor OpenAI-compatible single-tenant")
    _add_model(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=_positive, default=8000)
    p.add_argument("--detector", help="detector.json calibrado para este modelo")
    p.add_argument("--policy", help="política en JSON (campos de vigia.contracts.Policy)")
    p.add_argument("--api-key-env", help="variable de entorno con la API key del servidor")
    p.add_argument(
        "--rag-allowed-hosts",
        default="",
        help="hosts del retriever de rag_url separados por coma; habilita http y redes privadas",
    )
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("models", help="lista las arquitecturas soportadas")
    p.set_defaults(func=cmd_models)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrumpido: la corrida se reanuda con el mismo comando", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (VigiaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
