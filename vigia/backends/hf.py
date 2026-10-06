"""Backend HF transformers: generación token a token con hooks de señales (P-02).

Bucle de decodificación propio con cache KV (no usa `model.generate`, ni `output_router_logits`).
Cada token emitido sale como `TokenEvent` con features, logprob, IDs de expertos por capa y
offsets en code points dentro del texto de la respuesta.

Texto de la respuesta: la concatenación del `text` de los eventos con `channel == "final"` y
`is_special == False`. Los tokens especiales, los encabezados harmony y el canal `analysis` llevan
su texto en `text` pero tienen ancho cero (`start == end`) y nunca forman parte de la respuesta.
Un carácter UTF-8 partido entre tokens se asigna entero al token que lo completa; los tokens
previos quedan con `text == ""` (AT-S09). Los eventos se retienen mientras haya bytes pendientes
o mientras el final del texto pueda ser el comienzo de un string de parada.

El backend sirve un request por vez (lock): los hooks se instalan sobre el modelo compartido.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import torch

from vigia.adapters import find_moe_blocks, model_info
from vigia.contracts import Capabilities, Generation, InvalidRequestError, ModelInfo, Tier, TokenEvent
from vigia.extractor import SignalExtractor, TokenSignals, f2_unsupported_reason
from vigia.harmony import CONTROL, SCORED_CHANNEL, STOP_TOKENS, HarmonyParser, has_markup

DEFAULT_TIERS = (Tier.F0, Tier.F1)


class IncrementalDecoder:
    """Decodificación incremental que nunca emite U+FFFD por bytes UTF-8 aún incompletos.

    Ventana de prefijo al estilo TGI: compara `decode(ids[prefix:read])` con `decode(ids[prefix:])`
    y solo avanza cuando el texto nuevo no termina en U+FFFD.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0

    @property
    def pending(self) -> bool:
        return self._read < len(self._ids)

    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def push(self, token_id: int) -> str:
        self._ids.append(int(token_id))
        prefix_text = self._decode(self._ids[self._prefix : self._read])
        new_text = self._decode(self._ids[self._prefix :])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._prefix, self._read = self._read, len(self._ids)
            return new_text[len(prefix_text) :]
        return ""

    def flush(self) -> str:
        """Texto de los bytes pendientes (con U+FFFD si quedaron inválidos) y reinicia."""
        delta = ""
        if self.pending:
            prefix_text = self._decode(self._ids[self._prefix : self._read])
            delta = self._decode(self._ids[self._prefix :])[len(prefix_text) :]
        self.reset()
        return delta


class _Emitter:
    """Arma `TokenEvent` con offsets, canales y strings de parada a partir de IDs."""

    def __init__(self, tokenizer: Any, special_ids: set[int], harmony: bool, stop: list[str]) -> None:
        self.tokenizer = tokenizer
        self.special_ids = special_ids
        self.parser = HarmonyParser() if harmony else None
        self.decoder = IncrementalDecoder(tokenizer)
        self.stop = stop
        self.text = ""
        self.pending: list[TokenEvent] = []
        self.count = 0
        self.stopped = False
        self._pending_from: int | None = None  # primer evento cuyo texto espera bytes
        self._searched = 0

    def _special_text(self, token_id: int) -> str:
        return str(self.tokenizer.convert_ids_to_tokens(int(token_id)))

    def prime(self, prompt_ids: list[int]) -> None:
        """Lleva el parser harmony al estado en que termina el prompt (ej. `<|start|>assistant`)."""
        if self.parser is None:
            return
        for tid in prompt_ids:
            if tid in self.special_ids:
                self.parser.feed(self._special_text(tid), True)
            else:
                self.parser.feed(self.tokenizer.decode([tid], skip_special_tokens=False), False)

    def push(self, token_id: int, signals: TokenSignals | None, logprob: float | None) -> list[TokenEvent]:
        pos = len(self.text)
        common = {
            "index": self.count,
            "token_id": int(token_id),
            "features": signals.features if signals else {},
            "logprob": logprob,
            "expert_ids": signals.expert_ids if signals else None,
        }
        if token_id in self.special_ids:
            self._attach(self.decoder.flush())
            text = self._special_text(token_id)
            channel = self.parser.feed(text, True) if self.parser else CONTROL
            pos = len(self.text)
            ev = TokenEvent(text=text, start=pos, end=pos, channel=channel, is_special=True, **common)
        else:
            delta = self.decoder.push(token_id)
            channel = self.parser.feed(delta, False) if self.parser else SCORED_CHANNEL
            end = pos + len(delta) if channel == SCORED_CHANNEL else pos
            ev = TokenEvent(text=delta, start=pos, end=end, channel=channel, **common)
            if channel == SCORED_CHANNEL:
                self.text += delta
            if not self.decoder.pending:
                self._pending_from = None
            elif self._pending_from is None:
                self._pending_from = ev.index
        self.count += 1
        self.pending.append(ev)
        return self._release(final=False)

    def finish(self) -> list[TokenEvent]:
        if not self.stopped:
            self._attach(self.decoder.flush())
        return self._release(final=True)

    def _attach(self, leftover: str) -> None:
        """Asigna los bytes que quedaron pendientes al último token que los aportó."""
        self._pending_from = None
        if not leftover or not self.pending:
            return
        target = self.pending[-1]
        target.text += leftover
        if target.channel == SCORED_CHANNEL and not target.is_special:
            target.end += len(leftover)
            self.text += leftover

    def _find_stop(self) -> int | None:
        if not self.stop:
            return None
        start = max(0, self._searched - max(len(s) for s in self.stop) + 1)
        self._searched = len(self.text)
        hits = [i for i in (self.text.find(s, start) for s in self.stop) if i >= 0]
        return min(hits) if hits else None

    def _holdback(self) -> int:
        best = 0
        for s in self.stop:
            for n in range(min(len(s) - 1, len(self.text)), best, -1):
                if self.text.endswith(s[:n]):
                    best = n
                    break
        return best

    def _truncate(self, cut: int) -> None:
        self.text = self.text[:cut]
        kept = []
        for ev in self.pending:
            if ev.channel == SCORED_CHANNEL and not ev.is_special and ev.end > cut:
                if ev.start < cut:
                    ev.text = ev.text[: cut - ev.start]
                    ev.end = cut
                    kept.append(ev)
                break
            kept.append(ev)
        self.pending = kept

    def _release(self, final: bool) -> list[TokenEvent]:
        if not self.stopped:
            cut = self._find_stop()
            if cut is not None:
                self._truncate(cut)
                self.stopped = True
                final = True
        if final:
            out, self.pending = self.pending, []
            return out
        safe = len(self.text) - self._holdback()
        n = 0
        for ev in self.pending:
            if self._pending_from is not None and ev.index >= self._pending_from:
                break
            if ev.end > safe or ev.start > safe:
                break
            n += 1
        out, self.pending = self.pending[:n], self.pending[n:]
        return out


class GenerationStream(Iterator[TokenEvent]):
    """Iterador de `TokenEvent`. Al agotarse expone `finish_reason`, `text`, `timings` y conteos."""

    def __init__(self) -> None:
        self._gen: Iterator[TokenEvent] | None = None
        self.finish_reason: str | None = None
        self.text = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.timings: dict[str, float] = {}

    def __iter__(self) -> GenerationStream:
        return self

    def __next__(self) -> TokenEvent:
        assert self._gen is not None
        return next(self._gen)

    def close(self) -> None:
        """Cancela la generación y libera hooks y lock."""
        if self._gen is not None:
            self._gen.close()  # type: ignore[attr-defined]


def _validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages debe ser una lista no vacía")
    for m in messages:
        if (
            not isinstance(m, dict)
            or not isinstance(m.get("role"), str)
            or not isinstance(m.get("content"), str)
        ):
            raise ValueError("cada mensaje debe ser un dict con 'role' y 'content' de tipo str")
    return messages


def _sample(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator | None) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits))
    # float64 y máximo restado: con temperature subnormal (5e-324) el cociente queda en [-inf, 0]
    # y nunca da inf ni NaN; el token máximo conserva probabilidad 1 (equivale a greedy).
    z = logits.double()
    probs = torch.softmax((z - z.max()) / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        drop = torch.cumsum(sorted_p, dim=-1) - sorted_p > top_p
        sorted_p[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_p)
    return int(torch.multinomial(probs, 1, generator=generator))


class HFBackend:
    """Backend que cumple `vigia.contracts.Backend` sobre un modelo MoE de transformers."""

    def __init__(
        self,
        model_id_or_model: Any,
        tokenizer: Any = None,
        device: str = "cpu",
        dtype: Any = None,
        *,
        model_id: str | None = None,
        probe_layer: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Kwargs extra para apply_chat_template, ej. {"reasoning_effort": "low"} en gpt-oss.
        self._template_kwargs = dict(chat_template_kwargs or {})
        if isinstance(model_id_or_model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = model_id or model_id_or_model
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id_or_model)
            # Solo safetensors y sin código remoto: un checkpoint con pickle no se carga (supply chain).
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_model, dtype=dtype or "auto", use_safetensors=True, trust_remote_code=False
            )
        else:
            model = model_id_or_model
            if tokenizer is None:
                raise ValueError("con un modelo ya cargado hace falta pasar el tokenizer")
            if dtype is not None:
                model = model.to(dtype)
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.probe_layer = probe_layer
        self.model_id = model_id or str(getattr(model.config, "_name_or_path", "") or type(model).__name__)
        self._info = model_info(self.model, self.model_id)
        self._harmony = self._info.model_type == "gpt_oss"
        self._f2_reason = f2_unsupported_reason(find_moe_blocks(self.model))
        self._special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        self._special_ids |= {int(i) for i, tok in decoder.items() if getattr(tok, "special", False)}
        self._eos_ids = self._collect_eos()
        self._max_positions = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        self._lock = threading.Lock()

    def _collect_eos(self) -> set[int]:
        ids: set[int] = set()
        gen_cfg = getattr(self.model, "generation_config", None)
        for value in (getattr(gen_cfg, "eos_token_id", None), getattr(self.tokenizer, "eos_token_id", None)):
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, (list, tuple)):
                ids |= {int(v) for v in value}
        if self._harmony:
            vocab = self.tokenizer.get_vocab()
            ids |= {vocab[t] for t in STOP_TOKENS if t in vocab}
        return {i for i in ids if 0 <= i < self._info.vocab_size}

    # --- contrato -------------------------------------------------------------------------------

    def model_info(self) -> ModelInfo:
        return self._info

    def capabilities(self) -> Capabilities:
        tiers = [Tier.F0, Tier.F1, Tier.U, Tier.H]
        if self._f2_reason is None:
            tiers.insert(3, Tier.F2)
        disabled = (self._f2_reason,) if self._f2_reason else ()
        return Capabilities(tiers=tuple(tiers), streaming=True, disabled=disabled)

    def _check_tiers(self, tiers: tuple[Tier | str, ...]) -> tuple[Tier, ...]:
        wanted = tuple(dict.fromkeys(Tier(t) for t in tiers))
        missing = [t.value for t in wanted if t not in self.capabilities().tiers]
        if missing:
            raise ValueError(
                f"niveles no disponibles en este backend: {missing}; {list(self.capabilities().disabled)}"
            )
        return wanted

    def _prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        messages = _validate_messages(messages)
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self._template_kwargs
            )
        else:
            text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant: "
        ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if not ids:
            raise InvalidRequestError("el prompt quedó vacío después de tokenizar", code="invalid_prompt")
        if self._max_positions and len(ids) >= self._max_positions:
            raise InvalidRequestError(
                f"el prompt ({len(ids)} tokens) excede el contexto del modelo ({self._max_positions})",
                code="context_length_exceeded",
            )
        return ids

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop: list[str] | None = None,
        tiers: tuple[Tier, ...] = DEFAULT_TIERS,
        ignore_eos: bool = False,
    ) -> GenerationStream:
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens debe ser un entero >= 1")
        if not 0.0 <= float(temperature):
            raise ValueError("temperature debe ser >= 0")
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p debe estar en (0, 1]")
        stop = list(stop or [])
        if any(not isinstance(s, str) or not s for s in stop):
            raise ValueError("cada stop debe ser un string no vacío")
        prompt_ids = self._prompt_ids(messages)
        wanted = self._check_tiers(tiers)
        out = GenerationStream()
        out.prompt_tokens = len(prompt_ids)
        out._gen = self._run(
            out, prompt_ids, max_tokens, float(temperature), float(top_p), seed, stop, wanted, ignore_eos
        )
        return out

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> Generation:
        """Corre `stream` hasta el final y devuelve la `Generation` completa."""
        s = self.stream(messages, **kwargs)
        tokens = list(s)
        return Generation(
            text=s.text,
            tokens=tokens,
            finish_reason=s.finish_reason or "stop",  # type: ignore[arg-type]
            prompt_tokens=s.prompt_tokens,
            model=self._info,
            timings=dict(s.timings),
        )

    def _run(
        self,
        out: GenerationStream,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        stop: list[str],
        tiers: tuple[Tier, ...],
        ignore_eos: bool = False,
    ) -> Iterator[TokenEvent]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        emitter = _Emitter(self.tokenizer, self._special_ids, self._harmony, stop)
        emitter.prime(prompt_ids)
        limit = max_tokens
        if self._max_positions:
            limit = min(limit, self._max_positions - len(prompt_ids))
        # El modo de grad es local de cada hilo y StreamingResponse avanza este generador desde el
        # threadpool: ningún context manager de torch puede cruzar un `yield`. Cada tramo de cómputo
        # entre yields abre su propio inference_mode.
        with self._lock, SignalExtractor(self.model, tiers, self.probe_layer) as ex:
            t0 = time.perf_counter()
            with torch.inference_mode():
                ids = torch.tensor([prompt_ids], device=self.device)
                score = torch.zeros(ids.shape, dtype=torch.bool)
                score[0, -1] = True
                ex.begin_step(score_mask=score)
                res = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
            prefill_s = time.perf_counter() - t0
            decode_s = 0.0
            finish = "length"
            n = 0
            while True:
                with torch.inference_mode():
                    logits = res.logits[0, -1]
                    token = _sample(logits, temperature, top_p, generator)
                    signals = ex.end_step(res.logits[:, -1:], torch.tensor([[token]]))[0][0]
                    eos = token in self._eos_ids and not ignore_eos
                    if not eos:
                        logprob = float(torch.log_softmax(logits.float(), dim=-1)[token])
                        events = emitter.push(token, signals, logprob)
                if eos:
                    finish = "stop"
                    break
                yield from events
                n += 1
                if emitter.stopped:
                    finish = "stop"
                    break
                if n >= limit:
                    break
                t1 = time.perf_counter()
                with torch.inference_mode():
                    ex.begin_step()
                    res = self.model(
                        input_ids=torch.tensor([[token]], device=self.device),
                        past_key_values=res.past_key_values,
                        use_cache=True,
                    )
                decode_s += time.perf_counter() - t1
            yield from emitter.finish()
            out.finish_reason = finish
            out.text = emitter.text
            out.completion_tokens = n
            out.timings = {"prefill_s": prefill_s, "decode_s": decode_s, "total_s": time.perf_counter() - t0}

    def _final_header_ids(self, prompt_ids: list[int]) -> list[int]:
        """IDs del marcado harmony que lleva del final del prompt al contenido del canal `final`."""
        emitter = _Emitter(self.tokenizer, self._special_ids, self._harmony, [])
        emitter.prime(prompt_ids)
        assert emitter.parser is not None
        header = emitter.parser.final_header()
        if not header:
            return []
        return list(self.tokenizer(header, add_special_tokens=False)["input_ids"])

    def teacher_force(
        self, messages: list[dict[str, str]], response_text: str, tiers: tuple[Tier, ...] = DEFAULT_TIERS
    ) -> list[TokenEvent]:
        """Features de una respuesta dada con una sola pasada sobre prompt + respuesta (calibración).

        Los offsets de los eventos se refieren al texto del canal `final`. Con harmony, una
        respuesta en texto plano se puntúa como contenido de `final`: si el prompt no deja al
        parser en ese canal (la plantilla real de gpt-oss termina en `<|start|>assistant`), se
        antepone el encabezado que el modelo generaría (`<|channel|>final<|message|>`). Sus eventos
        tienen ancho cero (control y header), igual que en `stream`, y las features U acumulan esos
        tokens como en generación. Una respuesta con marcado harmony se usa tal cual;
        `vigia.harmony.final_view` da su texto final y lleva los spans a ese texto.
        """
        if not isinstance(response_text, str):
            raise ValueError("response_text debe ser str")
        prompt_ids = self._prompt_ids(messages)
        wanted = self._check_tiers(tiers)
        resp_ids = list(self.tokenizer(response_text, add_special_tokens=False)["input_ids"])
        if not resp_ids:
            return []
        if self._harmony and not has_markup(response_text):
            resp_ids = self._final_header_ids(prompt_ids) + resp_ids
        total = len(prompt_ids) + len(resp_ids)
        if self._max_positions and total > self._max_positions:
            raise InvalidRequestError(
                f"prompt + respuesta ({total} tokens) excede el contexto ({self._max_positions})",
                code="context_length_exceeded",
            )
        n_prompt, n_resp = len(prompt_ids), len(resp_ids)
        with self._lock, torch.inference_mode(), SignalExtractor(self.model, wanted, self.probe_layer) as ex:
            ids = torch.tensor([prompt_ids + resp_ids], device=self.device)
            score = torch.zeros(ids.shape, dtype=torch.bool)
            score[0, n_prompt - 1 : total - 1] = True
            ex.begin_step(score_mask=score)
            res = self.model(input_ids=ids, use_cache=False, logits_to_keep=n_resp + 1)
            targets = torch.tensor([resp_ids + [0]])
            signals = ex.end_step(res.logits, targets)[0]
            logp = torch.log_softmax(res.logits[0, :n_resp].float(), dim=-1)
            logprobs = logp.gather(-1, torch.tensor(resp_ids, device=logp.device)[:, None])[:, 0].tolist()
        emitter = _Emitter(self.tokenizer, self._special_ids, self._harmony, [])
        emitter.prime(prompt_ids)
        events: list[TokenEvent] = []
        for tid, sig, lp in zip(resp_ids, signals, logprobs, strict=True):
            events.extend(emitter.push(tid, sig, lp))
        events.extend(emitter.finish())
        return events
