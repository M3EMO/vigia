"""Gráficos SVG como funciones puras (ADR A-12).

Cada función devuelve un string SVG con `role="img"`, `<title>` y `<desc>` con la conclusión
principal. Cada gráfico tiene una función `*_table` con los mismos argumentos que devuelve la tabla
de datos HTML para un `<details>`.

Colores: Okabe-Ito para categorías y una rampa secuencial derivada de cividis; nunca rojo-verde.
Cada elemento lleva el color como atributo de presentación (fallback que no necesita CSS y cumple
una CSP sin `unsafe-inline`) y una clase que `CHART_CSS` redefine con variables CSS para modo
claro y oscuro. Todo texto se escapa para XML.
"""

from __future__ import annotations

import hashlib
import html
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
# Orden de series en modo claro: azul, bermellón, tinta, púrpura, verde azulado, celeste, naranja.
# El par principal es azul/bermellón (seguro para deficiencias rojo-verde). Celeste y naranja son
# variantes oscurecidas de Okabe-Ito: #56B4E9 y #E69F00 dan ~2,3:1 sobre blanco y WCAG 1.4.11
# pide 3:1 para trazos de datos. Cada serie conserva además su patrón de trazo (DASHES).
CATEGORY_LIGHT = ("#0072B2", "#D55E00", "#1F2328", "#CC79A7", "#009E73", "#3388C2", "#B07400")
CATEGORY_DARK = ("#56B4E9", "#E69F00", "#E6EDF3", "#CC79A7", "#009E73", "#F0E442", "#D55E00")
DASHES = ("", "6 3", "2 3", "8 3 2 3", "1 2", "10 4", "4 2 1 2")
# Diez muestras equiespaciadas de cividis (de azul oscuro a amarillo, L* monótono).
CIVIDIS = (
    "#00204D",
    "#00336F",
    "#39486B",
    "#575C6D",
    "#707173",
    "#8A8779",
    "#A69D75",
    "#C4B56C",
    "#E4CF5B",
    "#FFEA46",
)
INK, MUTED, GRID = "#1F2328", "#59636E", "#D1D9E0"


def _css_palette(cats: Sequence[str], ink: str, muted: str, grid: str, bg: str) -> str:
    items = [f"--vg-ink:{ink}", f"--vg-muted:{muted}", f"--vg-grid:{grid}", f"--vg-bg:{bg}"]
    items += [f"--vg-cat-{i + 1}:{c}" for i, c in enumerate(cats)]
    items.append(f"--vg-alert:{cats[1]}")
    return ";".join(items)


_DARK = _css_palette(CATEGORY_DARK, "#E6EDF3", "#9198A1", "#3D444D", "#0D1117")
CHART_CSS = (
    ":root{" + _css_palette(CATEGORY_LIGHT, INK, MUTED, GRID, "#FFFFFF") + "}"
    '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){' + _DARK + "}}"
    ':root[data-theme="dark"]{' + _DARK + "}"
    ".vg-chart{max-width:100%;height:auto}"
    ".vg-chart text{fill:var(--vg-ink," + INK + ");font-family:system-ui,sans-serif}"
    ".vg-chart .vg-muted{fill:var(--vg-muted," + MUTED + ")}"
    ".vg-axis{stroke:var(--vg-ink," + INK + ")}"
    ".vg-grid{stroke:var(--vg-grid," + GRID + ")}"
    ".vg-ref{stroke:var(--vg-muted," + MUTED + ")}"
    ".vg-chart .vg-alert{fill:var(--vg-alert,"
    + CATEGORY_LIGHT[1]
    + ")}"
    + "".join(
        f".vg-s{i + 1}{{stroke:var(--vg-cat-{i + 1},{c})}}.vg-f{i + 1}{{fill:var(--vg-cat-{i + 1},{c})}}"
        for i, c in enumerate(CATEGORY_LIGHT)
    )
    + ".vg-data .vg-table-wrap{overflow-x:auto;max-width:100%}"
    ".vg-data table{border-collapse:collapse}.vg-data th,.vg-data td{padding:2px 8px;text-align:right}"
    ".vg-data th[scope=row]{text-align:left}"
)

W, H = 480, 360
ML, MR, MT, MB = 56, 16, 16, 48


# --- utilidades -------------------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def fmt_num(x: Any, digits: int = 3) -> str:
    """Número con separador de miles `.` y coma decimal (locale es-AR); 's/d' si falta."""
    if not _finite(x):
        return "s/d"
    return f"{x:,.{digits}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_pct(x: Any, digits: int = 1) -> str:
    if not _finite(x):
        return "s/d"
    return fmt_num(x * 100, digits) + " %"


def cividis(t: float) -> str:
    """Color de la rampa secuencial para t en [0, 1] (interpolación lineal entre muestras)."""
    t = min(1.0, max(0.0, float(t))) * (len(CIVIDIS) - 1)
    i = min(int(t), len(CIVIDIS) - 2)
    frac = t - i
    a, b = CIVIDIS[i], CIVIDIS[i + 1]
    rgb = [round(int(a[k : k + 2], 16) * (1 - frac) + int(b[k : k + 2], 16) * frac) for k in (1, 3, 5)]
    return "#" + "".join(f"{v:02X}" for v in rgb)


def _uid(kind: str, payload: Any) -> str:
    return f"vg-{kind}-" + hashlib.sha1(repr(payload).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _cat(i: int) -> tuple[str, str, str]:
    """(índice de clase 1..7, color fallback, patrón de trazo) de la serie i."""
    k = i % len(CATEGORY_LIGHT)
    return str(k + 1), CATEGORY_LIGHT[k], DASHES[k]


def _open(kind: str, payload: Any, title: str, desc: str, width: int = W, height: int = H) -> str:
    uid = _uid(kind, payload)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-labelledby="{uid}-t {uid}-d" class="vg-chart vg-{kind}" font-size="12">'
        f'<title id="{uid}-t">{_esc(title)}</title><desc id="{uid}-d">{_esc(desc)}</desc>'
    )


def _text(
    x: float, y: float, text: str, *, anchor: str = "middle", cls: str = "", size: int = 12, rot: bool = False
) -> str:
    transform = f' transform="rotate(-90 {x:.1f} {y:.1f})"' if rot else ""
    klass = f' class="{cls}"' if cls else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
        f'fill="{INK}"{klass}{transform}>'
        f"{_esc(text)}</text>"
    )


def _line(
    x1: float, y1: float, x2: float, y2: float, cls: str, color: str, dash: str = "", width: float = 1
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke="{color}" '
        f'stroke-width="{width}"{dash_attr}/>'
    )


class _Frame:
    """Área de trazado con escalas lineales."""

    def __init__(
        self, x0: float, y0: float, w: float, h: float, xlim: tuple[float, float], ylim: tuple[float, float]
    ):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim = xlim, ylim

    def x(self, v: float) -> float:
        lo, hi = self.xlim
        return self.x0 + (v - lo) / ((hi - lo) or 1.0) * self.w

    def y(self, v: float) -> float:
        lo, hi = self.ylim
        return self.y0 + self.h - (v - lo) / ((hi - lo) or 1.0) * self.h

    def axes(
        self, xticks: Sequence[float], yticks: Sequence[float], xlabel: str, ylabel: str, xfmt: Any, yfmt: Any
    ) -> str:
        parts = []
        for t in yticks:
            parts.append(_line(self.x0, self.y(t), self.x0 + self.w, self.y(t), "vg-grid", GRID))
            parts.append(_text(self.x0 - 6, self.y(t) + 4, yfmt(t), anchor="end", size=11))
        for t in xticks:
            parts.append(_text(self.x(t), self.y0 + self.h + 16, xfmt(t), size=11))
        parts.append(_line(self.x0, self.y0 + self.h, self.x0 + self.w, self.y0 + self.h, "vg-axis", INK))
        parts.append(_line(self.x0, self.y0, self.x0, self.y0 + self.h, "vg-axis", INK))
        parts.append(_text(self.x0 + self.w / 2, self.y0 + self.h + 36, xlabel))
        parts.append(_text(14, self.y0 + self.h / 2, ylabel, rot=True))
        return "".join(parts)

    def polyline(self, xs: Sequence[float], ys: Sequence[float], i: int) -> str:
        k, color, dash = _cat(i)
        pts = " ".join(
            f"{self.x(a):.1f},{self.y(b):.1f}"
            for a, b in zip(xs, ys, strict=True)
            if _finite(a) and _finite(b)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pts}" fill="none" class="vg-s{k}" stroke="{color}" '
            f'stroke-width="2"{dash_attr}/>'
        )


UNIT_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
NICE_STEPS = (1.0, 2.0, 5.0, 10.0)


def nice_ticks(vmax: float, max_intervals: int = 5) -> list[float]:
    """Marcas 0, paso, 2·paso… hasta cubrir vmax, con paso 1, 2 o 5 × 10^k.

    Elige el paso más chico que cubre vmax con a lo sumo `max_intervals` intervalos. La última marca
    es el tope del eje (por ejemplo vmax = 0,89 da 0; 0,2; 0,4; 0,6; 0,8; 1,0).
    """
    if not _finite(vmax) or vmax <= 0:
        raise ValueError("vmax debe ser un número finito positivo")
    mag = 10.0 ** math.floor(math.log10(vmax / max_intervals))
    for m in NICE_STEPS:
        step = m * mag
        n = math.ceil(vmax / step - 1e-9)
        if n <= max_intervals:
            return [round(i * step, 12) for i in range(n + 1)]
    raise AssertionError("inalcanzable: el paso 10 × 10^k siempre cubre vmax")  # pragma: no cover


def _tick_digits(ticks: Sequence[float], scale: float = 1.0) -> int:
    """Decimales justos para escribir el paso de las marcas (multiplicado por `scale`) sin redondeo."""
    step = (ticks[1] - ticks[0]) * scale
    return max(0, math.ceil(-math.log10(step) - 1e-9))


def _unit_frame() -> _Frame:
    return _Frame(ML, MT, W - ML - MR, H - MT - MB, (0.0, 1.0), (0.0, 1.0))


def _legend(frame: _Frame, labels: Sequence[str], *, corner: str = "br") -> str:
    parts = []
    n = len(labels)
    x = frame.x0 + frame.w - 190 if corner.endswith("r") else frame.x0 + 10
    y = frame.y0 + frame.h - 14 * n - 6 if corner.startswith("b") else frame.y0 + 14
    for i, label in enumerate(labels):
        k, color, dash = _cat(i)
        yy = y + 14 * i
        parts.append(_line(x, yy - 4, x + 22, yy - 4, f"vg-s{k}", color, dash, 2))
        parts.append(_text(x + 28, yy, label, anchor="start", size=11))
    return "".join(parts)


def _empty(kind: str, title: str) -> str:
    desc = "Sin datos suficientes para dibujar el gráfico."
    return _open(kind, title, title, desc) + _text(W / 2, H / 2, desc, cls="vg-muted") + "</svg>"


def data_table(caption: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Tabla de datos accesible dentro de `<details>`. La primera columna es encabezado de fila.

    La tabla va en un contenedor enfocable con scroll horizontal propio: una tabla ancha no hace
    scroll en la página entera a 375 px.
    """
    head = "".join(f'<th scope="col">{_esc(h)}</th>' for h in headers)
    body = []
    for row in rows:
        cells = [f'<th scope="row">{_esc(row[0])}</th>'] + [f"<td>{_esc(c)}</td>" for c in row[1:]]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<details class="vg-data"><summary>Ver datos: {_esc(caption)}</summary>'
        f'<div class="vg-table-wrap" tabindex="0" role="region" aria-label="{_esc(caption)}">'
        f"<table><caption>{_esc(caption)}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div></details>"
    )


def figure(svg: str, table_html: str) -> str:
    """Gráfico y su tabla alternativa en un `<figure>`."""
    return f'<figure class="vg-figure">{svg}{table_html}</figure>'


# --- curva de confiabilidad -------------------------------------------------------------------------


def _clean_bins(bins: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        b
        for b in bins
        if _finite(b.get("mean_pred"))
        and _finite(b.get("frac_pos"))
        and _finite(b.get("count"))
        and b["count"] > 0
    ]


MIN_BIN_WIDTH = 0.01  # tope visual de la densidad: un bin más angosto cuenta como de ancho 0,01


def _bin_density(pts: Sequence[Mapping[str, Any]]) -> list[tuple[float, float, float, float]]:
    """(borde inferior, borde superior, densidad, puntaje medio) por bin, ordenados por puntaje medio.

    Los bordes caen a mitad de camino entre bins vecinos (`hi` del anterior y `lo` del siguiente,
    o sus puntajes medios si faltan); los extremos son `lo` del primero y `hi` del último, o 0 y 1.
    La densidad es fracción de tokens / ancho, con ancho mínimo `MIN_BIN_WIDTH`: 1 = uniforme.
    """
    bins = sorted(pts, key=lambda b: b["mean_pred"])
    lo = [b["lo"] if _finite(b.get("lo")) else b["mean_pred"] for b in bins]
    hi = [b["hi"] if _finite(b.get("hi")) else b["mean_pred"] for b in bins]
    first = lo[0] if _finite(bins[0].get("lo")) else 0.0
    last = hi[-1] if _finite(bins[-1].get("hi")) else 1.0
    edges = [first] + [(hi[i] + lo[i + 1]) / 2 for i in range(len(bins) - 1)] + [last]
    edges = [min(1.0, max(0.0, e)) for e in edges]
    total = float(sum(b["count"] for b in bins))
    out = []
    for i, b in enumerate(bins):
        a, z = edges[i], max(edges[i], edges[i + 1])
        out.append((a, z, b["count"] / total / max(z - a, MIN_BIN_WIDTH), b["mean_pred"]))
    return out


def _calibration_word(ece: float) -> str:
    if ece < 0.05:
        return "El puntaje está bien calibrado"
    if ece < 0.1:
        return "La calibración es moderada"
    return "El puntaje está mal calibrado"


def reliability_svg(
    bins: Sequence[Mapping[str, Any]], ece: float, *, title: str = "Curva de confiabilidad"
) -> str:
    """Puntaje medio por bin contra frecuencia observada, con diagonal, densidad de tokens y ECE.

    `bins`: dicts con `mean_pred`, `frac_pos`, `count` y opcionalmente `lo`, `hi`. Los bins vacíos se omiten.
    """
    pts = _clean_bins(bins)
    if not pts or not _finite(ece):
        return _empty("reliability", title)
    worst = max(pts, key=lambda b: abs(b["frac_pos"] - b["mean_pred"]))
    dens = _bin_density(pts)
    densest = max(dens, key=lambda e: e[2])
    desc = (
        f"ECE = {fmt_num(ece)} con {len(pts)} bins de igual masa. {_calibration_word(ece)}: la frecuencia "
        f"observada se aparta en promedio {fmt_num(ece * 100, 1)} puntos del puntaje. La mayor brecha "
        f"está en el bin de puntaje medio {fmt_num(worst['mean_pred'], 2)}, con frecuencia observada "
        f"{fmt_num(worst['frac_pos'], 2)}. La mayor densidad de tokens está en el bin de puntaje "
        f"medio {fmt_num(densest[3], 3)}."
    )
    frame = _Frame(ML, MT, W - ML - MR, 220, (0.0, 1.0), (0.0, 1.0))
    parts = [_open("reliability", (list(map(dict, pts)), ece), title, desc)]
    parts.append(
        frame.axes(
            UNIT_TICKS,
            UNIT_TICKS,
            "",
            "Frecuencia observada",
            lambda t: fmt_num(t, 1),
            lambda t: fmt_num(t, 1),
        )
    )
    parts.append(_line(frame.x(0), frame.y(0), frame.x(1), frame.y(1), "vg-ref", MUTED, "4 4"))
    xs, ys = [b["mean_pred"] for b in pts], [b["frac_pos"] for b in pts]
    parts.append(frame.polyline(xs, ys, 0))
    k, color, _ = _cat(0)
    for x, y in zip(xs, ys, strict=True):
        parts.append(
            f'<circle cx="{frame.x(x):.1f}" cy="{frame.y(y):.1f}" r="3" class="vg-f{k}" fill="{color}"/>'
        )
    parts.append(_text(frame.x0 + 8, frame.y0 + 14, f"ECE = {fmt_num(ece)}", anchor="start"))
    # Densidad por bin: con bins de igual masa el conteo es casi constante y no informa; la
    # densidad (fracción de tokens / ancho del bin, 1 = uniforme) muestra dónde se concentran.
    top = densest[2]
    hist = _Frame(ML, MT + 250, W - ML - MR, 50, (0.0, 1.0), (0.0, top))
    k2, color2, _ = _cat(2)
    for lo, hi, d, _ in dens:
        x0, x1 = hist.x(lo), hist.x(hi)
        y = hist.y(d)
        parts.append(
            f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(1.0, x1 - x0 - 1):.1f}" '
            f'height="{hist.y0 + hist.h - y:.1f}" class="vg-f{k2}" fill="{color2}" fill-opacity="0.5"/>'
        )
    parts.append(_line(hist.x0, hist.y0 + hist.h, hist.x0 + hist.w, hist.y0 + hist.h, "vg-axis", INK))
    parts.append(_line(hist.x0, hist.y0, hist.x0, hist.y0 + hist.h, "vg-axis", INK))
    for v in (0.0, top):
        parts.append(_text(hist.x0 - 6, hist.y(v) + 4, fmt_num(v, 1), anchor="end", size=11))
    for t in UNIT_TICKS:
        parts.append(_text(hist.x(t), hist.y0 + hist.h + 14, fmt_num(t, 1), size=11))
    parts.append(_text(W / 2, H - 4, "Puntaje medio del bin (barras: densidad de tokens, 1 = uniforme)"))
    return "".join(parts) + "</svg>"


def reliability_table(
    bins: Sequence[Mapping[str, Any]], ece: float, *, title: str = "Curva de confiabilidad"
) -> str:
    rows = [
        (i + 1, fmt_num(b["mean_pred"]), fmt_num(b["frac_pos"]), int(b["count"]))
        for i, b in enumerate(_clean_bins(bins))
    ]
    return data_table(
        f"{title} (ECE {fmt_num(ece)})", ("Bin", "Puntaje medio", "Frecuencia observada", "Tokens"), rows
    )


# --- ROC y PR ---------------------------------------------------------------------------------------


def _check_curves(
    curves: Sequence[Mapping[str, Any]], xkey: str, ykey: str, auckey: str
) -> list[Mapping[str, Any]]:
    out = []
    for c in curves:
        xs, ys = list(c.get(xkey, ())), list(c.get(ykey, ()))
        if len(xs) != len(ys):
            raise ValueError(f"la curva {c.get('name')!r} tiene {xkey} e {ykey} de distinto largo")
        if xs and _finite(c.get(auckey)):
            out.append(c)
    return out


def roc_svg(curves: Sequence[Mapping[str, Any]], *, title: str = "Curva ROC") -> str:
    """Curvas ROC. Cada curva: `name`, `fpr`, `tpr`, `auc`. La primera es el detector principal."""
    cs = _check_curves(curves, "fpr", "tpr", "auc")
    if not cs:
        return _empty("roc", title)
    best = max(cs, key=lambda c: c["auc"])
    ranking = "; ".join(f"{c['name']} {fmt_num(c['auc'])}" for c in sorted(cs, key=lambda c: -c["auc"]))
    desc = (
        f"Mayor AUROC: {best['name']} con {fmt_num(best['auc'])}. La diagonal es el azar (0,5). "
        f"AUROC por serie: {ranking}."
    )
    frame = _unit_frame()
    parts = [_open("roc", [(c["name"], c["auc"], len(c["fpr"])) for c in cs], title, desc)]
    parts.append(
        frame.axes(
            UNIT_TICKS,
            UNIT_TICKS,
            "Tasa de falsos positivos",
            "Tasa de verdaderos positivos",
            lambda t: fmt_num(t, 1),
            lambda t: fmt_num(t, 1),
        )
    )
    parts.append(_line(frame.x(0), frame.y(0), frame.x(1), frame.y(1), "vg-ref", MUTED, "4 4"))
    for i, c in enumerate(cs):
        parts.append(frame.polyline(c["fpr"], c["tpr"], i))
    parts.append(_legend(frame, [f"{c['name']} ({fmt_num(c['auc'])})" for c in cs]))
    return "".join(parts) + "</svg>"


def _tpr_at(c: Mapping[str, Any], fpr_max: float) -> float:
    best = 0.0
    for f, t in zip(c["fpr"], c["tpr"], strict=True):
        if _finite(f) and _finite(t) and f <= fpr_max + 1e-12:
            best = max(best, t)
    return best


def roc_table(curves: Sequence[Mapping[str, Any]], *, title: str = "Curva ROC") -> str:
    rows = [
        (c["name"], fmt_num(c["auc"]), fmt_num(_tpr_at(c, 0.1)))
        for c in _check_curves(curves, "fpr", "tpr", "auc")
    ]
    return data_table(title, ("Serie", "AUROC", "TPR con FPR ≤ 0,1"), rows)


def pr_svg(
    curves: Sequence[Mapping[str, Any]],
    prevalence: float,
    *,
    threshold_point: Mapping[str, Any] | None = None,
    title: str = "Curva precisión-recall",
) -> str:
    """Curvas PR con línea base de prevalencia y el punto del umbral actual.

    Cada curva: `name`, `recall`, `precision`, `ap`. `threshold_point`: `recall`, `precision`, `tau`.
    """
    cs = _check_curves(curves, "recall", "precision", "ap")
    if not cs or not _finite(prevalence):
        return _empty("pr", title)
    main = cs[0]
    lift = main["ap"] / prevalence if prevalence > 0 else float("nan")
    desc = (
        f"AUPRC de {main['name']} = {fmt_num(main['ap'])}, {fmt_num(lift, 1)} veces la prevalencia "
        f"({fmt_pct(prevalence)}), que es la base de un clasificador al azar."
    )
    tp = (
        threshold_point
        if threshold_point and all(_finite(threshold_point.get(k)) for k in ("recall", "precision", "tau"))
        else None
    )
    if tp:
        desc += (
            f" Con τ = {fmt_num(tp['tau'])}: precisión {fmt_num(tp['precision'])} "
            f"y recall {fmt_num(tp['recall'])}."
        )
    frame = _unit_frame()
    parts = [_open("pr", ([(c["name"], c["ap"]) for c in cs], prevalence, dict(tp or {})), title, desc)]
    parts.append(
        frame.axes(
            UNIT_TICKS, UNIT_TICKS, "Recall", "Precisión", lambda t: fmt_num(t, 1), lambda t: fmt_num(t, 1)
        )
    )
    yb = frame.y(prevalence)
    parts.append(_line(frame.x(0), yb, frame.x(1), yb, "vg-ref", MUTED, "4 4"))
    legend = [f"{c['name']} ({fmt_num(c['ap'])})" for c in cs]
    obstacles = [_legend_box(frame, legend)]
    base_at = len(parts)
    for i, c in enumerate(cs):
        parts.append(frame.polyline(c["recall"], c["precision"], i))
    if tp:
        x, y = frame.x(tp["recall"]), frame.y(tp["precision"])
        parts.append(
            f'<path d="M{x:.1f} {y - 6:.1f} L{x + 6:.1f} {y:.1f} '
            f'L{x:.1f} {y + 6:.1f} L{x - 6:.1f} {y:.1f} Z" '
            f'class="vg-s3 vg-f1" stroke="{INK}" fill="{CATEGORY_LIGHT[0]}" stroke-width="1.5"/>'
        )
        near_right = x > frame.x0 + frame.w - 80
        label = f"τ = {fmt_num(tp['tau'])}"
        lx, anchor = (x - 9, "end") if near_right else (x + 9, "start")
        parts.append(_text(lx, y - 8, label, anchor=anchor, size=11))
        obstacles += [(x - 6, y - 6, x + 6, y + 6), _text_box(lx, y - 8, label, anchor, 11)]
    parts.insert(base_at, _baseline_label(frame, yb, f"Base: prevalencia {fmt_pct(prevalence)}", obstacles))
    parts.append(_legend(frame, legend, corner="bl"))
    return "".join(parts) + "</svg>"


def _text_box(x: float, y: float, text: str, anchor: str, size: int) -> tuple[float, float, float, float]:
    """Caja estimada (x0, y0, x1, y1) de un texto con línea base en y."""
    w = _text_width(text, size)
    x0 = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
    return (x0, y - 0.8 * size, x0 + w, y + 0.25 * size)


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _legend_box(frame: _Frame, labels: Sequence[str]) -> tuple[float, float, float, float]:
    """Caja de la leyenda que `_legend` dibuja en la esquina inferior izquierda."""
    y = frame.y0 + frame.h - 14 * len(labels) - 6
    w = 28 + max((_text_width(t, 11) for t in labels), default=0.0)
    return (frame.x0 + 10, y - 9, frame.x0 + 10 + w, y + 14 * (len(labels) - 1) + 3)


def _baseline_label(frame: _Frame, yb: float, text: str, obstacles: Sequence[Sequence[float]]) -> str:
    """Etiqueta de la línea base: arriba a la derecha salvo que pise el umbral τ o la leyenda.

    Prueba arriba y abajo de la línea, primero a la derecha y después a la izquierda, y por último
    una fila más arriba a la derecha. Si ninguna posición queda libre dentro del área, usa la primera.
    """
    right, left = frame.x(1) - 4, frame.x(0) + 4
    spots = [
        (right, yb - 4, "end"),
        (right, yb + 16, "end"),
        (left, yb - 4, "start"),
        (left, yb + 16, "start"),
        (right, yb - 22, "end"),
    ]
    for x, y, anchor in spots:
        box = _text_box(x, y, text, anchor, 11)
        inside = frame.y0 <= box[1] and box[3] <= frame.y0 + frame.h
        if inside and not any(_overlaps(box, o) for o in obstacles):
            break
    else:
        x, y, anchor = spots[0]
    return _text(x, y, text, anchor=anchor, cls="vg-muted", size=11)


def pr_table(
    curves: Sequence[Mapping[str, Any]],
    prevalence: float,
    *,
    threshold_point: Mapping[str, Any] | None = None,
    title: str = "Curva precisión-recall",
) -> str:
    rows: list[tuple[Any, ...]] = [
        (c["name"], fmt_num(c["ap"]), fmt_pct(prevalence))
        for c in _check_curves(curves, "recall", "precision", "ap")
    ]
    if threshold_point:
        rows.append(
            (
                f"Umbral τ = {fmt_num(threshold_point.get('tau'))}",
                f"precisión {fmt_num(threshold_point.get('precision'))}",
                f"recall {fmt_num(threshold_point.get('recall'))}",
            )
        )
    return data_table(title, ("Serie", "AUPRC", "Prevalencia"), rows)


# --- histograma de puntajes por clase ---------------------------------------------------------------


def _check_hist(edges: Sequence[float], a: Sequence[float], b: Sequence[float]) -> None:
    if len(edges) != len(a) + 1 or len(a) != len(b):
        raise ValueError("edges debe tener un elemento más que cada vector de conteos")
    if any(not _finite(v) or v < 0 for v in list(a) + list(b)):
        raise ValueError("los conteos deben ser finitos y no negativos")


def _share_above(edges: Sequence[float], counts: Sequence[float], tau: float) -> float:
    total = sum(counts)
    if total <= 0:
        return float("nan")
    above = 0.0
    for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True):
        if lo >= tau:
            above += c
        elif hi > tau:
            above += c * (hi - tau) / (hi - lo)
    return above / total


def score_histogram_svg(
    edges: Sequence[float],
    counts_correct: Sequence[float],
    counts_error: Sequence[float],
    *,
    tau: float | None = None,
    title: str = "Distribución del puntaje por clase",
) -> str:
    """Barras agrupadas por bin con la proporción de cada clase y la línea vertical del umbral."""
    _check_hist(edges, counts_correct, counts_error)
    n_ok, n_err = sum(counts_correct), sum(counts_error)
    if not counts_correct or (n_ok == 0 and n_err == 0):
        return _empty("hist", title)
    share_ok = [c / n_ok if n_ok else 0.0 for c in counts_correct]
    share_err = [c / n_err if n_err else 0.0 for c in counts_error]
    desc = f"{int(n_err)} tokens erróneos y {int(n_ok)} correctos."
    if tau is not None and _finite(tau):
        desc += (
            f" Con τ = {fmt_num(tau)} queda marcado el "
            f"{fmt_pct(_share_above(edges, counts_error, tau))} de los tokens "
            f"erróneos y el {fmt_pct(_share_above(edges, counts_correct, tau))} de los correctos."
        )
    yticks = nice_ticks((max(share_ok + share_err) or 1.0) * 1.05)
    frame = _Frame(
        ML, MT + 18, W - ML - MR, H - MT - MB - 18, (float(edges[0]), float(edges[-1])), (0.0, yticks[-1])
    )
    pct_digits = _tick_digits(yticks, 100.0)
    parts = [_open("hist", (list(edges), list(counts_correct), list(counts_error), tau), title, desc)]
    parts.append(
        frame.axes(
            [float(e) for e in edges[:: max(1, len(edges) // 5)]],
            yticks,
            "Puntaje p_error",
            "Proporción de la clase",
            lambda t: fmt_num(t, 1),
            lambda t: fmt_pct(t, pct_digits),
        )
    )
    for lo, hi, so, se in zip(edges[:-1], edges[1:], share_ok, share_err, strict=True):
        x0, x1 = frame.x(lo), frame.x(hi)
        half = max(0.5, (x1 - x0) / 2 - 0.5)
        for j, share in enumerate((so, se)):
            k, color, _ = _cat(j)
            y = frame.y(share)
            pattern = ' stroke-dasharray="2 1"' if j else ""
            parts.append(
                f'<rect x="{x0 + j * half:.1f}" y="{y:.1f}" width="{half:.1f}" '
                f'height="{frame.y0 + frame.h - y:.1f}" '
                f'class="vg-f{k} vg-s3" fill="{color}" stroke="{INK}" stroke-width="0.5"{pattern}/>'
            )
    if tau is not None and _finite(tau):
        xt = frame.x(tau)
        parts.append(_line(xt, frame.y0, xt, frame.y0 + frame.h, "vg-axis", INK, "5 3", 1.5))
        parts.append(_text(xt + 4, frame.y0 + 12, f"τ = {fmt_num(tau)}", anchor="start", size=11))
    for j, label in enumerate(("Correctos", "Erróneos")):
        k, color, _ = _cat(j)
        x = ML + 10 + j * 110
        parts.append(f'<rect x="{x}" y="{MT}" width="12" height="12" class="vg-f{k}" fill="{color}"/>')
        parts.append(_text(x + 16, MT + 10, label, anchor="start", size=11))
    return "".join(parts) + "</svg>"


def score_histogram_table(
    edges: Sequence[float],
    counts_correct: Sequence[float],
    counts_error: Sequence[float],
    *,
    tau: float | None = None,
    title: str = "Distribución del puntaje por clase",
) -> str:
    _check_hist(edges, counts_correct, counts_error)
    rows = [
        (f"[{fmt_num(lo, 2)}; {fmt_num(hi, 2)})", int(a), int(b))
        for lo, hi, a, b in zip(edges[:-1], edges[1:], counts_correct, counts_error, strict=True)
    ]
    return data_table(title, ("Bin de puntaje", "Correctos", "Erróneos"), rows)


# --- latencia ---------------------------------------------------------------------------------------

PERCENTILES = ("p50", "p95", "p99")


def _overheads(
    baseline: Mapping[str, float], vigia: Mapping[str, float]
) -> list[tuple[str, float, float, float]]:
    out = []
    for p in PERCENTILES:
        b, v = baseline.get(p), vigia.get(p)
        if not (_finite(b) and _finite(v)) or b <= 0 or v < 0:
            raise ValueError(f"latencia {p} inválida: sin Vigía {b!r}, con Vigía {v!r}")
        out.append((p, float(b), float(v), v / b - 1.0))
    return out


def latency_svg(
    baseline: Mapping[str, float],
    vigia: Mapping[str, float],
    *,
    target: float = 0.05,
    unit: str = "ms",
    title: str = "Latencia por token",
) -> str:
    """Barras p50/p95/p99 sin y con Vigía, marca de meta (+target) y alerta si se supera."""
    rows = _overheads(baseline, vigia)
    ok = [p for p, _, _, o in rows if o <= target]
    bad = [p for p, _, _, o in rows if o > target]
    desc = (
        "Overhead " + ", ".join(f"{p} {fmt_pct(o)}" for p, _, _, o in rows) + f". Meta {fmt_pct(target, 0)}: "
    )
    desc += ("cumple en " + ", ".join(ok)) if ok else "no cumple en ningún percentil"
    if bad and ok:
        desc += "; no cumple en " + ", ".join(bad)
    desc += "."
    yticks = nice_ticks(max(max(v, b * (1 + target)) for _, b, v, _ in rows) * 1.15)
    frame = _Frame(ML, MT + 18, W - ML - MR, H - MT - MB - 18, (0.0, 3.0), (0.0, yticks[-1]))
    parts = [_open("latency", (rows, target, unit), title, desc)]
    digits = _tick_digits(yticks)
    parts.append(frame.axes([], yticks, "Percentil", f"Latencia ({unit})", str, lambda t: fmt_num(t, digits)))
    for i, (p, b, v, o) in enumerate(rows):
        cx = frame.x(i + 0.5)
        for j, value in enumerate((b, v)):
            k, color, _ = _cat(j)
            y = frame.y(value)
            parts.append(
                f'<rect x="{cx - 34 + j * 34:.1f}" y="{y:.1f}" width="32" '
                f'height="{frame.y0 + frame.h - y:.1f}" '
                f'class="vg-f{k}" fill="{color}"/>'
            )
        ym = frame.y(b * (1 + target))
        parts.append(_line(cx - 40, ym, cx + 40, ym, "vg-axis", INK, "4 2", 1.5))
        cls = "vg-alert" if o > target else ""
        label = f"+{fmt_pct(o)}" + (" ⚠ supera la meta" if o > target else "")
        parts.append(_text(cx, frame.y(max(b * (1 + target), v)) - 6, label, cls=cls, size=11))
        parts.append(_text(cx, frame.y0 + frame.h + 16, p, size=11))
    for j, label in enumerate(("Sin Vigía", "Con Vigía", f"Meta +{fmt_pct(target, 0)}")):
        x = ML + 10 + j * 120
        if j < 2:
            k, color, _ = _cat(j)
            parts.append(f'<rect x="{x}" y="{MT}" width="12" height="12" class="vg-f{k}" fill="{color}"/>')
        else:
            parts.append(_line(x, MT + 6, x + 12, MT + 6, "vg-axis", INK, "4 2", 1.5))
        parts.append(_text(x + 16, MT + 10, label, anchor="start", size=11))
    return "".join(parts) + "</svg>"


def latency_table(
    baseline: Mapping[str, float],
    vigia: Mapping[str, float],
    *,
    target: float = 0.05,
    unit: str = "ms",
    title: str = "Latencia por token",
) -> str:
    rows = [
        (p, fmt_num(b, 2), fmt_num(v, 2), fmt_pct(o), "cumple" if o <= target else "no cumple")
        for p, b, v, o in _overheads(baseline, vigia)
    ]
    return data_table(
        title,
        ("Percentil", f"Sin Vigía ({unit})", f"Con Vigía ({unit})", "Overhead", f"Meta {fmt_pct(target, 0)}"),
        rows,
    )


# --- tendencia de marcado ---------------------------------------------------------------------------


def _check_trend(labels: Sequence[str], rates: Sequence[float], recalibrations: Sequence[int]) -> None:
    if len(labels) != len(rates):
        raise ValueError("labels y rates deben tener el mismo largo")
    if any(not _finite(r) or not 0 <= r <= 1 for r in rates):
        raise ValueError("las tasas deben estar en [0, 1]")
    if any(
        not isinstance(i, numbers.Integral) or isinstance(i, bool) or not 0 <= i < len(rates)
        for i in recalibrations
    ):
        raise ValueError("cada recalibración es un índice válido de la serie")


def flag_trend_svg(
    labels: Sequence[str],
    rates: Sequence[float],
    *,
    recalibrations: Sequence[int] = (),
    title: str = "Tasa de tokens marcados",
) -> str:
    """Serie temporal de la tasa de marcado con líneas verticales en cada recalibración."""
    _check_trend(labels, rates, recalibrations)
    if not rates:
        return _empty("trend", title)
    imax = max(range(len(rates)), key=lambda i: rates[i])
    desc = (
        f"La tasa de marcado pasó de {fmt_pct(rates[0])} ({labels[0]}) "
        f"a {fmt_pct(rates[-1])} ({labels[-1]}). "
        f"Máximo {fmt_pct(rates[imax])} en {labels[imax]}. Recalibraciones marcadas: {len(recalibrations)}."
    )
    yticks = nice_ticks(max(rates) * 1.15 or 0.05)
    pct_digits = _tick_digits(yticks, 100.0)
    n = len(rates)
    # Un solo día: el punto va al centro del área y no sobre el eje Y.
    xlim = (-0.5, 0.5) if n == 1 else (0.0, float(n - 1))
    frame = _Frame(ML, MT, W - ML - MR, H - MT - MB, xlim, (0.0, yticks[-1]))
    step = max(1, n // 5)
    xticks = sorted(set(list(range(0, n, step)) + [n - 1]))
    parts = [_open("trend", (list(labels), list(rates), list(recalibrations)), title, desc)]
    parts.append(
        frame.axes(
            xticks,
            yticks,
            "Fecha",
            "Tokens marcados",
            lambda t: str(labels[int(t)]),
            lambda t: fmt_pct(t, pct_digits),
        )
    )
    for i in recalibrations:
        x = frame.x(i)
        parts.append(_line(x, frame.y0, x, frame.y0 + frame.h, "vg-ref", MUTED, "3 3", 1.5))
        parts.append(_text(x + 3, frame.y0 + 12, "recal.", anchor="start", cls="vg-muted", size=10))
    parts.append(frame.polyline(list(range(n)), rates, 0))
    k, color, _ = _cat(0)
    radius = 4 if n == 1 else 2.5  # sin línea que una los puntos, el marcador tiene que verse solo
    for i, r in enumerate(rates):
        parts.append(
            f'<circle cx="{frame.x(i):.1f}" cy="{frame.y(r):.1f}" r="{radius}" '
            f'class="vg-f{k}" fill="{color}"/>'
        )
    return "".join(parts) + "</svg>"


def flag_trend_table(
    labels: Sequence[str],
    rates: Sequence[float],
    *,
    recalibrations: Sequence[int] = (),
    title: str = "Tasa de tokens marcados",
) -> str:
    _check_trend(labels, rates, recalibrations)
    marks = set(recalibrations)
    rows = [
        (lab, fmt_pct(r, 2), "sí" if i in marks else "")
        for i, (lab, r) in enumerate(zip(labels, rates, strict=True))
    ]
    return data_table(title, ("Fecha", "Tasa de marcado", "Recalibración"), rows)


# --- heatmap capa x experto -------------------------------------------------------------------------


def _check_matrix(matrix: Sequence[Sequence[float]], rows: Sequence[Any], cols: Sequence[Any]) -> list[float]:
    if len(matrix) != len(rows) or any(len(r) != len(cols) for r in matrix):
        raise ValueError("la matriz debe ser filas x columnas según las etiquetas")
    return [v for r in matrix for v in r if _finite(v)]


HEATMAP_LEGEND_MIN_W = 120.0


def _text_width(text: str, size: int) -> float:
    """Ancho estimado de un texto: 0,62 em por carácter cubre system-ui y sans-serif con margen."""
    return len(text) * size * 0.62


def heatmap_svg(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[Any],
    col_labels: Sequence[Any],
    *,
    value_label: str = "uso",
    title: str = "Uso por capa y experto",
) -> str:
    """Heatmap capa (filas) x experto (columnas) con rampa cividis y leyenda de color."""
    values = _check_matrix(matrix, row_labels, col_labels)
    if not values:
        return _empty("heatmap", title)
    vmin, vmax = min(values), max(values)
    cells = [(i, j, v) for i, r in enumerate(matrix) for j, v in enumerate(r) if _finite(v)]
    imax = max(cells, key=lambda c: c[2])
    imin = min(cells, key=lambda c: c[2])
    desc = (
        f"{value_label.capitalize()} máximo {fmt_num(vmax)} en capa {row_labels[imax[0]]}, "
        f"experto {col_labels[imax[1]]}; "
        f"mínimo {fmt_num(vmin)} en capa {row_labels[imin[0]]}, experto {col_labels[imin[1]]}."
    )
    ncol, nrow = len(col_labels), len(row_labels)
    cell = max(4.0, min(24.0, 640.0 / ncol))
    ch = max(4.0, min(24.0, 480.0 / nrow))
    left, top = 64, 16
    # Leyenda: rampa de ancho lw, mínimo y máximo debajo y el nombre del valor en una línea propia.
    lw = max(HEATMAP_LEGEND_MIN_W, min(200.0, ncol * cell))
    legend_text = f"Color: {value_label}"
    content_w = max(ncol * cell, lw, _text_width(legend_text, 10))
    width = int(math.ceil(left + content_w + 16))
    height = int(top + nrow * ch + 84)
    parts = [
        _open(
            "heatmap",
            ([list(r) for r in matrix], list(row_labels), list(col_labels)),
            title,
            desc,
            width,
            height,
        )
    ]
    span = (vmax - vmin) or 1.0
    for i, r in enumerate(matrix):
        for j, v in enumerate(r):
            x, y = left + j * cell, top + i * ch
            fill = cividis((v - vmin) / span) if _finite(v) else "none"
            extra = "" if _finite(v) else f' stroke="{MUTED}" stroke-dasharray="1 1"'
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{ch:.1f}" fill="{fill}"{extra}/>'
            )
    rstep = max(1, int(12 // ch) + 1)
    for i in range(0, nrow, rstep):
        parts.append(_text(left - 4, top + i * ch + ch / 2 + 4, str(row_labels[i]), anchor="end", size=10))
    cstep = max(1, math.ceil(24 / cell - 1e-9))
    for j in range(0, ncol, cstep):
        parts.append(_text(left + j * cell + cell / 2, top + nrow * ch + 12, str(col_labels[j]), size=10))
    parts.append(_text(left + ncol * cell / 2, top + nrow * ch + 30, "Experto"))
    parts.append(_text(16, top + nrow * ch / 2, "Capa", rot=True))
    # Leyenda de color.
    ly = top + nrow * ch + 40
    for s in range(20):
        parts.append(
            f'<rect x="{left + s * lw / 20:.1f}" y="{ly:.1f}" width="{lw / 20 + 0.5:.1f}" '
            f'height="8" fill="{cividis(s / 19)}"/>'
        )
    parts.append(_text(left, ly + 20, fmt_num(vmin), anchor="start", size=10))
    parts.append(_text(left + lw, ly + 20, fmt_num(vmax), anchor="end", size=10))
    parts.append(_text(left, ly + 36, legend_text, anchor="start", size=10))
    return "".join(parts) + "</svg>"


def heatmap_table(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[Any],
    col_labels: Sequence[Any],
    *,
    value_label: str = "uso",
    title: str = "Uso por capa y experto",
) -> str:
    _check_matrix(matrix, row_labels, col_labels)
    headers = ["Capa"] + [f"Experto {c}" for c in col_labels]
    rows = [[f"Capa {lab}"] + [fmt_num(v) for v in r] for lab, r in zip(row_labels, matrix, strict=True)]
    return data_table(f"{title} ({value_label})", headers, rows)


# --- frontera detección / falsas alarmas por respuesta ----------------------------------------------


def _pareto_points(points: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Puntos con alpha, recall y false_alarm finitos (recall y false_alarm en [0, 1]), por falsa alarma."""
    clean = [
        p
        for p in points
        if all(_finite(p.get(k)) for k in ("alpha", "recall", "false_alarm"))
        and 0 <= p["recall"] <= 1
        and 0 <= p["false_alarm"] <= 1
    ]
    return sorted(clean, key=lambda p: (p["false_alarm"], p["recall"], p["alpha"]))


def _is_recommended(p: Mapping[str, Any], recommended_alpha: Any) -> bool:
    return _finite(recommended_alpha) and abs(p["alpha"] - recommended_alpha) < 1e-9


def pareto_svg(
    points: Sequence[Mapping[str, Any]],
    *,
    recommended_alpha: float | None = None,
    title: str = "Detección y falsas alarmas por respuesta",
) -> str:
    """Frontera por respuesta: falsas alarmas en respuestas buenas (x) contra detección de respuestas
    malas (y). Cada punto lleva su alpha; el recomendado va con un rombo.

    `points`: dicts con `alpha`, `recall`, `false_alarm` y opcionalmente `tau` y `flag_rate`.
    """
    pts = _pareto_points(points)
    if not pts:
        return _empty("pareto", title)
    first, last = pts[0], pts[-1]
    desc = (
        f"{len(pts)} alphas en la frontera. Va de {fmt_pct(first['false_alarm'])} de falsas alarmas "
        f"con {fmt_pct(first['recall'])} de detección (alpha {fmt_num(first['alpha'], 2)}) a "
        f"{fmt_pct(last['false_alarm'])} con {fmt_pct(last['recall'])} (alpha {fmt_num(last['alpha'], 2)})."
    )
    rec = next((p for p in pts if _is_recommended(p, recommended_alpha)), None)
    if rec is not None:
        desc += (
            f" El alpha recomendado, {fmt_num(rec['alpha'], 2)}, detecta el {fmt_pct(rec['recall'])} de "
            f"las respuestas malas y marca el {fmt_pct(rec['false_alarm'])} de las buenas: es la mayor "
            "diferencia entre las dos."
        )
    elif _finite(recommended_alpha):
        desc += f" El alpha recomendado, {fmt_num(recommended_alpha, 2)}, no está en esta frontera."
    frame = _unit_frame()
    parts = [
        _open(
            "pareto",
            ([(p["alpha"], p["recall"], p["false_alarm"]) for p in pts], recommended_alpha),
            title,
            desc,
        )
    ]
    parts.append(
        frame.axes(
            UNIT_TICKS,
            UNIT_TICKS,
            "Falsas alarmas en respuestas buenas",
            "Detección de respuestas malas",
            lambda t: fmt_pct(t, 0),
            lambda t: fmt_pct(t, 0),
        )
    )
    parts.append(_line(frame.x(0), frame.y(0), frame.x(1), frame.y(1), "vg-ref", MUTED, "4 4"))
    parts.append(frame.polyline([p["false_alarm"] for p in pts], [p["recall"] for p in pts], 0))
    k, color, _ = _cat(0)
    for p in pts:
        if p is not rec:
            parts.append(
                f'<circle cx="{frame.x(p["false_alarm"]):.1f}" cy="{frame.y(p["recall"]):.1f}" r="3.5" '
                f'class="vg-f{k}" fill="{color}"/>'
            )
    note = "Número junto a cada punto: alpha. Diagonal: azar."
    obstacles = [_text_box(frame.x0 + frame.w - 4, frame.y0 + frame.h - 8, note, "end", 10)]
    labels: list[str] = []
    # El recomendado se etiqueta primero; los demás, solo si no pisan una etiqueta ya puesta.
    for p in sorted(pts, key=lambda p: (p is not rec, p["alpha"])):
        x, y = frame.x(p["false_alarm"]), frame.y(p["recall"])
        text = f"Recomendado: alpha {fmt_num(p['alpha'], 2)}" if p is rec else fmt_num(p["alpha"], 2)
        size = 11 if p is rec else 10
        candidates = [(x + 7, y + 14, "start"), (x + 7, y - 7, "start"), (x - 7, y + 14, "end")]
        for lx, ly, anchor in candidates:
            box = _text_box(lx, ly, text, anchor, size)
            inside = frame.x0 <= box[0] and box[2] <= frame.x0 + frame.w and frame.y0 <= box[1]
            if inside and box[3] <= frame.y0 + frame.h and not any(_overlaps(box, o) for o in obstacles):
                break
        else:
            if p is not rec:
                continue
            lx, ly, anchor = candidates[0] if x < frame.x0 + frame.w / 2 else candidates[2]
            box = _text_box(lx, ly, text, anchor, size)
        obstacles.append(box)
        labels.append(_text(lx, ly, text, anchor=anchor, size=size))
    if rec is not None:
        x, y = frame.x(rec["false_alarm"]), frame.y(rec["recall"])
        k2, color2, _ = _cat(1)
        parts.append(
            f'<path d="M{x:.1f} {y - 7:.1f} L{x + 7:.1f} {y:.1f} L{x:.1f} {y + 7:.1f} '
            f'L{x - 7:.1f} {y:.1f} Z" class="vg-s3 vg-f{k2}" stroke="{INK}" fill="{color2}" '
            'stroke-width="1.5"/>'
        )
    parts += labels
    parts.append(
        _text(frame.x0 + frame.w - 4, frame.y0 + frame.h - 8, note, anchor="end", cls="vg-muted", size=10)
    )
    return "".join(parts) + "</svg>"


def pareto_table(
    points: Sequence[Mapping[str, Any]],
    *,
    recommended_alpha: float | None = None,
    title: str = "Detección y falsas alarmas por respuesta",
) -> str:
    pts = _pareto_points(points)
    with_rate = any(_finite(p.get("flag_rate")) for p in pts)
    headers = ["alpha", "τ", "Detección de respuestas malas", "Falsas alarmas en respuestas buenas"]
    if with_rate:
        headers.append("Tasa de marcado de tokens")
    rows = []
    for p in pts:
        row = [
            fmt_num(p["alpha"], 2) + (" (recomendado)" if _is_recommended(p, recommended_alpha) else ""),
            fmt_num(p.get("tau")),
            fmt_pct(p["recall"]),
            fmt_pct(p["false_alarm"]),
        ]
        if with_rate:
            row.append(fmt_pct(p.get("flag_rate")))
        rows.append(row)
    return data_table(title, headers, rows)
