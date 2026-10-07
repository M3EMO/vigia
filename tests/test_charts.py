import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vigia import charts

NS = {"svg": "http://www.w3.org/2000/svg"}

BINS = [
    {"mean_pred": 0.1, "frac_pos": 0.12, "count": 40, "lo": 0.0, "hi": 0.2},
    {"mean_pred": 0.5, "frac_pos": 0.45, "count": 40, "lo": 0.2, "hi": 0.7},
    {"mean_pred": 0.85, "frac_pos": 0.9, "count": 40, "lo": 0.7, "hi": 1.0},
]
ROC = [
    {"name": "Detector", "fpr": [0, 0.05, 0.2, 1], "tpr": [0, 0.6, 0.9, 1], "auc": 0.91},
    {"name": "Baseline out_entropy", "fpr": [0, 0.3, 1], "tpr": [0, 0.5, 1], "auc": 0.64},
]
PR = [{"name": "Detector", "recall": [0, 0.5, 1], "precision": [1, 0.8, 0.2], "ap": 0.7}]
# Frontera medida con Granite (barrido fino de alpha): 0,10 marca demasiadas buenas, 0,14 es el codo.
PARETO = [
    {"alpha": 0.1, "tau": 0.2, "recall": 0.99, "false_alarm": 0.6, "flag_rate": 0.5},
    {"alpha": 0.14, "tau": 0.3, "recall": 0.96, "false_alarm": 0.2, "flag_rate": 0.3},
    {"alpha": 0.32, "tau": 0.5, "recall": 0.85, "false_alarm": 0.15, "flag_rate": 0.2},
]


def _all_charts():
    return {
        "reliability": charts.reliability_svg(BINS, 0.031),
        "roc": charts.roc_svg(ROC),
        "pr": charts.pr_svg(PR, 0.2, threshold_point={"recall": 0.8, "precision": 0.6, "tau": 0.42}),
        "hist": charts.score_histogram_svg(
            [0, 0.25, 0.5, 0.75, 1], [50, 30, 15, 5], [5, 10, 25, 60], tau=0.5
        ),
        "latency": charts.latency_svg(
            {"p50": 10, "p95": 12, "p99": 20}, {"p50": 10.2, "p95": 12.4, "p99": 22}
        ),
        "trend": charts.flag_trend_svg(
            ["01-10", "02-10", "03-10", "04-10"], [0.04, 0.05, 0.07, 0.06], recalibrations=[2]
        ),
        "heatmap": charts.heatmap_svg([[0.1, 0.5, 0.2], [0.9, 0.0, 0.3]], [0, 1], [0, 1, 2]),
        "pareto": charts.pareto_svg(PARETO, recommended_alpha=0.14),
    }


def _parse(svg):
    root = ET.fromstring(svg)  # noqa: S314 - SVG generado por el propio módulo
    title, desc = root.find("svg:title", NS), root.find("svg:desc", NS)
    return root, title, desc


@pytest.mark.parametrize("kind", list(_all_charts()))
def test_every_chart_is_accessible_svg(kind):
    svg = _all_charts()[kind]
    root, title, desc = _parse(svg)
    assert root.get("role") == "img"
    assert root.get("aria-labelledby") == f"{title.get('id')} {desc.get('id')}"
    assert title.text and desc.text and len(desc.text) > 30
    assert root.get("viewBox") and root.get("width") == "100%"
    assert "style=" not in svg  # sin estilos en línea: compatible con CSP sin unsafe-inline


def test_descriptions_state_the_conclusion():
    c = _all_charts()
    assert (
        "ECE = 0,031" in _parse(c["reliability"])[2].text
        and "bien calibrado" in _parse(c["reliability"])[2].text
    )
    assert "Mayor AUROC: Detector con 0,910" in _parse(c["roc"])[2].text
    pr_desc = _parse(c["pr"])[2].text
    assert "3,5 veces la prevalencia (20,0 %)" in pr_desc and "τ = 0,420" in pr_desc
    hist_desc = _parse(c["hist"])[2].text
    assert "85,0 % de los tokens erróneos" in hist_desc and "20,0 % de los correctos" in hist_desc
    lat_desc = _parse(c["latency"])[2].text
    assert "cumple en p50, p95" in lat_desc and "no cumple en p99" in lat_desc
    assert "pasó de 4,0 % (01-10) a 6,0 % (04-10)" in _parse(c["trend"])[2].text
    assert "máximo 0,900 en capa 1, experto 0" in _parse(c["heatmap"])[2].text


def test_latency_alert_marks_percentiles_over_target():
    svg = charts.latency_svg({"p50": 10, "p95": 12, "p99": 20}, {"p50": 10.2, "p95": 12.4, "p99": 22})
    assert svg.count('class="vg-alert"') == 1 and "supera la meta" in svg
    with pytest.raises(ValueError):
        charts.latency_svg({"p50": 0, "p95": 1, "p99": 1}, {"p50": 1, "p95": 1, "p99": 1})


def test_text_is_escaped():
    evil = [{"name": '<script>alert("x")</script> & co', "fpr": [0, 1], "tpr": [0, 1], "auc": 0.5}]
    svg = charts.roc_svg(evil)
    assert "<script>" not in svg
    root, _, _ = _parse(svg)  # XML válido aun con el texto hostil
    texts = "".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert '<script>alert("x")</script> & co' in texts
    table = charts.roc_table(evil)
    assert "&lt;script&gt;" in table and "<script>" not in table


def test_colors_use_css_variables_with_fallback_and_no_red_green():
    css = charts.CHART_CSS
    assert "prefers-color-scheme: dark" in css and ':root[data-theme="dark"]' in css
    assert ".vg-s1{stroke:var(--vg-cat-1,#0072B2)}" in css
    svg = charts.roc_svg(ROC)
    assert 'class="vg-s1" stroke="#0072B2"' in svg and 'class="vg-s2" stroke="#D55E00"' in svg
    okabe = {c.upper() for c in charts.OKABE_ITO.values()}
    assert set(charts.CATEGORY_LIGHT[:5]) - {"#1F2328"} <= okabe  # 6 y 7: variantes oscurecidas
    assert set(charts.CATEGORY_DARK) - {"#E6EDF3"} <= okabe
    all_svg = "".join(_all_charts().values()).upper()
    for forbidden in ("#FF0000", "#00FF00", "#008000", "RED", "GREEN"):
        assert forbidden not in all_svg
    assert charts.CATEGORY_LIGHT[:2] == ("#0072B2", "#D55E00")  # par principal azul/bermellón


def test_cividis_ramp_and_heatmap_cells():
    assert charts.cividis(0) == "#00204D" and charts.cividis(1) == "#FFEA46"
    assert charts.cividis(-1) == "#00204D" and charts.cividis(2) == "#FFEA46"
    svg = charts.heatmap_svg([[0.0, float("nan")], [1.0, 0.5]], ["L0", "L1"], [0, 1])
    assert 'fill="#00204D"' in svg and 'fill="#FFEA46"' in svg and 'fill="none"' in svg
    with pytest.raises(ValueError):
        charts.heatmap_svg([[1.0]], [0, 1], [0])


def test_tables_are_accessible_details():
    table = charts.reliability_table(BINS, 0.031)
    assert table.startswith('<details class="vg-data"><summary>Ver datos: ')
    assert table.count('<th scope="row">') == 3 and table.count('<th scope="col">') == 4
    assert "0,450" in table
    for fn, args, kwargs in [
        (charts.roc_table, (ROC,), {}),
        (charts.pr_table, (PR, 0.2), {"threshold_point": {"recall": 0.8, "precision": 0.6, "tau": 0.42}}),
        (charts.score_histogram_table, ([0, 0.5, 1], [3, 1], [1, 3]), {"tau": 0.5}),
        (charts.latency_table, ({"p50": 1, "p95": 2, "p99": 3}, {"p50": 1, "p95": 2, "p99": 3.3}), {}),
        (charts.flag_trend_table, (["a", "b"], [0.1, 0.2]), {"recalibrations": [1]}),
        (charts.heatmap_table, ([[1.0, 2.0]], [0], [0, 1]), {}),
    ]:
        html = fn(*args, **kwargs)
        assert "<table>" in html and "<caption>" in html and re.search(r"<th scope=\"row\">", html)
    assert "no cumple" in charts.latency_table(
        {"p50": 1, "p95": 2, "p99": 3}, {"p50": 1, "p95": 2, "p99": 3.3}
    )
    fig = charts.figure(charts.roc_svg(ROC), charts.roc_table(ROC))
    assert fig.startswith('<figure class="vg-figure"><svg') and fig.endswith("</details></figure>")


def test_empty_and_invalid_inputs():
    assert "Sin datos" in charts.reliability_svg([], 0.1)
    assert "Sin datos" in charts.roc_svg([])
    assert "Sin datos" in charts.flag_trend_svg([], [])
    assert "Sin datos" in charts.heatmap_svg([[float("nan")]], [0], [0])
    with pytest.raises(ValueError):
        charts.roc_svg([{"name": "x", "fpr": [0, 1], "tpr": [0], "auc": 0.5}])
    with pytest.raises(ValueError):
        charts.score_histogram_svg([0, 1], [1, 2], [1])
    with pytest.raises(ValueError):
        charts.flag_trend_svg(["a"], [1.5])
    with pytest.raises(ValueError):
        charts.flag_trend_svg(["a"], [0.5], recalibrations=[3])


def test_charts_are_pure_and_deterministic():
    assert _all_charts() == _all_charts()
    a = charts.roc_svg(ROC)
    b = charts.roc_svg(ROC[:1])
    assert _parse(a)[1].get("id") != _parse(b)[1].get("id")  # ids distintos para datos distintos


def test_numpy_scalars_are_accepted():
    import numpy as np

    svg = charts.heatmap_svg([[np.int64(1), np.int64(2)]], [0], [0, 1])
    assert "Sin datos" not in svg and 'fill="#FFEA46"' in svg
    assert "recal." in charts.flag_trend_svg(["a", "b"], [np.float32(0.1), 0.2], recalibrations=[np.int64(1)])


def _texts(svg):
    root = ET.fromstring(svg)  # noqa: S314 - SVG generado por el propio módulo
    return root, list(root.iter("{http://www.w3.org/2000/svg}text"))


def _y_tick_labels(svg):
    x = f"{charts.ML - 6:.1f}"
    return [t.text for t in _texts(svg)[1] if t.get("text-anchor") == "end" and t.get("x") == x]


def test_y_axis_ticks_are_nice_numbers():
    """Regresión: "Las marcas de los ejes Y no son números redondos (85,0 / 63,7 / 42,5 / 21,2;
    3390,2 / 2542,6 / 1695,1 / 847,5)"."""
    assert charts.nice_ticks(0.8925) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert charts.nice_ticks(3390.2) == [0.0, 1000.0, 2000.0, 3000.0, 4000.0]
    assert charts.nice_ticks(0.0012) == [0.0, 0.0005, 0.001, 0.0015]
    trend = charts.flag_trend_svg(["a", "b", "c"], [0.6, 0.74, 0.3])
    assert _y_tick_labels(trend) == ["0 %", "20 %", "40 %", "60 %", "80 %", "100 %"]
    small = charts.flag_trend_svg(["a", "b"], [0.0004, 0.001])
    assert _y_tick_labels(small) == ["0,00 %", "0,05 %", "0,10 %", "0,15 %"]
    hist = _all_charts()["hist"]
    assert _y_tick_labels(hist) == ["0 %", "20 %", "40 %", "60 %", "80 %"]
    lat = charts.latency_svg({"p50": 1000, "p95": 2000, "p99": 2600}, {"p50": 1010, "p95": 2100, "p99": 2948})
    assert _y_tick_labels(lat) == ["0", "1.000", "2.000", "3.000", "4.000"]
    with pytest.raises(ValueError):
        charts.nice_ticks(0.0)


def test_heatmap_legend_fits_with_few_experts():
    """Regresión: "Leyenda del heatmap superpuesta y fuera del viewBox con pocos expertos (por
    ejemplo Mixtral con 8)"."""
    label = "fracción de selecciones"
    matrix = [[0.215, 0.25, 0.285, 0.25], [0.24, 0.26, 0.25, 0.25]]
    svg = charts.heatmap_svg(matrix, [0, 1], [0, 1, 2, 3], value_label=label)
    root, texts = _texts(svg)
    _, _, width, height = (float(v) for v in root.get("viewBox").split())
    by_text = {t.text: t for t in texts}
    # Con celdas de 24 entran todos los rótulos de expertos.
    assert all(str(j) in by_text for j in range(4))
    # El nombre del valor va en su propia línea y entra en el viewBox.
    legend = by_text[f"Color: {label}"]
    assert legend.get("text-anchor") == "start"
    assert float(legend.get("x")) + charts._text_width(legend.text, 10) <= width
    vmin, vmax = by_text["0,215"], by_text["0,285"]
    assert float(vmax.get("x")) - float(vmin.get("x")) >= charts.HEATMAP_LEGEND_MIN_W
    assert float(legend.get("y")) - float(vmax.get("y")) >= 14
    # "Experto" queda debajo de los rótulos de columna, sin pisarlos.
    assert float(by_text["Experto"].get("y")) - float(by_text["2"].get("y")) >= 16
    assert all(float(t.get("y")) <= height - 4 for t in texts)


def test_data_tables_scroll_inside_their_own_focusable_region():
    """Regresión: "Las tablas 'Ver datos' de los gráficos no tienen contenedor con scroll y provocan
    scroll horizontal de la página a 375 px y 320 px"."""
    table = charts.data_table("Latencia <por> request", ("Serie", "p50"), [("a", "1")])
    m = re.search(r'<details class="vg-data"><summary>[^<]*</summary>(<div [^>]*>)<table>', table)
    assert m, table
    wrap = m.group(1)
    assert 'class="vg-table-wrap"' in wrap and 'tabindex="0"' in wrap and 'role="region"' in wrap
    assert 'aria-label="Latencia &lt;por&gt; request"' in wrap
    assert table.endswith("</table></div></details>")
    assert ".vg-data .vg-table-wrap{overflow-x:auto;max-width:100%}" in charts.CHART_CSS


def test_numbers_use_thousands_separator_like_the_panel():
    """Regresión: "Los números de los gráficos y de sus tablas no usan separador de miles, a
    diferencia del resto del panel"."""
    assert charts.fmt_num(3176.666, 2) == "3.176,67"
    assert charts.fmt_num(6000, 0) == "6.000"
    assert charts.fmt_num(-1234567.5, 1) == "-1.234.567,5"
    assert charts.fmt_num(0.45) == "0,450"
    assert charts.fmt_pct(12.5, 0) == "1.250 %"
    lat = charts.latency_table({"p50": 3176.666, "p95": 4203.94, "p99": 5000}, {"p50": 1, "p95": 2, "p99": 3})
    assert "3.176,67" in lat and "4.203,94" in lat
    svg = charts.latency_svg({"p50": 1000, "p95": 5000, "p99": 5200}, {"p50": 1010, "p95": 5100, "p99": 5300})
    assert "6.000" in _y_tick_labels(svg)


def _box(t):
    size = int(t.get("font-size"))
    x, y, w = float(t.get("x")), float(t.get("y")), len(t.text) * size * 0.6
    x0 = {"end": x - w, "middle": x - w / 2}.get(t.get("text-anchor"), x)
    return (x0, y - 0.75 * size, x0 + w, y + 0.2 * size)


@pytest.mark.parametrize(
    ("prevalence", "point"),
    [
        (0.3, {"recall": 1.0, "precision": 0.3, "tau": 0.0}),  # τ = 0: precisión igual a la prevalencia
        (0.3, {"recall": 0.97, "precision": 0.33, "tau": 0.01}),
        (0.05, {"recall": 1.0, "precision": 0.05, "tau": 0.0}),  # base cerca del eje X
        (0.9, {"recall": 1.0, "precision": 0.9, "tau": 0.0}),
        (0.3, {"recall": 0.5, "precision": 0.32, "tau": 0.4}),
    ],
)
def test_pr_baseline_label_does_not_overlap_threshold_label(prevalence, point):
    """Regresión: "En la curva precisión-recall la etiqueta 'Base: prevalencia' se pisa con 'τ = …'
    cuando el punto del umbral cae en recall 1"."""
    curve = [{"name": "Detector", "recall": [0, 0.5, 1], "precision": [1, 0.6, prevalence], "ap": 0.6}]
    root, texts = _texts(charts.pr_svg(curve, prevalence, threshold_point=point))
    base = next(t for t in texts if t.text.startswith("Base: prevalencia"))
    tau = next(t for t in texts if t.text.startswith("τ = "))
    a, b = _box(base), _box(tau)
    assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), (base.attrib, tau.attrib)
    # La etiqueta de la base queda dentro del área de trazado.
    assert charts.MT <= a[1] and a[3] <= charts.H - charts.MB


def test_pr_baseline_label_keeps_its_place_when_threshold_is_far():
    root, texts = _texts(
        charts.pr_svg(PR, 0.2, threshold_point={"recall": 0.8, "precision": 0.6, "tau": 0.42})
    )
    base = next(t for t in texts if t.text.startswith("Base: prevalencia"))
    frame = charts._unit_frame()
    assert base.get("text-anchor") == "end"
    assert float(base.get("x")) == pytest.approx(round(frame.x(1) - 4, 1))
    assert float(base.get("y")) == pytest.approx(round(frame.y(0.2) - 4, 1))


def test_single_day_trend_point_is_centered_and_visible():
    """Regresión: "La tendencia de marcado con un solo día dibuja su único punto encima del eje Y"."""
    root, _ = _texts(charts.flag_trend_svg(["04-10"], [0.12]))
    circle = root.find("svg:circle", NS)
    frame = charts._unit_frame()
    assert float(circle.get("cx")) == pytest.approx(frame.x0 + frame.w / 2, abs=0.1)
    assert float(circle.get("r")) >= 4
    # Con más de un día, el primer punto sigue en el borde izquierdo y el radio no cambia.
    root, _ = _texts(charts.flag_trend_svg(["a", "b"], [0.1, 0.2]))
    first = root.find("svg:circle", NS)
    assert float(first.get("cx")) == pytest.approx(charts.ML) and first.get("r") == "2.5"


def _contrast(a, b):
    def lum(h):
        c = [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_light_series_have_3_to_1_contrast_on_white():
    """Regresión: "Dos series del gráfico ROC en tema claro tienen contraste no textual menor que
    3:1 (#56B4E9 y #E69F00 sobre blanco)". Cada trazo de datos llega a 3:1 (WCAG 1.4.11) en claro
    sobre #FFFFFF y en oscuro sobre #0D1117, y las 7 series siguen siendo colores distintos."""
    for color in charts.CATEGORY_LIGHT:
        assert _contrast(color, "#FFFFFF") >= 3.0, color
    for color in charts.CATEGORY_DARK:
        assert _contrast(color, "#0D1117") >= 3.0, color
    assert len(set(charts.CATEGORY_LIGHT)) == len(charts.CATEGORY_LIGHT)
    roc7 = [{"name": f"s{i}", "fpr": [0, 1], "tpr": [0, 1], "auc": 0.5} for i in range(7)]
    svg = charts.roc_svg(roc7)
    assert "#56B4E9" not in svg and "#E69F00" not in svg
    assert f'class="vg-s6" stroke="{charts.CATEGORY_LIGHT[5]}"' in svg


def _rect_heights(svg):
    root, _, _ = _parse(svg)
    return [float(r.get("height")) for r in root.iter("{http://www.w3.org/2000/svg}rect")]


def test_reliability_bars_encode_density_not_equal_mass_counts():
    """Regresión: "El histograma de la curva de confiabilidad es un bloque gris uniforme: con bins de
    igual masa todas las barras miden lo mismo". Con bins de igual masa y anchos distintos, la
    altura codifica la densidad (fracción / ancho) y el eje la rotula."""
    from vigia.evaluate import reliability_bins

    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0.0, 0.05, 500), rng.uniform(0.3, 0.95, 1000)])
    y = (rng.random(len(p)) < p).astype(int)
    bins = reliability_bins(y, p)
    assert len({b["count"] for b in bins}) <= 2  # igual masa: los conteos no informan
    svg = charts.reliability_svg(bins, 0.03)
    heights = _rect_heights(svg)
    assert len(heights) == len(bins)
    assert max(heights) == pytest.approx(50.0) and min(heights) < 10  # varía con la densidad
    assert "densidad de tokens" in svg and "conteo por bin" not in svg
    assert "La mayor densidad de tokens está en el bin de puntaje medio 0,0" in _parse(svg)[2].text
    # Bins contiguos sin superposición: cada barra empieza donde termina la anterior (± 1 px).
    root, _, _ = _parse(svg)
    xs = [(float(r.get("x")), float(r.get("width"))) for r in root.iter("{http://www.w3.org/2000/svg}rect")]
    for (x0, w0), (x1, _) in zip(xs, xs[1:], strict=False):
        assert x1 >= x0 + w0 - 1e-6
    # Densidad exacta en los bins de BINS: 1/3 de los tokens en anchos 0,2, 0,5 y 0,3 (bordes en
    # 0,2 y 0,7): la barra más alta es la del bin más angosto.
    hb = _rect_heights(charts.reliability_svg(BINS, 0.031))
    assert hb[0] == pytest.approx(50.0) and hb[1] == pytest.approx(50.0 * 0.2 / 0.5, abs=0.15)


def test_pareto_marks_recommended_alpha_and_states_the_tradeoff():
    svg = charts.pareto_svg(PARETO, recommended_alpha=0.14)
    desc = _parse(svg)[2].text
    assert "3 alphas en la frontera" in desc
    assert "El alpha recomendado, 0,14, detecta el 96,0 % de las respuestas malas" in desc
    assert "marca el 20,0 % de las buenas" in desc
    texts = [t.text for t in _texts(svg)[1]]
    assert "Recomendado: alpha 0,14" in texts and "0,10" in texts and "0,32" in texts
    assert "Falsas alarmas en respuestas buenas" in texts and "Detección de respuestas malas" in texts
    assert svg.count("<path ") == 1  # rombo del recomendado
    root = ET.fromstring(svg)  # noqa: S314 - SVG generado por el propio módulo
    boxes = [
        _box(t)
        for t in root.iter("{http://www.w3.org/2000/svg}text")
        if t.get("font-size") in ("10", "11")
        and (t.text or "").startswith(("0,", "Recomendado"))
        and "%" not in (t.text or "")
    ]
    for i, a in enumerate(boxes):  # las etiquetas de alpha no se pisan
        for b in boxes[i + 1 :]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    other = _parse(charts.pareto_svg(PARETO[:1], recommended_alpha=0.14))[2].text
    assert "no está en esta frontera" in other
    assert "Sin datos" in charts.pareto_svg([{"alpha": 0.1, "recall": None, "false_alarm": 0.2}])


def test_pareto_table_lists_alpha_detection_false_alarms_and_flag_rate():
    table = charts.pareto_table(PARETO, recommended_alpha=0.14)
    assert table.startswith('<details class="vg-data">')
    for header in (
        "alpha",
        "Detección de respuestas malas",
        "Falsas alarmas en respuestas buenas",
        "Tasa de marcado de tokens",
    ):
        assert f'<th scope="col">{header}</th>' in table
    assert '<th scope="row">0,14 (recomendado)</th>' in table and "96,0 %" in table
    assert "Tasa de marcado" not in charts.pareto_table(
        [{k: v for k, v in PARETO[0].items() if k != "flag_rate"}]
    )
