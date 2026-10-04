import re
import xml.etree.ElementTree as ET

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
    assert set(charts.CATEGORY_LIGHT) - {"#1F2328"} <= okabe
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
