"""Casos cerrados de las fórmulas de `vigia.features` (AT-S03)."""

import math

import numpy as np
import pytest

from vigia import features as fx

N = 8


def test_at_s03_router_entropy_uniform_and_one_hot():
    """AT-S03: router uniforme → r_ent_n = 1; router one-hot → r_ent_n = 0."""
    uniform = np.zeros((3, N))
    one_hot = np.zeros((3, N))
    one_hot[:, 2] = 1e4
    assert np.allclose(fx.router_entropy_n(uniform), 1.0)
    assert np.allclose(fx.router_entropy_n(one_hot), 0.0, atol=1e-12)


def test_at_s03_usage_closed_values():
    """AT-S03: uso uniforme y one-hot dan sus valores cerrados de u_ihhi, u_gini y u_ent."""
    uniform = np.ones(N)
    one_hot = np.zeros(N)
    one_hot[5] = 3.0
    u = fx.usage_stats(np.stack([uniform, one_hot]))
    assert u["u_ihhi"] == pytest.approx([1.0, 1.0 / N])
    assert u["u_gini"] == pytest.approx([1.0 - 1.0 / N, 0.0])
    assert u["u_ent"] == pytest.approx([1.0, 0.0])


def test_usage_null_is_nan():
    u = fx.usage_stats(np.zeros(N))
    assert all(math.isnan(float(v)) for v in u.values())


def test_usage_counts_fixes_n_even_if_top_experts_unused():
    counts = fx.usage_counts(
        np.array([[0, 1], [1, 0]]), n_experts=6, weights=np.array([[0.7, 0.3], [0.6, 0.4]])
    )
    assert counts.shape == (2, 6)
    assert counts[0, :2] == pytest.approx([0.7, 0.3])
    assert fx.usage_counts(np.array([[0, 1]]), 4)[0].tolist() == [1, 1, 0, 0]
    with pytest.raises(ValueError):
        fx.usage_counts(np.array([[0, 9]]), 4)


def test_output_features_uniform_and_peaked():
    v = 50
    f = fx.output_features(np.zeros((1, v)), np.array([3]))
    assert f["out_entropy"][0] == pytest.approx(math.log(v))
    assert f["out_nll"][0] == pytest.approx(math.log(v))
    assert f["out_margin"][0] == pytest.approx(0.0)
    assert f["out_top20_entropy"][0] == pytest.approx(math.log(20))
    peaked = np.zeros((1, v))
    peaked[0, 7] = 1e4
    g = fx.output_features(peaked, np.array([7]))
    assert g["out_entropy"][0] == pytest.approx(0.0, abs=1e-12)
    assert g["out_nll"][0] == pytest.approx(0.0, abs=1e-12)
    assert g["out_margin"][0] == pytest.approx(1.0)


def test_output_features_match_direct_formula():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 30))
    targets = np.array([0, 5, 9, 29])
    f = fx.output_features(logits, targets)
    q = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    assert f["out_entropy"] == pytest.approx(-(q * np.log(q)).sum(-1))
    assert f["out_nll"] == pytest.approx(-np.log(q[np.arange(4), targets]))
    s = np.sort(q, -1)[:, ::-1]
    assert f["out_margin"] == pytest.approx(s[:, 0] - s[:, 1])
    top = s[:, :20] / s[:, :20].sum(-1, keepdims=True)
    assert f["out_top20_entropy"] == pytest.approx(-(top * np.log(top)).sum(-1))


def test_topk_weight_entropy_and_mass_and_margins():
    assert fx.topk_weight_entropy(np.array([0.25, 0.25])) == pytest.approx(1.0)
    assert fx.topk_weight_entropy(np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert fx.topk_weight_entropy(np.array([1.0])) == pytest.approx(0.0)
    p = np.array([0.5, 0.3, 0.2])
    assert fx.topk_mass(p, np.array([0, 2])) == pytest.approx(0.7)
    m12, mkk1 = fx.logit_margins(np.array([3.0, 1.0, 2.5, 0.0]), k=2)
    assert m12 == pytest.approx(0.5)
    assert mkk1 == pytest.approx(1.5)
    _, none = fx.logit_margins(np.array([1.0, 0.0]), k=2)
    assert math.isnan(float(none))


def test_route_softmax_topk_with_and_without_renorm():
    z = np.array([[2.0, 0.0, 1.0, -1.0]])
    p = fx.softmax(z)
    idx, w = fx.route(z, "softmax_topk", 2)
    assert idx.tolist() == [[0, 2]]
    assert w[0] == pytest.approx(p[0, [0, 2]])
    _, wn = fx.route(z, "softmax_topk", 2, renormalize=True)
    assert wn.sum() == pytest.approx(1.0)


def test_route_topk_softmax():
    z = np.array([[0.1, 3.0, 2.0, -5.0]])
    idx, w = fx.route(z, "topk_softmax", 2)
    assert idx.tolist() == [[1, 2]]
    assert w[0] == pytest.approx(fx.softmax(np.array([3.0, 2.0])))


def test_route_sigmoid_group_uses_bias_for_choice_and_groups():
    z = np.array([[2.0, 1.9, -3.0, 0.5]])
    bias = np.array([0.0, 0.0, 10.0, 0.0])
    idx, w = fx.route(z, "sigmoid_group", 1, correction_bias=bias, scaling=2.0)
    assert idx.tolist() == [[2]]  # el sesgo cambia la elección
    assert w[0, 0] == pytest.approx(2.0 / (1 + math.exp(3.0)))  # pero no el peso
    # Dos grupos {0,1} y {2,3}; con topk_group=1 gana el grupo {0,1}.
    idx2, w2 = fx.route(z, "sigmoid_group", 2, n_group=2, topk_group=1, renormalize=True)
    assert sorted(idx2[0].tolist()) == [0, 1]
    assert w2.sum() == pytest.approx(1.0)


def test_route_sparsemixer_top2():
    z = np.array([[1.0, 3.0, 2.9, -2.0]])
    idx, w = fx.route(z, "sparsemixer", 2, jitter_eps=0.01)
    assert idx.tolist() == [[1, 2]]
    assert w[0, 0] == pytest.approx(1.0)  # los demás quedan fuera del umbral relativo
    with pytest.raises(ValueError):
        fx.route(z, "sparsemixer", 3)


def test_route_rejects_bad_k_and_gating():
    with pytest.raises(ValueError):
        fx.route(np.zeros((1, 4)), "softmax_topk", 5)
    with pytest.raises(ValueError):
        fx.route(np.zeros((1, 4)), "otro", 2)  # type: ignore[arg-type]


def test_router_stats_keys():
    stats = fx.router_stats(np.random.default_rng(1).normal(size=(5, N)), "softmax_topk", 2)
    assert set(stats) == {"r_ent_n", "r_ent_k", "r_topk_mass", "r_margin12", "r_margin_kk1", "idx", "w"}
    assert stats["idx"].shape == (5, 2)


def test_route_switch():
    assert fx.route_switch(np.array([1, 2]), np.array([2, 1])) == pytest.approx(0.0)
    assert fx.route_switch(np.array([1, 2]), np.array([3, 4])) == pytest.approx(1.0)
    assert fx.route_switch(np.array([1, 2]), np.array([2, 5])) == pytest.approx(0.5)


def test_expert_cosine_and_dispersion_closed_cases():
    e = np.ones((3, 4))
    w = np.array([0.5, 0.3, 0.2])
    assert fx.expert_cosine(e, w) == pytest.approx(1.0)
    assert fx.expert_dispersion(e, w) == pytest.approx(0.0)
    opposite = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert fx.expert_cosine(opposite, np.array([0.5, 0.5])) == pytest.approx(-1.0)
    assert math.isnan(float(fx.expert_dispersion(opposite, np.array([0.5, 0.5]))))
    assert math.isnan(float(fx.expert_cosine(np.ones((1, 3)), np.array([1.0]))))


def test_expert_dispersion_formula():
    rng = np.random.default_rng(2)
    e = rng.normal(size=(3, 5))
    w = np.array([2.0, 1.0, 1.0])
    wn = w / w.sum()
    mean = (wn[:, None] * e).sum(0)
    expected = (wn * ((e - mean) ** 2).sum(-1)).sum() / (mean @ mean)
    assert fx.expert_dispersion(e, w) == pytest.approx(expected)


def test_weighted_jsd_bounds_and_closed_cases():
    p = np.array([[0.2, 0.8], [0.2, 0.8]])
    assert fx.weighted_jsd(p, np.array([0.5, 0.5])) == pytest.approx(0.0, abs=1e-12)
    disjoint = np.eye(3)
    assert fx.weighted_jsd(disjoint, np.ones(3) / 3) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    q = rng.dirichlet(np.ones(10), size=(20, 4))
    vals = fx.weighted_jsd(q, rng.dirichlet(np.ones(4), size=20))
    assert np.all((vals >= 0) & (vals <= 1))


def test_layer_aggregates_ignore_nan():
    a = fx.layer_aggregates(np.array([1.0, np.nan, 3.0]))
    assert a["mean"] == pytest.approx(2.0)
    assert a["max"] == pytest.approx(3.0)
    assert a["std"] == pytest.approx(1.0)
    b = fx.layer_aggregates(np.array([np.nan, np.nan]))
    assert all(math.isnan(float(v)) for v in b.values())


@pytest.mark.parametrize(
    ("gating", "kw"),
    [
        ("softmax_topk", {"renormalize": True}),
        ("softmax_topk", {}),
        ("topk_softmax", {}),
        ("sigmoid_group", {"n_group": 2, "topk_group": 1, "renormalize": True, "scaling": 2.5}),
        ("sparsemixer", {"jitter_eps": 0.01}),
    ],
)
def test_router_stats_equals_composition(gating, kw):
    """La ruta rápida de `router_stats` coincide con componer las fórmulas individuales."""
    z = np.random.default_rng(4).normal(size=(3, 5, 8))
    stats = fx.router_stats(z, gating, 2, **kw)
    idx, w = fx.route(z, gating, 2, **kw)
    m12, mkk1 = fx.logit_margins(z, 2)
    assert np.array_equal(stats["idx"], idx) and np.allclose(stats["w"], w)
    assert np.allclose(stats["r_ent_n"], fx.router_entropy_n(z))
    assert np.allclose(stats["r_ent_k"], fx.topk_weight_entropy(w))
    assert np.allclose(stats["r_topk_mass"], fx.topk_mass(fx.softmax(z), idx))
    assert np.allclose(stats["r_margin12"], m12) and np.allclose(stats["r_margin_kk1"], mkk1)
