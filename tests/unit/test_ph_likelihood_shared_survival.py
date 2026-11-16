"""Shared PH survival retains the defining law and safe inputs (spec 4.4/7)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL, TAIL_RTOL

import survjax

POINTS = {
    "exponential": [[0.75, 0.25], [1.25, -0.375]],
    "weibull": [[0.75, 1.5, 0.25], [1.75, 2.0, -0.375]],
    "lognormal": [[0.25, 1.25, 0.25], [-0.375, 0.875, -0.375]],
    "loglogistic": [[0.75, 1.5, 0.25], [2.5, 2.0, -0.375]],
}


def _distribution(family, p):
    if family == "exponential":
        base = survjax.Exponential(rate=p[0])
    elif family == "weibull":
        base = survjax.Weibull(shape=p[0], scale=p[1])
    elif family == "lognormal":
        base = survjax.LogNormal(meanlog=p[0], sdlog=p[1])
    else:
        base = survjax.LogLogistic(shape=p[0], scale=p[1])
    return survjax.ProportionalHazards(baseline=base, log_hazard_ratio=p[-1])


def _reference(family, point):
    # Independent densities/survivals, then f_PH=e^eta*f0*S0^(e^eta-1).
    def row(t, exact, *p):
        if t == 0:
            return mp.mpf(0)
        if family == "exponential":
            s, f = mp.exp(-p[0] * t), p[0] * mp.exp(-p[0] * t)
        elif family == "weibull":
            h = (t / p[1]) ** p[0]
            s, f = mp.exp(-h), p[0] * h * mp.exp(-h) / t
        elif family == "lognormal":
            z = (mp.log(t) - p[0]) / p[1]
            s = mp.erfc(z / mp.sqrt(2)) / 2
            f = mp.exp(-(z**2) / 2) / (t * p[1] * mp.sqrt(2 * mp.pi))
        else:
            h = (t / p[1]) ** p[0]
            s, f = 1 / (1 + h), p[0] * h / (t * (1 + h) ** 2)
        return mp.log(mp.exp(p[-1]) * f / s) * exact + mp.exp(p[-1]) * mp.log(s)

    values, jacobian, hessian = [], [], []
    with mp.workdps(80):
        p = tuple(mp.mpf(float(q)) for q in point)
        basis = [tuple(int(i == j) for j in range(len(p))) for i in range(len(p))]
        for t, exact in ((1, True), (2, False), (0, False), (3, True)):

            def fn(*q, t=t, exact=exact):
                return row(t, exact, *q)

            values.append(float(fn(*p)))
            jacobian.append([float(mp.diff(fn, p, d)) for d in basis])
            hessian.append(
                [
                    [
                        float(
                            mp.diff(
                                fn, p, tuple(a + b for a, b in zip(i, j, strict=True))
                            )
                        )
                        for j in basis
                    ]
                    for i in basis
                ]
            )
    return tuple(np.asarray(v) for v in (values, jacobian, hessian))


@pytest.mark.parametrize("family", POINTS)
def test_shared_values_and_joint_derivatives_match_independent_law(family, x64_enabled):
    """Spec 4.4/7.3: baseline/eta Jacobians and all mixed Hessians, both AD modes."""
    data = survjax.SurvivalData([1.0, 2.0, 0.0, 3.0], [True, False, False, True])
    points = jnp.asarray(POINTS[family])
    references = [_reference(family, point) for point in points]
    expected = [np.stack([r[order] for r in references]) for order in range(3)]
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}

    def rows(p):
        return _distribution(family, p).log_likelihood(data)

    with jax.debug_nans(True):
        rows(points[0])
        jax.grad(lambda p: rows(p).sum())(points[0])
    transforms = [(lambda f: f, 0), (jax.jacfwd, 1), (jax.jacrev, 1)]
    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            transforms.append((lambda f, o=outer, i=inner: o(i(f)), 2))
    for transform, order in transforms:
        evaluate = transform(rows)
        np.testing.assert_allclose(evaluate(points[0]), expected[order][0], **tol)
        result = jax.jit(jax.vmap(evaluate))(points)
        np.testing.assert_allclose(result, expected[order], **tol)
        np.testing.assert_array_equal(result[:, 2], 0)


@survjax.register_family
@dataclass(frozen=True)
class RootBaseline(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t):
        return -self.rate * jnp.sqrt(t)

    def reference_time(self):
        return jnp.ones_like(self.rate)


def test_inactive_generic_hazard_uses_interior_time(x64_enabled):
    """Spec 7.2: right censoring at zero must not differentiate sqrt there."""
    data = survjax.SurvivalData([1.0, 0.0], [True, False])

    def rows(p):
        dist = survjax.ProportionalHazards(
            baseline=RootBaseline(rate=p[0]), log_hazard_ratio=p[1]
        )
        return dist.log_likelihood(data)

    p = jnp.asarray([0.75, 0.25])

    # At t=1, log f_PH=log(r/2)+eta-r*exp(eta); the second row is constant zero.
    def formula(q):
        return jnp.stack(
            [jnp.log(q[0] / 2) + q[1] - q[0] * jnp.exp(q[1]), jnp.asarray(0.0)]
        )

    with jax.debug_nans(True):
        for transform in (lambda f: f, jax.jacfwd, jax.jacrev, jax.hessian):
            np.testing.assert_allclose(
                transform(rows)(p),
                transform(formula)(p),
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )


@pytest.mark.parametrize("eta_sign", [-1, 1])
def test_extreme_scaling_keeps_ph_values_without_equivalent_parameters(
    eta_sign, x64_enabled
):
    """Spec 4.4/7.1: preserve lower-tail recovery and overflowing closure values."""
    # Keep H normal: CPU exp flushes subnormal outputs even in the old route.
    eta = (
        (720.0 if x64_enabled else 100.0)
        if eta_sign > 0
        else (-600.0 if x64_enabled else -60.0)
    )
    time = (1e-300 if x64_enabled else 1e-30) if eta > 0 else 1.0
    shape = 0.125 if eta < 0 else 2.0
    data = survjax.SurvivalData([time, time, 0.0], [True, False, False])
    dist = survjax.ProportionalHazards(
        baseline=survjax.Weibull(shape=shape, scale=1.0), log_hazard_ratio=eta
    )
    with mp.workdps(100):
        t = mp.mpf(float(data.lower[0]))
        h = mp.exp(eta) * t**shape
        exact = mp.log(shape) + (shape - 1) * mp.log(t) + eta - h
        expected = [float(exact), float(-h), 0.0]
    for evaluate in (lambda d: dist.log_likelihood(d), jax.jit(dist.log_likelihood)):
        np.testing.assert_allclose(
            evaluate(data), expected, atol=0, rtol=TAIL_RTOL[x64_enabled]
        )


def test_dynamic_observations_reuse_trace_and_keep_validation():
    """Spec 3.2/7.1–7.2: eligibility cannot remove stored or current checks."""
    calls = []

    @jax.jit
    def evaluate(time, event):
        calls.append(None)
        return survjax.ProportionalHazards(
            baseline=survjax.Exponential(rate=1.0), log_hazard_ratio=0.0
        ).log_likelihood(time=time, event=event)

    for t, e, expected in (
        ([1.0, 2.0], [1, 0], [-1, -2]),
        ([2.0, 0.0], [0, 0], [-2, 0]),
        ([1.0, 2.0], [1, 2], [-1, np.nan]),
    ):
        np.testing.assert_array_equal(
            evaluate(jnp.asarray(t), jnp.asarray(e)), expected
        )
    assert len(calls) == 1
    data = survjax.SurvivalData([1.0, 2.0], [True, False])
    leaves, tree = jax.tree.flatten(data)
    leaves[4] = leaves[4].at[1].set(-1.0)
    rebuilt = jax.tree.unflatten(tree, leaves)
    assert rebuilt.issues.ok
    dist = survjax.ProportionalHazards(
        baseline=survjax.Exponential(rate=1.0), log_hazard_ratio=0.0
    )
    np.testing.assert_array_equal(dist.log_likelihood(rebuilt), [-1, np.nan])


@survjax.register_family
@dataclass(frozen=True)
class InheritedPH(survjax.ProportionalHazards):
    pass


@pytest.mark.parametrize("mode", ["left", "interval", "entry", "exit"])
def test_disqualifying_modes_retain_density_hook(mode, monkeypatch):
    """Spec 3.2/7.2: each censoring/truncation exclusion preserves the old path."""
    lower, upper, window = {
        "left": ([1.0, 2.0, 0.0], [1.0, np.inf, 3.0], {}),
        "interval": ([1.0, 2.0, 0.5], [1.0, np.inf, 3.0], {}),
        "entry": ([1.0, 2.0], [1.0, np.inf], {"left_trunc": 0.25}),
        "exit": ([1.0, 2.0], [1.0, np.inf], {"right_trunc": 4.0}),
    }[mode]
    data = survjax.SurvivalData.from_bounds(
        lower, upper, exact=[True] + [False] * (len(lower) - 1), **window
    )
    calls = []
    original = survjax.ProportionalHazards.log_prob

    def record(self, t):
        calls.append(None)
        return original(self, t)

    monkeypatch.setattr(survjax.ProportionalHazards, "log_prob", record)
    dist = survjax.ProportionalHazards(
        baseline=survjax.Exponential(rate=1.0), log_hazard_ratio=0.0
    )
    assert jnp.isfinite(jax.jit(dist.log_likelihood)(data)).all()
    assert calls


def test_shared_route_uses_one_survival_and_preserves_subclass_density(monkeypatch):
    """Spec 4.4/7.2: one shared survival call; other concrete types retain hooks."""
    data = survjax.SurvivalData([1.0, 2.0], [True, False])
    calls = []
    original = survjax.ProportionalHazards.log_survival

    def record(self, t):
        calls.append(None)
        return original(self, t)

    monkeypatch.setattr(survjax.ProportionalHazards, "log_survival", record)

    def forbidden(*args):
        raise AssertionError("unneeded density call")

    monkeypatch.setattr(survjax.ProportionalHazards, "log_prob", forbidden)
    dist = survjax.ProportionalHazards(
        baseline=survjax.Exponential(rate=1.0), log_hazard_ratio=0.0
    )
    np.testing.assert_array_equal(dist.log_likelihood(data), [-1, -2])
    assert len(calls) == 1
    calls.clear()
    original_density = InheritedPH.log_prob

    def density(self, t):
        calls.append("density")
        return original_density(self, t)

    monkeypatch.setattr(InheritedPH, "log_prob", density)
    dist = InheritedPH(baseline=survjax.Exponential(rate=1.0), log_hazard_ratio=0.0)
    np.testing.assert_array_equal(dist.log_likelihood(data), [-1, -2])
    assert "density" in calls
