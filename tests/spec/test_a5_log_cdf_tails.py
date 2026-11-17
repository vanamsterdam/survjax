"""A5: `log_cdf` in both tails for the built-in families (spec 4.1, 4.2, 7.2, 12.1; rev 63).

Spec 4.1, rev 63: Exponential and Weibull compute `log_cdf` from `log H`
(`log r + log t`; `k*(log t - log b)`), "`log H` itself where `H` is tiny,
otherwise `log1mexp(-H)`". "The lower tail keeps relative precision wherever
`log H` is representable, even where `H` or `F` underflows. The upper tail
(`log F ~ -S`) has `log1mexp`'s absolute accuracy (section 7.2) and may round
to zero once `S` is below the dtype's epsilon". Spec 7.2, rev 63: `log1mexp`
is `log(-expm1(x))`, absolute error "within a few units of the dtype's epsilon
for every `x < 0`", and "its first and second derivatives keep relative
precision". Mächler's two-branch split "is not required", so the rev 25
branch-split value checks are gone.

Spec 12.1, rev 63 ("`log_cdf` tails"): against mpmath `log1p(-exp(-H))`, "at
`H` from below `log 2` through about 20, 33 and 55 and deeper in each dtype's
normal range (for example `H=70` in float32, `H=600` in float64), eager, under
`jit` and `vmap`, and through `log_cdf_grid`: values within the ordinary
absolute tolerance (zero is acceptable where the true value is smaller in
magnitude), and first and second parameter derivatives with relative accuracy
...; built-in lower tails where `H` or `F` underflows, with relative accuracy
of values and derivatives". Upper-tail values are the 12.1 exception to zero
absolute tolerance ("values of `log1mexp` and its callers within rounding of
zero"), so they use `assert_close`; derivatives and lower tails keep zero
absolute tolerance.

Cases (``H`` is recomputed from the dtype-rounded inputs):

- Exponential ``r = 0.5`` at ``t = 2H`` (exact in both dtypes), ``H = r t``.
- Weibull ``k = 2.5``, ``b = 3`` (non-unit shape and scale) at
  ``t = 3 H**(1/k)``, ``H = (t/b)**k`` in mpmath.
- ConstHazard (``log S = -rate t``, generic `log_cdf` default) at ``t = 2H``,
  for `log_cdf_grid` only; its direct-method tail tests are A4
  (`test_a4_log_cdf_upper_tail.py`).

Derivatives of ``log F = log1mexp(-H)``: with ``g1 = 1/expm1(H)`` and
``g2 = -exp(H)/expm1(H)**2`` (``d/dH`` and ``d2/dH2``, evaluated as
``exp(-H)/(-expm1(-H))`` and ``-exp(-H)/expm1(-H)**2`` so that the float64
reference does not overflow at ``H = 600``):
Exponential ``d/dr = t g1`` (spec: ``t/expm1(H)``), ``d2/dr2 = t**2 g2``.
Weibull, ``L = log(t/b)``: ``dH/dk = H L``, ``d2H/dk2 = H L**2``,
``dH/db = -k H/b``, ``d2H/db2 = k (k+1) H/b**2``; then
``d/dp log F = g1 H'`` and ``d2/dp2 log F = g2 H'**2 + g1 H''``.

Lower tail where ``H`` underflows: ``H/expm1(H)`` and ``H**2 exp(H)/expm1(H)**2``
equal 1 to far below any dtype's epsilon, so the exponential gives
``d/dr log F = 1/r`` and ``d2/dr2 = -1/r**2``; the Weibull gives
``d/dk log F = L``, ``d/db = -k/b`` and ``d2/db2 = k/b**2`` (``d2/dk2 = O(H L**2)``
underflows and is not compared).
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    INF,
    X64,
    assert_close,
    assert_tail_close,
    f,
    log1mexp,
    mp_log_cdf,
    mp_weibull_cumulative_hazard,
    tail_rtol,
)

import survjax

# Spec 12.1, rev 63: "from below `log 2` through about 20, 33 and 55 and deeper".
LOG_CDF_H = (0.5, 0.69, 0.70, 1.2, 20.0, 33.0, 55.0, 600.0 if X64 else 70.0)

WEIBULL_K, WEIBULL_B = 2.5, 3.0


def _exponential_case(h):
    t = f(2.0 * h)
    return survjax.Exponential(rate=f(0.5)), t, 0.5 * float(t)


def _weibull_case(h):
    t = f(WEIBULL_B * h ** (1.0 / WEIBULL_K))
    hk = mp_weibull_cumulative_hazard(WEIBULL_K, WEIBULL_B, t)
    return survjax.Weibull(shape=f(WEIBULL_K), scale=f(WEIBULL_B)), t, hk


def _generic_case(h):
    t = f(2.0 * h)
    return fam.const_hazard()(rate=f(0.5)), t, 0.5 * float(t)


CASES = {
    "exponential": _exponential_case,
    "weibull": _weibull_case,
}


def _vector_case(name, hs):
    """One scalar distribution, a time vector, and the exact ``H`` per time."""
    make = {**CASES, "generic": _generic_case}[name]
    parts = [make(h) for h in hs]
    t = np.array([p[1] for p in parts], dtype=parts[0][1].dtype)
    return parts[0][0], t, [p[2] for p in parts]


def _assert_values(out, hs_exact, hs_nominal):
    """12.1, rev 63: ordinary tolerances, with their absolute floor."""
    out = np.asarray(out)
    for i, (hx, hn) in enumerate(zip(hs_exact, hs_nominal, strict=True)):
        assert_close(out[i], mp_log_cdf(hx), err_msg=f"H={hn}")


# --------------------------------------------------------------- values, eager


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", LOG_CDF_H)
def test_exponential_log_cdf_is_accurate_in_absolute_terms(h):
    """spec 4.1/7.2/12.1, rev 63: Exponential `log F` against mpmath `log1p(-exp(-H))` within the ordinary absolute tolerance."""
    dist, t, hx = _exponential_case(h)
    assert_close(dist.log_cdf(t), mp_log_cdf(hx))


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", LOG_CDF_H)
def test_weibull_log_cdf_is_accurate_in_absolute_terms(h):
    """spec 4.1/7.2/12.1, rev 63: Weibull (`k=2.5`, `b=3`) `log F` against mpmath within the ordinary absolute tolerance."""
    dist, t, hx = _weibull_case(h)
    assert_close(dist.log_cdf(t), mp_log_cdf(hx))


# --------------------------------------------------------------- values, transformed


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", list(CASES))
def test_log_cdf_values_under_jit(name):
    """spec 12.1, rev 63: the values hold under `jit`."""
    dist, t, hx = _vector_case(name, LOG_CDF_H)
    out = jax.jit(lambda d, s: d.log_cdf(s))(dist, t)
    _assert_values(out, hx, LOG_CDF_H)


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", list(CASES))
def test_log_cdf_values_under_vmap_over_time(name):
    """spec 12.1, rev 63: the values hold under `vmap` over scalar times."""
    dist, t, hx = _vector_case(name, LOG_CDF_H)
    out = jax.vmap(dist.log_cdf)(t)
    _assert_values(out, hx, LOG_CDF_H)


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", [*CASES, "generic"])
def test_log_cdf_values_through_grid(name):
    """spec 5/12.1, rev 63: the values hold "through `log_cdf_grid`" (built-ins and the generic default)."""
    dist, t, hx = _vector_case(name, LOG_CDF_H)
    out = dist.log_cdf_grid(t)
    assert np.shape(out) == (len(LOG_CDF_H),)
    _assert_values(out, hx, LOG_CDF_H)


# --------------------------------------------------------------- lower tail where H or F underflows


@pytest.mark.step("A5")
def test_exponential_log_cdf_where_cumulative_hazard_underflows():
    """spec 4.1, rev 63: lower tail "wherever `log H` is representable, even where `H` or `F` underflows".

    ``r = t = 1e-30`` (float32) or ``1e-200`` (float64): ``H = r t`` underflows
    (``1e-60``, ``1e-400``), but ``log F = log H - H/2 + ... = log r + log t``
    (``-138.2``, ``-921.0``) at the dtype-rounded inputs.
    """
    x = f(1e-200 if X64 else 1e-30)
    out = np.asarray(survjax.Exponential(rate=x).log_cdf(x))
    assert np.isfinite(out)
    assert_close(out, 2.0 * np.log(float(x)))


@pytest.mark.step("A5")
def test_weibull_log_cdf_where_cumulative_hazard_underflows():
    """spec 4.1, rev 63: Weibull `log F = k (log t - log b)` where `H = (t/b)**k` underflows.

    ``k = 2.5``, ``b = 3``, ``t = 3e-150`` (float64; ``H = 1e-375``) or
    ``3e-20`` (float32; ``H = 1e-50``).
    """
    t = f(3e-150 if X64 else 3e-20)
    out = np.asarray(survjax.Weibull(shape=f(WEIBULL_K), scale=f(WEIBULL_B)).log_cdf(t))
    assert np.isfinite(out)
    assert_close(out, WEIBULL_K * (np.log(float(t)) - np.log(WEIBULL_B)))


@pytest.mark.step("A5")
def test_exponential_log_cdf_derivatives_where_cumulative_hazard_underflows():
    """spec 4.1/12.1, rev 63: lower tail "with relative accuracy of values and derivatives": `d/dr log F = 1/r`, `d2/dr2 = -1/r**2`.

    ``r = 1e-130``, ``t = 1e-200`` (float64; ``H = 1e-330``) or ``r = 1e-18``,
    ``t = 1e-30`` (float32; ``H = 1e-48``): ``H`` underflows while ``1/r**2``
    (``1e260``, ``1e36``) stays finite.
    """
    r, t = (f(1e-130), f(1e-200)) if X64 else (f(1e-18), f(1e-30))
    rr = float(r)
    _check_derivatives(lambda p: survjax.Exponential(rate=p).log_cdf(t), r, 1.0 / rr, -1.0 / rr**2, None)


@pytest.mark.step("A5")
def test_weibull_log_cdf_derivatives_where_cumulative_hazard_underflows():
    """spec 4.1/12.1, rev 63: Weibull `k=2.5`, `b=3` at `t = 3e-150` / `3e-20`: `d/dk log F = log(t/b)`, `d/db = -k/b`, `d2/db2 = k/b**2`."""
    k, b = WEIBULL_K, WEIBULL_B
    t = f(3e-150 if X64 else 3e-20)
    lz = np.log(float(t) / b)
    rtol = tail_rtol(1.0)

    def by_k(kk):
        return survjax.Weibull(shape=kk, scale=f(b)).log_cdf(t)

    def by_b(bb):
        return survjax.Weibull(shape=f(k), scale=bb).log_cdf(t)

    assert_tail_close(jax.grad(by_k)(f(k)), lz, rtol)
    assert_tail_close(jax.jacfwd(by_k)(f(k)), lz, rtol)
    _check_derivatives(by_b, f(b), -k / b, k / b**2, None)


# --------------------------------------------------------------- parameter derivatives


def _g(h):
    """``d/dH`` and ``d2/dH2`` of ``log1mexp(-H)``, in a form that does not overflow."""
    em1 = np.expm1(-h)
    return np.exp(-h) / -em1, -np.exp(-h) / em1**2


def _check_derivatives(fn, p, d1, d2, h):
    """First and second derivatives, reverse and forward mode; relative accuracy, zero atol (12.1, rev 63).

    ``h`` is the nominal ``H`` for ``tail_rtol``; ``None`` (lower tail) uses
    the 12.1 ``rtol``.
    """
    rtol = tail_rtol(1.0 if h is None else max(h, 1.0))

    def check(a, e):
        assert_tail_close(a, e, rtol)

    check(jax.grad(fn)(p), d1)
    check(jax.jacfwd(fn)(p), d1)
    check(jax.grad(jax.grad(fn))(p), d2)
    check(jax.jacfwd(jax.jacrev(fn))(p), d2)


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", LOG_CDF_H)
def test_exponential_log_cdf_rate_derivatives(h):
    """spec 7.2/7.3/12.1, rev 63: `d/dr log F = t/expm1(H)`, `d2/dr2 = -t**2 exp(H)/expm1(H)**2`, `H = r t`."""
    t = f(2.0 * h)
    tt = float(t)
    g1, g2 = _g(0.5 * tt)
    _check_derivatives(
        lambda r: survjax.Exponential(rate=r).log_cdf(t), f(0.5), tt * g1, tt**2 * g2, h
    )


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", LOG_CDF_H)
def test_weibull_log_cdf_shape_derivatives(h):
    """spec 7.2/7.3/12.1, rev 63: `d/dk log F = g1 H L`, `d2/dk2 = g2 (H L)**2 + g1 H L**2`, `L = log(t/b)`."""
    k, b = WEIBULL_K, WEIBULL_B
    t = f(b * h ** (1.0 / k))
    lz = np.log(float(t) / b)
    hh = float(mp_weibull_cumulative_hazard(k, b, t))
    g1, g2 = _g(hh)
    d1 = g1 * hh * lz
    d2 = g2 * (hh * lz) ** 2 + g1 * hh * lz**2
    _check_derivatives(
        lambda kk: survjax.Weibull(shape=kk, scale=f(b)).log_cdf(t), f(k), d1, d2, h
    )


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", LOG_CDF_H)
def test_weibull_log_cdf_scale_derivatives(h):
    """spec 7.2/7.3/12.1, rev 63: `d/db log F = -g1 k H/b`, `d2/db2 = g2 (k H/b)**2 + g1 k (k+1) H/b**2`."""
    k, b = WEIBULL_K, WEIBULL_B
    t = f(b * h ** (1.0 / k))
    hh = float(mp_weibull_cumulative_hazard(k, b, t))
    g1, g2 = _g(hh)
    d1 = -g1 * k * hh / b
    d2 = g2 * (k * hh / b) ** 2 + g1 * k * (k + 1.0) * hh / b**2
    _check_derivatives(
        lambda bb: survjax.Weibull(shape=f(k), scale=bb).log_cdf(t), f(b), d1, d2, h
    )


# --------------------------------------------------------------- jax_debug_nans on mixed vectors


@pytest.mark.step("A5")
def test_exponential_log_cdf_mixed_times_under_debug_nans():
    """spec 4.1/12.1, rev 63: `t = (0, 2, 66, +inf)` with `r = 0.5` (`H = 0, 1, 33, inf`) raises no NaN under `jax_debug_nans`.

    Expected ``(-inf, log1mexp(-1), log1mexp(-33), 0)``; the upper-tail entry
    within the ordinary absolute tolerance (12.1, rev 63).
    """
    with jax.debug_nans(True):
        out = np.asarray(
            survjax.Exponential(rate=f(0.5)).log_cdf(f([0.0, 2.0, 66.0, INF]))
        )
    np.testing.assert_array_equal(out[[0, 3]], [-INF, 0.0])
    assert_close(out[1], log1mexp(-1.0))
    assert_close(out[2], mp_log_cdf(33.0))


@pytest.mark.step("A5")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_log_cdf_mixed_times_under_debug_nans(k):
    """spec 4.1/12.1, rev 63: Weibull `b = 3`, `t = (0, 3, 3*33**(1/k), +inf)` (`H ~ 0, 1, 33, inf`) under `jax_debug_nans`."""
    b = 3.0
    t = f([0.0, b, b * 33.0 ** (1.0 / k), INF])
    with jax.debug_nans(True):
        out = np.asarray(survjax.Weibull(shape=f(k), scale=f(b)).log_cdf(t))
    np.testing.assert_array_equal(out[[0, 3]], [-INF, 0.0])
    assert_close(out[1], log1mexp(-1.0))
    hx = mp_weibull_cumulative_hazard(k, b, t[2])
    assert_close(out[2], mp_log_cdf(hx))
