"""observation.censoring: tails, tiny masses, narrow intervals and `jax_debug_nans` for left/interval rows (spec 4.1, 7.1, 7.2, 12.1, 12.2; rev 51, 63).

12.2 makes "narrow intervals, interval cancellation ... Slice B evidence";
12.1 (Numerical behaviour): "tiny nonzero masses, both tails, narrow
intervals, cancellation, boundaries, inactive branches; no floors ... the
likelihood suite run eagerly under `jax_debug_nans` on valid mixed batches".
7.2: "for a finite exponential interval `(a,b]`, `-r*a + log(-expm1(-r*(b-a)))`;
Weibull uses cumulative-hazard increments computed without subtracting nearly
equal powers. Never clip probabilities upward, floor `Z`, add an epsilon to
data, or turn a narrow interval into a density."

References are 50-digit mpmath values of the spec 2 table at the
dtype-rounded inputs (``observation_reference``). Comparisons use zero
absolute tolerance (12.1: "Comparisons of tiny log probabilities ... use zero
absolute tolerance, so that an erroneous zero cannot pass"). The relative
scale is the 12.1 ``rtol`` unless stated: every value here is a sum of a few
terms each accurate to a few ulps times at most ``1 + k |log(t/b)|`` (the
``z**k`` amplification of ``helpers._reference``), and that bound is below
the 12.1 ``rtol`` for all cases.

Rev 63 (7.2, 12.1): upper-tail ``log F`` and other `log1mexp` values within
rounding of zero are accurate in absolute terms only, and 12.1 compares them
with "the ordinary absolute tolerance" (zero acceptable where the true value is
smaller); 12.1 adds "likelihood rows with wide intervals and upper-tail left
censoring, values and first and second derivatives", the derivatives with
relative accuracy (``helpers.tail_rtol``, zero absolute tolerance).

Hand derivatives, Exponential ``r = 1`` (as in
``test_observation_censoring_derivatives.py``): left at ``c``:
``dl/dr = c/expm1(c)``, ``d2l/dr2 = -c**2 exp(c)/expm1(c)**2``; interval
``(a, a + w]``: ``dl/dr = -a + w/expm1(w)``, ``d2l/dr2 = -w**2 exp(w)/expm1(w)**2``.
Each ``exp(x)/expm1(x)**k`` is evaluated as ``exp(-x)/(-expm1(-x))**k`` so
that the float64 reference does not overflow at 600. Weibull left at ``c``,
``H = (c/b)**k``, ``L = log(c/b)``, ``g1 = 1/expm1(H)``,
``g2 = -exp(H)/expm1(H)**2``: ``dl/dk = g1 H L``,
``d2l/dk2 = g2 (H L)**2 + g1 H L**2`` (as in A5).
"""

import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import (
    INF,
    NAN,
    X64,
    assert_close,
    assert_tail_close,
    f,
    flag_data,
    ll,
    mp_log_cdf,
    mp_weibull_cumulative_hazard,
    tail_rtol,
)
from tolerances import TOLERANCES

import survjax

FEATURE = pytest.mark.feature("observation.censoring")
RTOL = TOLERANCES[np.dtype(np.float64 if X64 else np.float32)]["rtol"]


def _exp(r=1.0):
    return survjax.Exponential(rate=f(r))


def _weib(k, s):
    return survjax.Weibull(shape=f(k), scale=f(s))


# ------------------------------------------------------------------- lower tail


@FEATURE
def test_exponential_left_row_at_tiny_time_keeps_relative_precision():
    """spec 4.1 and 12.1, rev 51: left at `c = 1e-30`, `r = 1`: `log F = log(-expm1(-1e-30)) ~ -69.08`, never `-inf`."""
    out = ll(_exp(), flag_data(["left"], [1e-30]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 0.0, 1e-30)], RTOL)


@FEATURE
def test_weibull_left_row_where_the_cumulative_hazard_underflows():
    """spec 4.1 and 12.1, rev 51: Weibull `k=2.5`, `b=2`, left at `1e-30`: `H = (5e-31)**2.5` underflows float32, but `log F ~ log H ~ -174.43` is representable and required.

    4.1: the lower tail keeps relative precision "wherever `log H` is
    representable, even where `H` or `F` underflows".
    """
    out = ll(_weib(2.5, 2.0), flag_data(["left"], [1e-30]))
    assert np.isfinite(out).all()
    assert_tail_close(out, [ref.contribution("weibull", (2.5, 2.0), 0.0, 1e-30)], RTOL)


@FEATURE
def test_exponential_interval_at_tiny_times():
    """spec 7.2 and 12.1, rev 51: `(1e-30, 2e-30]`, `r=1`: `-r a + log(-expm1(-r w)) ~ -69.08`, a tiny nonzero mass with no floor."""
    out = ll(_exp(), flag_data(["interval"], [1e-30], [2e-30]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 1e-30, 2e-30)], RTOL)


@FEATURE
def test_weibull_interval_at_tiny_times():
    """spec 7.2 and 12.1, rev 51: Weibull `k=1.5`, `b=2`, `(1e-10, 2e-10]`: mass about `6.46e-16` (log about `-34.975`)."""
    out = ll(_weib(1.5, 2.0), flag_data(["interval"], [1e-10], [2e-10]))
    assert_tail_close(out, [ref.contribution("weibull", (1.5, 2.0), 1e-10, 2e-10)], RTOL)


# ------------------------------------------------------------------- upper tail

UPPER_H = [20.0, 33.0, 55.0, 600.0 if X64 else 70.0]


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_exponential_left_row_in_the_upper_tail(h):
    """spec 4.1, 7.2 and 12.1, rev 63: left at `c = H` (`r = 1`): `log F ~ -exp(-H)` within the ordinary absolute tolerance."""
    (c,) = ref.rounded(h)
    out = ll(_exp(), flag_data(["left"], [c]))
    assert_close(out, [mp_log_cdf(c)])


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_weibull_left_row_in_the_upper_tail(h):
    """spec 4.1, 7.2 and 12.1, rev 63: Weibull `k=2`, `b=1`, left at `sqrt(H)`: `log F ~ -exp(-H)` within the ordinary absolute tolerance."""
    (c,) = ref.rounded(np.sqrt(h))
    big_h = mp_weibull_cumulative_hazard(2.0, 1.0, c)
    out = ll(_weib(2.0, 1.0), flag_data(["left"], [c]))
    assert_close(out, [mp_log_cdf(big_h)])


def _derivatives(fn, p):
    """First and second derivatives in reverse and forward mode."""
    return (
        (jax.grad(fn)(p), jax.jacfwd(fn)(p)),
        (jax.grad(jax.grad(fn))(p), jax.jacfwd(jax.jacrev(fn))(p)),
    )


def _assert_derivatives(fn, p, d1, d2, h):
    rtol = tail_rtol(max(h, 1.0))
    (g_rev, g_fwd), (h_rev, h_fwd) = _derivatives(fn, p)
    for got in (g_rev, g_fwd):
        assert_tail_close(got, d1, rtol)
    for got in (h_rev, h_fwd):
        assert_tail_close(got, d2, rtol)


def _sum_ll(make, data):
    return lambda p: make(p).log_likelihood(data).sum()


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_exponential_left_row_in_the_upper_tail_rate_derivatives(h):
    """spec 7.2, 7.3 and 12.1, rev 63: left at `c = H` (`r = 1`): `c/expm1(c)` and `-c**2 exp(c)/expm1(c)**2`, relative accuracy."""
    (c,) = ref.rounded(h)
    em1 = np.expm1(-c)
    d1, d2 = c * np.exp(-c) / -em1, -(c**2) * np.exp(-c) / em1**2
    data = flag_data(["left"], [c])
    _assert_derivatives(_sum_ll(lambda r: survjax.Exponential(rate=r), data), f(1.0), d1, d2, h)


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_weibull_left_row_in_the_upper_tail_shape_derivatives(h):
    """spec 7.2, 7.3 and 12.1, rev 63: Weibull `k=2`, `b=1`, left at `sqrt(H)`: `g1 H L` and `g2 (H L)**2 + g1 H L**2`, relative accuracy."""
    (c,) = ref.rounded(np.sqrt(h))
    hh = float(mp_weibull_cumulative_hazard(2.0, 1.0, c))
    lz = np.log(c)
    em1 = np.expm1(-hh)
    g1, g2 = np.exp(-hh) / -em1, -np.exp(-hh) / em1**2
    d1 = g1 * hh * lz
    d2 = g2 * (hh * lz) ** 2 + g1 * hh * lz**2
    data = flag_data(["left"], [c])
    _assert_derivatives(_sum_ll(lambda k: survjax.Weibull(shape=k, scale=f(1.0)), data), f(2.0), d1, d2, h)


# Wide intervals ``(0.25, 0.25 + H]`` (endpoints exact in both dtypes), Exponential ``r = 1``:
# ``l = -0.25 + log1mexp(-H)``, the `log1mexp` term within rounding of zero.
WIDE_A = 0.25


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_exponential_wide_interval_value(h):
    """spec 7.2 and 12.1, rev 63: interval `(0.25, 0.25 + H]`, `r = 1`: `-0.25 + log(1 - e**-H)` within the ordinary tolerances."""
    out = ll(_exp(), flag_data(["interval"], [WIDE_A], [WIDE_A + h]))
    assert_close(out, [ref.contribution("exponential", (1.0,), WIDE_A, WIDE_A + h)])


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_exponential_wide_interval_rate_derivatives(h):
    """spec 7.2, 7.3 and 12.1, rev 63: interval `(0.25, 0.25 + H]`: `-0.25 + H/expm1(H)` and `-H**2 exp(H)/expm1(H)**2`, relative accuracy."""
    em1 = np.expm1(-h)
    d1 = -WIDE_A + h * np.exp(-h) / -em1
    d2 = -(h**2) * np.exp(-h) / em1**2
    data = flag_data(["interval"], [WIDE_A], [WIDE_A + h])
    _assert_derivatives(_sum_ll(lambda r: survjax.Exponential(rate=r), data), f(1.0), d1, d2, h)


@FEATURE
def test_wide_interval_and_upper_tail_left_rows_in_one_batch():
    """spec 7.2 and 12.1, rev 63: a batch of a wide interval and an upper-tail left row (`H = 33`): values and the summed rate derivatives.

    Sum of the two rows' hand derivatives (Exponential ``r = 1``).
    """
    h = 33.0
    data = flag_data(["interval", "left"], [WIDE_A, h], [WIDE_A + h, NAN])
    out = ll(_exp(), data)
    assert_close(out, [ref.contribution("exponential", (1.0,), WIDE_A, WIDE_A + h), mp_log_cdf(h)])
    em1 = np.expm1(-h)
    g1, g2 = h * np.exp(-h) / -em1, -(h**2) * np.exp(-h) / em1**2
    _assert_derivatives(_sum_ll(lambda r: survjax.Exponential(rate=r), data), f(1.0), -WIDE_A + 2 * g1, 2 * g2, h)


@FEATURE
def test_exponential_interval_deep_in_the_upper_tail():
    """spec 7.2 and 12.1, rev 51: `(1000, 1001]`, `r=1`: `S(a)` and `S(b)` underflow, but `-1000 + log(1 - e**-1)` is finite and required."""
    out = ll(_exp(), flag_data(["interval"], [1000.0], [1001.0]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 1000.0, 1001.0)], RTOL)


@FEATURE
def test_weibull_interval_deep_in_the_upper_tail():
    """spec 7.2 and 12.1, rev 51: Weibull `k=2`, `b=1`, `(30, 30.5]`: `H(a) = 900`; value about `-900`, from increments in log space."""
    out = ll(_weib(2.0, 1.0), flag_data(["interval"], [30.0], [30.5]))
    assert_tail_close(out, [ref.contribution("weibull", (2.0, 1.0), 30.0, 30.5)], RTOL)


# ------------------------------------------------------------- narrow intervals

NARROW = [2.0**-20] + ([2.0**-40] if X64 else [])


@FEATURE
@pytest.mark.parametrize("width", NARROW, ids=lambda w: f"2^{int(np.log2(w))}")
def test_exponential_narrow_interval(width):
    """spec 7.2 and 12.2, rev 51: `(1, 1 + w]` with `w = 2**-20` (both dtypes) or `2**-40` (float64): `-1 + log(-expm1(-w))`.

    The endpoints are exact in the dtype, so ``b - a = w`` exactly; a narrow
    interval must not become a density or zero.
    """
    out = ll(_exp(), flag_data(["interval"], [1.0], [1.0 + width]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 1.0, 1.0 + width)], RTOL)


@FEATURE
@pytest.mark.parametrize("width", NARROW, ids=lambda w: f"2^{int(np.log2(w))}")
@pytest.mark.parametrize("k", [0.5, 1.5, 4.0])
def test_weibull_narrow_interval_uses_cumulative_hazard_increments(k, width):
    """spec 7.2, rev 51: Weibull `b=2`, `(1, 1 + w]`: "cumulative-hazard increments computed without subtracting nearly equal powers".

    ``H(b) - H(a) = H(a) expm1(k log1p(w/a))`` is about ``k w H(a)/a``; a
    naive ``H(b) - H(a)`` in the dtype has relative error ``eps H(a)/(H(b)-H(a))
    ~ eps/(k w)`` (about ``0.1`` in float32 for ``w = 2**-20``), far outside
    the 12.1 ``rtol`` on ``log(H(b) - H(a)) ~ log(k w/2**k)``.
    """
    out = ll(_weib(k, 2.0), flag_data(["interval"], [1.0], [1.0 + width]))
    assert_tail_close(out, [ref.contribution("weibull", (k, 2.0), 1.0, 1.0 + width)], RTOL)


# --------------------------------------------------------- jax_debug_nans (12.1)


def _mixed_rows(scale):
    """Valid mixed rows at time scale ``scale`` (the upper-tail time, ``H`` about 1e3).

    exact interior, right interior, right at 0 (uninformative), left at 1e-30,
    left interior, left upper tail, interval narrow, interval interior,
    interval upper tail.
    """
    kinds = ["exact", "right", "right", "left", "left", "left", "interval", "interval", "interval"]
    time = [1.5, 2.0, 0.0, 1e-30, 1.2, scale, 1.0, 0.5, scale]
    time2 = [NAN, NAN, NAN, NAN, NAN, NAN, 1.0 + 2.0**-20, 2.5, 1.5 * scale]
    return flag_data(kinds, time, time2)


def _run(make, params, data):
    def total(p):
        out = make(**p).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, has_aux=True)(params)
    values = np.asarray(values)
    assert np.isfinite(values).all(), values
    np.testing.assert_array_equal(values[2], 0.0)
    for g in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(g)).all(), g
    return values


@FEATURE
def test_exponential_mixed_censored_batch_under_debug_nans():
    """spec 7.2 and 12.1, rev 51: Exponential `r=2` on the mixed rows: no NaN in any primitive, value or rate gradient."""
    _run(survjax.Exponential, {"rate": jnp.asarray(f(2.0))}, _mixed_rows(500.0))


@FEATURE
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_mixed_censored_batch_under_debug_nans(k):
    """spec 7.2 and 12.1, rev 51: Weibull `b=2` and shape `k` on the mixed rows: no NaN in values or `(k, b)` gradients."""
    params = {"shape": jnp.asarray(f(k)), "scale": jnp.asarray(f(2.0))}
    _run(survjax.Weibull, params, _mixed_rows(2.0 * 1e3 ** (1.0 / k)))


@FEATURE
def test_weibull_per_row_parameters_under_debug_nans():
    """spec 5, 7.2 and 12.1, rev 51: per-row shapes and scales on the mixed rows raise no NaN."""
    params = {
        "shape": jnp.asarray(f([0.5, 2.0, 0.5, 3.0, 1.0, 2.0, 0.7, 1.5, 2.0])),
        "scale": jnp.asarray(f([1.0, 2.0, 1e3, 1.0, 1e-2, 1.0, 2.0, 1.0, 1.0])),
    }
    _run(survjax.Weibull, params, _mixed_rows(40.0))


@FEATURE
def test_custom_family_mixed_censored_batch_under_debug_nans():
    """spec 4.2, 7.2 and 12.1, rev 51: `const_hazard` on generic defaults raises no NaN on the mixed rows."""
    _run(fam.const_hazard(), {"rate": jnp.asarray(f(2.0))}, _mixed_rows(500.0))


@FEATURE
def test_bounds_form_censored_batch_under_debug_nans():
    """spec 3.1, 7.2 and 12.1, rev 51: bounds-form left `(0, c]` and interval rows with exact rows raise no NaN."""
    data = survjax.SurvivalData.from_bounds(
        f([1.5, 0.0, 0.0, 0.5, 2.0]),
        f([1.5, 1e-30, 1.2, 2.5, INF]),
        exact=jnp.asarray([True, False, False, False, False]),
    )

    def total(k, s):
        out = survjax.Weibull(shape=k, scale=s).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, argnums=(0, 1), has_aux=True)(
            jnp.asarray(f(0.5)), jnp.asarray(f(2.0))
        )
    assert np.isfinite(np.asarray(values)).all()
    for g in grads:
        assert np.isfinite(np.asarray(g))
