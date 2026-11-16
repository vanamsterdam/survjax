"""A4: the generic `log_cdf` default from below `log 2` into the upper tail (spec 4.2, 7.2, 12.1; rev 63).

Spec 4.2, rev 63: the default is "`log1mexp(log S(t))` (section 7.2). In the
upper tail (`log F ~ -S`) it is accurate in absolute terms and may round to
zero once `S` is below the dtype's epsilon". Spec 7.2, rev 63: `log1mexp` is
`log(-expm1(x))`; "Its absolute error is within a few units of the dtype's
epsilon for every `x < 0`, and its first and second derivatives keep relative
precision". Spec 12.1, rev 63 ("`log_cdf` tails"), for "a custom family that
relies on the generic default, at `H` from below `log 2` through about 20, 33
and 55 and deeper in each dtype's normal range ... eager, under `jit` and
`vmap`, and through `log_cdf_grid`: values within the ordinary absolute
tolerance (zero is acceptable where the true value is smaller in magnitude),
and first and second parameter derivatives with relative accuracy". The
values are the 12.1 exception to zero absolute tolerance, so they use
`assert_close`; derivatives keep zero absolute tolerance. `log_cdf_grid` on
this family is A5 (`test_a5_log_cdf_tails.py`).

ConstHazard: ``log S = -rate t``, so ``H = rate t``; ``rate = 0.5`` at
``t = 2H`` (exact in both dtypes). Derivatives as for the exponential:
``d/dr log F = t/expm1(H)``, ``d2/dr2 = -t**2 exp(H)/expm1(H)**2``, evaluated
as ``t exp(-H)/(-expm1(-H))`` and ``-t**2 exp(-H)/expm1(-H)**2`` so that the
float64 reference does not overflow at ``H = 600``.
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
    tail_rtol,
)

# Spec 12.1, rev 63: "from below `log 2` through about 20, 33 and 55 and deeper".
LOG_CDF_H = (0.5, 0.69, 0.70, 1.2, 20.0, 33.0, 55.0, 600.0 if X64 else 70.0)


def _dist(rate=0.5):
    return fam.const_hazard()(rate=f(rate))


def _assert_values(out, hs):
    """12.1, rev 63: ordinary tolerances (absolute floor) at ``H = 0.5 t`` of the rounded ``t``."""
    out = np.asarray(out)
    for i, h in enumerate(hs):
        assert_close(out[i], mp_log_cdf(h), err_msg=f"H={h}")


@pytest.mark.step("A4")
@pytest.mark.parametrize("h", LOG_CDF_H)
def test_default_log_cdf_is_accurate_in_absolute_terms(h):
    """spec 4.2/7.2/12.1, rev 63: the default against mpmath `log1p(-exp(-H))` within the ordinary absolute tolerance."""
    t = f(2.0 * h)
    assert_close(_dist().log_cdf(t), mp_log_cdf(0.5 * float(t)))


@pytest.mark.step("A4")
def test_default_log_cdf_values_under_jit():
    """spec 12.1, rev 63: the default's values hold under `jit`."""
    t = f([2.0 * h for h in LOG_CDF_H])
    out = jax.jit(lambda d, s: d.log_cdf(s))(_dist(), t)
    _assert_values(out, [0.5 * float(x) for x in t])


@pytest.mark.step("A4")
def test_default_log_cdf_values_under_vmap_over_time():
    """spec 12.1, rev 63: the default's values hold under `vmap` over scalar times."""
    t = f([2.0 * h for h in LOG_CDF_H])
    out = jax.vmap(_dist().log_cdf)(t)
    _assert_values(out, [0.5 * float(x) for x in t])


@pytest.mark.step("A4")
@pytest.mark.parametrize("h", LOG_CDF_H)
def test_default_log_cdf_rate_derivatives(h):
    """spec 7.2/7.3/12.1, rev 63: `d/dr log F = t/expm1(H)`, `d2/dr2 = -t**2 exp(H)/expm1(H)**2`, relative accuracy, zero atol."""
    t = f(2.0 * h)
    tt = float(t)
    hh = 0.5 * tt
    em1 = np.expm1(-hh)
    d1, d2 = tt * np.exp(-hh) / -em1, -(tt**2) * np.exp(-hh) / em1**2
    rtol = tail_rtol(h)

    def fn(r):
        return fam.const_hazard()(rate=r).log_cdf(t)

    assert_tail_close(jax.grad(fn)(f(0.5)), d1, rtol)
    assert_tail_close(jax.jacfwd(fn)(f(0.5)), d1, rtol)
    assert_tail_close(jax.grad(jax.grad(fn))(f(0.5)), d2, rtol)
    assert_tail_close(jax.jacfwd(jax.jacrev(fn))(f(0.5)), d2, rtol)


@pytest.mark.step("A4")
def test_default_log_cdf_mixed_times_under_debug_nans():
    """spec 4.2/12.1, rev 63: `t = (0, 2, 66, +inf)` (`H = 0, 1, 33, inf`) raises no NaN under `jax_debug_nans`.

    Expected ``(log1mexp(0), log1mexp(-1), log1mexp(-33), log1mexp(-inf)) =
    (-inf, -0.4587, -4.66e-15, 0)``; the upper-tail entry within the ordinary
    absolute tolerance (12.1, rev 63).
    """
    with jax.debug_nans(True):
        out = np.asarray(_dist().log_cdf(f([0.0, 2.0, 66.0, INF])))
    np.testing.assert_array_equal(out[[0, 3]], [-INF, 0.0])
    assert_close(out[1], log1mexp(-1.0))
    assert_close(out[2], mp_log_cdf(33.0))
