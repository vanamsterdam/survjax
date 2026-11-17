"""Revision-63 private log arithmetic (spec 7.2, 12.1)."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL, TAIL_ATOL, TAIL_RTOL

from survjax.distributions._log_math import safe_log1mexp


def test_log1mexp_values_match_mpmath(x64_enabled):
    """Spec 7.2/12.1: values have ordinary absolute accuracy in both tails."""
    inputs = jnp.array([-1e-6, -0.5, -1.0, -20.0, -33.0, -70.0])
    with mp.workdps(100):
        expected = [float(mp.log(-mp.expm1(float(x)))) for x in inputs]
    for method in (safe_log1mexp, jax.jit(safe_log1mexp), jax.vmap(safe_log1mexp)):
        actual = method(inputs)
        assert actual.dtype == inputs.dtype
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_log1mexp_endpoints_and_invalid_values():
    """Spec 7.2: endpoint values are supplied safely; invalid operands stay NaN."""
    with jax.debug_nans(True):
        actual = jax.jit(safe_log1mexp)(jnp.array([0.0, -jnp.inf]))
    np.testing.assert_array_equal(actual, [-np.inf, 0.0])
    assert np.isnan(safe_log1mexp(jnp.array(1.0)))
    assert np.isnan(safe_log1mexp(jnp.array(jnp.nan)))


@pytest.mark.parametrize("first", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
@pytest.mark.parametrize(
    "second", [None, jax.jacfwd, jax.jacrev], ids=["first", "fwd", "rev"]
)
def test_log1mexp_mixed_vector_derivatives(first, second, x64_enabled):
    """Spec 7.2/12.1: interior derivatives retain relative accuracy near zero values."""
    deep = -600.0 if x64_enabled else -70.0
    inputs = jnp.array([0.0, -0.5, -33.0, deep, -jnp.inf])
    derivative = first(safe_log1mexp)
    if second is not None:
        derivative = second(derivative)
    # Hand differentiation: f'=-q/(1-q), f''=-q/(1-q)^2, q=exp(x).
    with mp.workdps(100):
        power = 1 if second is None else 2
        expected = [
            float(-mp.exp(float(x)) / (-mp.expm1(float(x))) ** power)
            for x in inputs[1:4]
        ]
    with jax.debug_nans(True):
        mapped = jax.vmap(derivative)
        for actual in (mapped(inputs), jax.jit(mapped)(inputs)):
            np.testing.assert_allclose(
                actual[1:4],
                expected,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
            )
