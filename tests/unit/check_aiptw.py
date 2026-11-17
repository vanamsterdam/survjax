"""Optional: AIPTW tutorial checks (JAX_ENABLE_X64=1 uv run --group examples pytest -q this-file)."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as distributions
import pytest

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.aiptw_rmst import analysis  # noqa: E402
from examples.aiptw_rmst.cohort import Cohort, resample, simulate_cohort  # noqa: E402
from examples.aiptw_rmst.follow_up import FOLLOW_UP, UniformFollowUp  # noqa: E402

KEY = jax.random.PRNGKey(1)

pytestmark = pytest.mark.skipif(
    not jax.config.x64_enabled, reason="the tutorial runs in float64"
)


def test_uniform_follow_up_matches_closed_forms():
    """Uniform(4, 8): S = (8 - t)/4, f = 1/4 and h = 1/(8 - t) inside, by hand."""
    t = jnp.array([3.0, 4.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(FOLLOW_UP.survival(t), [1.0, 1.0, 0.75, 0.25, 0.0])
    np.testing.assert_allclose(
        FOLLOW_UP.log_prob(t), [-np.inf, *np.log([0.25, 0.25, 0.25]), -np.inf]
    )
    np.testing.assert_allclose(FOLLOW_UP.hazard(t), [0.0, 0.25, 1 / 3, 1.0, np.inf])


def test_uniform_follow_up_gradients_are_finite_on_both_sides_of_the_support():
    """Spec 7.2: substituted inputs keep unselected branches out of the gradient."""

    def total(end):
        follow_up = UniformFollowUp(start=4.0, end=end)
        t = jnp.array([3.0, 5.0, 9.0])
        return jnp.sum(jnp.where(t < end, follow_up.log_survival(t), 0.0))

    # d/d end of log((end - 5)/(end - 4)) at end = 8: 1/3 - 1/4, by hand.
    np.testing.assert_allclose(jax.grad(total)(8.0), 1 / 3 - 1 / 4)


def test_propensity_likelihood_is_the_bernoulli_formula():
    """NumPyro's Bernoulli(logits) sum equals sum(A eta - log(1 + exp eta))."""
    cohort = simulate_cohort(KEY, n=50)
    theta = jnp.array([0.3, -0.5, -0.8, 0.1])
    eta = theta[0] + cohort.covariates @ theta[1:]
    numpyro_sum = distributions.Bernoulli(logits=eta).log_prob(cohort.treated).sum()
    by_hand = jnp.sum(cohort.treated * eta - jnp.logaddexp(0.0, eta))
    np.testing.assert_allclose(numpyro_sum, by_hand, rtol=1e-12)


def test_fits_reach_zero_score():
    """Each maximum likelihood fit has a zero gradient at its optimum."""
    cohort = simulate_cohort(KEY)
    theta, converged = analysis.fit_propensity(cohort)
    assert converged
    X = analysis.outcome_design_matrix(cohort.treated, cohort.covariates)
    outcome_theta, outcome_converged = analysis.fit_weibull_ph(cohort.survival, X)
    assert outcome_converged

    def outcome_nll(t):
        return -analysis.weibull_ph(t, X).log_likelihood(cohort.survival).sum()

    assert jnp.max(jnp.abs(jax.grad(outcome_nll)(outcome_theta))) < 1e-5


def test_aiptw_with_known_propensity_and_no_censoring_is_the_mean_per_arm():
    """By hand: with m = 0, IPCW = 1 and e = 1/2, AIPTW is 2 * mean(1{A=a} Y)."""
    outcome = jnp.array([1.0, 2.0, 3.0, 5.0])
    treated = jnp.array([0.0, 1.0, 0.0, 1.0])
    zero = jnp.zeros(4)
    estimate = analysis.aiptw(outcome, jnp.ones(4), treated, zero + 0.5, zero, zero)
    np.testing.assert_allclose(estimate, [2.0, 3.5])


def test_censoring_weights_drop_unknown_status_and_upweight_the_rest():
    """G(min(t, 5)) from Uniform(4, 8): 1 before 4, 3/4 at 5; unknown status gets 0."""
    survival = survjax.SurvivalData(
        [2.0, 4.5, 4.5, 6.0], event=[True, True, False, False]
    )
    np.testing.assert_allclose(
        analysis.censoring_weights(survival), [1.0, 1 / 0.875, 0.0, 1 / 0.75]
    )


def test_vmapped_bootstrap_matches_a_loop():
    """jit(vmap(analyse)) over three resamples agrees with analysing them one by one."""
    cohort = simulate_cohort(KEY)
    key = jax.random.PRNGKey(2)
    batched = analysis.bootstrap(key, cohort, 3)
    indices = jax.random.randint(key, (3, len(cohort.treated)), 0, len(cohort.treated))
    for b, idx in enumerate(indices):
        single = analysis.analyse(resample(cohort, idx))
        np.testing.assert_allclose(batched.aiptw[b], single.aiptw, rtol=1e-8)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "aiptw_rmst"


def read_fixture_cohort():
    """The committed CSV of export_cohort.py, as a Cohort."""
    table = jnp.asarray(np.loadtxt(FIXTURE / "cohort.csv", delimiter=",", skiprows=1))
    survival = survjax.SurvivalData(table[:, 0], event=table[:, 1] == 1)
    return table, Cohort(survival, table[:, 2], table[:, 3:])


def test_r_fixture_cohort_is_the_tutorial_cohort():
    """The CSV holds simulate_cohort(PRNGKey(1) data key) at 17 significant digits."""
    table, _ = read_fixture_cohort()
    data_key, _ = jax.random.split(KEY)
    cohort = simulate_cohort(data_key, n=500)
    expected = np.column_stack(
        [
            cohort.survival.lower,
            cohort.survival.exact,
            cohort.treated,
            cohort.covariates,
        ]
    )
    np.testing.assert_array_equal(table, expected)  # %.17g round-trips float64


def test_analysis_matches_r_survreg_on_the_same_cohort():
    """survjax + NumPyro + optimistix against glm + survreg (generate_aiptw_r_estimates.R)."""
    _, cohort = read_fixture_cohort()
    estimates = analysis.analyse(cohort)
    expected = json.loads((FIXTURE / "r_estimates.json").read_text())
    for name in ("naive", "g_formula", "iptw", "aiptw"):
        # Both optimisers converge far below this (observed: 1e-9 relative);
        # it is the project float64 rtol, not a loosened one.
        np.testing.assert_allclose(getattr(estimates, name), expected[name], rtol=1e-8)
