"""Doubly robust (AIPTW) estimate of the 5-year RMST difference; not library API."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro.distributions as distributions
import optimistix as optx

import survjax
from examples.aiptw_rmst.cohort import FOLLOW_UP, TAU, Cohort, resample


class Estimates(NamedTuple):
    """Per-arm RMST(TAU) in years, shape (2,) as (surgery alone, chemotherapy)."""

    naive: jax.Array
    g_formula: jax.Array
    iptw: jax.Array
    aiptw: jax.Array
    outcome_theta: jax.Array  # (7,) fitted outcome model, for effect-by-SRS curves
    converged: jax.Array  # () True when all three fits converged


def minimise(
    objective: Callable[[jax.Array], jax.Array], start: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """BFGS minimum of a smooth objective; returns (theta, converged)."""
    solution = optx.minimise(
        lambda theta, _: objective(theta),
        optx.BFGS(rtol=1e-8, atol=1e-8),
        start,
        max_steps=500,
        throw=False,  # report failure in `converged` instead, also under vmap
    )
    return solution.value, solution.result == optx.RESULTS.successful


# Propensity score: logistic regression -------------------------------------


def with_intercept(covariates: jax.Array) -> jax.Array:
    """(N, 1 + p) design: a column of ones, then the covariates."""
    return jnp.column_stack([jnp.ones(len(covariates)), covariates])


def fit_propensity(cohort: Cohort) -> tuple[jax.Array, jax.Array]:
    """Logistic regression of treatment on all three covariates."""
    design = with_intercept(cohort.covariates)

    def negative_log_likelihood(theta):
        treatment = distributions.Bernoulli(logits=design @ theta)
        return -treatment.log_prob(cohort.treated).sum()

    return minimise(negative_log_likelihood, jnp.zeros(design.shape[1]))


def propensity_scores(theta: jax.Array, covariates: jax.Array) -> jax.Array:
    """(N,) fitted probability of chemotherapy."""
    return jax.nn.sigmoid(with_intercept(covariates) @ theta)


# Outcome model: Weibull proportional hazards ---------------------------------


def outcome_design(treated: jax.Array, covariates: jax.Array) -> jax.Array:
    """(N, 5): chemo, age, frail, SRS and the prespecified chemo x SRS term."""
    treated = jnp.broadcast_to(treated, covariates.shape[:1])
    srs = covariates[:, 2]
    return jnp.column_stack([treated, covariates, treated * srs])


def weibull_ph(theta: jax.Array, design: jax.Array) -> survjax.ProportionalHazards:
    """theta = (log shape, log scale, coefficients): one distribution per row."""
    baseline = survjax.Weibull.from_log(log_shape=theta[0], log_scale=theta[1])
    return survjax.ProportionalHazards(
        baseline=baseline, log_hazard_ratio=design @ theta[2:]
    )


def fit_weibull_ph(
    survival: survjax.SurvivalData, design: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Maximum likelihood Weibull PH fit of right-censored survival on a design."""

    def negative_log_likelihood(theta):
        return -weibull_ph(theta, design).log_likelihood(survival).sum()

    start = jnp.zeros(2 + design.shape[1]).at[1].set(jnp.log(survival.lower.mean()))
    return minimise(negative_log_likelihood, start)


def counterfactual_rmst(
    theta: jax.Array, covariates: jax.Array, treated: float
) -> jax.Array:
    """(N,) model-based RMST(TAU) for every patient had they all had `treated`."""
    return weibull_ph(theta, outcome_design(treated, covariates)).rmst(TAU)


# Censoring: known administrative follow-up -----------------------------------


def censoring_weights(survival: survjax.SurvivalData) -> jax.Array:
    """(N,) inverse probability of censoring weights, 0 when the status at TAU is unknown."""
    time = survival.lower
    known_by_tau = survival.exact | (time >= TAU)
    survived_censoring = FOLLOW_UP.survival(jnp.minimum(time, TAU))  # >= 0.75
    return jnp.where(known_by_tau, 1.0 / survived_censoring, 0.0)


# Estimators -------------------------------------------------------------------


def aiptw(outcome, ipcw, treated, propensity, rmst_0, rmst_1) -> jax.Array:
    """(2,) augmented IPTW mean of min(T, TAU) per arm."""
    estimates = []
    for arm, model_rmst, p_arm in (
        (0.0, rmst_0, 1 - propensity),
        (1.0, rmst_1, propensity),
    ):
        weight = (treated == arm) * ipcw / p_arm
        estimates.append(jnp.mean(model_rmst + weight * (outcome - model_rmst)))
    return jnp.stack(estimates)


def iptw(outcome, ipcw, treated, propensity) -> jax.Array:
    """(2,) weighted (Hajek) mean of min(T, TAU) per arm."""
    estimates = []
    for arm, p_arm in ((0.0, 1 - propensity), (1.0, propensity)):
        weight = (treated == arm) * ipcw / p_arm
        estimates.append(jnp.sum(weight * outcome) / jnp.sum(weight))
    return jnp.stack(estimates)


def naive_rmst(cohort: Cohort) -> tuple[jax.Array, jax.Array]:
    """(2,) RMST per arm from a Weibull PH model with treatment only."""
    theta, converged = fit_weibull_ph(cohort.survival, cohort.treated[:, None])
    per_arm = weibull_ph(theta, jnp.array([[0.0], [1.0]]))
    return per_arm.rmst(TAU), converged


def analyse(data: Cohort) -> Estimates:
    """The whole analysis of one cohort: fit, predict and estimate.

    A pure function of the data, so it can be jitted and vmapped over
    bootstrap resamples.
    """
    ps_theta, ps_converged = fit_propensity(data)
    propensity = propensity_scores(ps_theta, data.covariates)

    design = outcome_design(data.treated, data.covariates)
    outcome_theta, outcome_converged = fit_weibull_ph(data.survival, design)
    rmst_0 = counterfactual_rmst(outcome_theta, data.covariates, 0.0)
    rmst_1 = counterfactual_rmst(outcome_theta, data.covariates, 1.0)

    outcome = jnp.minimum(data.survival.lower, TAU)  # min(T, TAU) where it is known
    ipcw = censoring_weights(data.survival)
    naive, naive_converged = naive_rmst(data)

    return Estimates(
        naive=naive,
        g_formula=jnp.stack([rmst_0.mean(), rmst_1.mean()]),
        iptw=iptw(outcome, ipcw, data.treated, propensity),
        aiptw=aiptw(outcome, ipcw, data.treated, propensity, rmst_0, rmst_1),
        outcome_theta=outcome_theta,
        converged=ps_converged & outcome_converged & naive_converged,
    )


def bootstrap(key: jax.Array, data: Cohort, n_boot: int) -> Estimates:
    """Analyse n_boot resamples with replacement in one vmapped, compiled call."""
    n = len(data.treated)
    indices = jax.random.randint(key, (n_boot, n), 0, n)
    return jax.jit(jax.vmap(lambda idx: analyse(resample(data, idx))))(indices)
