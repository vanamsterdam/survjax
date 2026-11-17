"""Doubly robust (AIPTW) estimate of the 5-year RMST difference; not library API."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro.distributions as distributions
import optimistix as optx

import survjax
from examples.aiptw_rmst.cohort import TAU, resample
from examples.aiptw_rmst.follow_up import FOLLOW_UP


class Estimates(NamedTuple):
    """Per-arm RMST(TAU) in years, shape (2,) as (surgery alone, chemotherapy)."""

    naive: jax.Array
    g_formula: jax.Array
    iptw: jax.Array
    aiptw: jax.Array
    outcome_theta: jax.Array  # (7,) fitted outcome model, for effect-by-SRS curves
    converged: jax.Array  # () True when all three fits converged


def minimise(objective, start):
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


def fit_propensity(cohort):
    """Logistic regression of treatment on an intercept and the three covariates."""
    n = len(cohort.treated)
    X_matrix = jnp.column_stack([jnp.ones(n), cohort.covariates])  # (N, 4)

    def negative_log_likelihood(theta):
        treatment = distributions.Bernoulli(logits=X_matrix @ theta)
        return -treatment.log_prob(cohort.treated).sum()

    return minimise(negative_log_likelihood, jnp.zeros(X_matrix.shape[1]))


def propensity_scores(theta, covariates):
    """(N,) fitted probability of chemotherapy: intercept plus covariate effects."""
    return jax.nn.sigmoid(theta[0] + covariates @ theta[1:])


# Outcome model: Weibull proportional hazards ---------------------------------


def outcome_design_matrix(treated, covariates):
    """(N, 5) columns: chemo, age, frail, SRS and the chemo x SRS interaction."""
    treated = jnp.broadcast_to(treated, covariates.shape[:1])  # a scalar sets everyone
    srs = covariates[:, 2]
    return jnp.column_stack([treated, covariates, treated * srs])


def weibull_ph(theta, design_matrix):
    """theta = (log shape, log scale, coefficients): one distribution per row."""
    log_shape, log_scale = theta[:2]
    coefficients = theta[2:]
    baseline = survjax.Weibull.from_log(log_shape=log_shape, log_scale=log_scale)
    return survjax.ProportionalHazards(
        baseline=baseline, log_hazard_ratio=design_matrix @ coefficients
    )


def fit_weibull_ph(survival, design_matrix):
    """Maximum likelihood Weibull PH fit of right-censored survival times."""

    def negative_log_likelihood(theta):
        return -weibull_ph(theta, design_matrix).log_likelihood(survival).sum()

    # Start at shape 1, scale = mean observed time, and no covariate effects.
    start = jnp.zeros(2 + design_matrix.shape[1])
    start = start.at[1].set(jnp.log(survival.lower.mean()))
    return minimise(negative_log_likelihood, start)


def counterfactual_rmst(theta, covariates, treated):
    """(N,) model-based RMST(TAU) for every patient, had they all had `treated`."""
    # Same covariates, but the treatment column set to `treated` for everyone.
    counterfactual_X = outcome_design_matrix(treated, covariates)
    return weibull_ph(theta, counterfactual_X).rmst(TAU)


# Censoring: known administrative follow-up -----------------------------------


def censoring_weights(survival):
    """(N,) inverse probability of censoring weights, 0 when the status at TAU is unknown."""
    time = survival.lower
    known_by_tau = survival.exact | (time >= TAU)
    survived_censoring = FOLLOW_UP.survival(jnp.minimum(time, TAU))  # >= 0.75
    return jnp.where(known_by_tau, 1.0 / survived_censoring, 0.0)


# Estimators -------------------------------------------------------------------


def aiptw(outcome, ipcw, treated, propensity, rmst_0, rmst_1):
    """(2,) augmented IPTW mean of min(T, TAU): (surgery alone, chemotherapy)."""
    weight_0 = (1 - treated) * ipcw / (1 - propensity)
    weight_1 = treated * ipcw / propensity
    psi_0 = jnp.mean(rmst_0 + weight_0 * (outcome - rmst_0))
    psi_1 = jnp.mean(rmst_1 + weight_1 * (outcome - rmst_1))
    return jnp.stack([psi_0, psi_1])


def iptw(outcome, ipcw, treated, propensity):
    """(2,) weighted (Hajek) mean of min(T, TAU): (surgery alone, chemotherapy)."""
    weight_0 = (1 - treated) * ipcw / (1 - propensity)
    weight_1 = treated * ipcw / propensity
    psi_0 = jnp.sum(weight_0 * outcome) / jnp.sum(weight_0)
    psi_1 = jnp.sum(weight_1 * outcome) / jnp.sum(weight_1)
    return jnp.stack([psi_0, psi_1])


def naive_rmst(cohort):
    """(2,) RMST per arm from a Weibull PH model with treatment only."""
    theta, converged = fit_weibull_ph(cohort.survival, cohort.treated[:, None])
    per_arm = weibull_ph(theta, jnp.array([[0.0], [1.0]]))
    return per_arm.rmst(TAU), converged


def analyse(data):
    """The whole analysis of one cohort: fit, predict and estimate.

    A pure function of the data, so it can be jitted and vmapped over
    bootstrap resamples.
    """
    ps_theta, ps_converged = fit_propensity(data)
    propensity = propensity_scores(ps_theta, data.covariates)

    observed_X = outcome_design_matrix(data.treated, data.covariates)
    outcome_theta, outcome_converged = fit_weibull_ph(data.survival, observed_X)
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


def bootstrap(key, data, n_boot):
    """Analyse n_boot resamples with replacement in one vmapped, compiled call."""
    n = len(data.treated)
    indices = jax.random.randint(key, (n_boot, n), 0, n)  # one row per replicate

    def analyse_resample(idx):
        return analyse(resample(data, idx))

    analyse_all = jax.vmap(analyse_resample)  # over the rows of `indices`
    return jax.jit(analyse_all)(indices)  # compiled once, for all replicates
