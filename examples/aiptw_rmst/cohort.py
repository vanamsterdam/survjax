"""Synthetic stage III colon cancer cohort and its true RMST; not library API."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import survjax
from examples.aiptw_rmst.follow_up import FOLLOW_UP

TAU = 5.0  # restriction time, years

# Treatment: older and frail patients are less often given chemotherapy.
TREATMENT_INTERCEPT = 0.7
TREATMENT_AGE = -0.6  # per 10 years
TREATMENT_FRAIL = -1.0

# Death: Weibull proportional hazards, time in years.
DEATH_SHAPE = 1.3
DEATH_SCALE = 8.4  # untreated, aged 66, not frail, average SRS
# Log hazard ratios for (chemo, age per 10 years, frail, SRS, chemo x SRS).
DEATH_LOG_HR = (math.log(0.7), 0.35, 0.7, 0.2, 0.15)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Cohort:
    """One row per patient: outcome, treatment and baseline covariates.

    Attributes:
        survival: (N,) overall survival in years, exact or right-censored.
        treated: (N,) 1.0 for adjuvant chemotherapy, 0.0 for surgery alone.
        covariates: (N, 3) columns: age as (years - 66) / 10, frail (0/1), SRS.
    """

    survival: survjax.SurvivalData
    treated: jax.Array
    covariates: jax.Array


def resample(cohort: Cohort, idx: jax.Array) -> Cohort:
    """Select patients by (N,) row indices, for example a bootstrap draw."""
    return jax.tree.map(lambda leaf: leaf[idx], cohort)


def sample_covariates(key: jax.Array, n: int) -> jax.Array:
    """Draw (n, 3) covariates: age 66 +- 10 years (scaled), frailty, SRS ~ N(0, 1)."""
    age_key, frail_key, srs_key = jax.random.split(key, 3)
    age = jax.random.normal(age_key, (n,))
    frail = jax.random.bernoulli(frail_key, jax.nn.sigmoid(-1.4 + 0.6 * age))
    srs = jax.random.normal(srs_key, (n,))
    return jnp.stack([age, frail.astype(age.dtype), srs], axis=-1)


def true_propensity(covariates: jax.Array) -> jax.Array:
    """(N,) probability of chemotherapy; the SRS was never seen by the clinicians."""
    age, frail = covariates[:, 0], covariates[:, 1]
    return jax.nn.sigmoid(
        TREATMENT_INTERCEPT + TREATMENT_AGE * age + TREATMENT_FRAIL * frail
    )


def true_death_model(
    covariates: jax.Array, treated: ArrayLike
) -> survjax.ProportionalHazards:
    """Batch (N,) of the true per-patient survival distributions."""
    age, frail, srs = covariates[:, 0], covariates[:, 1], covariates[:, 2]
    treated = jnp.broadcast_to(treated, age.shape)
    design = jnp.stack([treated, age, frail, srs, treated * srs], axis=-1)
    return survjax.ProportionalHazards(
        baseline=survjax.Weibull(shape=DEATH_SHAPE, scale=DEATH_SCALE),
        log_hazard_ratio=design @ jnp.asarray(DEATH_LOG_HR),
    )


def simulate_cohort(key: jax.Array, n: int = 500) -> Cohort:
    """Simulate n patients with confounded treatment and administrative censoring."""
    covariate_key, treat_key, death_key, follow_key = jax.random.split(key, 4)
    covariates = sample_covariates(covariate_key, n)
    treated = jax.random.bernoulli(treat_key, true_propensity(covariates))
    treated = treated.astype(covariates.dtype)

    # Weibull PH by inversion: T = b_i (-log U)^(1/k), b_i = b0 exp(-eta_i / k).
    eta = true_death_model(covariates, treated).log_hazard_ratio
    scale = DEATH_SCALE * jnp.exp(-eta / DEATH_SHAPE)
    u = jax.random.uniform(death_key, (n,))
    death_time = scale * (-jnp.log(u)) ** (1 / DEATH_SHAPE)
    follow_up = jax.random.uniform(
        follow_key, (n,), minval=FOLLOW_UP.start, maxval=FOLLOW_UP.end
    )

    time = jnp.minimum(death_time, follow_up)
    survival = survjax.SurvivalData(time, event=death_time <= follow_up)
    return Cohort(survival, treated, covariates)


def true_rmst(key: jax.Array, n_draws: int = 1_000_000) -> dict[str, jax.Array]:
    """Monte Carlo truth over the covariate distribution, using closed-form RMST.

    Returns:
        ``causal``: (2,) E[RMST(TAU | a, X)] for a = 0, 1. ``naive``: (2,)
        E[RMST(TAU | 0, X) | A = 0] and E[RMST(TAU | 1, X) | A = 1], the
        target of an unadjusted comparison.
    """
    covariates = sample_covariates(key, n_draws)
    rmst = jnp.stack(
        [true_death_model(covariates, a).rmst(TAU) for a in (0.0, 1.0)], axis=-1
    )
    p = true_propensity(covariates)
    arm_weight = jnp.stack([1 - p, p], axis=-1)
    naive = jnp.sum(arm_weight * rmst, axis=0) / jnp.sum(arm_weight, axis=0)
    return {"causal": rmst.mean(axis=0), "naive": naive}
