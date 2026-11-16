"""Synthetic stage III colon cancer cohort, its follow-up and its true RMST; not library API."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import survjax

ROOT_KEY = 20261025  # shared by all use cases; this one folds in 4
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


@survjax.register_family
@dataclass(frozen=True)
class UniformFollowUp(survjax.SurvivalDistribution):
    """Administrative censoring: potential follow-up is uniform on [start, end].

    Patients enter uniformly over ``end - start`` years and the analysis is
    ``end`` years after the first entry, so nobody is censored before ``start``.
    """

    start: jax.Array = survjax.param()
    end: jax.Array = survjax.param()

    def _inside(self, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Mask of start <= t < end, and t with outside values replaced by the midpoint."""
        inside = (t >= self.start) & (t < self.end)
        # spec 7.2: substitute before the logs so unselected branches stay finite
        return inside, jnp.where(inside, t, (self.start + self.end) / 2)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        t = jnp.asarray(t)
        inside, safe_t = self._inside(t)
        log_s = jnp.log1p(-(safe_t - self.start) / (self.end - self.start))
        return jnp.where(t < self.start, 0.0, jnp.where(inside, log_s, -jnp.inf))

    # Hard support edges: override density and hazard rather than differentiate.
    def log_prob(self, t: ArrayLike) -> jax.Array:
        t = jnp.asarray(t)
        inside, _ = self._inside(t)
        return jnp.where(inside, -jnp.log(self.end - self.start), -jnp.inf)

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        # Hazard 1 / (end - t) inside; at and beyond end it keeps its limit, +inf.
        t = jnp.asarray(t)
        inside, safe_t = self._inside(t)
        log_h = -jnp.log(self.end - safe_t)
        return jnp.where(t < self.start, -jnp.inf, jnp.where(inside, log_h, jnp.inf))

    def reference_time(self) -> jax.Array:
        return jnp.broadcast_to((self.start + self.end) / 2, self.batch_shape)


FOLLOW_UP = UniformFollowUp(start=4.0, end=8.0)


def use_case_key(stream: int) -> jax.Array:
    """Key for stream 0 (cohort), 1 (bootstrap) or 2 (truth by Monte Carlo)."""
    return jax.random.fold_in(jax.random.fold_in(jax.random.key(ROOT_KEY), 4), stream)


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


def true_rmst(n_draws: int = 1_000_000) -> dict[str, jax.Array]:
    """Monte Carlo truth over the covariate distribution, using closed-form RMST.

    Returns:
        ``causal``: (2,) E[RMST(TAU | a, X)] for a = 0, 1. ``naive``: (2,)
        E[RMST(TAU | 0, X) | A = 0] and E[RMST(TAU | 1, X) | A = 1], the
        target of an unadjusted comparison.
    """
    covariates = sample_covariates(use_case_key(2), n_draws)
    rmst = jnp.stack(
        [true_death_model(covariates, a).rmst(TAU) for a in (0.0, 1.0)], axis=-1
    )
    p = true_propensity(covariates)
    arm_weight = jnp.stack([1 - p, p], axis=-1)
    naive = jnp.sum(arm_weight * rmst, axis=0) / jnp.sum(arm_weight, axis=0)
    return {"causal": rmst.mean(axis=0), "naive": naive}
