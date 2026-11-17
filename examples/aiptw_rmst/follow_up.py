"""Administrative follow-up as a custom survjax distribution; not library API."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import survjax


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
