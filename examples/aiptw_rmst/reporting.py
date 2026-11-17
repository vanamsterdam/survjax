"""Table helpers for the AIPTW tutorial page and notebook; not library API."""

import jax.numpy as jnp
from IPython.display import Markdown


def markdown_table(header, rows):
    """A Markdown table, rendered like the hand-written tables on the page."""
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return Markdown("\n".join(lines))


def standardized_difference(x, treated, weight):
    """Weighted standardized mean difference, chemotherapy minus surgery alone."""
    means, variances = [], []
    for arm in (0.0, 1.0):
        w = weight * (treated == arm)
        mean = jnp.sum(w * x) / jnp.sum(w)
        means.append(mean)
        variances.append(jnp.sum(w * (x - mean) ** 2) / jnp.sum(w))
    return (means[1] - means[0]) / jnp.sqrt((variances[0] + variances[1]) / 2)
