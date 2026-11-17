"""Write a simulated cohort to CSV, so that R analyses the very same patients.

From the repository root:

    JAX_ENABLE_X64=1 uv run --group examples python -m examples.aiptw_rmst.export_cohort \
        tests/fixtures/aiptw_rmst/cohort.csv --n 500
"""

import argparse

import jax
import numpy as np

from examples.aiptw_rmst.cohort import simulate_cohort

COLUMNS = ("time", "event", "treated", "z_age", "frail", "srs")


def tutorial_data_key():
    """The data key of the tutorial page: PRNGKey(1), split into data and bootstrap."""
    data_key, _bootstrap_key = jax.random.split(jax.random.PRNGKey(1))
    return data_key


def cohort_table(cohort):
    """(N, 6) host array with the columns in COLUMNS."""
    survival = cohort.survival
    columns = [survival.lower, survival.exact, cohort.treated, *cohort.covariates.T]
    return np.column_stack([np.asarray(column, dtype=np.float64) for column in columns])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="CSV file to write")
    parser.add_argument("--n", type=int, default=500, help="number of patients")
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        raise SystemExit("set JAX_ENABLE_X64=1: the tutorial cohort is float64")

    table = cohort_table(simulate_cohort(tutorial_data_key(), n=args.n))
    np.savetxt(args.path, table, fmt="%.17g", delimiter=",",
               header=",".join(COLUMNS), comments="")  # fmt: skip


if __name__ == "__main__":
    main()
