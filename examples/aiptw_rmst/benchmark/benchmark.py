"""Time the survjax analysis: one cohort, then bootstraps of growing size.

Prints one CSV row per timing. Compilation is timed apart from running, and
each run is the median of a few repeats. See run.sh.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding, PartitionSpec

import survjax
from examples.aiptw_rmst.analysis import analyse
from examples.aiptw_rmst.cohort import Cohort, resample


def read_cohort(path):
    """The CSV written by export_cohort.py, as a float64 Cohort."""
    table = jnp.asarray(np.loadtxt(path, delimiter=",", skiprows=1))
    time_, event, treated = table[:, 0], table[:, 1], table[:, 2]
    survival = survjax.SurvivalData(time_, event=event == 1)
    return Cohort(survival, treated, table[:, 3:])


def median_seconds(run, repeats):
    """Median wall time of `run()`, waiting for JAX to finish each time."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(run())
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def timed_compile(function, *args):
    """(compiled function, seconds spent compiling it)."""
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*args).compile()
    return compiled, time.perf_counter() - start


def analyse_resamples(data, indices):
    """The bootstrap of analysis.py, with the data as an argument so it compiles once."""
    return jax.vmap(lambda idx: analyse(resample(data, idx)))(indices)


def shard_over_devices(indices):
    """Spread the bootstrap rows over all CPU devices; jit then splits the work."""
    mesh = jax.make_mesh(
        (jax.device_count(),), ("replicate",), axis_types=(AxisType.Auto,)
    )
    return jax.device_put(indices, NamedSharding(mesh, PartitionSpec("replicate")))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--n-boot", type=int, nargs="+", default=[120, 1200])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        raise SystemExit("set JAX_ENABLE_X64=1: the analysis runs in float64")

    print("engine,n,n_boot,compile_s,run_s,converged,delta_aiptw,ci_low,ci_high")
    for path in args.csv:
        data = read_cohort(path)
        n = len(data.treated)
        single, compile_s = timed_compile(analyse, data)
        estimates = single(data)
        run_s = median_seconds(lambda: single(data), 10 * args.repeats)  # noqa: B023
        delta = float(estimates.aiptw[1] - estimates.aiptw[0])
        print(f"jax,{n},1,{compile_s:.3f},{run_s:.5f},{int(estimates.converged)},"
              f"{delta:.6f},,")  # fmt: skip

        for n_boot in args.n_boot:
            key = jax.random.PRNGKey(n_boot)
            indices = jax.random.randint(key, (n_boot, n), 0, n)
            indices = shard_over_devices(indices)
            boot, compile_s = timed_compile(analyse_resamples, data, indices)
            replicates = boot(data, indices)
            run_s = median_seconds(lambda: boot(data, indices), args.repeats)  # noqa: B023
            draws = replicates.aiptw[:, 1] - replicates.aiptw[:, 0]
            low, high = np.quantile(np.asarray(draws), [0.025, 0.975])
            engine = f"jax ({jax.device_count()} devices)"
            print(f"{engine},{n},{n_boot},{compile_s:.3f},{run_s:.4f},"
                  f"{int(replicates.converged.sum())},,{low:.4f},{high:.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
