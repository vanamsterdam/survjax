"""PH.PERF.1: CPU PH-Weibull costs; run each JAX_ENABLE_X64 mode separately."""

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import runpy
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import survjax

SEED = 20261116
TOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tests/impl_tolerances.py")
)


@survjax.register_family
@dataclass(frozen=True)
class UnsharedPH(survjax.ProportionalHazards):
    """Inherited PH primitives through the unchanged generic evaluator."""


def _timed(call):
    start = time.perf_counter()
    result = jax.block_until_ready(call())
    return result, time.perf_counter() - start


def _inputs(n, mix, dtype):
    rng = np.random.default_rng(SEED)
    t = rng.uniform(0.75, 2.5, n).astype(dtype)
    arm = rng.integers(0, 2, n).astype(dtype)
    if mix in ("exact", "exact_right"):
        event = np.ones(n, bool) if mix == "exact" else rng.random(n) < 0.65
        data = survjax.SurvivalData(t, event)
    else:
        kind = np.arange(n) % 4
        lower = np.where(kind == 2, 0, t)
        upper = np.where(kind == 1, np.inf, np.where(kind == 3, t + 0.5, t))
        windows = {}
        if mix == "truncated":
            window = (np.arange(n) // 4) % 4
            windows = {
                "left_trunc": np.where(window % 2 == 1, 0.5, 0).astype(dtype),
                "right_trunc": np.where(window >= 2, 3, np.inf).astype(dtype),
            }
        data = survjax.SurvivalData.from_bounds(
            lower, upper, exact=kind == 0, **windows
        )
    assert bool(data.issues.ok)
    return data, jnp.asarray(arm)


def _reference(p, data, arm):
    # Independent analytical derivatives in (log k, log b, gamma).
    # H'=H*(k*log(t/b), -k, arm); log f=log k-log t+log H-H.
    k, log_b, eta = np.exp(float(p[0])), float(p[1]), float(p[2]) * np.asarray(arm)

    def hazard(t):
        t = np.asarray(t, dtype=np.float64)
        interior = (t > 0) & np.isfinite(t)
        u = np.log(np.where(interior, t, 1)) - log_b
        h = np.exp(eta + k * u)
        g = np.stack((k * u, np.full_like(u, -k), arm), axis=-1)
        return np.where(t == 0, 0, h), np.where((t == 0)[:, None], 0, h[:, None] * g), g

    def mass(a, b):
        ha, da, _ = hazard(a)
        hb, db, _ = hazard(b)
        finite = np.isfinite(b)
        delta = np.where(finite, hb - ha, 1)
        # log(S(a)-S(b))=-H(a)+log(1-exp(-(H(b)-H(a)))).
        value = -ha + np.where(finite, np.log(-np.expm1(-delta)), 0)
        grad = -da + np.where(finite[:, None], (db - da) / np.expm1(delta)[:, None], 0)
        return value, grad

    exact = np.asarray(data.exact)
    t = np.where(exact, data.lower, 1)
    h, _, g = hazard(t)
    density = np.log(k) - np.log(t) + eta + k * (np.log(t) - log_b) - h
    density_grad = (1 - h[:, None]) * g + np.array([1, 0, 0])
    left, right = np.asarray(data.left_trunc), np.asarray(data.right_trunc)
    lo = np.where(exact, left, np.maximum(data.lower, left))
    hi = np.where(exact, right, np.minimum(data.upper, right))
    interval, interval_grad = mass(lo, hi)
    normalizer, normalizer_grad = mass(left, right)
    values = np.where(exact, density, interval) - normalizer
    grads = np.where(exact[:, None], density_grad, interval_grad) - normalizer_grad
    return values.sum(), grads.sum(axis=0)


def _objective(route, operation, p, data, arm):
    k, b, eta = jnp.exp(p[0]), jnp.exp(p[1]), p[2] * arm
    if route == "unchecked":
        # Moderate exact/right control only: no validation or tail guarantees.
        log_h = eta + k * (jnp.log(data.lower) - p[1])
        log_s = -jnp.exp(log_h)
        density = p[0] - jnp.log(data.lower) + log_h + log_s
        values = jnp.where(data.exact, density, log_s)
        if operation != "likelihood":
            values = log_s if operation == "log_survival" else density
    else:
        baseline = survjax.Weibull(shape=k, scale=b)
        ph_type = UnsharedPH if route == "ph_generic" else survjax.ProportionalHazards
        dist = (
            ph_type(baseline=baseline, log_hazard_ratio=eta)
            if route in ("ph", "ph_generic")
            else survjax.Weibull(shape=k, scale=b * jnp.exp(-eta / k))
        )
        values = (
            dist.log_likelihood(data)
            if operation == "likelihood"
            else getattr(dist, operation)(data.lower)
        )
    return values.sum()


def _measure(n, mix, operation, dtype, warmups, repeats):
    (data, arm), preparation = _timed(lambda: _inputs(n, mix, dtype))
    p = jnp.asarray([np.log(1.3), np.log(2.0), 0.25], dtype=dtype)
    routes = ["ph", "equivalent"]
    if operation == "likelihood":
        routes.insert(1, "ph_generic")
    if mix in ("exact", "exact_right"):
        routes.append("unchecked")
    cases = []
    for route in routes:

        def objective(parameters, observations, covariates, route=route):
            return _objective(route, operation, parameters, observations, covariates)

        for derivative in (False, True):
            function = jax.grad(objective) if derivative else objective
            lowered, lowering = _timed(
                lambda function=function: jax.jit(function).lower(p, data, arm)
            )
            compiled, compilation = _timed(lowered.compile)
            result, first = _timed(lambda compiled=compiled: compiled(p, data, arm))
            # Check two parameter points on the same executable, outside timing.
            for point in (p, p + jnp.asarray([0.05, -0.025, 0.125], dtype=dtype)):
                if operation == "likelihood":
                    expected = _reference(point, data, np.asarray(arm))[int(derivative)]
                else:

                    def control(q):
                        return _objective("unchecked", operation, q, data, arm)

                    expected = (
                        jax.grad(control)(point) if derivative else control(point)
                    )
                np.testing.assert_allclose(
                    jax.block_until_ready(compiled(point, data, arm)),
                    expected,
                    atol=TOL["ATOL"][bool(jax.config.x64_enabled)],
                    rtol=TOL["RTOL"][bool(jax.config.x64_enabled)],
                )
            for _ in range(warmups):
                jax.block_until_ready(compiled(p, data, arm))
            cases.append(
                (
                    compiled,
                    {
                        "n": n,
                        "mix": mix,
                        "operation": operation,
                        "route": route,
                        "derivative": derivative,
                        "flags": list(data._may_contain),
                        "preparation_seconds": preparation,
                        "lowering_seconds": lowering,
                        "compile_seconds": compilation,
                        "first_execution_seconds": first,
                        "result": np.asarray(result).tolist(),
                        "warm_seconds": [],
                    },
                )
            )
    # Rotate order to distribute drift; synchronization includes dispatch cost.
    for i in range(repeats):
        order = cases[i % len(cases) :] + cases[: i % len(cases)]
        for compiled, row in order:
            row["warm_seconds"].append(
                _timed(lambda compiled=compiled: compiled(p, data, arm))[1]
            )
    rows = []
    for _, row in cases:
        row["warm_median_seconds"] = float(np.median(row["warm_seconds"]))
        rows.append(row)
        print(
            f"N={n} {mix} {operation} {row['route']} grad={row['derivative']}: "
            f"{row['warm_median_seconds'] * 1e6:.1f} us",
            flush=True,
        )
    return rows


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 30_000])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if min(args.sizes) < 16 or args.warmups < 1 or args.repeats < 2:
        parser.error("require sizes >=16, warmups >=1 and repeats >=2")
    jax.config.update("jax_enable_compilation_cache", False)
    device = jax.devices("cpu")[0]
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    processor = platform.processor()
    if sys.platform == "darwin":
        try:
            processor = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass  # Restricted environments retain platform.processor() metadata.
    report = {
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src"], text=True
        ).strip(),
        "command": f"JAX_ENABLE_X64={int(jax.config.x64_enabled)} JAX_PLATFORMS=cpu uv run python {shlex.join(sys.argv)}",
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
        "device": str(device),
        "dtype": np.dtype(dtype).name,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "survjax")
        },
        "seed": SEED,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "parameters": ["log_shape=log(1.3)", "log_scale=log(2)", "gamma=0.25"],
        "method": "sum objective and grad separately; dynamic prepared data and arm; block_until_ready every call; persistent compilation cache disabled",
        "limitations": "moderate representable Weibull control only; unchecked lacks guards; direct controls use AD, likelihood uses analytical NumPy derivatives; preparation includes generation, placement and construction; peak memory not measured",
    }
    rows = []
    with jax.default_device(device):
        for n in args.sizes:
            for mix in ("exact", "exact_right", "mixed", "truncated"):
                rows.extend(
                    _measure(n, mix, "likelihood", dtype, args.warmups, args.repeats)
                )
            for operation in ("log_survival", "log_prob"):
                rows.extend(
                    _measure(
                        n, "exact_right", operation, dtype, args.warmups, args.repeats
                    )
                )
    # One JSON record per line keeps raw samples compact and reviewable.
    args.output.write_text(
        '{"environment": '
        + json.dumps(report, allow_nan=False)
        + ',\n"rows": [\n'
        + ",\n".join(json.dumps(row, allow_nan=False) for row in rows)
        + "\n]}\n"
    )


if __name__ == "__main__":
    _main()
