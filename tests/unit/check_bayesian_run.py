"""Optional: the Bayesian runner passes its acceptance checks and reproduces the committed draws."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "examples/bayesian_rmst/results/posterior.npz"
NOTEBOOK = ROOT / "examples/bayesian_rmst/bayesian-rmst.ipynb"


def test_notebook_trial_matches_shared_generator(tmp_path):
    """02: the standalone trial preserves the shared generator's exact draws."""
    notebook = json.loads(NOTEBOOK.read_text())
    trial_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "event_key, censor_key" in "".join(cell["source"])
    )
    program = (
        "import jax, jax.numpy as jnp, numpy as np, survjax\n"
        "from examples.bayesian_rmst.model import make_trial\n"
        + trial_cell
        + "\ntrial = make_trial()\n"
        "np.testing.assert_array_equal(data.lower, trial.data.lower)\n"
        "np.testing.assert_array_equal(data.exact, trial.data.exact)\n"
        "np.testing.assert_array_equal(arm, trial.arm)\n"
        "assert censor_limit == trial.censor_limit\n"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=dict(os.environ, JAX_ENABLE_X64="1"),
        check=True,
        timeout=60,
    )


def test_notebook_runs_outside_checkout(tmp_path):
    """02: the notebook runs in a fresh kernel with no example package available.

    Run this optional check with both the docs and examples dependency groups.
    """
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "import importlib.util\nassert importlib.util.find_spec('examples') is None"
        ),
    )
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()


def test_run_reproduces_committed_posterior(tmp_path):
    """02: a fresh float64 run passes the acceptance asserts and matches the draws bit for bit."""
    output = tmp_path / "run"
    # A separate process: the runner sets the device count and float64 before JAX starts.
    subprocess.run(
        [sys.executable, "-m", "examples.bayesian_rmst.run", str(output)],
        cwd=ROOT,
        env=dict(os.environ, JAX_ENABLE_X64="1"),
        check=True,
        capture_output=True,
        timeout=600,
    )
    with np.load(output / "posterior.npz") as new, np.load(COMMITTED) as old:
        assert sorted(new.files) == sorted(old.files)
        for name in old.files:
            np.testing.assert_array_equal(new[name], old[name], err_msg=name)
