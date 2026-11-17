"""Optional: the AIPTW tutorial notebook (needs the docs and examples groups)."""

import ast
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/aiptw_rmst"
NOTEBOOK = EXAMPLE / "aiptw-rmst.ipynb"


def definitions(source):
    """{name: source} of the top-level functions and classes in Python source."""
    return {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }


def test_notebook_defines_the_functions_of_analysis_py():
    """Every analysis.py definition appears in a cell of its own, unchanged."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    in_notebook = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.startswith(("def ", "class ")):
            in_notebook.update(definitions(cell.source))
    analysis = definitions((EXAMPLE / "analysis.py").read_text())
    assert set(analysis) <= set(in_notebook)
    for name, source in analysis.items():
        assert in_notebook[name] == source, name


def test_notebook_runs_top_to_bottom():
    """A fresh kernel runs every cell, including the 1000-replicate bootstrap."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    # Raises CellExecutionError, naming the cell, on the first failing cell.
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
