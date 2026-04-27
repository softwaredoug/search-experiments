from pathlib import Path

import nbformat
import yaml


NOTEBOOK_CONFIG_PAIRS = [
    ("notebooks/bm25_ms_marco.ipynb", "configs/bm25.yml"),
    ("notebooks/agentic_ecom_2tools_wands.ipynb", "configs/agentic_ecom_2tools_gpt5_mini.yml"),
    ("notebooks/agentic_msmarco_minimarco.ipynb", "configs/agentic_msmarco.yml"),
]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def test_notebook_first_cell_includes_config_description():
    for notebook_path, config_path in NOTEBOOK_CONFIG_PAIRS:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        description = config.get("strategy", {}).get("description")
        if not description:
            continue

        nb = nbformat.read(Path(notebook_path), as_version=4)
        assert nb.cells, f"Notebook has no cells: {notebook_path}"
        first = nb.cells[0]
        assert first.get("cell_type") == "markdown", (
            f"First cell must be markdown in {notebook_path}"
        )
        first_text = "".join(first.get("source", []))
        assert _normalize(description) in _normalize(first_text), (
            f"Notebook {notebook_path} first cell missing description from {config_path}"
        )
