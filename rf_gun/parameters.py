"""UH reference parameters (copy of the Orchestrator master file)."""

from __future__ import annotations

from pathlib import Path

import yaml

REFERENCE_PARAMETERS_PATH = Path(__file__).resolve().parents[1] / "parameters" / "UH_reference_parameters.yaml"


def load_reference_parameters(path: str | Path | None = None) -> dict:
    """Return the reference-parameter YAML as a dict; defaults to the repo copy."""
    with open(REFERENCE_PARAMETERS_PATH if path is None else path, encoding="utf-8") as f:
        return yaml.safe_load(f)
