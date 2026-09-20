from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "UH_gun_tracking_demo.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_sources() -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    ]


def _named_assignments(tree: ast.AST) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
    return assignments


def _constant_key_map(node: ast.AST) -> dict[str, ast.AST]:
    assert isinstance(node, ast.Dict)
    result: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        result[key.value] = value
    return result


def _completion_marker_ast() -> tuple[str, ast.AST, dict[str, ast.AST]]:
    cells = [source for source in _code_sources() if "rg.atomic_write_json(" in source]
    assert len(cells) == 1
    source = cells[0]
    tree = ast.parse(source)
    marker_writes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "atomic_write_json"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "completion_marker"
        ):
            marker_writes.append(node.args[1])
    assert len(marker_writes) == 1
    return source, marker_writes[0], _named_assignments(tree)


@pytest.mark.parametrize("name", ["UH_gun_tracking_demo.ipynb", "XFdtd_field_map_preprocessing.ipynb"])
def test_notebook_is_clean_and_all_code_cells_parse(name: str) -> None:
    notebook = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
    code = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        source = "".join(cell.get("source", []))
        code.append(source)
        ast.parse(source)
    executable_source = "\n".join(code).lower()
    assert "load_fieldmap_mat" not in executable_source
    # Inspect literal input paths, not attribute names such as
    # cathode_origin_check.material_sample_beam_z_m.
    for node in ast.walk(ast.parse("\n".join(code))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.lower().endswith(".mat")


def test_notebook_marker_uses_schema_v2_required_output_hash_contract() -> None:
    source, marker_node, assignments = _completion_marker_ast()
    marker = _constant_key_map(marker_node)

    assert isinstance(marker["schema_version"], ast.Constant)
    assert marker["schema_version"].value == 2
    assert isinstance(marker["stage"], ast.Constant)
    assert marker["stage"].value == "transport"
    assert isinstance(marker["status"], ast.Constant)
    assert marker["status"].value == "complete"
    assert isinstance(marker["validation_status"], ast.Constant)
    assert marker["validation_status"].value == "ok"

    required_marker_keys = {
        "run_family",
        "scan_tags",
        "run_config_sha256",
        "canonical_config",
        "canonical_config_sha256",
        "field_artifact_sha256",
        "field_artifact_payload_sha256",
        "field_input_sha256",
        "field_identity",
        "required_outputs",
    }
    assert required_marker_keys <= set(marker)
    assert isinstance(marker["required_outputs"], ast.Name)
    assert marker["required_outputs"].id == "_required_outputs"

    outputs = _constant_key_map(assignments["_required_outputs"])
    assert {
        "run_config",
        "run_results",
        "validation",
        "back_bombardment_events",
    } <= set(outputs)
    for name in ("run_config", "run_results", "validation"):
        assert set(_constant_key_map(outputs[name])) == {"file", "sha256"}
    event = _constant_key_map(outputs["back_bombardment_events"])
    assert set(event) == {"file", "sha256", "schema_version"}
    assert isinstance(event["schema_version"], ast.Constant)
    assert event["schema_version"].value == "back_bombardment_events_v2"

    assert "sha256_file(run_config_path)" in source
    assert "sha256_file(run_results_path)" in source
    assert "sha256_file(validation_path)" in source
    assert "sha256_file(_bb_events_v2_path)" in source
    assert 'for _output_path in sorted(RUN_DIR.rglob("*"))' in source
    assert '"sha256": sha256_file(_output_path)' in source
    assert "completion_marker.unlink(missing_ok=True)" in source
    assert source.index("completion_marker.unlink(missing_ok=True)") < source.index(
        'if validation_report["status"] == "ok"'
    )

    all_source = "\n".join(_code_sources())
    early_invalidation = '(RUN_DIR / ".run_complete").unlink(missing_ok=True)'
    assert early_invalidation in all_source
    assert all_source.index(early_invalidation) < all_source.index(
        "rg.run_transport_with_progress("
    )
    assert "RUN_DIR.mkdir(parents=True, exist_ok=False)" in all_source
    assert "FIGURES_DIR.mkdir(exist_ok=False)" in all_source
    assert "SCREENS_DIR.mkdir(exist_ok=False)" in all_source


def test_notebook_validation_requires_requested_completion_bundle() -> None:
    source, _, _ = _completion_marker_ast()
    for check_name in (
        "requested_openpmd_beams_written",
        "requested_screen_hdf5_written",
        "requested_lost_particle_diagnostics_written",
        "requested_figures_written",
        "requested_macropulse_outputs_written",
        "requested_emission_iteration_data_written",
    ):
        assert f'validation_checks["{check_name}"]' in source
    assert "len(z_snaps) == _requested_screen_count" in source
    assert "len(saved_screen_paths) == _requested_screen_count" in source
    assert "path.parent.resolve() == BB_STUDY_DIR.resolve()" in source


def test_notebook_marker_preserves_full_field_power_and_tail_identity() -> None:
    _, marker_node, assignments = _completion_marker_ast()
    marker = _constant_key_map(marker_node)
    assert isinstance(marker["field_identity"], ast.Name)
    assert marker["field_identity"].id == "_field_identity"

    field_identity = _constant_key_map(assignments["_field_identity"])
    assert {
        "kind",
        "path",
        "file_sha256",
        "payload_sha256",
        "payload_verification",
        "source_sha256",
        "processing_config_sha256",
        "rf_magnetic_field_mode",
        "rf_magnetic_field_enabled",
        "source_power_w",
        "target_power_w",
        "common_field_scale",
        "field_tail_policy",
        "artifact_variant",
    } <= set(field_identity)

    all_source = "\n".join(_code_sources())
    assert "_field_artifact_file_sha256 = sha256_file(FIELD_ARTIFACT_PATH)" in all_source
    assert '"file_sha256": _field_artifact_file_sha256' in all_source
    assert '"payload_verification": "payload"' in all_source
    assert '"back_bombardment_events_v2_written"' in all_source
