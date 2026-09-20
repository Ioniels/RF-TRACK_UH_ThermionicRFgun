"""The pure field-map package must not require the RF-Track Python binding."""

from __future__ import annotations

import subprocess
import sys


def test_fieldmap_submodule_import_does_not_load_rftrack() -> None:
    code = r'''
import importlib.abc
import sys

class BlockRFTrack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "RF_Track":
            raise AssertionError("field-map import attempted to load RF_Track")
        return None

sys.meta_path.insert(0, BlockRFTrack())
import rf_gun.fieldmaps.xfdtd_h5
assert "RF_Track" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_field_analysis_plotting_import_does_not_load_rftrack() -> None:
    code = r'''
import importlib.abc
import sys

class BlockRFTrack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "RF_Track":
            raise AssertionError("field-analysis plotting import attempted to load RF_Track")
        return None

sys.meta_path.insert(0, BlockRFTrack())
import rf_gun.plotting.field_analysis
assert "RF_Track" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
