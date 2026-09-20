"""Integrity regressions for shared file hashing and concurrent JSON writes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from rf_gun.fieldmaps.artifact import canonical_json
from rf_gun.io import atomic_write_json
from rf_gun.provenance import canonical_json_sha256, sha256_file


def test_streamed_file_hash_is_independent_of_chunk_size(tmp_path: Path):
    payload = bytes(range(256)) * 257
    path = tmp_path / "source.bin"
    path.write_bytes(payload)
    for chunk_size in (1, 71, 8192):
        assert sha256_file(path, chunk_size) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="positive"):
        sha256_file(path, 0)


def test_canonical_hash_preserves_identity_and_rejects_nonfinite_values():
    assert canonical_json_sha256({"a": 1, "b": [2]}) == canonical_json_sha256({"b": [2], "a": 1})
    with pytest.raises(ValueError):
        canonical_json_sha256({"a": float("nan")})
    value = {"phasors": np.array([1 + 2j]), "scalar": np.complex64(3 + 4j)}
    assert json.loads(canonical_json(value)) == {
        "phasors": [{"real": 1, "imag": 2}], "scalar": {"real": 3, "imag": 4},
    }


def test_concurrent_atomic_writers_leave_one_complete_record(tmp_path: Path):
    path = tmp_path / "run.json"
    barrier = Barrier(4)

    def write(index):
        barrier.wait()
        atomic_write_json(path, {"index": index, "samples": [index] * 1000})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    result = json.loads(path.read_text())
    assert result["samples"] == [result["index"]] * 1000
    assert list(tmp_path.iterdir()) == [path]


def test_failed_json_write_preserves_previous_record(tmp_path: Path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text('{"previous": true}')

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr("rf_gun.io.json.dump", fail)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_json(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"previous": True}
    assert list(tmp_path.iterdir()) == [path]
