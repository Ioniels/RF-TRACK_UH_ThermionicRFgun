"""RF-Track configuration and imports.

RF-Track is loaded lazily.  Field-map inspection/reduction is deliberately usable on analysis
machines that do not have the proprietary tracking binding installed; the first operation that
actually asks for an RF-Track attribute still performs the same version gate as before.
"""
import importlib
import os

from typing import Optional

#: Every RF-Track-facing code path in this package (BeamLoadingSW's constructor signature,
#: Volume.set_sc_engine(), and everything tested in tests/) was developed and verified against
#: this exact version. A real KOA production batch hit `Volume has no set_sc_engine()` because the
#: cluster's venv had RF-Track 2.6.3 installed -- an older API generation, not a bug in this
#: codebase -- after ~100s of field-map interpolation had already run. Gate on the version at
#: import time instead, so a mismatch fails in under a second with an actionable message.
_MIN_SUPPORTED_RF_TRACK_VERSION = (2, 7)


def _check_rf_track_version(module: object) -> None:
    version_str = str(getattr(module, "version", "unknown"))
    parts = version_str.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        print(
            f"Warning: could not parse RF-Track version {version_str!r} (expected 'X.Y[.Z]') -- "
            f"skipping the version compatibility check. This package is tested against RF-Track "
            f"{'.'.join(str(v) for v in _MIN_SUPPORTED_RF_TRACK_VERSION)}.x."
        )
        return
    if (major, minor) < _MIN_SUPPORTED_RF_TRACK_VERSION:
        raise RuntimeError(
            f"Installed RF-Track version is {version_str}, but this package requires >= "
            f"{'.'.join(str(v) for v in _MIN_SUPPORTED_RF_TRACK_VERSION)} (e.g. Volume."
            "set_sc_engine(), used whenever space charge is enabled, does not exist before 2.7). "
            "Upgrade the active virtual environment with: pip install --upgrade RF_Track"
        )

class _LazyRFTrackModule:
    """Small transparent proxy that imports and validates :mod:`RF_Track` on first use."""

    _module: object | None = None

    def _load(self) -> object:
        if self._module is None:
            try:
                module = importlib.import_module("RF_Track")
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "RF-Track is required for particle tracking but is not installed. "
                    "Field-map preprocessing remains available through rf_gun.fieldmaps; "
                    "install RF-Track >=2.7 before using simulation functions."
                ) from exc
            _check_rf_track_version(module)
            self._module = module
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._load())))

    def __repr__(self) -> str:
        if self._module is None:
            return "<lazy RF_Track module (not loaded)>"
        return repr(self._module)


rft = _LazyRFTrackModule()


def show_versions():
    """Display RF-Track version and threading info."""
    print("RF-Track version:", rft.version)
    print("Max threads:", rft.max_number_of_threads)



def resolve_threads(requested: Optional[int] = None, default: int = 1) -> int:
    """Resolve runtime thread count from explicit value or scheduler environment.

    A malformed `requested` value or a garbage `SLURM_CPUS_PER_TASK` is a real misconfiguration
    (a typo'd CLI flag, a broken scheduler environment) -- it prints a warning and falls back to
    `default` rather than silently masking it, so a user relying on `--threads` actually taking
    effect finds out if it didn't.
    """
    if requested is not None:
        try:
            return max(1, int(requested))
        except (TypeError, ValueError):
            print(f"Warning: resolve_threads: requested={requested!r} is not a valid thread count; "
                  f"using default={default}.", flush=True)
            return max(1, int(default))

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is None:
        return max(1, int(default))
    try:
        return max(1, int(slurm_cpus))
    except (TypeError, ValueError):
        print(f"Warning: resolve_threads: SLURM_CPUS_PER_TASK={slurm_cpus!r} is not a valid thread "
              f"count; using default={default}.", flush=True)
        return max(1, int(default))


def set_thread_environment(threads: int, pin_blas_threads: bool = True) -> None:
    """Set RF-Track and BLAS/OpenMP thread env vars for reproducible runs."""
    os.environ["RF_TRACK_NUMBER_OF_THREADS"] = str(max(1, int(threads)))
    if pin_blas_threads:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
