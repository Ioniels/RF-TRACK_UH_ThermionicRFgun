"""RF-Track configuration and imports."""
import os

import RF_Track as rft
from typing import Optional


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
