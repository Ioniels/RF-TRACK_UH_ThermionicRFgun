"""Standalone progress-percentage printer, run in its own `multiprocessing.get_context("spawn")`
process by `rf_gun.simulation.run_transport_with_progress`.

Lives outside the `rf_gun` package on purpose: importing any `rf_gun` submodule imports
RF_Track (a heavy, banner-printing import) via `rf_gun/__init__.py`. A spawned child re-imports
its target function's module from scratch, so keeping this printer in a plain top-level module
means the child only imports the stdlib.

RF-Track's tracking call is a single blocking C++ call with no per-step Python hook, so progress
is a wall-clock estimate printed concurrently on a timer. A plain in-process thread for this can
be starved of the GIL for the whole call and stop ticking; a separate OS process has no GIL to
share with the tracking call, so it keeps printing on schedule regardless. This only helps a
consumer reading the process's output directly (e.g. a redirected log file) -- forwarding that
output live into a Jupyter cell still goes through the kernel's own GIL-bound relay, so inside a
notebook this backend freezes the same way an in-process thread does.
"""
from __future__ import annotations

import math
import time


def progress_proxy_pct(elapsed_s: float, est_s: float) -> float:
    """Wall-clock proxy percentage against a runtime estimate -- not real tracking progress.
    Asymptotes toward 99% if the run overruns the estimate. Duplicated from
    `rf_gun.simulation._progress_proxy_pct`: this module must not import from `rf_gun`."""
    est = max(1e-9, float(est_s))
    if elapsed_s <= est:
        return min(98.0, 100.0 * elapsed_s / est)
    over = (elapsed_s - est) / est
    return min(99.0, 98.0 + (1.0 - math.exp(-over)))


#: Force a heartbeat line at least this often, even if the rounded percentage hasn't changed
#: (it can sit at the same value for a long stretch once the run overruns its estimate).
_MAX_SECONDS_BETWEEN_PRINTS = 15.0


def spawn_progress_target(start_s: float, est_for_proxy_s: float, poll_interval_s: float) -> None:
    """Process entry point: print `tracking NN%` on a timer until the parent terminates it."""
    last_pct = -1
    last_print_s = -math.inf
    while True:
        elapsed_s = time.time() - start_s
        pct = int(progress_proxy_pct(elapsed_s, est_for_proxy_s))
        overdue = (elapsed_s - last_print_s) >= _MAX_SECONDS_BETWEEN_PRINTS
        if pct != last_pct or overdue:
            suffix = " (still running -- overran the time estimate)" if overdue and pct == last_pct else ""
            print(f"tracking {pct}% | elapsed={elapsed_s:,.1f}s est={float(est_for_proxy_s):,.1f}s{suffix}", flush=True)
            last_pct = pct
            last_print_s = elapsed_s
        time.sleep(max(0.1, float(poll_interval_s)))
