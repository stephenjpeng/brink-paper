"""Aggregate resident-memory monitor for multiprocess sweeps.

:class:`RssMonitor` samples the resident set size of a parent process and its
pool workers; each sweep records the peak in its summary.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3


def aggregate_rss_gib(pids: Iterable[int], timeout_s: float = 10.0) -> float:
    """Sum of resident set sizes of the given processes, in GiB.

    Uses ``ps`` (reports KiB); processes that have exited are skipped. The
    ``ps`` call has a timeout: in the double-integrator sweep a ``ps`` child was stopped
    (state T) for hours and, without one, blocked the monitor thread and kept
    the parent from exiting after every row was written.

    Args:
        pids: Process ids to include.
        timeout_s: Seconds to wait for ``ps`` before giving up.

    Returns:
        Aggregate RSS in GiB, or NaN when ``ps`` did not answer in time (the
        caller records the missed sample; a timed-out child is killed).
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        ).stdout
    except subprocess.TimeoutExpired:
        logger.warning("ps did not answer within %.0f s; RSS sample skipped", timeout_s)
        return float("nan")
    return sum(int(v) for v in out.split()) * 1024 / BYTES_PER_GB


class RssMonitor(threading.Thread):
    """Daemon thread sampling this process plus its multiprocessing children.

    Shared library pages are counted once per process, so the sum is a
    conservative (high) estimate of the footprint.
    interval sampling can miss sub-second spikes; the chunked
    posterior keeps per-block transients near 160 MiB, well under the margin.

    Attributes:
        interval_s: Seconds between samples.
        peak_gib: Largest aggregate RSS sampled so far.
        samples: (elapsed_s, gib) per sample.
        missed_samples: Samples skipped because ``ps`` timed out; a nonzero
            count means ``peak_gib`` may understate the true peak.
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        """Create the monitor; call ``start()`` to begin sampling.

        Args:
            interval_s: Seconds between samples.
        """
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak_gib = 0.0
        self.samples: list[tuple[float, float]] = []
        self.missed_samples = 0
        self._halt = threading.Event()
        self._t0 = time.monotonic()

    def run(self) -> None:
        """Sample until :meth:`stop` is called."""
        while not self._halt.is_set():
            pids = [os.getpid(), *(p.pid for p in mp.active_children())]
            gib = aggregate_rss_gib(pids)
            if gib != gib:  # NaN: ps timed out; record the gap, keep sampling
                self.missed_samples += 1
            else:
                self.samples.append(
                    (round(time.monotonic() - self._t0, 1), round(gib, 3))
                )
                self.peak_gib = max(self.peak_gib, gib)
            self._halt.wait(self.interval_s)

    def stop(self) -> None:
        """Stop sampling and wait for the thread to finish."""
        self._halt.set()
        self.join()
