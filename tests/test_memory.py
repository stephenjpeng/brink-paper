"""Tests for the measured-RSS helpers."""

import os
import time

from brink.memory import RssMonitor, aggregate_rss_gib


def test_aggregate_rss_of_self_is_positive():
    assert aggregate_rss_gib([os.getpid()]) > 0.0


def test_rss_monitor_records_samples():
    mon = RssMonitor(interval_s=0.05)
    mon.start()
    time.sleep(0.2)
    mon.stop()
    assert mon.peak_gib > 0.0
    assert len(mon.samples) >= 1
    assert not mon.is_alive()


def test_stuck_ps_cannot_hang_the_monitor(monkeypatch):
    """A stopped or hung `ps` child must not block RssMonitor.stop().

    Regression for the double-integrator sweep (2026-10-23): a `ps` child sat in state T
    for 2.7 h, the monitor thread blocked on it, and the parent never exited
    even though all 750 rows had been written.
    """
    import subprocess

    import brink.memory as memory

    def hung_run(*args, **kwargs):
        timeout = kwargs.get("timeout")
        assert timeout is not None, "ps must be called with a timeout"
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=timeout)

    monkeypatch.setattr(memory.subprocess, "run", hung_run)
    assert aggregate_rss_gib([os.getpid()]) != aggregate_rss_gib([os.getpid()])  # nan
    mon = RssMonitor(interval_s=0.05)
    mon.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    mon.stop()
    assert time.monotonic() - t0 < 5.0
    assert not mon.is_alive()
    assert mon.missed_samples >= 1
