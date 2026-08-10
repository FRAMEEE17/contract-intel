"""Unit test for the Timed proxy — records time, preserves behavior, doesn't nest."""
from __future__ import annotations

import time

from profile_latency import Timed


class _Port:
    def slow(self, x):
        time.sleep(0.01)
        return x * 2

    def passthrough(self, y):
        return y + 1


def test_timed_records_and_preserves_behavior():
    sink: dict = {}
    p = Timed(_Port(), sink)

    assert p.slow(3) == 6          # same return value
    assert p.passthrough(4) == 5
    assert p.slow(5) == 10

    assert sink["slow"] >= 0.02    # two ~10ms calls accumulated
    assert "passthrough" in sink
    assert sink["passthrough"] < sink["slow"]


def test_timed_passes_through_non_callable_attrs():
    class _WithAttr:
        value = 42
    assert Timed(_WithAttr(), {}).value == 42
