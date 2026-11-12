"""Tests for the online two-speed regime layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from regime_partitioning.online import (  # noqa: E402
    FastPathDetector,
    FeatureBuilder,
    HMMTracker,
    RegimeStreamingDetector,
    WindowRule,
    WindowStateMachine,
)

TS = pd.Timestamp("2024-01-01")


class StubTracker:
    def __init__(self, regime="risk_on", p_state1=0.1):
        self.regime = regime
        self.p_state1 = p_state1

    def score_step(self, obs_row):
        return {
            "p_state0": 1.0 - self.p_state1,
            "p_state1": self.p_state1,
            "map_state": 0 if self.regime == "risk_on" else 1,
            "regime": self.regime,
            "no_data": False,
        }


def _bar(close, high=None, low=None, open_=None, t=None):
    high = close if high is None else high
    low = close if low is None else low
    open_ = close if open_ is None else open_
    return {"t": t or TS, "open": open_, "high": high, "low": low, "close": close, "volume": 1.0}


def test_hysteresis_requires_confirmation_and_dwell():
    sm = WindowStateMachine(WindowRule(open_p=0.8, close_p=0.5, confirm_open=2, confirm_close=2, min_dwell_bars=2))
    assert sm.on_prob(TS, "risk_on", 0.1)  # initial window opens immediately
    sm.on_prob(TS + pd.Timedelta(days=1), "risk_on", 0.1)
    assert not sm.on_prob(TS + pd.Timedelta(days=2), "risk_off", 0.9)  # single bar
    assert not sm.on_prob(TS + pd.Timedelta(days=3), "risk_on", 0.1)  # streak reset
    events = sm.on_prob(TS + pd.Timedelta(days=4), "risk_off", 0.9)
    events += sm.on_prob(TS + pd.Timedelta(days=5), "risk_off", 0.95)
    assert events, "two confirmed bars should flip the label"
    assert sm.label == "risk_off"


def test_missing_data_is_explicit():
    fb = FeatureBuilder()
    row = fb.on_bar({"close": float("nan")})
    assert pd.isna(row["ret"]) and pd.isna(row["rv_20d"])
    tracker = HMMTracker.__new__(HMMTracker)
    tracker.cols = ("ret", "rv_20d")
    scored = HMMTracker.score_step(tracker, pd.Series({"ret": np.nan, "rv_20d": np.nan}))
    assert scored["no_data"] is True
    assert scored["regime"] == "NO DATA"


def test_seeded_vol_spike_trips_fast_path_but_slow_label_stays():
    detector = RegimeStreamingDetector(tracker=StubTracker("risk_on"), bar_minutes=1440)
    for i in range(40):
        detector.on_bar(_bar(1.10, high=1.1005, low=1.0995, t=TS + pd.Timedelta(days=i)))
    quiet = detector.on_bar(_bar(1.10, high=1.1005, low=1.0995, t=TS + pd.Timedelta(days=40)))
    assert quiet["fast_flag"] is False
    spike_day = 0
    for i in range(41, 45):
        # A large range and gap over two confirmed daily bars.
        out = detector.on_bar(
            _bar(1.16, high=1.19, low=1.09, t=TS + pd.Timedelta(days=i))
        )
        spike_day = i
        if out["fast_flag"]:
            break
    assert out["fast_flag"] is True, f"fast flag did not trip by day {spike_day}"
    assert out["fast_reasons"]
    assert out["regime"] == "risk_on", "slow label must remain stable during the spike"


def test_quiet_window_stability():
    detector = RegimeStreamingDetector(tracker=StubTracker("risk_on"), bar_minutes=1440)
    labels = []
    for i in range(60):
        out = detector.on_bar(
            _bar(1.10 + 0.00001 * i, high=1.1005 + 0.00001 * i, low=1.0995 + 0.00001 * i, t=TS + pd.Timedelta(days=i))
        )
        labels.append(out["regime"])
        assert out["fast_flag"] is False
    assert set(labels) == {"risk_on"}


def test_signal_contract_fields():
    detector = RegimeStreamingDetector(tracker=StubTracker("risk_off", p_state1=0.9))
    out = detector.on_bar(_bar(1.10))
    for field in (
        "as_of",
        "regime",
        "p_states",
        "classifier_version",
        "fast_flag",
        "fast_reasons",
        "stale",
        "no_data",
    ):
        assert field in out, field
    assert isinstance(out["fast_reasons"], list)
    assert isinstance(out["fast_flag"], bool)
    assert isinstance(out["no_data"], bool)


def test_signal_matches_json_schema():
    import json

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = REPO / "docs" / "regime-signal.schema.json"
    schema = json.loads(schema_path.read_text())
    sample = {
        "as_of": "2024-12-30",
        "symbol": "NZDUSD",
        "regime": "risk_off",
        "p_states": {"risk_on": 0.08, "risk_off": 0.92},
        "classifier_version": "regime-partitioning-online-v1",
        "fast_flag": False,
        "fast_reasons": [],
        "stale": False,
        "no_data": False,
    }
    jsonschema.validate(sample, schema)
