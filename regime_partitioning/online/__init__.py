"""Online two-speed regime tracking layer.

Public surface:

- :class:`RegimeStreamingDetector` - ``on_bar`` facade;
- :class:`HMMTracker` - causal forward-filter scorer;
- :class:`FastPathDetector` - intra-session conditions-changed flag;
- :class:`FeatureBuilder` - ``ret``/``rv_20d`` from bars;
- :class:`DCUpdater` - directional-change intrinsic-time clock;
- :class:`WindowStateMachine` - hysteresis/debounce.

The layer is read-only: it emits labels and flags and never places orders.
"""

from .dc import DCEvent, DCState, DCUpdater
from .fast_path import FastPathDetector, confirmation_bars
from .features import FeatureBuilder
from .hmm_tracker import HMMTracker, WalkForwardHMMTracker
from .streaming import RegimeSignal, RegimeStreamingDetector
from .windows import Window, WindowRule, WindowStateMachine

__all__ = [
    "DCEvent",
    "DCState",
    "DCUpdater",
    "FastPathDetector",
    "confirmation_bars",
    "FeatureBuilder",
    "HMMTracker",
    "WalkForwardHMMTracker",
    "RegimeSignal",
    "RegimeStreamingDetector",
    "Window",
    "WindowRule",
    "WindowStateMachine",
]
