Plan and scaffold, wired to your repo.

# What you already have

* PELT wrapper and 2-state HMM fitter in `regime_partitioning/processing.py` (PELT via `ruptures`, HMM via `hmmlearn`).   
* Example usage in `fx-regime-partitioning.py` showing PELT on `rate_diff_2y` and `rv_20d`, plus HMM outputs.  
* `rv_20d(symbol, …)` and 2-year rate differentials already defined.  
* Book references that justify HMM (EM/Baum-Welch) and online tracking with a classifier over DC features.   

# Minimal additions (modules + interfaces)

Create a thin “online layer” that consumes OHLCV bars and emits regime windows. Keep training offline.

```
regime_partitioning/
  online/
    __init__.py
    dc.py                 # directional-change state machine
    hmm_tracker.py        # forward-filter scoring
    bayes_tracker.py      # optional Naïve Bayes on (TMV, T)
    windows.py            # debounced open/close logic
    streaming.py          # on_bar() facade
    features.py           # ret, rv_20d (fallback), adapters to your data
```

## `online/dc.py`

Purpose: maintain DC intrinsic-time features in real time.

```python
class DCState:
    def __init__(self, theta_pct: float = 0.4):
        ...

class DCEvent(NamedTuple):
    t: pd.Timestamp
    r: float       # time-adjusted return or R
    tlen: int      # T (bars) or seconds if tick
    tmv: float     # total move value

class DCUpdater:
    def __init__(self, theta_pct: float = 0.4):
        self.state = DCState(theta_pct)
    def update(self, t: pd.Timestamp, price: float) -> list[DCEvent]:
        """Feed latest price. Return zero or more completed DCEvents."""
```

Notes: no need to restate the DC algorithm; it mirrors the book’s DC→R/T/TMV indicators used for regime tracking. The tracker in the book evaluates probabilities over these live indicators.  

## `online/hmm_tracker.py`

Purpose: turn your fitted `GaussianHMM` into an online scorer.

```python
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class HMMTracker:
    def __init__(self, model: GaussianHMM, scaler: StandardScaler, feature_cols: tuple[str, ...] = ("ret","rv_20d")):
        self.model = model
        self.scaler = scaler
        self.cols = feature_cols
        # maintain rolling buffer for scaling shape
        self._X_last = None

    def score_step(self, obs_row: pd.Series) -> dict:
        """
        obs_row: Series indexed by self.cols for time t.
        Returns: {'p_state0': float, 'p_state1': float, 'map_state': int}
        Uses model.predict_proba on standardized single-step appended to history.
        """
```

Rationale: reuse the fitted 2-state diagonal‐cov HMM and its scaler you already produce in `fit_2state_hmm`. We don’t re-derive EM; we call the library’s forward/posterior.   

## `online/bayes_tracker.py` (optional)

Purpose: Naïve Bayes over `(TMV, T)` if you prefer the book’s online classifier.

```python
class NaiveBayesTracker:
    def __init__(self, priors: dict, cond_models: dict):
        ...
    def score_step(self, tmv: float, tlen: float) -> dict:
        """Return {'p_regime1':..., 'p_regime2':...}"""
```

This matches the book’s Chapter 5 “regime tracking” with p(C_k | x_t).  

## `online/windows.py`

Purpose: debounce and emit regime windows.

```python
@dataclass
class WindowRule:
    open_p: float = 0.80
    close_p: float = 0.50
    confirm_open: int = 2     # k
    confirm_close: int = 2    # k'
    min_trends: int = 2       # L_min

@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp | None
    label: str                # 'regime_2' etc.

class WindowStateMachine:
    def __init__(self, rule: WindowRule):
        self.rule = rule
        self.current: Window | None = None
        self._open_streak = 0
        self._close_streak = 0
        self._trend_count = 0

    def on_prob(self, t: pd.Timestamp, p_regime2: float, dc_event: bool) -> list[Window]:
        """
        Call at each DC event or bar. Returns list of windows closed/started at t.
        Implements: open when p>=open_p for confirm_open events; close when
        p<=close_p for confirm_close events and min_trends satisfied.
        """
```

## `online/features.py`

Purpose: supply per-bar features for the tracker.

```python
class FeatureBuilder:
    def __init__(self, symbol: str, rv_fallback: bool = True):
        ...
    def on_bar(self, bar: dict) -> pd.Series:
        """
        bar: {'t', 'open','high','low','close','volume'}
        Returns Series with ['ret','rv_20d'] at t.
        Uses your existing rv_20d for realized-vol fallback when live DC is quiet.
        """
```

Your `rv_20d` definition and return name `rv_20d` align here. 

## `online/streaming.py`

Single façade for the rest of the codebase.

```python
class RegimeStreamingDetector:
    def __init__(
        self,
        hmm_model: GaussianHMM,
        scaler: StandardScaler,
        dc_theta_pct: float = 0.4,
        rule: WindowRule = WindowRule(),
        use_bayes: bool = False
    ):
        self.features = FeatureBuilder(symbol="EURUSD")
        self.dc = DCUpdater(theta_pct=dc_theta_pct)
        self.tracker = HMMTracker(hmm_model, scaler)  # or NaiveBayesTracker
        self.windows = WindowStateMachine(rule)

    def on_bar(self, bar: dict) -> list[Window]:
        """
        1) update DC with bar['close']; collect any completed DCEvents
        2) build features; get p(Regime 2) from tracker
        3) on DC event boundaries, call windows.on_prob to open/close
        Returns any windows that opened/closed at this bar.
        """
```

# Offline training and signals you already run

Keep your current notebook/script for fitting and PELT. It already shows:

* `pelt_changepoints` to extract macro-driven events from `rate_diff_2y` and `rv_20d`. Use these as context or filters. 
* `fit_2state_hmm` yielding `(model, scaler, hmm_df)` with `state`, `p_state0`, `p_state1`, and `regime` labels. Persist `model` and `scaler` for the online detector.  

Example glue stays almost identical to your existing script:

```python
# offline_fit.py
from regime_partitioning.processing import fit_2state_hmm
# df_fx must have 'ret' and 'rv_20d' (keep your pipeline)
model, scaler, hmm_df = fit_2state_hmm(df_fx, cols=("ret","rv_20d"))
joblib.dump((model, scaler), "artifacts/eurusd_hmm.pkl")
```

# Real-time usage (main interface)

You asked for “ingest OHLCV on bar, output windows.” That is the interface below.

```python
# live_loop.py
(hmm, scaler) = joblib.load("artifacts/eurusd_hmm.pkl")
det = RegimeStreamingDetector(hmm, scaler, dc_theta_pct=0.4, rule=WindowRule())

def on_new_bar(bar: dict):
    """
    bar = {'t': pd.Timestamp, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
    """
    windows_changed = det.on_bar(bar)
    for w in windows_changed:
        # publish window open/close to your strategy layer
        pass
```

Output contract:

* Open events: `Window(start=t, end=None, label='regime_2')`
* Close events: previous window returned with `end=t`

# Where the extra packages fit

* `ruptures`: already in use; keep as is for PELT on `rate_diff_2y` and `rv_20d`. 
* `arch`: optional builder for alt RV or EGARCH/GJR inputs to HMM; plug into `features.py` when `rv_20d` isn’t enough. No API duplication here since your `rv_20d` already exists. 
* `fredapi`: optional for offline data refresh feeding `rate_diff_2y`; you already standardize cleaned 2-year yield CSVs under `data/yields/clean`. 
* `pomegranate`: optional replacement for `hmmlearn` with friendlier stepwise posteriors; keep the same `HMMTracker.score_step` signature.

# Strategy integration

* Use the emitted windows as a binary filter. Example: only trade when the active window is `regime_2`.
* Or use `p₂ = p(Regime 2)` directly as a continuous bias.
* Combine with PELT macro events: require alignment with a recent PELT breakpoint in `rate_diff_2y` to arm the filter.

# Data expectations and alignment

* `rate_diff_2y` builder and combined DataFrame already exist. Keep explicit column naming for easy joins. 
* `rv_20d` name matches the HMM training column in your fitter.  

# Summary

* Keep your offline pipeline as is for PELT + HMM fit.
* Add the online layer above. It consumes bars, maintains DC, queries the trained HMM (or NB), and debounces windows.
* Output is a deterministic stream of window open/close events you can wire into your strategy runner.

If you want, I can draft concrete class stubs in those files next.

