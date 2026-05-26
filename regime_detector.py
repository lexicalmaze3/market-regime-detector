"""
regime_detector.py — standalone market regime classifier.

Usage:
    from regime_detector import get_current_regime
    regime, prob = get_current_regime()
"""

import pickle
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── paths relative to this file so imports work from any cwd ─────────────────
_ROOT = Path(__file__).parent
_SCALER_PATH = _ROOT / "models" / "hmm_scaler.pkl"
_MODEL_PATH  = _ROOT / "models" / "hmm_model.pkl"
_HISTORY_CSV = _ROOT / "data"   / "spy_regimes_hmm.csv"

FEATURE_COLS = [
    "returns", "vol_5", "vol_21", "vol_63",
    "trend_5", "trend_21", "range", "vol_ratio", "momentum_63",
]

RECOMMENDATIONS = {
    "Bull Quiet":    {"position_size": 1.0,  "trade": True,  "note": "Optimal conditions"},
    "Bear Quiet":    {"position_size": 0.5,  "trade": True,  "note": "Reduce size, trend weak"},
    "Choppy":        {"position_size": 0.25, "trade": False, "note": "Avoid, high whipsaw risk"},
    "Bear Volatile": {"position_size": 0.0,  "trade": False, "note": "Go to cash, crisis conditions"},
}

# ── label map derived once from training history ──────────────────────────────

def _build_label_map() -> dict[int, str]:
    """Derive regime-id → label mapping from the saved training history."""
    df = pd.read_csv(_HISTORY_CSV, index_col="date", parse_dates=True)
    rows = []
    for r in df["regime"].unique():
        mask = df["regime"] == r
        rows.append({
            "regime":      r,
            "mean_return": df.loc[mask, "returns"].mean(),
            "mean_vol":    df.loc[mask, "vol_21"].mean(),
        })
    stats = pd.DataFrame(rows).set_index("regime")

    labels: dict[int, str] = {}
    remaining = set(stats.index)

    bq = stats.loc[list(remaining), "mean_return"].idxmax()
    labels[bq] = "Bull Quiet"; remaining.discard(bq)

    bv = stats.loc[list(remaining), "mean_vol"].idxmax()
    labels[bv] = "Bear Volatile"; remaining.discard(bv)

    bqt = stats.loc[list(remaining), "mean_return"].idxmin()
    labels[bqt] = "Bear Quiet"; remaining.discard(bqt)

    labels[remaining.pop()] = "Choppy"
    return labels


_LABEL_MAP: dict[int, str] | None = None


def _label_map() -> dict[int, str]:
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _LABEL_MAP = _build_label_map()
    return _LABEL_MAP


# ── lazy model/scaler loader ──────────────────────────────────────────────────

_scaler = None
_model  = None


def _load_artifacts():
    global _scaler, _model
    if _scaler is None:
        with open(_SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)


# ── feature engineering (mirrors data/features.py) ───────────────────────────

def _engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with lowercase columns: close, high, low.
    Returns a DataFrame of FEATURE_COLS with NaN rows dropped.
    """
    close = raw["close"]
    high  = raw["high"]
    low   = raw["low"]

    feat = pd.DataFrame(index=raw.index)
    feat["returns"]     = np.log(close / close.shift(1))
    feat["vol_5"]       = feat["returns"].rolling(5).std()
    feat["vol_21"]      = feat["returns"].rolling(21).std()
    feat["vol_63"]      = feat["returns"].rolling(63).std()
    feat["trend_5"]     = feat["returns"].rolling(5).mean()
    feat["trend_21"]    = feat["returns"].rolling(21).mean()
    feat["range"]       = (high - low) / close
    feat["vol_ratio"]   = feat["vol_5"] / feat["vol_21"]
    feat["momentum_21"] = close / close.shift(21) - 1
    feat["momentum_63"] = close / close.shift(63) - 1

    feat.dropna(inplace=True)
    return feat[FEATURE_COLS]


def _download_spy(start: str, end: str) -> pd.DataFrame:
    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw.index.name = "date"
    return raw


# ── public API ────────────────────────────────────────────────────────────────

def get_current_regime(lookback_days: int = 120) -> tuple[str, float]:
    """
    Download the last lookback_days trading days of SPY, engineer features,
    and return (regime_label, confidence) for the most recent day.
    """
    _load_artifacts()

    # download enough calendar days to guarantee lookback_days trading rows
    # after feature engineering (63-day rolling window eats the first rows)
    buffer_days = (lookback_days + 90) * 2
    start = (date.today() - timedelta(days=buffer_days)).isoformat()
    end   = date.today().isoformat()

    raw  = _download_spy(start, end)
    feat = _engineer_features(raw)

    # keep only the last lookback_days rows
    feat = feat.iloc[-lookback_days:]

    X       = _scaler.transform(feat.values)
    proba   = _model.predict_proba(X)
    regimes = _model.predict(X)

    last_regime = int(regimes[-1])
    last_conf   = float(proba[-1].max())
    label       = _label_map()[last_regime]

    return label, last_conf


def get_regime_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Return a DataFrame with columns [regime, confidence] for the given date range.
    Downloads extra data before start_date so rolling features are valid.
    """
    _load_artifacts()

    buffer_start = (
        pd.Timestamp(start_date) - pd.offsets.BDay(90)
    ).strftime("%Y-%m-%d")

    raw  = _download_spy(buffer_start, end_date)
    feat = _engineer_features(raw)
    feat = feat.loc[start_date:end_date]

    X       = _scaler.transform(feat.values)
    proba   = _model.predict_proba(X)
    regimes = _model.predict(X)

    lmap = _label_map()
    return pd.DataFrame({
        "date":       feat.index,
        "regime":     [lmap[int(r)] for r in regimes],
        "confidence": proba.max(axis=1),
    }).set_index("date")


def get_regime_recommendation(regime_label: str) -> dict:
    """Return trading recommendations for the given regime label."""
    return dict(RECOMMENDATIONS[regime_label])


def print_regime_report() -> None:
    """Print a one-page summary: current regime, recommendation, recent transitions."""
    label, conf = get_current_regime()
    rec = get_regime_recommendation(label)

    today = date.today().isoformat()

    print("=" * 52)
    print("  MARKET REGIME REPORT")
    print(f"  Date:    {today}")
    print(f"  Regime:  {label}")
    print(f"  Conf:    {conf:.1%}")
    print("-" * 52)
    print("  RECOMMENDATION")
    print(f"  Trade:         {'Yes' if rec['trade'] else 'No'}")
    print(f"  Position size: {rec['position_size']:.0%}")
    print(f"  Note:          {rec['note']}")
    print("-" * 52)
    print("  LAST 10 REGIME CHANGES")

    history = pd.read_csv(_HISTORY_CSV, index_col="date", parse_dates=True)
    lmap = _label_map()
    history["label"] = history["regime"].map(lmap)

    # find transition dates: rows where label differs from the previous row
    transitions = history["label"][history["label"] != history["label"].shift(1)]
    last_10 = transitions.tail(10)

    for dt, lbl in last_10.items():
        print(f"  {dt.strftime('%Y-%m-%d')}  →  {lbl}")

    print("=" * 52)


if __name__ == "__main__":
    print_regime_report()
