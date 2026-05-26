import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.hmm_model import RegimeHMM

FEATURES_PATH = "data/spy_features.csv"
RAW_PATH = "data/spy_raw.csv"
REGIMES_PATH = "data/spy_regimes_hmm.csv"
MODEL_PATH = "models/hmm_model.pkl"
SCALER_PATH = "models/hmm_scaler.pkl"

FEATURE_COLS = [
    "returns", "vol_5", "vol_21", "vol_63",
    "trend_5", "trend_21", "range", "vol_ratio", "momentum_63",
]


def label_regimes(stats: pd.DataFrame) -> dict[int, str]:
    """
    Assign a human-readable label to each regime based on return and vol_21.
    Stats index is regime id; columns include mean_return and mean_vol.
    """
    labels = {}
    remaining = set(stats.index)

    bull_quiet = stats.loc[list(remaining), "mean_return"].idxmax()
    labels[bull_quiet] = "Bull Quiet"
    remaining.discard(bull_quiet)

    bear_volatile = stats.loc[list(remaining), "mean_vol"].idxmax()
    labels[bear_volatile] = "Bear Volatile"
    remaining.discard(bear_volatile)

    low_ret = stats.loc[list(remaining), "mean_return"].idxmin()
    labels[low_ret] = "Bear Quiet"
    remaining.discard(low_ret)

    labels[remaining.pop()] = "Choppy"

    return labels


def print_regime_stats(df: pd.DataFrame, n_regimes: int) -> None:
    total = len(df)
    stats_rows = []
    for r in range(n_regimes):
        mask = df["regime"] == r
        sub = df[mask]
        stats_rows.append({
            "regime": r,
            "mean_return": sub["returns"].mean(),
            "mean_vol": sub["vol_21"].mean(),
            "mean_momentum_63": sub["momentum_63"].mean(),
            "n_days": mask.sum(),
        })
    stats = pd.DataFrame(stats_rows).set_index("regime")

    regime_labels = label_regimes(stats)

    print("\n=== Regime Summary ===")
    for r in range(n_regimes):
        row = stats.loc[r]
        pct = row["n_days"] / total * 100
        lbl = regime_labels[r]
        print(
            f"  Regime {r} [{lbl}]: {int(row['n_days'])} days ({pct:.1f}%) | "
            f"ret={row['mean_return']:.5f}  vol21={row['mean_vol']:.5f}  mom63={row['mean_momentum_63']:.4f}"
        )


def print_transition_matrix(model: RegimeHMM, n_regimes: int) -> None:
    T = model._model.transmat_
    header = "        " + "  ".join(f"→R{j}" for j in range(n_regimes))
    print(f"\n=== Transition Matrix ===\n{header}")
    for i in range(n_regimes):
        row = "  ".join(f"{T[i, j]:.4f}" for j in range(n_regimes))
        print(f"  R{i}  |  {row}")


def main():
    feat = pd.read_csv(FEATURES_PATH, index_col="date", parse_dates=True)
    raw = pd.read_csv(RAW_PATH, index_col="date", parse_dates=True)

    feat = feat[FEATURE_COLS]

    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {SCALER_PATH}")

    model = RegimeHMM(n_regimes=4)
    model.fit(X)
    model.save(MODEL_PATH)
    print(f"Model saved  → {MODEL_PATH}")

    regimes = model.predict(X)
    proba = model.predict_proba(X)
    regime_prob = proba.max(axis=1)

    close = raw["close"].reindex(feat.index)

    out = pd.DataFrame({
        "date": feat.index,
        "close": close.values,
        "regime": regimes,
        "regime_prob": regime_prob,
    }).set_index("date")

    out["returns"] = feat["returns"].values
    out["vol_21"] = feat["vol_21"].values
    out["momentum_63"] = feat["momentum_63"].values

    out.to_csv(REGIMES_PATH)
    print(f"Regimes saved → {REGIMES_PATH}")

    print_regime_stats(out, n_regimes=4)
    print_transition_matrix(model, n_regimes=4)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
