import os
import pickle
import numpy as np
import pandas as pd

from models.gmm_model import RegimeGMM

FEATURES_PATH = "data/spy_features.csv"
RAW_PATH = "data/spy_raw.csv"
HMM_REGIMES_PATH = "data/spy_regimes_hmm.csv"
REGIMES_PATH = "data/spy_regimes_gmm.csv"
MODEL_PATH = "models/gmm_model.pkl"
SCALER_PATH = "models/hmm_scaler.pkl"

FEATURE_COLS = [
    "returns", "vol_5", "vol_21", "vol_63",
    "trend_5", "trend_21", "range", "vol_ratio", "momentum_63",
]


def label_regimes(stats: pd.DataFrame) -> dict[int, str]:
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


def compute_regime_stats(df: pd.DataFrame, n_regimes: int) -> tuple[pd.DataFrame, dict[int, str]]:
    total = len(df)
    rows = []
    for r in range(n_regimes):
        mask = df["regime"] == r
        sub = df[mask]
        rows.append({
            "regime": r,
            "mean_return": sub["returns"].mean(),
            "mean_vol": sub["vol_21"].mean(),
            "mean_momentum_63": sub["momentum_63"].mean(),
            "n_days": mask.sum(),
            "pct": mask.sum() / total * 100,
        })
    stats = pd.DataFrame(rows).set_index("regime")
    return stats, label_regimes(stats)


def print_regime_stats(stats: pd.DataFrame, regime_labels: dict[int, str]) -> None:
    print("\n=== Regime Summary ===")
    for r in stats.index:
        row = stats.loc[r]
        lbl = regime_labels[r]
        print(
            f"  Regime {r} [{lbl}]: {int(row['n_days'])} days ({row['pct']:.1f}%) | "
            f"ret={row['mean_return']:.5f}  vol21={row['mean_vol']:.5f}  mom63={row['mean_momentum_63']:.4f}"
        )


def print_comparison(hmm_df: pd.DataFrame, gmm_df: pd.DataFrame,
                     hmm_labels: dict[int, str], gmm_labels: dict[int, str],
                     n_regimes: int) -> None:
    def label_pcts(df: pd.DataFrame, labels: dict[int, str]) -> dict[str, float]:
        n = len(df)
        return {
            labels[r]: (df["regime"] == r).sum() / n * 100
            for r in range(n_regimes)
        }

    hmm_pcts = label_pcts(hmm_df, hmm_labels)
    gmm_pcts = label_pcts(gmm_df, gmm_labels)

    aligned = hmm_df[["regime"]].join(gmm_df[["regime"]], lsuffix="_hmm", rsuffix="_gmm", how="inner")
    hmm_named = aligned["regime_hmm"].map(hmm_labels)
    gmm_named = aligned["regime_gmm"].map(gmm_labels)
    agreement = (hmm_named == gmm_named).mean() * 100

    label_order = ["Bull Quiet", "Bear Volatile", "Choppy", "Bear Quiet"]
    col_w = 10

    print("\n=== Model Comparison ===")
    print(f"{'Metric':<28}| {'HMM':>{col_w}} | {'GMM':>{col_w}}")
    print("-" * (28 + col_w * 2 + 7))
    for lbl in label_order:
        hmm_val = hmm_pcts.get(lbl, 0.0)
        gmm_val = gmm_pcts.get(lbl, 0.0)
        print(f"  {lbl + ' days (%)':<26}| {hmm_val:>{col_w}.1f}% | {gmm_val:>{col_w}.1f}%")
    print(f"  {'Agreement rate (%)':<26}| {'—':>{col_w}} | {agreement:>{col_w}.1f}%")


def main():
    feat = pd.read_csv(FEATURES_PATH, index_col="date", parse_dates=True)
    raw = pd.read_csv(RAW_PATH, index_col="date", parse_dates=True)

    feat = feat[FEATURE_COLS]

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(feat.values)
    print(f"Scaler loaded ← {SCALER_PATH}")

    model = RegimeGMM(n_regimes=4)
    model.fit(X)
    model.save(MODEL_PATH)
    print(f"Model saved   → {MODEL_PATH}")

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

    gmm_stats, gmm_labels = compute_regime_stats(out, n_regimes=4)
    print_regime_stats(gmm_stats, gmm_labels)

    hmm_df = pd.read_csv(HMM_REGIMES_PATH, index_col="date", parse_dates=True)
    hmm_stats, hmm_labels = compute_regime_stats(hmm_df, n_regimes=4)
    print_comparison(hmm_df, out, hmm_labels, gmm_labels, n_regimes=4)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
