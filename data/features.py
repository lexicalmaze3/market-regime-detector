import numpy as np
import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "spy_raw.csv")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "spy_features.csv")


def build_features():
    df = pd.read_csv(RAW_PATH, index_col="date", parse_dates=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    feat = pd.DataFrame(index=df.index)
    feat["returns"] = np.log(close / close.shift(1))
    feat["vol_5"] = feat["returns"].rolling(5).std()
    feat["vol_21"] = feat["returns"].rolling(21).std()
    feat["vol_63"] = feat["returns"].rolling(63).std()
    feat["trend_5"] = feat["returns"].rolling(5).mean()
    feat["trend_21"] = feat["returns"].rolling(21).mean()
    feat["range"] = (high - low) / close
    feat["vol_ratio"] = feat["vol_5"] / feat["vol_21"]
    feat["momentum_21"] = close / close.shift(21) - 1
    feat["momentum_63"] = close / close.shift(63) - 1

    feat.dropna(inplace=True)
    feat.to_csv(FEATURES_PATH)

    print(f"Shape:      {feat.shape}")
    print(f"Date range: {feat.index[0].date()} → {feat.index[-1].date()}")
    print("\nFeature stats:")
    print(feat.describe().loc[["mean", "std", "min", "max"]].to_string())
    print("\nCorrelation matrix:")
    print(feat.corr().to_string())


if __name__ == "__main__":
    build_features()
