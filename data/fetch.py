import yfinance as yf
from datetime import date
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "spy_raw.csv")


def fetch_spy():
    today = date.today().isoformat()
    df = yf.download("SPY", start="2000-01-01", end=today, auto_adjust=True, progress=False)

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.index.name = "date"

    df.to_csv(RAW_PATH)

    print(f"Shape:      {df.shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())


if __name__ == "__main__":
    fetch_spy()
