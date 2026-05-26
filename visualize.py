import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
HMM_CSV = "data/spy_regimes_hmm.csv"
GMM_CSV = "data/spy_regimes_gmm.csv"
HMM_PKL = "models/hmm_model.pkl"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── shared style ─────────────────────────────────────────────────────────────
LABEL_ORDER = ["Bull Quiet", "Bear Volatile", "Bear Quiet", "Choppy"]
COLORS = {
    "Bull Quiet":    "#2ecc71",
    "Bear Volatile": "#e74c3c",
    "Bear Quiet":    "#e67e22",
    "Choppy":        "#3498db",
}
EVENTS = {
    "Dot-com\nbottom":  "2002-10-09",
    "GFC\nbottom":      "2009-03-09",
    "COVID\ncrash":     "2020-03-23",
    "2022\nbear":       "2022-10-12",
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         120,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def label_regimes(df: pd.DataFrame) -> dict[int, str]:
    total = len(df)
    rows = []
    for r in df["regime"].unique():
        mask = df["regime"] == r
        rows.append({
            "regime": r,
            "mean_return": df.loc[mask, "returns"].mean(),
            "mean_vol":    df.loc[mask, "vol_21"].mean(),
            "n_days":      mask.sum(),
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


def add_regime_bands(ax, df: pd.DataFrame, regime_col: str,
                     label_map: dict[int, str], alpha: float = 0.25) -> None:
    """Shade background of ax by regime."""
    dates = df.index
    regimes = df[regime_col].values
    i = 0
    while i < len(dates):
        j = i + 1
        while j < len(dates) and regimes[j] == regimes[i]:
            j += 1
        lbl = label_map[regimes[i]]
        ax.axvspan(dates[i], dates[j - 1], color=COLORS[lbl], alpha=alpha, linewidth=0)
        i = j


def add_events(ax, ypos: float = 0.97) -> None:
    for name, date in EVENTS.items():
        ax.axvline(pd.Timestamp(date), color="#555555", linewidth=0.8,
                   linestyle="--", zorder=5)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * ypos, name,
                fontsize=6, ha="center", va="top", color="#555555",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def regime_legend(label_map: dict[int, str]) -> list[mpatches.Patch]:
    seen: set[str] = set()
    patches = []
    for lbl in LABEL_ORDER:
        if lbl not in seen:
            patches.append(mpatches.Patch(color=COLORS[lbl], label=lbl))
            seen.add(lbl)
    return patches


# ── load data ─────────────────────────────────────────────────────────────────

hmm = pd.read_csv(HMM_CSV, index_col="date", parse_dates=True)
gmm = pd.read_csv(GMM_CSV, index_col="date", parse_dates=True)

hmm_labels = label_regimes(hmm)
gmm_labels = label_regimes(gmm)

hmm["label"] = hmm["regime"].map(hmm_labels)
gmm["label"] = gmm["regime"].map(gmm_labels)

with open(HMM_PKL, "rb") as f:
    hmm_model = pickle.load(f)


# ── Plot 1: regime_overlay.png ────────────────────────────────────────────────

def plot_regime_overlay():
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [4, 1]}, sharex=True
    )
    fig.subplots_adjust(hspace=0.04)

    # price panel
    ax1.plot(hmm.index, hmm["close"], color="#1a1a2e", linewidth=0.8, zorder=3)
    add_regime_bands(ax1, hmm, "regime", hmm_labels, alpha=0.3)
    ax1.set_ylabel("SPY Close Price ($)", fontsize=10)
    ax1.set_title("SPY Price History with HMM Market Regimes (2000–2026)", fontsize=13, pad=10)
    ax1.spines["bottom"].set_visible(False)
    ax1.yaxis.set_minor_locator(plt.NullLocator())
    ax1.set_axisbelow(False)

    ax1.legend(handles=regime_legend(hmm_labels), loc="upper left",
               fontsize=8, framealpha=0.85, edgecolor="none")

    # after limits are set, add event lines
    ax1.set_xlim(hmm.index[0], hmm.index[-1])
    for name, date in EVENTS.items():
        ts = pd.Timestamp(date)
        ax1.axvline(ts, color="#555555", linewidth=0.8, linestyle="--", zorder=5)
        ax1.text(ts, ax1.get_ylim()[1] * 0.97, name, fontsize=6,
                 ha="center", va="top", color="#555555",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # regime band strip
    dates = hmm.index
    regimes = hmm["regime"].values
    i = 0
    while i < len(dates):
        j = i + 1
        while j < len(dates) and regimes[j] == regimes[i]:
            j += 1
        lbl = hmm_labels[regimes[i]]
        ax2.axvspan(dates[i], dates[j - 1], color=COLORS[lbl], alpha=0.9, linewidth=0)
        i = j

    ax2.set_yticks([])
    ax2.set_ylabel("Regime", fontsize=9)
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

    path = os.path.join(PLOTS_DIR, "regime_overlay.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 2: regime_stats.png ──────────────────────────────────────────────────

def plot_regime_stats():
    total = len(hmm)
    stats = []
    for r in range(4):
        mask = hmm["regime"] == r
        sub = hmm[mask]
        lbl = hmm_labels[r]
        stats.append({
            "label":        lbl,
            "mean_return":  sub["returns"].mean() * 100,
            "mean_vol":     sub["vol_21"].mean() * 100,
            "mean_mom63":   sub["momentum_63"].mean() * 100,
            "pct":          mask.sum() / total * 100,
        })
    df_s = pd.DataFrame(stats).set_index("label").reindex(LABEL_ORDER)
    pal = [COLORS[l] for l in LABEL_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Regime Characteristics — HMM", fontsize=14, y=1.01)

    def bar(ax, col, title, ylabel):
        bars = ax.bar(df_s.index, df_s[col], color=pal, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticklabels(df_s.index, rotation=15, ha="right", fontsize=9)
        ax.axhline(0, color="#aaaaaa", linewidth=0.6)
        ax.spines["bottom"].set_visible(False)
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (0.003 if v >= 0 else -0.006),
                    f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    bar(axes[0, 0], "mean_return",  "Mean Daily Return (%)", "Return (%)")
    bar(axes[0, 1], "mean_vol",     "Mean Vol-21 (%)",       "Volatility (%)")
    bar(axes[1, 0], "mean_mom63",   "Mean Momentum-63 (%)",  "Momentum (%)")

    # pie
    ax_pie = axes[1, 1]
    wedges, texts, autotexts = ax_pie.pie(
        df_s["pct"], labels=df_s.index,
        colors=pal, autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.2),
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax_pie.set_title("Regime Distribution", fontsize=11)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "regime_stats.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 3: transition_heatmap.png ────────────────────────────────────────────

def plot_transition_heatmap():
    T = hmm_model._model.transmat_
    tick_labels = [hmm_labels[i] for i in range(4)]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        T, annot=True, fmt=".4f", cmap="YlOrRd",
        xticklabels=tick_labels, yticklabels=tick_labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Transition Probability"},
        ax=ax,
    )
    ax.set_title("Regime Transition Probabilities", fontsize=13, pad=12)
    ax.set_xlabel("To Regime", fontsize=10)
    ax.set_ylabel("From Regime", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    path = os.path.join(PLOTS_DIR, "transition_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 4: hmm_vs_gmm.png ────────────────────────────────────────────────────

def plot_hmm_vs_gmm():
    start = "2018-01-01"
    hmm_z = hmm.loc[start:]
    gmm_z = gmm.loc[start:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    fig.suptitle("HMM vs GMM Regime Detection — SPY 2018–2026", fontsize=13, y=1.02)

    for ax, df, lmap, title in [
        (ax1, hmm_z, hmm_labels, "HMM"),
        (ax2, gmm_z, gmm_labels, "GMM"),
    ]:
        ax.plot(df.index, df["close"], color="#1a1a2e", linewidth=0.9, zorder=3)
        add_regime_bands(ax, df, "regime", lmap, alpha=0.35)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Date", fontsize=9)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

    ax1.set_ylabel("SPY Close Price ($)", fontsize=10)
    ax1.legend(handles=regime_legend(hmm_labels), loc="upper left",
               fontsize=8, framealpha=0.85, edgecolor="none")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "hmm_vs_gmm.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 5: regime_returns_dist.png ──────────────────────────────────────────

def plot_returns_dist():
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle("Daily Return Distributions by HMM Regime", fontsize=13, y=1.02)

    for ax, lbl in zip(axes, LABEL_ORDER):
        subset = hmm[hmm["label"] == lbl]["returns"] * 100
        color = COLORS[lbl]

        sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.35,
                    linewidth=1.8, cut=3)

        ax.axvline(subset.mean(), color=color, linewidth=1.2,
                   linestyle="--", label=f"μ={subset.mean():.3f}%")
        ax.axvline(0, color="#aaaaaa", linewidth=0.7, linestyle=":")

        ax.set_title(lbl, fontsize=11, color=color)
        ax.set_xlabel("Daily Return (%)", fontsize=9)
        ax.set_ylabel("Density" if ax == axes[0] else "", fontsize=9)
        ax.legend(fontsize=8, framealpha=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        n = len(subset)
        std = subset.std()
        ax.text(0.97, 0.95, f"n={n}\nσ={std:.3f}%",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                color="#555555")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "regime_returns_dist.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_regime_overlay()
    plot_regime_stats()
    plot_transition_heatmap()
    plot_hmm_vs_gmm()
    plot_returns_dist()
    print(f"\nAll plots saved to {PLOTS_DIR}/")
