"""
EMA Crossover Strategy Visualiser
===================================
Reads three CSVs written by ema_crossover_strategy.py — no recalculation,
no parameters to re-enter.

  ema_series.csv          : bar-level OHLCV + ema_short + ema_long + spread
  trade_book.csv          : one row per completed trade
  performance_metrics.csv : single-row metrics summary

6-panel dark dashboard:
  [1] Price + EMA lines + long/short entry & exit markers  (full width)
  [2] Equity curve (capital_return weighted) vs buy-and-hold
  [3] Drawdown curve (capital_return weighted)
  [4] Monthly return bars (capital_return weighted)
  [5] Trade return histogram (capital_return weighted)
  [6] EMA spread

Usage
-----
  python ema_visualizer.py
  python ema_visualizer.py ema_series.csv trade_book.csv performance_metrics.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = "#0d0f14"
SURFACE     = "#12151c"
SURFACE2    = "#1a1f2b"
BORDER      = "#252c3a"
TEXT_PRI    = "#e4e8f0"
TEXT_SEC    = "#7a8494"
TEXT_DIM    = "#454e60"
C_SHORT_EMA = "#6b9fff"   # EMA-short line
C_LONG_EMA  = "#ffb347"   # EMA-long line
C_LONG_ENT  = "#3dd68c"   # long entry marker  (green up triangle)
C_LONG_EXT  = "#3dd68c"   # long exit marker   (green down triangle, dimmer)
C_SHORT_ENT = "#ff5c6e"   # short entry marker (red down triangle)
C_SHORT_EXT = "#ff5c6e"   # short exit marker  (red up triangle, dimmer)
C_BULL      = "#3dd68c"
C_BEAR      = "#ff5c6e"
C_SPREAD    = "#b08fff"
C_EQ        = "#6b9fff"
C_BH        = "#454e60"
C_DD        = "#ff5c6e"
C_WIN       = "#3dd68c"
C_LOSE      = "#ff5c6e"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str)
    )
    return df.sort_values("_datetime").reset_index(drop=True)


def load_book(path: str) -> pd.DataFrame:
    book = pd.read_csv(path)
    book["_enter_dt"] = pd.to_datetime(
        book["enter_date"].astype(str) + " " + book["enter_time"].astype(str)
    )
    book["_exit_dt"] = pd.to_datetime(
        book["exit_date"].astype(str) + " " + book["exit_time"].astype(str)
    )
    return book


def load_metrics(path: str) -> dict:
    return pd.read_csv(path).iloc[0].to_dict()


# ── Style helpers ─────────────────────────────────────────────────────────────

def style(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_SEC, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    if title:
        ax.set_title(title, color=TEXT_PRI, fontsize=10.5,
                     fontweight="bold", loc="left", pad=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=8.5)


def fmt_xdate(ax, interval=3):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=7.5, color=TEXT_SEC)


def mlegend(ax):
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.12,
              labelcolor=TEXT_PRI, facecolor=SURFACE2, edgecolor=BORDER)


# ── Panel 1: Price ────────────────────────────────────────────────────────────

def panel_price(ax, series: pd.DataFrame, book: pd.DataFrame,
                short_w: int, long_w: int):
    style(ax, title=f"Price  ·  EMA-{short_w} / EMA-{long_w}  ·  Long & Short signals",
          ylabel="Price ($)")

    # Candlesticks — last 600 bars
    candles = series.tail(600).reset_index(drop=True)
    dt_num  = mdates.date2num(candles["_datetime"].dt.to_pydatetime())
    w       = max((dt_num[-1] - dt_num[0]) / len(dt_num) * 0.55, 5e-5)
    for d, row in zip(dt_num, candles.itertuples()):
        o, h, l, c = row.open, row.high, row.low, row.close
        col = C_BULL if c >= o else C_BEAR
        ax.plot([d, d], [l, h], color=col, lw=0.45, alpha=0.7)
        ax.add_patch(mpatches.FancyBboxPatch(
            (d - w / 2, min(o, c)), w, max(abs(c - o), 1e-4),
            boxstyle="square,pad=0", lw=0, fc=col, alpha=0.88))

    # EMA lines
    dt = series["_datetime"]
    ax.plot(dt, series["ema_short"], color=C_SHORT_EMA, lw=1.25,
            label=f"EMA-{short_w}", alpha=0.92, zorder=3)
    ax.plot(dt, series["ema_long"],  color=C_LONG_EMA,  lw=1.25,
            label=f"EMA-{long_w}",  alpha=0.92, zorder=3)

    if book is not None and len(book):
        longs  = book[book["signal"] == "LONG_EMA_CROSS"]
        shorts = book[book["signal"] == "SHORT_EMA_CROSS"]

        # Vertical reference lines (faint) for all trades
        for _, t in book.iterrows():
            color = C_LONG_ENT if t["signal"] == "LONG_EMA_CROSS" else C_SHORT_ENT
            ax.axvline(t["_enter_dt"], color=color, lw=0.35, alpha=0.15, zorder=1)
            ax.axvline(t["_exit_dt"],  color=color, lw=0.35, alpha=0.10, zorder=1)

        # Long entries: green up triangle at enter_price
        if len(longs):
            ax.scatter(longs["_enter_dt"], longs["enter_price"],
                       marker="^", s=50, color=C_LONG_ENT,
                       zorder=5, label="Long entry")
            # Long exits: green down triangle at exit_price (dimmer)
            ax.scatter(longs["_exit_dt"], longs["exit_price"],
                       marker="v", s=35, color=C_LONG_EXT,
                       alpha=0.55, zorder=5, label="Long exit")

        # Short entries: red down triangle at enter_price
        if len(shorts):
            ax.scatter(shorts["_enter_dt"], shorts["enter_price"],
                       marker="v", s=50, color=C_SHORT_ENT,
                       zorder=5, label="Short entry")
            # Short exits: red up triangle at exit_price (dimmer)
            ax.scatter(shorts["_exit_dt"], shorts["exit_price"],
                       marker="^", s=35, color=C_SHORT_EXT,
                       alpha=0.55, zorder=5, label="Short exit")

    fmt_xdate(ax)
    mlegend(ax)


# ── Panel 2: Equity curve ─────────────────────────────────────────────────────

def panel_equity(ax, book: pd.DataFrame, series: pd.DataFrame):
    style(ax, title="Equity curve  vs  buy & hold  (capital-weighted)", ylabel="Growth of $1")

    # Capital-weighted equity — uses capital_return column
    equity = (1 + book["capital_return"]).cumprod()
    x      = np.arange(len(equity))
    ax.plot(x, equity.values, color=C_EQ, lw=1.5, label="Strategy", zorder=3)

    # Buy-and-hold from first entry to last exit
    start_dt  = book["_enter_dt"].iloc[0]
    bh_prices = series.loc[series["_datetime"] >= start_dt, "close"].reset_index(drop=True)
    if len(bh_prices):
        bh     = bh_prices / bh_prices.iloc[0]
        bh_idx = np.linspace(0, len(bh) - 1, len(equity)).astype(int)
        ax.plot(x, bh.iloc[bh_idx].values, color=C_BH, lw=1.1,
                linestyle="--", label="Buy & hold", zorder=2, alpha=0.7)

    ax.axhline(1, color=BORDER, lw=0.8)
    ax.fill_between(x, equity.values, 1,
                    where=equity.values >= 1, color=C_WIN,  alpha=0.08)
    ax.fill_between(x, equity.values, 1,
                    where=equity.values <  1, color=C_LOSE, alpha=0.12)

    step = max(1, len(x) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(i + 1) for i in x[::step]], fontsize=7, color=TEXT_SEC)
    ax.set_xlabel("Trade #", color=TEXT_SEC, fontsize=8.5)
    mlegend(ax)


# ── Panel 3: Drawdown ─────────────────────────────────────────────────────────

def panel_drawdown(ax, book: pd.DataFrame):
    style(ax, title="Drawdown curve  (capital-weighted)", ylabel="Drawdown (%)")

    # Capital-weighted drawdown
    equity = (1 + book["capital_return"]).cumprod()
    dd_pct = ((equity - equity.cummax()) / equity.cummax()).values * 100
    x      = np.arange(len(dd_pct))

    ax.fill_between(x, dd_pct, 0, color=C_DD, alpha=0.35)
    ax.plot(x, dd_pct, color=C_DD, lw=1.0, alpha=0.9)
    ax.axhline(0, color=BORDER, lw=0.8)

    max_dd = dd_pct.min()
    ax.axhline(max_dd, color=C_DD, lw=0.7, linestyle=":", alpha=0.6)
    ax.text(len(x) * 0.01, max_dd - 0.3,
            f"Max DD: {max_dd:.2f}%", color=C_DD, fontsize=7.5, va="top")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    step = max(1, len(x) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(i + 1) for i in x[::step]], fontsize=7, color=TEXT_SEC)
    ax.set_xlabel("Trade #", color=TEXT_SEC, fontsize=8.5)


# ── Panel 4: Monthly returns ──────────────────────────────────────────────────

def panel_monthly(ax, book: pd.DataFrame):
    style(ax, title="Monthly capital return  (summed)", ylabel="Capital Return (%)")

    # Sum capital_return per month — reflects actual portfolio impact
    monthly = (book.groupby(["year", "month"])["capital_return"]
               .sum().reset_index())
    monthly["label"] = monthly.apply(
        lambda r: f"{int(r.year)}-{int(r.month):02d}", axis=1)
    monthly["pct"] = monthly["capital_return"] * 100

    x    = np.arange(len(monthly))
    cols = [C_WIN if v >= 0 else C_LOSE for v in monthly["pct"]]
    ax.bar(x, monthly["pct"], color=cols, alpha=0.85, width=0.7, zorder=2)
    ax.axhline(0, color=BORDER, lw=0.8)

    step = max(1, len(x) // 28)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(monthly["label"].iloc[::step].tolist(),
                       rotation=45, ha="right", fontsize=6.5, color=TEXT_SEC)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))


# ── Panel 5: Histogram ────────────────────────────────────────────────────────

def panel_histogram(ax, book: pd.DataFrame):
    style(ax, title="Capital return distribution  (per trade)",
          ylabel="Frequency", xlabel="Capital Return (%)")

    # Use capital_return — shows actual portfolio impact of each trade
    pnl  = book["capital_return"].values * 100
    bins = min(40, max(10, len(pnl) // 3))
    n, edges, patches = ax.hist(pnl, bins=bins, edgecolor=SURFACE, lw=0.4)
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(C_WIN if left >= 0 else C_LOSE)
        patch.set_alpha(0.82)

    ax.axvline(0,              color=BORDER,      lw=1.0)
    ax.axvline(pnl.mean(),     color=C_LONG_EMA,  lw=1.2, linestyle="--",
               label=f"Mean {pnl.mean():.2f}%")
    ax.axvline(np.median(pnl), color=C_SHORT_EMA, lw=1.0, linestyle=":",
               label=f"Median {np.median(pnl):.2f}%")
    mlegend(ax)


# ── Panel 6: EMA spread ───────────────────────────────────────────────────────

def panel_spread(ax, series: pd.DataFrame, short_w: int, long_w: int):
    style(ax, title=f"EMA spread  (EMA-{short_w} − EMA-{long_w})", ylabel="Spread")

    dt = series["_datetime"]
    ax.axhline(0, color=BORDER, lw=1.0, zorder=1)
    ax.fill_between(dt, series["spread"], 0,
                    where=series["spread"] > 0, color=C_WIN,  alpha=0.18, label="Bullish")
    ax.fill_between(dt, series["spread"], 0,
                    where=series["spread"] < 0, color=C_LOSE, alpha=0.18, label="Bearish")
    ax.plot(dt, series["spread"], color=C_SPREAD, lw=0.75, alpha=0.85)
    fmt_xdate(ax)
    mlegend(ax)


# ── Metrics footer ────────────────────────────────────────────────────────────

def draw_metrics_box(fig, m: dict, book: pd.DataFrame):
    if not m:
        return

    def fmt(k, fmt_str):
        v = m.get(k)
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
               else format(v, fmt_str)

    n_long  = int((book["signal"] == "LONG_EMA_CROSS").sum())  if book is not None else 0
    n_short = int((book["signal"] == "SHORT_EMA_CROSS").sum()) if book is not None else 0

    lines = [
        f"Trades: {int(m.get('num_trades', 0))} ({n_long}L / {n_short}S)",
        f"Total return: {fmt('total_return_pct', '.2f')}%",
        f"Ann. return: {fmt('annualized_return_pct', '.2f')}%",
        f"Sharpe: {fmt('sharpe_ratio', '.2f')}",
        f"Max DD: {fmt('max_drawdown_pct', '.2f')}%",
        f"Win rate: {fmt('win_rate_pct', '.1f')}%",
        f"Avg profit: ${fmt('avg_profit_per_trade', '.4f')}",
        f"Avg hold: {fmt('avg_holding_minutes', '.0f')} min",
        f"Avg size: {fmt('avg_position_size_pct', '.1f')}%",
        f"Exposure: {fmt('exposure_pct', '.1f')}%",
    ]
    fig.text(0.5, 0.003, "   ".join(lines), ha="center", va="bottom",
             color=TEXT_SEC, fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE2,
                       ec=BORDER, lw=0.6, alpha=0.9))


# ── Main ──────────────────────────────────────────────────────────────────────

def visualise(series_path="ema_series.csv",
              book_path="trade_book.csv",
              metrics_path="performance_metrics.csv",
              save_path=None):

    print(f"Reading {series_path} ...")
    series = load_series(series_path)

    book = None
    if Path(book_path).exists():
        book = load_book(book_path)
        n_long  = (book["signal"] == "LONG_EMA_CROSS").sum()
        n_short = (book["signal"] == "SHORT_EMA_CROSS").sum()
        print(f"Reading {book_path}  ({len(book)} trades: {n_long} long, {n_short} short)")
    else:
        print(f"No trade book found at {book_path} — price chart only.")

    metrics = {}
    if Path(metrics_path).exists():
        metrics = load_metrics(metrics_path)
        print(f"Reading {metrics_path}")
    else:
        print(f"No metrics file found at {metrics_path}.")

    short_w = int(metrics.get("short_window", 20))
    long_w  = int(metrics.get("long_window",  50))

    # ── Figure — fit to screen ────────────────────────────────────────────────
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        screen_w_px = root.winfo_screenwidth()
        screen_h_px = root.winfo_screenheight()
        root.destroy()
        dpi   = plt.rcParams["figure.dpi"]
        fig_w = min(screen_w_px * 0.92 / dpi, 18)
        fig_h = min(screen_h_px * 0.88 / dpi, 10)
    except Exception:
        fig_w, fig_h = 16, 9

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    fig.suptitle(
        f"AAPL  ·  EMA Crossover (Long + Short)  ·  EMA({short_w}) / EMA({long_w})",
        color=TEXT_PRI, fontsize=12, fontweight="bold", y=0.993,
    )

    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[2.8, 1.6, 1.6],
                  hspace=0.62, wspace=0.32,
                  left=0.06, right=0.975, top=0.962, bottom=0.07)

    ax_price = fig.add_subplot(gs[0, :])
    ax_eq    = fig.add_subplot(gs[1, :2])
    ax_dd    = fig.add_subplot(gs[1, 2])
    ax_mon   = fig.add_subplot(gs[2, 0])
    ax_hist  = fig.add_subplot(gs[2, 1])
    ax_sprd  = fig.add_subplot(gs[2, 2])

    panel_price(ax_price, series, book, short_w, long_w)

    if book is not None and len(book):
        panel_equity(ax_eq,      book, series)
        panel_drawdown(ax_dd,    book)
        panel_monthly(ax_mon,    book)
        panel_histogram(ax_hist, book)
    else:
        for ax in [ax_eq, ax_dd, ax_mon, ax_hist]:
            style(ax)
            ax.text(0.5, 0.5, "No trade data", transform=ax.transAxes,
                    color=TEXT_DIM, ha="center", va="center", fontsize=9)

    panel_spread(ax_sprd, series, short_w, long_w)
    draw_metrics_box(fig, metrics, book)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {save_path}")
    else:
        plt.tight_layout(rect=[0, 0.04, 1, 0.99])
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    visualise(
        series_path  = sys.argv[1] if len(sys.argv) > 1 else "ema_series.csv",
        book_path    = sys.argv[2] if len(sys.argv) > 2 else "trade_book.csv",
        metrics_path = sys.argv[3] if len(sys.argv) > 3 else "performance_metrics.csv",
    )