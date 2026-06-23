"""
EMA Crossover Momentum Strategy  —  Single File
=================================================
Inputs  : CSV with columns [Date, Time, open, high, low, close, volume]

Outputs (6 CSV files):
  ema_series.csv          : bar-level OHLCV + EMA values (read by visualizer)
  trade_book.csv          : one row per completed trade (gross prices/profit)
  trade_book_costs.csv    : same trades with before/after cost columns
  performance_metrics.csv : strategy metrics (read by visualizer)
  baseline_comparison.csv : strategy vs buy-and-hold side by side
  cost_summary.csv        : aggregate cost breakdown

═══════════════════════════════════════════════════════════════════
Timeframe resampling
═══════════════════════════════════════════════════════════════════
  resample_ohlcv() converts the 1-minute CSV into any larger
  timeframe before the strategy runs. Change the timeframe
  string in __main__ to switch bars:
    "1min"  original 1-minute bars (no change)
    "5min"  5-minute
    "15min" 15-minute
    "30min" 30-minute
    "1h"    1-hour
    "1D"    daily
    "1W"    weekly

═══════════════════════════════════════════════════════════════════
Market hours filter
═══════════════════════════════════════════════════════════════════
  Only bars with time >= 09:30 and time < 16:00 are eligible for
  signal detection and trade execution. Bars outside this window
  are present in ema_series.csv for charting but signals are
  masked so they cannot fire on out-of-hours bars.

  Any open position at the last valid bar of each session is
  force-closed at that bar's close to avoid overnight exposure.

═══════════════════════════════════════════════════════════════════
Signal logic  —  long + short, always in market
═══════════════════════════════════════════════════════════════════
  The strategy is always in a position (long or short) during
  market hours after the first crossover signal fires.

  cross_up   (short EMA crosses above long EMA):
    → Exit SHORT if currently short
    → Enter LONG

  cross_down (short EMA crosses below long EMA):
    → Exit LONG if currently long
    → Enter SHORT

  Signal detected on bar N → filled on bar N+1 close (no look-ahead).
  The same bar N+1 close is used for both the exit and new entry,
  so the flip is instantaneous with no gap between trades.

  profit for LONG  : exit_price − enter_price  (per share)
  profit for SHORT : enter_price − exit_price  (per share)

═══════════════════════════════════════════════════════════════════
Position sizing  —  spread-magnitude fractional sizing
═══════════════════════════════════════════════════════════════════
  Each trade's allocation is scaled by how strong the crossover is
  relative to its rolling 200-bar history (percentile rank).
    rank 0.0 → min_pct (20%)   weakest crossover
    rank 1.0 → max_pct (100%)  strongest crossover
  The rank series is shifted forward 1 bar (no look-ahead).
  capital_return = (percent / 100) × profit_pct

═══════════════════════════════════════════════════════════════════
EMA computation
═══════════════════════════════════════════════════════════════════
  pandas ewm(span=N, adjust=False) — standard recursive EMA:
    EMA_t = α × close_t + (1 − α) × EMA_{t-1},  α = 2 / (N + 1)
  First long_window rows are dropped (warmup period).

═══════════════════════════════════════════════════════════════════
Cost model
═══════════════════════════════════════════════════════════════════
  Slippage + commission applied on every fill (entry and exit).
  LONG  entry: fill = close + slippage
  LONG  exit : fill = close − slippage
  SHORT entry: fill = close − slippage
  SHORT exit : fill = close + slippage
  Default: 1 tick × $0.01 + $0.005 commission = $0.03/share rt

═══════════════════════════════════════════════════════════════════
Baseline comparison
═══════════════════════════════════════════════════════════════════
  Buy-and-hold over the same window (first entry → last exit).
  Comparable metrics: total return, annualised return,
  Sharpe, Sortino, max drawdown.
"""

import sys
from datetime import time as dtime
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Timeframe resampling
# ═══════════════════════════════════════════════════════════════════════════════

def resample_ohlcv(df: pd.DataFrame,
                   timeframe: str = "1min",
                   date_col:  str = "Date",
                   time_col:  str = "Time",
                   session_start: str = "09:30",
                   session_end:   str = "16:00") -> pd.DataFrame:
    """
    Resample 1-minute OHLCV into a larger timeframe before running the strategy.

    Market hours are filtered FIRST (default 09:30–16:00) so bars are
    anchored to the session open. This ensures hourly bars start at
    09:30, 10:30, 11:30 ... 14:30, 15:30 — matching TradingView behaviour.
    Without this filter, pandas anchors to midnight and produces bars at
    10:00, 11:00, 12:00 which do not align with the session.

    The last bar of each day may be shorter than the requested timeframe
    (e.g. 15:30–16:00 is 30 min when using "1h") — this is correct and
    expected given the 6.5-hour US session.

    timeframe options:
      "1min"  → 1-minute  (original, no change)
      "5min"  → 5-minute
      "15min" → 15-minute
      "30min" → 30-minute
      "1h"    → 1-hour
      "1D"    → daily
      "1W"    → weekly
    """
    data = df.copy()
    data["_datetime"] = pd.to_datetime(
        data[date_col].astype(str) + " " + data[time_col].astype(str)
    )
    data = data.sort_values("_datetime").reset_index(drop=True)

    # ── Filter to market hours before resampling ──────────────────────────────
    # This anchors bars to the session open (09:30) rather than midnight,
    # producing bars at 09:30, 10:30, 11:30 ... matching TradingView.
    t         = data["_datetime"].dt.time
    t_start   = pd.Timestamp(f"2000-01-01 {session_start}").time()
    t_end     = pd.Timestamp(f"2000-01-01 {session_end}").time()
    in_hours  = (t >= t_start) & (t < t_end)
    data      = data[in_hours].copy()

    data = data.set_index("_datetime").sort_index()

    # ── Group by date + intraday offset from session open ─────────────────────
    # For intraday timeframes we resample within each day separately so bars
    # never straddle the overnight gap.
    if timeframe in ("1D", "1W"):
        # Daily/weekly — just use standard resample across all days
        resampled = data.resample(timeframe, label="left", closed="left").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["close"])
    else:
        # Intraday — resample each day separately and concatenate.
        # This guarantees bars start at 09:30 every day.
        daily_groups = []
        for date, group in data.groupby(data.index.date):
            resampled_day = group.resample(
                timeframe, label="left", closed="left",
                origin=group.index[0]          # anchor to first bar of the day
            ).agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["close"])
            daily_groups.append(resampled_day)
        resampled = pd.concat(daily_groups)

    resampled = resampled.reset_index()
    resampled["Date"] = resampled["_datetime"].dt.strftime("%m/%d/%Y")
    resampled["Time"] = resampled["_datetime"].dt.strftime("%H:%M:%S")
    resampled = resampled.drop(columns=["_datetime"])

    return resampled


# ═══════════════════════════════════════════════════════════════════════════════
# Cost model
# ═══════════════════════════════════════════════════════════════════════════════

class CostModel:
    """
    Parameters
    ----------
    slippage_ticks       : Ticks of adverse slippage per fill (default 1).
    tick_size            : Dollar value of one tick (default $0.01).
    commission_per_share : Flat commission per share per fill (default $0.005).
    """
    def __init__(self,
                 slippage_ticks:       float = 1.0,
                 tick_size:            float = 0.01,
                 commission_per_share: float = 0.005):
        self.slippage_per_fill    = slippage_ticks * tick_size
        self.commission_per_share = commission_per_share

    @property
    def round_trip_slippage(self)   -> float: return self.slippage_per_fill * 2
    @property
    def round_trip_commission(self) -> float: return self.commission_per_share * 2
    @property
    def round_trip_total(self)      -> float: return self.round_trip_slippage + self.round_trip_commission

    def long_entry(self,  p: float) -> float: return p + self.slippage_per_fill
    def long_exit(self,   p: float) -> float: return p - self.slippage_per_fill
    def short_entry(self, p: float) -> float: return p - self.slippage_per_fill
    def short_exit(self,  p: float) -> float: return p + self.slippage_per_fill

    def describe(self) -> str:
        return (f"Slippage {self.slippage_per_fill*100:.3f}¢/fill  |  "
                f"Commission {self.commission_per_share*100:.3f}¢/share/fill  |  "
                f"Round-trip {self.round_trip_total*100:.3f}¢/share")


# ═══════════════════════════════════════════════════════════════════════════════
# Position sizing
# ═══════════════════════════════════════════════════════════════════════════════

class SpreadSizer:
    """
    Fractional sizing: ranks abs(spread) in a rolling window and maps
    the percentile rank to [min_pct, max_pct]. Shifted 1 bar forward
    internally so no future spread values influence sizing.

    Parameters
    ----------
    lookback : Rolling window in bars for percentile rank (default 200).
    min_pct  : Minimum allocation % — weakest crossovers (default 20).
    max_pct  : Maximum allocation % — strongest crossovers (default 100).
    """
    def __init__(self, lookback: int = 200,
                 min_pct: float = 20.0, max_pct: float = 100.0):
        if not (0 < min_pct <= max_pct <= 100):
            raise ValueError("Require 0 < min_pct <= max_pct <= 100")
        self.lookback = lookback
        self.min_pct  = min_pct
        self.max_pct  = max_pct

    def compute(self, spread_series: pd.Series) -> pd.Series:
        abs_spread = spread_series.abs()
        rank = abs_spread.rolling(self.lookback, min_periods=1).apply(
            lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else 0.5,
            raw=True,
        )
        rank = rank.shift(1).fillna(0.5)
        return (self.min_pct + rank * (self.max_pct - self.min_pct)).round(2)

    def describe(self) -> str:
        return (f"Spread-magnitude sizing  |  lookback={self.lookback} bars  |  "
                f"range=[{self.min_pct}%, {self.max_pct}%]")


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_strategy_metrics(book: pd.DataFrame, data: pd.DataFrame,
                              pct_col:    str = "profit_pct",
                              profit_col: str = "profit") -> dict:
    """
    Capital-weighted metrics. Each trade's equity contribution is:
        r_capital = (percent / 100) × profit_pct
    All metrics are derived from this series.
    """
    if len(book) == 0:
        return {}
    n = len(book)
    t_start = pd.to_datetime(book["enter_date"].iloc[0])
    t_end   = pd.to_datetime(book["exit_date"].iloc[-1])
    years   = max((t_end - t_start).days / 365.25, 1 / 365.25)

    r_cap        = (book["percent"] / 100.0) * book[pct_col]
    equity       = (1 + r_cap).cumprod()
    total_return = equity.iloc[-1] - 1
    ann_ret      = (1 + total_return) ** (1 / years) - 1

    tpy       = n / max(years, 1e-9)
    std_r     = r_cap.std(ddof=1)
    sharpe    = (r_cap.mean() / std_r) * np.sqrt(tpy) if std_r > 0 else None
    downside  = r_cap[r_cap < 0]
    dd_std    = downside.std(ddof=1) if len(downside) > 1 else None
    sortino   = (r_cap.mean() / dd_std) * np.sqrt(tpy) if dd_std and dd_std > 0 else None
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()

    return {
        "num_trades":             n,
        "total_return_pct":       round(total_return * 100, 4),
        "annualized_return_pct":  round(ann_ret * 100, 4),
        "sharpe_ratio":           round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio":          round(sortino, 4) if sortino is not None else None,
        "max_drawdown_pct":       round(max_dd * 100, 4),
        "win_rate_pct":           round((book[pct_col] > 0).sum() / n * 100, 2),
        "avg_profit_per_trade":   round(book[profit_col].mean(), 6),
        "avg_holding_minutes":    round(book["holding_minutes"].mean(), 1),
        "avg_position_size_pct":  round(book["percent"].mean(), 2),
        "exposure_pct":           round(book["holding_bars"].sum() / len(data) * 100, 2),
    }


def compute_bh_metrics(data: pd.DataFrame, book: pd.DataFrame,
                       close_col: str = "close",
                       bars_per_year: int = 252 * 390) -> dict:
    """
    Buy-and-hold metrics over the same window as the strategy.
    bars_per_year should match the resampled timeframe:
      1-min: 252*390=98280  1-hour: 252*6=1512  daily: 252  weekly: 52
    """
    if len(book) == 0:
        return {}
    t_start = pd.to_datetime(book["enter_date"].iloc[0])
    t_end   = pd.to_datetime(book["exit_date"].iloc[-1])
    years   = max((t_end - t_start).days / 365.25, 1 / 365.25)

    mask   = (data["_datetime"] >= t_start) & (data["_datetime"] <= t_end)
    prices = data.loc[mask, close_col].reset_index(drop=True)
    if len(prices) < 2:
        return {}

    bh_return  = (prices.iloc[-1] / prices.iloc[0]) - 1
    bh_ann_ret = (1 + bh_return) ** (1 / years) - 1
    bar_rets   = prices.pct_change().dropna()
    bh_sharpe  = (bar_rets.mean() / bar_rets.std(ddof=1)) * np.sqrt(bars_per_year) \
                 if bar_rets.std(ddof=1) > 0 else None
    bh_downside = bar_rets[bar_rets < 0]
    bh_dd_std   = bh_downside.std(ddof=1) if len(bh_downside) > 1 else None
    bh_sortino  = (bar_rets.mean() / bh_dd_std) * np.sqrt(bars_per_year) \
                  if bh_dd_std and bh_dd_std > 0 else None
    equity     = (1 + bar_rets).cumprod()
    bh_maxdd   = ((equity - equity.cummax()) / equity.cummax()).min()

    return {
        "total_return_pct":       round(bh_return * 100, 4),
        "annualized_return_pct":  round(bh_ann_ret * 100, 4),
        "sharpe_ratio":           round(bh_sharpe, 4) if bh_sharpe is not None else None,
        "sortino_ratio":          round(bh_sortino, 4) if bh_sortino is not None else None,
        "max_drawdown_pct":       round(bh_maxdd * 100, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Trade record helper
# ═══════════════════════════════════════════════════════════════════════════════

def _make_trade(signal, enter_dt, exit_dt, enter_p, exit_p,
                pos_pct, hold_bars, hold_min, cost_model, direction):
    """
    Build one gross record and one costs record for a completed trade.

    direction : +1 for LONG, -1 for SHORT
      LONG  profit = exit − enter
      SHORT profit = enter − exit
    """
    profit  = direction * (exit_p - enter_p)
    pct     = profit / enter_p
    cap_ret = (pos_pct / 100.0) * pct

    if direction == 1:
        enter_net = cost_model.long_entry(enter_p)
        exit_net  = cost_model.long_exit(exit_p)
    else:
        enter_net = cost_model.short_entry(enter_p)
        exit_net  = cost_model.short_exit(exit_p)

    slip_cost  = cost_model.round_trip_slippage
    comm_cost  = cost_model.round_trip_commission
    total_cost = slip_cost + comm_cost
    profit_net = profit - total_cost
    pct_net    = profit_net / enter_p

    gross = {
        "signal":          signal,
        "enter_date":      enter_dt.date(),
        "year":            enter_dt.year,
        "month":           enter_dt.month,
        "enter_time":      enter_dt.time(),
        "enter_price":     round(enter_p,  6),
        "exit_date":       exit_dt.date(),
        "exit_time":       exit_dt.time(),
        "exit_price":      round(exit_p,   6),
        "profit":          round(profit,   6),
        "profit_pct":      round(pct,      6),
        "capital_return":  round(cap_ret,  6),
        "holding_bars":    hold_bars,
        "holding_minutes": hold_min,
        "percent":         pos_pct,
    }

    costs = {
        "signal":               signal,
        "enter_date":           enter_dt.date(),
        "year":                 enter_dt.year,
        "month":                enter_dt.month,
        "enter_time":           enter_dt.time(),
        "exit_date":            exit_dt.date(),
        "exit_time":            exit_dt.time(),
        "holding_bars":         hold_bars,
        "holding_minutes":      hold_min,
        "percent":              pos_pct,
        "enter_price_gross":    round(enter_p,                6),
        "exit_price_gross":     round(exit_p,                 6),
        "enter_price_net":      round(enter_net,              6),
        "exit_price_net":       round(exit_net,               6),
        "slippage_cost":        round(slip_cost,              6),
        "commission_cost":      round(comm_cost,              6),
        "total_cost":           round(total_cost,             6),
        "profit_gross":         round(profit,                 6),
        "profit_pct_gross":     round(pct,                    6),
        "capital_return_gross": round((pos_pct/100.0)*pct,    6),
        "profit_net":           round(profit_net,             6),
        "profit_pct_net":       round(pct_net,                6),
        "capital_return_net":   round((pos_pct/100.0)*pct_net,6),
    }
    return gross, costs


# ═══════════════════════════════════════════════════════════════════════════════
# Main strategy
# ═══════════════════════════════════════════════════════════════════════════════

MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def run_ema_crossover(
    df: pd.DataFrame,
    cost_model:    CostModel,
    sizer:         SpreadSizer,
    short_window:  int = 20,
    long_window:   int = 50,
    bars_per_year: int = 252 * 390,   # adjust when resampling (see table in docstring)
    date_col:      str = "Date",
    time_col:      str = "Time",
    open_col:      str = "open",
    high_col:      str = "high",
    low_col:       str = "low",
    close_col:     str = "close",
    volume_col:    str = "volume",
    series_csv:      str = "ema_series.csv",
    book_csv:        str = "trade_book.csv",
    costs_csv:       str = "trade_book_costs.csv",
    metrics_csv:     str = "performance_metrics.csv",
    baseline_csv:    str = "baseline_comparison.csv",
    costsummary_csv: str = "cost_summary.csv",
) -> dict:

    # ── 1. Validate & prepare ────────────────────────────────────────────────
    required = [date_col, time_col, open_col, high_col, low_col, close_col, volume_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df.copy()
    data["_datetime"] = pd.to_datetime(
        data[date_col].astype(str) + " " + data[time_col].astype(str)
    )
    data = data.sort_values("_datetime").reset_index(drop=True)

    if data[close_col].isna().any():
        raise ValueError("close column contains NaN values.")

    # ── 2. Compute EMAs (full series, before warmup drop) ────────────────────
    data["ema_short"] = data[close_col].ewm(span=short_window, adjust=False).mean()
    data["ema_long"]  = data[close_col].ewm(span=long_window,  adjust=False).mean()
    data["spread"]    = data["ema_short"] - data["ema_long"]

    # ── 3. Fractional position sizes (no look-ahead) ─────────────────────────
    data["position_pct"] = sizer.compute(data["spread"])

    # ── 4. Write ema_series.csv ───────────────────────────────────────────────
    ema_out = data[[
        "_datetime", open_col, high_col, low_col, close_col, volume_col,
        "ema_short", "ema_long", "spread", "position_pct",
    ]].copy()
    ema_out.insert(0, "Date", data[date_col])
    ema_out.insert(1, "Time", data[time_col])
    ema_out = ema_out.drop(columns=["_datetime"])
    ema_out["in_warmup"] = True
    ema_out.iloc[long_window:, ema_out.columns.get_loc("in_warmup")] = False
    ema_out.to_csv(series_csv, index=False)
    print(f"EMA series       → {series_csv}  ({len(ema_out):,} bars)")

    # ── 5. Crossover detection — no look-ahead ───────────────────────────────
    data["spread_prev"]  = data["spread"].shift(1)
    data["cross_up"]     = (data["spread"] > 0) & (data["spread_prev"] <= 0)
    data["cross_down"]   = (data["spread"] < 0) & (data["spread_prev"] >= 0)
    data["entry_signal"] = data["cross_up"].shift(1).fillna(False)
    data["exit_signal"]  = data["cross_down"].shift(1).fillna(False)

    # ── 6. Market hours filter ────────────────────────────────────────────────
    # For intraday bars apply the 09:30-16:00 window.
    # For daily/weekly bars every bar IS the session so the intraday
    # time check is skipped entirely — all bars are marked as in-hours.
    bar_time = data["_datetime"].dt.time
    intraday = bar_time.nunique() > 1   # multiple distinct times = intraday

    if intraday:
        data["_in_hours"] = (bar_time >= MARKET_OPEN) & (bar_time < MARKET_CLOSE)
    else:
        data["_in_hours"] = True

    data.loc[~data["_in_hours"], "entry_signal"] = False
    data.loc[~data["_in_hours"], "exit_signal"]  = False

    # Last valid bar detection.
    # Intraday: last bar of each calendar session (e.g. 15:30 on hourly).
    # Daily/weekly: only the very last bar of the entire dataset — positions
    # are allowed to carry across days, closing only on a signal or end of data.
    data["_date"] = data["_datetime"].dt.date
    if intraday:
        data["_last_bar"] = (
            data["_in_hours"] &
            (~data["_in_hours"].shift(-1, fill_value=False) |
             (data["_date"] != data["_date"].shift(-1,
              fill_value=data["_date"].iloc[0])))
        )
    else:
        data["_last_bar"] = False
        if len(data) > 0:
            data.loc[data.index[-1], "_last_bar"] = True

    # Drop warmup rows
    data = data.iloc[long_window:].reset_index(drop=True)

    # ── 7. Build trade books — long + short ──────────────────────────────────
    trades_gross = []
    trades_costs = []
    position     = 0       # 0=flat, 1=long, -1=short
    entry_row    = None
    entry_idx    = None

    for idx, row in data.iterrows():
        if not row["_in_hours"]:
            continue

        fill_p  = row[close_col]    # fill at bar N+1 close
        is_eod  = row["_last_bar"]

        # ── Force-close at end of session ─────────────────────────────────
        if is_eod and position != 0:
            enter_dt  = entry_row["_datetime"]
            exit_dt   = row["_datetime"]
            hold_bars = idx - entry_idx
            hold_min  = round((exit_dt - enter_dt).total_seconds() / 60, 1)
            sig       = "LONG_EMA_CROSS" if position == 1 else "SHORT_EMA_CROSS"
            g, c = _make_trade(sig, enter_dt, exit_dt,
                               entry_row[close_col], fill_p,
                               entry_row["position_pct"],
                               hold_bars, hold_min, cost_model, position)
            trades_gross.append(g)
            trades_costs.append(c)
            position  = 0
            entry_row = None
            entry_idx = None
            continue

        # ── Cross UP: exit short → enter long ─────────────────────────────
        if row["entry_signal"]:
            if position == -1:
                enter_dt  = entry_row["_datetime"]
                exit_dt   = row["_datetime"]
                hold_bars = idx - entry_idx
                hold_min  = round((exit_dt - enter_dt).total_seconds() / 60, 1)
                g, c = _make_trade("SHORT_EMA_CROSS", enter_dt, exit_dt,
                                   entry_row[close_col], fill_p,
                                   entry_row["position_pct"],
                                   hold_bars, hold_min, cost_model, -1)
                trades_gross.append(g)
                trades_costs.append(c)
            position  = 1
            entry_row = row
            entry_idx = idx

        # ── Cross DOWN: exit long → enter short ───────────────────────────
        elif row["exit_signal"]:
            if position == 1:
                enter_dt  = entry_row["_datetime"]
                exit_dt   = row["_datetime"]
                hold_bars = idx - entry_idx
                hold_min  = round((exit_dt - enter_dt).total_seconds() / 60, 1)
                g, c = _make_trade("LONG_EMA_CROSS", enter_dt, exit_dt,
                                   entry_row[close_col], fill_p,
                                   entry_row["position_pct"],
                                   hold_bars, hold_min, cost_model, 1)
                trades_gross.append(g)
                trades_costs.append(c)
            position  = -1
            entry_row = row
            entry_idx = idx

    # ── 8. Assemble & write trade books ──────────────────────────────────────
    gross_cols = [
        "signal", "enter_date", "year", "month", "enter_time", "enter_price",
        "exit_date", "exit_time", "exit_price", "profit", "profit_pct",
        "capital_return", "holding_bars", "holding_minutes", "percent",
    ]
    costs_cols = [
        "signal", "enter_date", "year", "month", "enter_time",
        "exit_date", "exit_time", "holding_bars", "holding_minutes", "percent",
        "enter_price_gross", "exit_price_gross", "enter_price_net", "exit_price_net",
        "slippage_cost", "commission_cost", "total_cost",
        "profit_gross", "profit_pct_gross", "capital_return_gross",
        "profit_net", "profit_pct_net", "capital_return_net",
    ]

    book_g = pd.DataFrame(trades_gross)[gross_cols] if trades_gross \
             else pd.DataFrame(columns=gross_cols)
    book_c = pd.DataFrame(trades_costs)[costs_cols] if trades_costs \
             else pd.DataFrame(columns=costs_cols)

    book_g.to_csv(book_csv,  index=False)
    book_c.to_csv(costs_csv, index=False)
    n_long  = (book_g["signal"] == "LONG_EMA_CROSS").sum()
    n_short = (book_g["signal"] == "SHORT_EMA_CROSS").sum()
    print(f"Trade book       → {book_csv}  "
          f"({len(book_g)} trades: {n_long} long, {n_short} short)")
    print(f"Trade book costs → {costs_csv}")

    # ── 9. Metrics ────────────────────────────────────────────────────────────
    m_gross = compute_strategy_metrics(book_c, data,
                                       pct_col="profit_pct_gross",
                                       profit_col="profit_gross")
    m_net   = compute_strategy_metrics(book_c, data,
                                       pct_col="profit_pct_net",
                                       profit_col="profit_net")
    m_bh    = compute_bh_metrics(data, book_g, close_col=close_col,
                                 bars_per_year=bars_per_year)

    m_gross["short_window"] = short_window
    m_gross["long_window"]  = long_window
    pd.DataFrame([m_gross]).to_csv(metrics_csv, index=False)
    print(f"Metrics          → {metrics_csv}")

    # ── 10. Baseline comparison CSV ───────────────────────────────────────────
    comparable = {
        "total_return_pct":      "Total return (%)",
        "annualized_return_pct": "Annualised return (%)",
        "sharpe_ratio":          "Sharpe ratio",
        "sortino_ratio":         "Sortino ratio",
        "max_drawdown_pct":      "Max drawdown (%)",
    }
    strategy_only = {
        "num_trades":            "Number of trades",
        "win_rate_pct":          "Win rate (%)",
        "avg_profit_per_trade":  "Avg profit / trade ($)",
        "avg_holding_minutes":   "Avg holding time (min)",
        "avg_position_size_pct": "Avg position size (%)",
        "exposure_pct":          "Exposure (%)",
    }

    baseline_rows = []
    for k, label in comparable.items():
        g = m_gross.get(k); n = m_net.get(k); b = m_bh.get(k)
        try:    vs_bh = round(g - b, 4)
        except: vs_bh = None
        baseline_rows.append({"metric": label, "strategy_gross": g,
                               "strategy_net": n, "buy_and_hold": b,
                               "gross_vs_bh": vs_bh})
    for k, label in strategy_only.items():
        baseline_rows.append({"metric": label,
                               "strategy_gross": m_gross.get(k),
                               "strategy_net":   m_net.get(k),
                               "buy_and_hold": "—", "gross_vs_bh": "—"})
    pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
    print(f"Baseline         → {baseline_csv}")

    # ── 11. Cost summary CSV ──────────────────────────────────────────────────
    if len(book_c):
        pd.DataFrame([{
            "slippage_per_fill_$":     cost_model.slippage_per_fill,
            "commission_per_fill_$":   cost_model.commission_per_share,
            "round_trip_total_$":      cost_model.round_trip_total,
            "num_trades":              len(book_c),
            "total_slippage_$":        book_c["slippage_cost"].sum().round(4),
            "total_commission_$":      book_c["commission_cost"].sum().round(4),
            "total_cost_all_trades_$": book_c["total_cost"].sum().round(4),
            "avg_cost_per_trade_$":    book_c["total_cost"].mean().round(6),
            "cost_drag_pct":           round(
                m_gross["total_return_pct"] - m_net["total_return_pct"], 4),
        }]).to_csv(costsummary_csv, index=False)
        print(f"Cost summary     → {costsummary_csv}")

    # ── 12. Console report ────────────────────────────────────────────────────
    w = 14
    print(f"\n{'═'*82}")
    print(f"  {'Metric':<32} {'Gross':>{w}} {'Net (w/ costs)':>{w}} "
          f"{'Buy & Hold':>{w}} {'Gross vs B&H':>{w}}")
    print(f"  {'-'*78}")
    for k, label in comparable.items():
        g = m_gross.get(k); n = m_net.get(k); b = m_bh.get(k)
        try:    diff = f"{g - b:+.4f}"
        except: diff = "—"
        print(f"  {label:<32} {str(g):>{w}} {str(n):>{w}} {str(b):>{w}} {diff:>{w}}")
    print(f"\n  {'Metric':<32} {'Value':>{w}}")
    print(f"  {'-'*48}")
    for k, label in strategy_only.items():
        print(f"  {label:<32} {str(m_gross.get(k)):>{w}}")
    print(f"\n  Market hours : {MARKET_OPEN.strftime('%H:%M')} – "
          f"{MARKET_CLOSE.strftime('%H:%M')}  (force-close at session end)")
    print(f"  Sizing       : {sizer.describe()}")
    print(f"  Costs        : {cost_model.describe()}")
    print(f"{'═'*82}\n")

    return {
        "trade_book":       book_g,
        "trade_book_costs": book_c,
        "metrics_gross":    m_gross,
        "metrics_net":      m_net,
        "metrics_bh":       m_bh,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "aapl_10y_1m.csv"

    # ── Timeframe ─────────────────────────────────────────────────────────────
    # Change this one line to switch bar size.
    # Also update bars_per_year below to match:
    #   "1min"  → 252 * 390 = 98280
    #   "5min"  → 252 * 78  = 19656
    #   "15min" → 252 * 26  = 6552
    #   "30min" → 252 * 13  = 3276
    #   "1h"    → 252 * 7   = 1764
    #   "1D"    → 252
    #   "1W"    → 52
    TIMEFRAME     = "15min"
    BARS_PER_YEAR = 252 * 26   # 15-min bars per year (252 days × 26 bars/day)

    # ── Cost model ────────────────────────────────────────────────────────────
    costs = CostModel(
        slippage_ticks       = 1,
        tick_size            = 0.01,
        commission_per_share = 0.005,
    )

    # ── Position sizer ────────────────────────────────────────────────────────
    sizer = SpreadSizer(
        lookback = 200,
        min_pct  = 20.0,
        max_pct  = 100.0,
    )

    raw = pd.read_csv(csv_path)
    print(f"Loaded {len(raw):,} rows from {csv_path}")

    if TIMEFRAME != "1min":
        raw = resample_ohlcv(raw, timeframe=TIMEFRAME)
        print(f"Resampled to {TIMEFRAME}  →  {len(raw):,} bars\n")
    else:
        print(f"Using original 1-min bars  →  {len(raw):,} bars\n")

    results = run_ema_crossover(
        df            = raw,
        cost_model    = costs,
        sizer         = sizer,
        short_window  = 30,
        long_window   = 60,
        bars_per_year = BARS_PER_YEAR,
        date_col      = "Date",
        time_col      = "Time",
    )

    print("── Trade book sample (first 5 rows) ──")
    print(results["trade_book"][
        ["signal", "enter_date", "enter_time", "enter_price",
         "exit_price", "profit", "profit_pct", "capital_return", "percent"]
    ].head().to_string(index=False))