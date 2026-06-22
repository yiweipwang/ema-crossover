# EMA Crossover Momentum Strategy

Single-file long/short EMA crossover backtest on intraday AAPL data, with full cost modelling and a companion visualizer.

## Files

| File | Purpose |
|------|---------|
| `ema_ls_kbar_close.py` | Strategy engine — resamples 1-min bars, generates signals, simulates trades, writes 6 output CSVs |
| `ema_visualizer.py` | Reads the output CSVs and produces charts |
| `performance_metrics.csv` | Sample run output |
| `baseline_comparison.csv` | Strategy vs buy-and-hold |
| `cost_summary.csv` | Slippage + commission breakdown |
| `trade_book.csv` / `trade_book_costs.csv` | Per-trade log (gross and net) |

## Data

Supply a CSV with columns `[Date, Time, open, high, low, close, volume]` at 1-minute resolution.
The strategy resamples to any timeframe (default: weekly bars). Change the `timeframe` string in `__main__` to switch.

## Quick start

```bash
pip install pandas matplotlib
python ema_ls_kbar_close.py          # runs on your data CSV, writes output CSVs
python ema_visualizer.py             # reads output CSVs, produces charts
```

## Strategy summary

- **Signal:** EMA(20) / EMA(50) crossover on close prices
- **Direction:** Long when fast > slow; short when slow > fast — always in market
- **Market hours filter:** Signals only fire 09:30–16:00; open positions force-closed at session end
- **Cost model:** $0.01 slippage + $0.005 commission per fill

## Sample results (AAPL 10Y, weekly bars)

| Metric | Strategy (gross) | Buy-and-hold |
|--------|-----------------|--------------|
| Total return | 20.5% | 639.5% |
| Sharpe | 0.29 | 0.86 |
| Max drawdown | -11.0% | -38.6% |
| Trades | 46 | — |
