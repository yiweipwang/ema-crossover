# EMA Crossover Momentum Strategy

Long/short EMA(10)/EMA(50) crossover backtested on 10 years of 1-minute intraday AAPL data
resampled to 30-minute bars. Includes a full transaction cost model, fractional position
sizing, and a decoupled visualization suite.

Parameters selected via grid search across 8 EMA pairs × 5 timeframes (40 combinations).

## Files

| File | Purpose |
|------|---------|
| `ema_ls_kbar_close.py` | Strategy engine — resamples 1-min bars, generates signals, simulates trades, writes 6 output CSVs |
| `ema_visualizer.py` | Reads the output CSVs and produces equity curve, EMA overlay, and trade distribution charts |
| `performance_metrics.csv` | Sample run output |
| `baseline_comparison.csv` | Strategy vs buy-and-hold |
| `cost_summary.csv` | Slippage + commission breakdown |
| `trade_book.csv` / `trade_book_costs.csv` | Per-trade log (gross and net) |

## Data

Supply a CSV with columns `[Date, Time, open, high, low, close, volume]` at 1-minute resolution.
The strategy resamples to the target timeframe (default: 30-minute bars). Change `TIMEFRAME`
and `BARS_PER_YEAR` in `__main__` to switch:

| Timeframe | `BARS_PER_YEAR` |
|-----------|-----------------|
| `"1min"`  | 252 × 390 = 98 280 |
| `"30min"` | 252 × 13  = 3 276  |
| `"1h"`    | 252 × 7   = 1 764  |
| `"1D"`    | 252               |

## Quick start

```bash
pip install pandas numpy matplotlib
python ema_ls_kbar_close.py your_data.csv   # writes 6 output CSVs
python ema_visualizer.py                    # reads CSVs, produces charts
```

## Strategy summary

- **Signal:** EMA(10) / EMA(50) crossover on 30-minute bars
- **Direction:** Long when fast > slow; short when slow > fast — always in market
- **Position sizing:** Spread-magnitude fractional sizing (20%–100% of capital), percentile-ranked over a rolling 200-bar window — no look-ahead
- **Market hours filter:** Signals only fire 09:30–16:00 ET; open positions force-closed at session end
- **Cost model:** $0.01 slippage + $0.005 commission per fill ($0.03 round-trip)

## Sample results (AAPL 10Y, 30-min bars, EMA 10/50)

Grid search winner across 40 parameter combinations by Sharpe ratio.

| Metric | Strategy (gross) | Buy-and-hold |
|--------|-----------------|--------------|
| Total return | 21.9% | 613.9% |
| Annualised return | 2.01% | 21.78% |
| **Sharpe ratio** | **1.03** | 0.84 |
| Max drawdown | **−2.54%** | −38.84% |
| Win rate | 54.4% | — |
| Trades | 733 | — |
| Cost drag (gross → net) | −10.9 pp | — |

The strategy's gross Sharpe of 1.03 exceeds buy-and-hold (0.84) while limiting max drawdown
to 2.54% vs 38.8% — demonstrating the short leg's risk-reduction value even as it
sacrifices absolute return on a secular growth stock.
