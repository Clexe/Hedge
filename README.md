# Hedge — Automated Momentum Trading System

A self-built, open-source automated momentum trading system for US equities / ETFs.
Runs overnight, rebalances based on cross-sectional momentum factors, and executes
trades through a broker API.

Inspired by the Quant Science workflow (QSConnect → QSResearch → QSWorkflow → Omega),
but built entirely from free / open-source tools you can audit, understand, and extend.

---

## 1. Recommended Tech Stack (2026)

| Stage | Commercial Tool | Our Open-Source Replacement |
|---|---|---|
| **Data Ingestion** | QSConnect | `yfinance` + Parquet files (or Polygon / Alpaca data API for higher quality) |
| **Research Database** | QSConnect DB | Parquet per-ticker on local disk (upgrade: DuckDB or ClickHouse) |
| **Signal Research** | QSResearch | `pandas` + `numpy` + `scipy` (add `scikit-learn` / `lightgbm` in Phase 2) |
| **Backtesting** | Zipline | Custom vectorised engine in this repo (upgrade: `vectorbt` or `NautilusTrader`) |
| **Automation** | QSWorkflow | `APScheduler` in-process (upgrade: Prefect or system cron) |
| **Execution** | Omega | `alpaca-py` for Alpaca Markets (alternative: `ib_insync` for Interactive Brokers) |
| **Dashboard** | N/A | Streamlit + Plotly (Phase 2) |

### Why this stack

- **yfinance**: Free, no API key, handles adjusted prices. Limitation: rate limits,
  delayed data, may break without warning. For production, supplement with Polygon or
  Alpaca's market data API.
- **Parquet**: Columnar, compressed, fast reads. One file per ticker makes incremental
  updates trivial. Scales to thousands of tickers on a laptop.
- **Custom backtest engine**: You understand every line. No hidden magic. The vectorised
  approach is fast enough for daily bars on 500 names (< 1 second).
- **APScheduler**: Zero infrastructure. Runs inside your Python process. For more
  robustness, use system cron or Prefect (adds task retry, observability, alerting).
- **Alpaca**: Commission-free equities, fractional shares, good Python SDK, built-in
  paper trading endpoint. The paper API is identical to live — critical for testing.

---

## 2. Project Structure

```
Hedge/
├── config/
│   ├── default.yaml            # All tunable parameters (checked into git)
│   └── secrets.yaml.example    # Template for API keys (NEVER commit real keys)
│
├── hedge/                      # Main Python package
│   ├── __init__.py
│   ├── __main__.py             # python -m hedge
│   ├── cli.py                  # Argument parsing & command dispatch
│   │
│   ├── data/                   # STEP 1 — Data Ingestion & Storage
│   │   ├── universe.py         #   S&P 500 ticker list, liquidity filter
│   │   └── loader.py           #   Download, cache, serve OHLCV bars
│   │
│   ├── signals/                # STEP 2 — Signal Generation
│   │   └── momentum.py         #   12-1 momentum, vol adjustment, ranking
│   │
│   ├── portfolio/              # STEP 3 — Portfolio Construction
│   │   └── optimizer.py        #   Inv-vol, risk parity, constraints
│   │
│   ├── backtest/               # STEP 4 — Historical Simulation
│   │   └── engine.py           #   Vectorised backtest with realistic costs
│   │
│   ├── execution/              # STEP 5 — Order Management & Broker
│   │   ├── broker.py           #   Paper + Alpaca broker abstraction
│   │   └── order_manager.py    #   Weight → order conversion, safety checks
│   │
│   ├── pipeline/               # STEP 6 — Orchestration
│   │   └── orchestrator.py     #   Nightly ETL → Signal → Portfolio → Execute
│   │
│   └── utils/
│       ├── config.py           #   YAML loader with env-var overrides
│       ├── logging.py          #   Structured rotating file + console logger
│       └── risk.py             #   Drawdown breaker, stop losses, corr guard
│
├── tests/                      # pytest test suite
├── notebooks/                  # Jupyter research notebooks
├── data/                       # Local data store (gitignored)
│   ├── raw/
│   ├── processed/              #   Parquet files live here
│   └── cache/
├── logs/                       # Rotating log files (gitignored)
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    NIGHTLY PIPELINE                         │
│                  (runs at 5 AM UTC)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. UNIVERSE          Fetch S&P 500 tickers                 │
│       │                                                     │
│       ▼                                                     │
│  2. DATA REFRESH      Download new bars for each ticker     │
│       │               Store as Parquet (incremental)        │
│       ▼                                                     │
│  3. LIQUIDITY FILTER  Drop tickers below $1M avg volume     │
│       │                                                     │
│       ▼                                                     │
│  4. SIGNAL GEN        12-1 momentum ÷ trailing vol          │
│       │               Cross-sectional rank → top decile     │
│       ▼                                                     │
│  5. PORTFOLIO OPT     Inverse-volatility weighting          │
│       │               Position cap (5%), vol target (12%)   │
│       │               Cash buffer (2%)                      │
│       ▼                                                     │
│  6. RISK CHECKS       Drawdown breaker, stop losses         │
│       │               Correlation guard                     │
│       ▼                                                     │
│  7. ORDER GEN         Diff target vs current positions      │
│       │               Cap individual order size ($25K)      │
│       ▼                                                     │
│  8. EXECUTION         Submit to Alpaca (or paper broker)    │
│       │               Sells first, then buys                │
│       ▼                                                     │
│  9. LOGGING           Full audit trail in logs/hedge.log    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Momentum Strategy: The Core Idea

### The 12-1 Factor (Jegadeesh & Titman, 1993)

```
momentum(t) = price(t - 21) / price(t - 252) - 1
```

- Look at each stock's total return over the past **12 months** (~252 trading days).
- **Skip the most recent month** (~21 days). Why? The last month contains short-term
  reversal noise that hurts momentum. This is one of the best-documented anomalies in
  finance.
- Rank all stocks cross-sectionally. Buy the top decile. (Optionally short the bottom
  decile — we default to long-only for simplicity and lower risk.)

### Risk-Adjusted Momentum

Raw momentum favours high-beta lottery tickets. Dividing by trailing 3-month volatility
gives a Sharpe-like signal that prefers *steady* compounders:

```
risk_adj_momentum(t) = momentum(t) / annualised_vol(t, lookback=63)
```

### Portfolio Construction

After selecting the top decile (~50 names from S&P 500), we weight by **inverse
volatility**: lower-vol names get more capital. Then we apply:

- **5% position cap** — no single bet dominates
- **12% annualised vol target** — scale gross exposure up/down to maintain consistent
  risk
- **2% cash buffer** — always have cash for margin calls or opportunities

---

## 5. Quick Start

```bash
# Clone and install
git clone https://github.com/your-repo/Hedge.git
cd Hedge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download market data (~15 min for full S&P 500, 5 years)
python -m hedge download

# Run a backtest
python -m hedge backtest --output equity_curve.csv

# Generate today's signals (no trading)
python -m hedge signals

# Paper trade: full pipeline, dry run (no orders submitted)
python -m hedge run --dry-run

# Paper trade: full pipeline, orders submitted to paper broker
python -m hedge run

# Start automated nightly schedule
python -m hedge schedule
```

---

## 6. Critical Gotchas (Be Brutally Honest)

### What Will Bite You

1. **Survivorship bias** — Using today's S&P 500 list to backtest historically is wrong.
   Companies that went bankrupt, got delisted, or were removed from the index are
   missing. Your backtest will be ~1-3% per year too optimistic. Fix: use point-in-time
   constituent data (Sharadar, Compustat).

2. **Look-ahead bias** — The most insidious bug. If ANY piece of data from the future
   leaks into a signal calculation, your backtest is meaningless. This includes using
   end-of-day prices to generate signals that are "traded" at that same close price.
   Our engine explicitly uses `prices.loc[:dt]` to prevent this.

3. **Transaction costs** — Academic papers often assume zero costs. In reality,
   commissions (0-5 bps) + slippage (5-20 bps) + spread (5-30 bps) eat into returns
   heavily, especially with monthly rebalancing of 50 names. We model 20 bps round-trip.

4. **Momentum crashes** — Momentum has spectacular drawdowns (2009: -50%+). The strategy
   can lose badly during sharp market reversals when last year's losers snap back
   violently. Mitigation: vol targeting (scale down when vol spikes), which we implement.

5. **Edge decay** — Momentum has been well-known since the 1990s. Every hedge fund runs
   some variant. The premium has compressed. Expect 2-5% annualised alpha in a good
   implementation, not the 10%+ from old papers.

6. **Data quality** — yfinance data has errors, missing bars, wrong split adjustments.
   Always eyeball your data. For production, pay for Polygon or similar.

7. **Overnight gaps** — You generate signals at night, but prices open differently in
   the morning. Slippage between signal price and execution price is real and variable.

8. **Live vs backtest divergence** — Your backtest will always look better than live.
   Guaranteed. The question is by how much. Budget 30-50% return degradation.

### What Is Actually Easy

- Setting up the data pipeline (a weekend).
- Computing momentum signals (an afternoon).
- Running a backtest (a day to get right).

### What Is Actually Hard

- Getting execution right (fills, partial fills, rejected orders, API outages).
- Debugging live production issues at 3 AM when your scheduler crashed.
- Maintaining the system for months after the initial excitement fades.
- Handling corporate actions, stock splits, ticker changes, and delistings.
- Resisting the urge to over-fit after seeing 50 backtest variations.
- Accepting that your live performance will trail your backtest.

---

## 7. Security & Safety Checklist

- [ ] **NEVER commit API keys.** Use `config/secrets.yaml` (gitignored) or env vars.
- [ ] **Paper trade for at least 3 months** before considering real money.
- [ ] **Set `paper_only: true`** in config. This is the default.
- [ ] **Max order size** is capped at $25K by default. Change deliberately.
- [ ] **Drawdown circuit breaker** flattens everything at -15%.
- [ ] **Code review every change** before it touches the live pipeline.
- [ ] **Monitor logs daily** — don't set and forget.
- [ ] **Start small** — even when going live, use a fraction of your capital.

---

## 8. Phased Roadmap

### Phase 1: MVP (You Are Here)

- [x] Data pipeline: yfinance → Parquet (S&P 500, 5 years daily)
- [x] Signal: 12-1 risk-adjusted momentum, top decile
- [x] Portfolio: inverse-volatility, 5% position cap, 12% vol target
- [x] Backtest: vectorised, realistic costs (20 bps round-trip)
- [x] Execution: paper broker (simulated) + Alpaca paper API
- [x] CLI: download / signals / backtest / run / schedule
- [x] Risk: drawdown breaker, stop losses, correlation guard
- [ ] Paper trade for 3+ months, compare live fills vs backtest

### Phase 2: Improvements

- [ ] **ML signals**: Train gradient-boosted model on momentum + value + quality features
- [ ] **Sector neutralisation**: Long-short within each GICS sector
- [ ] **Multi-timeframe**: Combine 1-month, 6-month, 12-month momentum
- [ ] **Intraday execution**: VWAP/TWAP for better fills
- [ ] **Dashboard**: Streamlit app showing equity curve, positions, risk metrics
- [ ] **Alerting**: Slack/email on pipeline failure, large drawdowns
- [ ] **Point-in-time universe**: Eliminate survivorship bias
- [ ] **Multi-asset**: Add fixed income ETFs, commodities, international

### Phase 3: Production

- [ ] **Redundant scheduling**: Prefect + dead-man's switch
- [ ] **Infrastructure**: Docker container, cloud VM with monitoring
- [ ] **Data quality pipeline**: Automated checks for stale/bad data
- [ ] **Performance attribution**: Factor decomposition (market, size, value, momentum)
- [ ] **Tax-lot optimisation**: Harvest losses, manage holding periods

---

## 9. Difficulty & Time Investment (Honest Assessment)

| Phase | Effort | What you'll actually spend time on |
|---|---|---|
| Phase 1 MVP | 2-4 weeks | 20% coding, 80% debugging data issues and backtest edge cases |
| Paper trading | 3-6 months | Watching, waiting, fixing bugs that only appear live |
| Phase 2 | 2-3 months | ML feature engineering, dashboard, better execution |
| Going live | Ongoing | Monitoring, maintenance, resisting over-fitting |

**The uncomfortable truth**: Most retail systematic trading efforts fail to produce
consistent alpha after costs. Institutional players have better data, lower latency,
more capital, and teams of PhDs. The realistic outcomes are:

1. **You learn a ton** about quantitative finance, software engineering, and markets.
   This is genuinely valuable regardless of P&L.
2. **Your strategy roughly matches the market** (beta ≈ 1, alpha ≈ 0) after costs.
   Still useful as a disciplined, emotion-free investment approach.
3. **You find a small edge** (1-3% alpha) that decays over time and requires constant
   maintenance. This is the realistic best case.
4. **You lose money** during a momentum crash and abandon the project. This is common
   and not a personal failing — the strategy has structural risks.

Build this to learn. If it also makes money, that's a bonus.

---

## License

MIT
