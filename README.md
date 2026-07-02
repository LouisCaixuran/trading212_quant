# Quant signals for a Trading 212 GBP account

A small personal toolkit that produces daily buy/sell signals for a
tech-focused basket of US-listed shares, optionally places the orders
through the Trading 212 API, keeps a monthly record, and prepares an
offline pack you can hand to any LLM for a periodic strategy review.

**This is a mechanical screening tool, not investment advice.** Daily
rule-based signals on liquid US equities have no demonstrated edge for a
retail account once spreads, the 0.15% FX conversion fee and taxes are
included. The strategy is concentrated and high-turnover, and its
drawdowns can exceed a third of the account. Read the *Honesty and risk*
section before trading real money.

---

## What is in the project

| File | Role |
|---|---|
| `quant_signals.py` | Daily engine: download prices, rank the universe, size positions, print the report, save a CSV, and (optionally) place orders via Trading 212. |
| `monthly_report.py` | Reads the saved daily CSVs and execution logs and writes one Markdown report per month. |
| `strategy_review.py` | Offline. Builds a recent-months report and a self-contained prompt you paste into any LLM to get a strategy critique. Never contacts the cloud itself. |
| `README.md` | This file. |

### Folder layout

The scripts read and write under a `Report/` folder that sits next to
them, so they work regardless of the directory you launch them from:

```
Report/
  Daily/       signals_YYYY-MM-DD.csv      (written by quant_signals.py)
  Execution/   executions_YYYY-MM-DD.jsonl (written on --execute)
  Monthly/     report_YYYY-MM.md           (written by monthly_report.py)
  Review/      review_report_*.md,
               llm_review_prompt_*.md      (written by strategy_review.py)
.yf_cache/     yfinance timezone cache
```

You can point the reporting scripts at a different location with `--base`.

---

## Setup

Python 3.10 or newer.

```bash
pip install yfinance pandas numpy      # required for the daily engine
```

Price data comes from Yahoo Finance through `yfinance`; no key is needed
for that. To let the engine read your live positions and cash, and to
place orders, create a Trading 212 API key in the app or web platform
(Settings, then API (Beta), then generate a key). Grant **account data**
and **portfolio** for read access; add **orders:execute** only if you
intend to use `--execute`. Export the credentials in your shell rather
than putting them in the file:

```bash
export T212_API_KEY="..."
export T212_API_SECRET="..."
```

In `quant_signals.py`, set `T212_ENV = "demo"` to use a paper account
(with its own demo key) or `"live"` for the real one. A demo key does not
work against the live host and vice versa.

Verify the connection without running a full report:

```bash
python quant_signals.py --t212-test
```

It prints your positions, cash, and any pending orders. Compare them with
the app before trusting a run.

---

## Daily use

```bash
python quant_signals.py            # report + saved CSV, no orders
python quant_signals.py --execute  # report, then place orders after a typed confirmation
```

What the engine does, in order: keep only names above their 200-day
trend and above a liquidity floor; rank the survivors by a blend of 3-,
6- and 12-month momentum; hold the top `TOP_N`, capped per name and per
sector; size by inverse volatility; and keep about 5% in cash. Existing
positions are only adjusted when they drift more than 3 percentage points
from target, which limits churn and the FX fee it would cost.

The report lists each name with a BUY, SELL or HOLD action and the GBP
amount to trade. Orders are plain market orders placed by GBP value; the
share counts shown are indicative, based on the last close. When you run
with `--execute`, sells are placed first and buys are sized against the
cash that is actually available afterwards, so a rotation does not
overdraw the account.

Timing: run after the US close (from about 21:00 UK in summer) for the
most stable signals; the engine drops a still-forming daily bar so the
output does not depend on the minute you run it. Orders placed while the
US market is closed queue for the next open. Extended-hours filling is
off by default; set `EXTENDED_HOURS = True` to allow pre-market and
after-hours fills, accepting wider spreads on the less liquid names.

Keep positions accurate: with the API configured, holdings and cash are
read from the account each run. Without it, update `HOLDINGS_GBP` and
`CASH_GBP` in the config after every fill.

### A useful check

```bash
python quant_signals.py --backtest
```

runs a rough monthly-rebalance backtest with the same rules. Treat its
numbers with heavy scepticism: the universe was chosen with hindsight, so
the backtest flatters the strategy. See *Honesty and risk*.

---

## Monthly record

```bash
python monthly_report.py             # latest month
python monthly_report.py --months 3  # last three months
```

Each report gives the account value at the start and end of the month,
every position held with an approximate profit or loss, a log of the
BUY/SELL actions (marked `[executed]` when a matching entry exists in the
execution logs, `[proposed only]` otherwise), and totals for proposed
volume and estimated FX fees. Deposits and withdrawals are not tracked
and appear inside the month's value change. The per-position profit is
exact only if every proposed order filled at the signal price, so treat
it as indicative.

Cash is included in the totals only for days whose CSV contains a `CASH`
row. Files saved by newer versions of the engine have it; older files do
not, and the report says so for those months.

---

## Periodic strategy review (offline)

```bash
python strategy_review.py            # last 3 months
python strategy_review.py --months 6
```

This writes two files under `Report/Review/`: a combined recent-month
report, and a single self-contained prompt. Paste the whole prompt file
into any LLM in one message; it already contains the report and the full
`quant_signals.py`. The LLM is asked to either say `NO CHANGE
RECOMMENDED` or return a replacement for the strategy section. You then
edit `quant_signals.py` yourself. The script does not contact any service
and never edits the code for you.

Before pasting a returned section in, you can check it:

```bash
python strategy_review.py --check candidate_strategy.py
```

This accepts either raw section code or a ```` ```python-strategy ````
block, runs a static check (imports limited to numpy/pandas/math/
statistics; no file, network or system access; `select_portfolio`
present), splices it in a temporary copy, and runs a synthetic smoke
test. It reports pass or fail and still does not modify your file.

### The strategy section

`quant_signals.py` contains one clearly marked region:

```
# ========================= STRATEGY SECTION -- BEGIN =========================
...strategy constants and the selection functions...
# ========================== STRATEGY SECTION -- END ==========================
```

This is the only part intended to change during a review. Anything a
replacement must keep is written at the top of that region: the same
constants (other code reads them), a `select_portfolio(close,
dollar_volume)` with the same return structure, imports limited to
numpy/pandas/math/statistics, and no input or output. The `--check`
command enforces these before you commit to a change.

A sensible loop: back up `quant_signals.py`, paste the new section
between the markers, run `--check` (or rely on the backup), run
`--backtest`, and paper-test in the demo environment before trading the
change.

---

## Honesty and risk

- The signals are mechanical. There is no evidence that this class of
  daily rule earns a retail account money after costs.
- The backtest carries strong survivorship and selection bias, because
  the watchlist is today's list of prominent names. Its historical return
  is not an expectation. A realistic prior for this family of strategy,
  on an unbiased universe through full cycles, is modest at best with
  occasional 40 to 50% drawdowns.
- Momentum concentrates into whatever has already run hardest, which is
  also where sharp reversals happen. The per-sector cap and cash buffer
  reduce this but do not remove it.
- The Trading 212 order endpoint is not idempotent: a repeated request
  can create duplicate orders. The engine never retries a failed order
  automatically; if you see a failure, check the app before running
  again.
- Market orders can slip from the last close, more so in extended hours
  and on mid-cap names.
- Corporate actions happen. A ticker that returns no data every day is
  usually delisted or renamed; remove it from `SECTORS`, or add a mapping
  for a duplicate listing (see `T212_SUFFIX1_TICKERS`).
- Keep API keys out of the source files and out of version control. Use a
  read-only key for daily reports and a separate key with execute rights
  only when placing orders.

Nothing here is financial advice. You are responsible for every order the
tool helps you place.