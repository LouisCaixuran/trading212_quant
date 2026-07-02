#!/usr/bin/env python3
"""
quant_signals.py -- daily rule-based (quantitative) signal generator for a
small portfolio of US-listed shares held in a GBP-denominated Trading 212
account.

Pipeline (all parameters in the CONFIG block below):
  1. Liquidity floor : names must have a 60-day median daily traded value of
                       at least MIN_DOLLAR_VOLUME to be selectable.
  2. Trend filter    : only names trading above their 200-day SMA are eligible
                       (120-day fallback for short histories, e.g. recent IPOs).
  3. Momentum rank   : blend of 3-month, 6-month and 12-minus-1-month total
                       returns, cross-sectionally z-scored; the top TOP_N
                       eligible names are kept.
  4. Sizing          : inverse 63-day realised volatility, capped at MAX_WEIGHT
                       per name, then scaled so CASH_BUFFER (5%) of the
                       portfolio is always kept as cash; any further
                       unallocated remainder is also held as cash.
  5. Orders          : full entries/exits trade whenever they exceed
                       MIN_TRADE_GBP; adjustments to existing positions trade
                       only when the weight deviates from target by more than
                       REBALANCE_BAND, so Trading 212's 0.15% FX fee is not
                       bled away on small rebalances.
  6. Execution       : signals are intended as plain market orders placed by
                       GBP amount; no limit prices are produced. The estimated
                       FX fee per order is shown in the report.

Costs modelled (Trading 212 Invest/ISA, GBP base currency, US stocks):
  - commission: zero
  - FX conversion fee: FX_FEE_RATE per side (0.15% at the time of writing)
  - bid-ask spread: rough SPREAD_BPS estimate, used in the backtest only

This is a mechanical screening tool, not investment advice. Daily rule-based
signals on liquid US equities have no demonstrated edge for retail accounts
after spreads, FX fees and taxes. The optional backtest carries survivorship
and selection bias, because the watchlist was chosen with hindsight.

Usage:
  pip install yfinance pandas numpy
  export T212_API_KEY="..."          # auto-load positions/cash (scopes:
  export T212_API_SECRET="..."       #   account + portfolio; add
                                     #   orders:execute only for --execute)
  python quant_signals.py --t212-test  # quick API connectivity check
  python quant_signals.py              # daily signal report + CSV
  python quant_signals.py --backtest   # rough monthly-rebalance backtest
  python quant_signals.py --execute    # report, then place the orders as
                                       # market orders after typed approval

Execution notes: the order endpoint is NOT idempotent, so failed or
timed-out orders are never retried automatically -- check the app before
re-running. Orders placed while the US market is closed queue for the next
open. Test with T212_ENV = "demo" (paper account, own key) before live.
With API credentials set, positions and free cash come from your account
at each run; otherwise update HOLDINGS_GBP and CASH_GBP after every fill.
If Trading 212 does not offer a watchlist ticker, delete it from SECTORS.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- edit this block
# ---------------------------------------------------------------------------

# --- Trading 212 API (optional) -------------------------------------------
# When credentials are present, current positions and free cash are pulled
# from your account automatically at each run, and HOLDINGS_GBP/CASH_GBP
# below become a fallback only. Generate a key in the app or web platform:
# Settings -> API (Beta) -> Generate API key. Grant READ-ONLY permissions
# (account data + portfolio) and leave order permissions off -- this script
# never places orders. Store the credentials in environment variables, not
# in this file. For --execute the key additionally needs the
# orders:execute permission; consider a separate key kept only for that.
#   export T212_API_KEY="..."
#   export T212_API_SECRET="..."
T212_KEY_ENV = "T212_API_KEY"
T212_SECRET_ENV = "T212_API_SECRET"
T212_ENV = "live"       # "live" (real account) or "demo" (paper account)

# Yahoo symbols whose Trading 212 instrument ticker carries a trailing '1'
# before '_US_EQ' because the plain symbol is taken by another listing
# (e.g. SNDK -> SNDK1_US_EQ). Used only when BUYING a name not yet held;
# once held, the account's own reported ticker is used instead. Extend as
# such names appear.
T212_SUFFIX1_TICKERS = {"SNDK"}

# Fallback positions in GBP, used only when the API is not configured or a
# request fails. Keep these up to date if you rely on them.
HOLDINGS_GBP = {
    "LITE": 1024.29,
    "AAOI": 924.52,
    "MCD":  639.99,
    "ORCL": 567.35,
    "LLY":  538.98,
    "MU":   467.50,
    "FIG":  271.23,
}
CASH_GBP = 0.0          # uninvested cash available in the account

# Tech watchlist by area, ~200 US-listed names. All should trade in USD on
# NYSE/Nasdaq (including ADRs) so the single GBP->USD FX fee model applies.
# Verify availability on Trading 212 and delete any ticker it does not offer.
SECTORS = {
    "semiconductors: design & compute": [
        "NVDA", "AMD", "INTC", "QCOM", "AVGO", "MRVL", "NXPI", "ADI", "TXN",
        "MCHP", "ON", "MPWR", "SWKS", "QRVO", "LSCC", "AMBA", "RMBS", "MTSI",
        "ALAB", "CRDO", "SLAB", "SYNA",
    ],
    "semiconductors: equipment, foundry, memory & EDA": [
        "ASML", "TSM", "UMC", "GFS", "ARM", "AMAT", "LRCX", "KLAC", "TER",
        "ENTG", "ONTO", "CAMT", "ACLS", "MKSI", "SNPS", "CDNS", "KEYS",
        "STX", "WDC", "SNDK",
    ],
    "networking, optical & hardware": [
        "CIEN", "ANET", "CSCO", "COHR", "FN", "UI", "FFIV", "EXTR", "HPE",
        "HPQ", "DELL", "SMCI", "VRT", "CLS", "FLEX", "JBL", "NTAP", "P",
        "ZBRA", "LOGI",
    ],
    "software platforms & mega-caps": [
        "MSFT", "GOOGL", "META", "AAPL", "AMZN", "IBM", "SAP", "ADBE", "CRM",
        "NOW", "INTU", "WDAY", "TEAM", "HUBS", "SHOP",
    ],
    "cybersecurity": [
        "PANW", "CRWD", "FTNT", "ZS", "OKTA", "S", "TENB", "QLYS",
        "GEN", "VRNS", "CHKP", "AKAM", "NET", "SAIL",
    ],
    "data, cloud infrastructure & dev tools": [
        "SNOW", "DDOG", "MDB", "ESTC", "GTLB", "TWLO", "PATH", "DT",
        "DOCN", "NTNX", "RBRK", "IOT", "FROG", "MNDY", "ZM", "DBX", "DOCU",
        "PLTR", "AI", "CRWV",
    ],
    "fintech & payments": [
        "V", "MA", "PYPL", "XYZ", "FISV", "FIS", "GPN", "AFRM", "COIN", "HOOD",
        "SOFI", "TOST", "NU", "BILL", "CRCL",
    ],
    "internet, consumer & media": [
        "NFLX", "SPOT", "PINS", "SNAP", "RDDT", "TTD", "APP", "ROKU", "DUOL",
        "MTCH", "EXPE", "BKNG", "TCOM", "EBAY", "ETSY", "CHWY", "MELI", "SE",
        "BABA", "PDD", "JD", "CPNG", "UBER", "ABNB", "DASH",
    ],
    "enterprise, vertical software & IT services": [
        "ACN", "CTSH", "INFY", "EPAM", "GLOB", "IT", "ADP", "PAYX", "PAYC",
        "PCTY", "ADSK", "PTC", "BSY", "TRMB", "DSGX", "ROP", "FICO",
        "TYL", "VEEV", "GWRE", "MANH", "PEGA", "CVLT", "DOCS", "TEM",
        "CACI", "BAH", "LDOS",
    ],
    "gaming & entertainment tech": [
        "EA", "TTWO", "RBLX", "U", "NTES", "BILI", "SONY",
    ],
    "space, robotics & mobility": [
        "TSLA", "RIVN", "ISRG", "SYM", "RKLB", "PL", "AVAV", "KTOS", "ACHR",
        "JOBY", "LUNR",
    ],
    "quantum computing": [
        "IONQ", "RGTI", "QBTS",
    ],
    "energy & power tech": [
        "FSLR", "ENPH", "GEV", "OKLO", "SMR",
    ],
}

# Sector labels for current holdings that are not in the watchlist above.
HOLDING_SECTORS = {
    "LITE": "networking, optical & hardware",
    "AAOI": "networking, optical & hardware",
    "MU":   "semiconductors: equipment, foundry, memory & EDA",
    "ORCL": "software platforms & mega-caps",
    "FIG":  "software platforms & mega-caps",
    "MCD":  "non-tech (held)",
    "LLY":  "non-tech (held)",
}

WATCHLIST = sorted({t for names in SECTORS.values() for t in names})
SECTOR_OF = {t: s for s, names in SECTORS.items() for t in names}
SECTOR_OF.update(HOLDING_SECTORS)
BENCHMARK = "QQQ"       # used only in the backtest, never selected

# Strategy parameters (TOP_N, weight caps, bands, cash buffer, momentum/
# trend windows, liquidity floor) live in the STRATEGY SECTION below.

FX_FEE_RATE = 0.0015    # Trading 212 currency-conversion fee per side (0.15%)
SPREAD_BPS = 3.0        # rough bid-ask spread estimate per side (liquid names)
COST_BPS_PER_SIDE = FX_FEE_RATE * 1e4 + SPREAD_BPS  # backtest cost per side


# --- execution (used only with --execute; see the docstring) ---------------
EXECUTE_QTY_DECIMALS = 2   # rounding for computed share quantities
MAX_ORDER_GBP = 800.0      # refuse execution if any single order exceeds this
MAX_RUN_GBP = 2500.0       # refuse execution if a run's total exceeds this
ORDER_PAUSE = 1.5          # seconds between order placements (50 req/min cap)
EXTENDED_HOURS = False     # False: orders fill only in the regular US
                           # session (14:30-21:00 UK); placed outside it
                           # they queue for the next open. True: orders may
                           # also fill in pre-market/after-hours -- thinner
                           # books, wider spreads, and not every instrument
                           # supports it.

CORR_WINDOW = 60        # window for the pairwise-correlation risk flag
CORR_FLAG = 0.75        # flag pairs above this correlation
SECTOR_FLAG = 0.40      # flag any single area above this target exposure

REBALANCE_EVERY = 21       # backtest rebalance frequency in trading days
HISTORY = "3y"             # price history to download

CHUNK_SIZE = 25            # tickers per download request
CHUNK_PAUSE = 1.5          # seconds between chunk requests
MAX_RETRY_ROUNDS = 2       # retry rounds for tickers returning no data
RETRY_PAUSE = 20.0         # seconds before each retry round
MIN_COVERAGE = 40          # abort if fewer tickers than this have data

UNIVERSE = sorted(set(HOLDINGS_GBP) | set(WATCHLIST) | {BENCHMARK})

# Output layout (relative to this script), matching the repo folders:
#   Report/Daily/     signals_YYYY-MM-DD.csv
#   Report/Execution/ executions_YYYY-MM-DD.jsonl
#   Report/Monthly/   report_YYYY-MM.md  (written by monthly_report.py)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_DIR = os.path.join(_BASE_DIR, "Report", "Daily")
EXEC_DIR = os.path.join(_BASE_DIR, "Report", "Execution")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _us_session_open_now(now=None):
    """True while orders can fill: the regular session (09:30-16:00 ET,
    Mon-Fri), widened to 04:00-20:00 ET (pre-market + after-hours) when
    EXTENDED_HOURS is enabled. The overnight session and US holidays are
    treated as closed -- conservative, so buys are deferred rather than
    placed against proceeds that do not exist yet."""
    try:
        from zoneinfo import ZoneInfo
        ny = now or datetime.now(ZoneInfo("America/New_York"))
        if ny.weekday() >= 5:
            return False
        lo, hi = (((4, 0), (20, 0)) if EXTENDED_HOURS
                  else ((9, 30), (16, 0)))
        return lo <= (ny.hour, ny.minute) < hi
    except Exception:
        return True


def _drop_incomplete_bar(close, volume):
    """If the newest row is dated today in New York and the US session has
    not yet closed (before 16:15 ET), drop it: its prices are intraday and
    would make signals depend on the time of day the script is run."""
    try:
        from zoneinfo import ZoneInfo
        ny = datetime.now(ZoneInfo("America/New_York"))
        if (close.index[-1].date() == ny.date()
                and (ny.hour, ny.minute) < (16, 15)):
            print("note: dropped today's incomplete bar (US market still "
                  "open); signals use the last completed close")
            return close.iloc[:-1], volume.iloc[:-1]
    except Exception:
        pass
    return close, volume


def _fix_ca_bundle():
    """Work around broken conda SSL setups (curl error 77: 'error setting
    certificate verify locations') by pointing curl/requests at certifi's
    CA bundle whenever the configured one is missing."""
    try:
        import certifi
        pem = certifi.where()
    except Exception:
        return
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        cur = os.environ.get(var)
        if not cur or not os.path.exists(cur):
            os.environ[var] = pem


def _extract(data, tickers):
    """Return (close, volume) from a yf.download result, handling both
    single- and multi-ticker column layouts."""
    if data is None or len(data) == 0:
        return None, None
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"], data["Volume"]
    c = data[["Close"]].rename(columns={"Close": tickers[0]})
    v = data[["Volume"]].rename(columns={"Volume": tickers[0]})
    return c, v


def fetch_fx():
    """Fetch the current GBPUSD rate (USD per 1 GBP) with retries."""
    _fix_ca_bundle()
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("yfinance is required: pip install yfinance") from exc
    fx = np.nan
    for _ in range(3):
        try:
            fxd = yf.download("GBPUSD=X", period="10d", interval="1d",
                              auto_adjust=True, progress=False)
            fx = float(np.asarray(fxd["Close"].dropna()).ravel()[-1])
            break
        except Exception:
            time.sleep(5.0)
    if not np.isfinite(fx):
        raise SystemExit("could not download the GBPUSD rate; rerun shortly")
    return fx


def fetch_prices():
    """Download adjusted daily closes and dollar volume for the universe,
    plus the GBPUSD rate.

    Yahoo throttles bursts of requests, and yfinance's threaded downloader
    can exhaust DNS threads and its own sqlite cache when ~200 tickers are
    requested in one call (symptoms: spurious 'possibly delisted' errors,
    'unable to open database file', curl getaddrinfo failures). Downloads
    are therefore made in small throttled chunks, with sequential retry
    rounds for anything that returns no data. Expect the full pass to take
    a couple of minutes.
    """
    _fix_ca_bundle()
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("yfinance is required: pip install yfinance") from exc

    # keep yfinance's timezone cache in a local, writable folder
    try:
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             ".yf_cache")
        os.makedirs(cache, exist_ok=True)
        yf.set_tz_cache_location(cache)
    except Exception:
        pass

    # warm-up: one small request initialises the cache before any burst
    try:
        yf.download(BENCHMARK, period="5d", progress=False, auto_adjust=True)
    except Exception:
        pass

    got_close: dict[str, pd.Series] = {}
    got_vol: dict[str, pd.Series] = {}

    def _absorb(chunk, threads):
        try:
            data = yf.download(chunk, period=HISTORY, interval="1d",
                               auto_adjust=True, group_by="column",
                               progress=False, threads=threads)
        except Exception as exc:
            print(f"  chunk of {len(chunk)} failed "
                  f"({type(exc).__name__}); will retry")
            return
        c, v = _extract(data, chunk)
        if c is None:
            return
        for col in c.columns:
            if col not in got_close and c[col].notna().sum() > 0:
                got_close[col] = c[col]
                got_vol[col] = v[col]

    todo = list(UNIVERSE)
    n_chunks = (len(todo) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(todo), CHUNK_SIZE):
        print(f"downloading chunk {i // CHUNK_SIZE + 1}/{n_chunks} ...")
        _absorb(todo[i:i + CHUNK_SIZE], threads=False)
        time.sleep(CHUNK_PAUSE)

    missing = [t for t in UNIVERSE if t not in got_close]
    for round_ in range(1, MAX_RETRY_ROUNDS + 1):
        if not missing:
            break
        print(f"retrying {len(missing)} tickers with no data "
              f"(round {round_}/{MAX_RETRY_ROUNDS}, sequential) ...")
        time.sleep(RETRY_PAUSE)
        for i in range(0, len(missing), CHUNK_SIZE):
            _absorb(missing[i:i + CHUNK_SIZE], threads=False)
            time.sleep(CHUNK_PAUSE)
        missing = [t for t in UNIVERSE if t not in got_close]

    if len(got_close) < MIN_COVERAGE:
        raise SystemExit(
            f"only {len(got_close)} tickers downloaded -- Yahoo appears to "
            f"be rate-limiting this connection; wait a few minutes and rerun")
    if missing:
        print(f"warning: still no data for {missing} after retries -- "
              f"dropping them for today. A ticker that fails every day is "
              f"likely delisted or renamed; delete it from SECTORS.")

    close = pd.DataFrame(got_close).sort_index()
    volume = pd.DataFrame(got_vol).sort_index()
    close, volume = _drop_incomplete_bar(close, volume)
    dollar_volume = close * volume
    return close, dollar_volume, fetch_fx()

# ---------------------------------------------------------------------------
# Trading 212 account state (endpoints per the official OpenAPI spec)
# ---------------------------------------------------------------------------

_T212_TRANSPORT = None   # test hook: fn(method, path, payload) -> parsed JSON


def _t212_call(method, path, payload=None):
    """Authenticated Trading 212 API request (HTTP Basic: key as username,
    secret as password). Raises on any HTTP or network error. Order
    placement is NOT idempotent, so callers must never retry POSTs."""
    if _T212_TRANSPORT is not None:
        return _T212_TRANSPORT(method, path, payload)
    import base64
    import json as _json
    import urllib.error as _er
    import urllib.request as _rq
    key = os.environ.get(T212_KEY_ENV, "").strip()
    secret = os.environ.get(T212_SECRET_ENV, "").strip()
    if not key or not secret:
        raise RuntimeError("Trading 212 credentials not set")
    base = ("https://live.trading212.com" if T212_ENV == "live"
            else "https://demo.trading212.com")
    data = _json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": "Basic " + base64.b64encode(
        f"{key}:{secret}".encode()).decode()}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = _rq.Request(base + path, data=data, headers=headers, method=method)
    try:
        with _rq.urlopen(req, timeout=30) as resp:
            return _json.loads(resp.read().decode())
    except _er.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code} on {path}: {body}") from None


def _t212_to_yahoo(ticker: str):
    """Map a Trading 212 instrument ticker to its Yahoo symbol. Trading 212
    appends a '1' to the stem when the plain symbol is taken by another
    listing (e.g. 'SNDK1_US_EQ' is SanDisk), so a single trailing '1' is
    stripped. Returns None for non-US instruments.
    Note: a US symbol that genuinely ended in '1' would be mis-stripped;
    none in the current watchlist does."""
    if ticker.endswith("_US_EQ"):
        ticker = ticker[: -len("_US_EQ")]
        if ticker.endswith("1"):
            ticker = ticker[:-1]
        return ticker
    return None


def _yahoo_to_t212(yahoo: str, state=None):
    """Best Trading 212 instrument ticker for a Yahoo symbol: the account's
    own reported ticker when the position is held, else '<SYM>1_US_EQ' for
    the known duplicate-listing names in T212_SUFFIX1_TICKERS, else the
    default '<SYM>_US_EQ'."""
    if state and yahoo in state.get("t212_ticker", {}):
        return state["t212_ticker"][yahoo]
    if yahoo in T212_SUFFIX1_TICKERS:
        return f"{yahoo}1_US_EQ"
    return f"{yahoo}_US_EQ"


def fetch_t212_state(fx):
    """Fetch open positions (GET /equity/positions) and cash
    (GET /equity/account/summary). Position `currentPrice` is quoted in the
    instrument currency (USD here) per the spec, so values convert to GBP
    at the live rate. Returns a state dict, or None when credentials are
    absent or a request fails (the config constants are then used)."""
    have_creds = (os.environ.get(T212_KEY_ENV, "").strip()
                  and os.environ.get(T212_SECRET_ENV, "").strip())
    if not have_creds and _T212_TRANSPORT is None:
        print(f"note: {T212_KEY_ENV} / {T212_SECRET_ENV} are not set in this "
              f"shell, so positions come from the config constants; export "
              f"both variables to use the Trading 212 API")
        return None
    try:
        positions = _t212_call("GET", "/api/v0/equity/positions")
        time.sleep(1.2)          # positions: 1 req/1s; summary: 1 req/5s
        summary = _t212_call("GET", "/api/v0/equity/account/summary")
    except Exception as exc:
        msg = str(exc)
        if "401" in msg:
            hint = (" | 401 = bad key: check both variables are exported in "
                    f"this shell and that T212_ENV ('{T212_ENV}') matches "
                    "the environment the key was generated in -- demo keys "
                    "only work against the demo host")
        elif "403" in msg:
            hint = (" | 403 = the key lacks a permission: enable the "
                    "'account data' and 'portfolio' scopes (and "
                    "'orders:execute' if using --execute)")
        else:
            hint = ""
        print(f"warning: Trading 212 API request failed ({exc}){hint}; "
              f"using HOLDINGS_GBP/CASH_GBP from the config instead")
        return None

    currency = str(summary.get("currency") or "")
    if currency and currency != "GBP":
        print(f"warning: account currency is {currency}, but this script's "
              f"FX handling assumes a GBP account; using config constants")
        return None

    holdings, qty, sellable, pie_qty = {}, {}, {}, {}
    t212_ticker_of = {}
    skipped = []
    for p in positions or []:
        inst = p.get("instrument") or {}
        t212_ticker = str(inst.get("ticker") or "")
        y = _t212_to_yahoo(t212_ticker)
        if y is None or str(inst.get("currency") or "USD") != "USD":
            skipped.append(t212_ticker or "?")
            continue
        q = float(p.get("quantity") or 0.0)
        px = float(p.get("currentPrice") or 0.0)
        if q <= 0 or px <= 0:
            continue
        t212_ticker_of.setdefault(y, t212_ticker)
        holdings[y] = holdings.get(y, 0.0) + q * px / fx   # USD -> GBP
        qty[y] = qty.get(y, 0.0) + q
        avail = p.get("quantityAvailableForTrading")
        sellable[y] = sellable.get(y, 0.0) + float(q if avail is None
                                                   else avail)
        in_pies = float(p.get("quantityInPies") or 0.0)
        if in_pies:
            pie_qty[y] = pie_qty.get(y, 0.0) + in_pies
    if skipped:
        print(f"note: ignoring non-US instruments from Trading 212: {skipped}")

    cash_gbp = float(((summary.get("cash") or {})
                      .get("availableToTrade")) or 0.0)
    print(f"Trading 212 ({T212_ENV}): {len(holdings)} US positions, cash "
          f"available to trade {cash_gbp:,.2f} {currency or 'GBP'}")
    return {"holdings_gbp": holdings, "cash_gbp": cash_gbp, "qty": qty,
            "sellable_qty": sellable, "pie_qty": pie_qty,
            "t212_ticker": t212_ticker_of}

# ========================= STRATEGY SECTION -- BEGIN =========================
# Everything between the BEGIN/END markers defines the trading strategy and
# is the ONLY region that the automated review (strategy_review.py) may
# replace. Contract for any replacement:
#   * may import only numpy / pandas / math / statistics
#   * must define every constant in this section (other code reads them)
#   * must define select_portfolio(close, dollar_volume) returning
#     (weights, diag): weights is a pd.Series of non-negative target
#     weights over tickers in close.columns (excluding BENCHMARK),
#     summing to <= 1.0; diag is a dict with keys mom_z, trend_ok, dist,
#     vol, liq_ok (Series), rank (dict), skipped_area (dict),
#     n_ranked (int), fallback (Series)
#   * no I/O, no network access, no reference to the trading/execution code

TOP_N = 8               # number of names in the target portfolio
MAX_PER_AREA = 3        # max names from any single area among the picks
                        # (guards against the whole portfolio chasing one theme)
MAX_WEIGHT = 0.15       # cap per name (fraction of total portfolio)
CASH_BUFFER = 0.05      # fraction of the portfolio always kept as cash, to
                        # absorb price moves, slippage and FX fees between
                        # signal time and execution
MIN_TRADE_GBP = 40.0    # ignore entries/exits smaller than this
REBALANCE_BAND = 0.03   # adjust an existing position only when its weight
                        # deviates from target by > 3 percentage points

MIN_DOLLAR_VOLUME = 20e6  # min 60-day median daily traded value (USD)

TREND_SMA = 200         # main trend filter window (trading days)
FALLBACK_SMA = 120      # used when a name has < TREND_SMA observations
MOM_WINDOWS = (63, 126)         # 3m and 6m momentum lookbacks
MOM_LONG = (252, 21)            # 12-month momentum, skipping the last month
VOL_WINDOW = 63         # realised-volatility window for sizing


def cross_sectional_z(col: pd.Series) -> pd.Series:
    v = col.dropna()
    if len(v) < 3:
        return pd.Series(np.nan, index=col.index)
    s = v.std(ddof=0)
    if not np.isfinite(s) or s == 0:
        return (col - v.mean()) * 0.0
    return (col - v.mean()) / s

# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def capped_inverse_vol(vol: pd.Series, cap: float) -> pd.Series:
    """Inverse-volatility weights with a per-name cap. May sum to < 1
    (the remainder is cash) if the cap binds on every name."""
    iv = 1.0 / vol.replace(0.0, np.nan)
    iv = iv.fillna(iv.mean() if iv.notna().any() else 1.0)
    w = iv / iv.sum()
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = w < cap - 1e-12
        if not under.any() or excess <= 0:
            break
        w.loc[under] = w.loc[under] + excess * (w.loc[under] / w.loc[under].sum())
    return w.clip(upper=cap)


def select_portfolio(close: pd.DataFrame, dollar_volume: pd.DataFrame | None):
    """Apply liquidity floor -> trend filter -> momentum ranking ->
    inverse-vol sizing to the data as of the last row of `close`.
    Returns (weights, diagnostics)."""
    last = close.iloc[-1]
    obs = close.notna().sum()

    if dollar_volume is not None:
        med_dv = (dollar_volume.rolling(60, min_periods=20)
                               .median().iloc[-1])
        liq_ok = med_dv >= MIN_DOLLAR_VOLUME  # NaN compares as False
    else:
        liq_ok = pd.Series(True, index=close.columns)

    sma_main = close.rolling(TREND_SMA, min_periods=TREND_SMA).mean().iloc[-1]
    sma_fb = close.rolling(FALLBACK_SMA, min_periods=FALLBACK_SMA).mean().iloc[-1]
    use_fb = sma_main.isna() & (obs >= FALLBACK_SMA)
    sma_used = sma_main.where(~use_fb, sma_fb)
    trend_ok = (last > sma_used).fillna(False)
    dist = last / sma_used - 1.0

    comps = {}
    for w in MOM_WINDOWS:
        comps[f"r{w}"] = (close / close.shift(w)).iloc[-1] - 1.0
    lw, skip = MOM_LONG
    comps[f"r{lw}_{skip}"] = (close.shift(skip) / close.shift(lw)).iloc[-1] - 1.0
    comp_df = pd.DataFrame(comps)
    z = comp_df.apply(cross_sectional_z, axis=0)
    mom_z = z.mean(axis=1)  # NaN components are skipped per name

    logret = np.log(close / close.shift(1))
    vol = (logret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2)
                 .std().iloc[-1] * np.sqrt(252.0))

    selectable = [t for t in close.columns
                  if t != BENCHMARK
                  and bool(liq_ok.get(t, False))
                  and bool(trend_ok.get(t, False))
                  and np.isfinite(mom_z.get(t, np.nan))]
    ranked = sorted(selectable, key=lambda t: float(mom_z[t]), reverse=True)
    picks, area_count, skipped_area = [], {}, {}
    for t in ranked:
        a = SECTOR_OF.get(t, "other")
        if area_count.get(a, 0) >= MAX_PER_AREA:
            skipped_area[t] = a
            continue
        picks.append(t)
        area_count[a] = area_count.get(a, 0) + 1
        if len(picks) == TOP_N:
            break

    if picks:
        v = vol[picks].copy()
        v = v.fillna(float(v.mean()) if v.notna().any() else 1.0)
        weights = capped_inverse_vol(v, MAX_WEIGHT) * (1.0 - CASH_BUFFER)
    else:
        weights = pd.Series(dtype=float)

    diag = {
        "mom_z": mom_z, "trend_ok": trend_ok, "dist": dist, "vol": vol,
        "liq_ok": liq_ok,
        "rank": {t: i + 1 for i, t in enumerate(ranked)},
        "skipped_area": skipped_area,
        "n_ranked": len(ranked), "fallback": use_fb,
    }
    return weights, diag

# ========================== STRATEGY SECTION -- END ==========================

# ---------------------------------------------------------------------------
# Daily report
# ---------------------------------------------------------------------------

def build_report(close, dollar_volume, fx, holdings_gbp, cash_gbp):
    weights, diag = select_portfolio(close, dollar_volume)
    last = close.iloc[-1]
    total = float(sum(holdings_gbp.values()) + cash_gbp)

    names = sorted(set(weights.index) | set(holdings_gbp))
    rows = []
    for t in names:
        w = float(weights.get(t, 0.0))
        cur = float(holdings_gbp.get(t, 0.0))
        cur_w = cur / total if total > 0 else 0.0
        tgt = w * total
        diff = tgt - cur
        price = float(last.get(t, np.nan))
        mz = float(diag["mom_z"].get(t, np.nan))
        shares = np.nan
        diff_exec = 0.0
        reason = ""

        if not np.isfinite(price):
            action = "NO DATA"
            reason = "no price data -- if newly held, add the ticker to SECTORS"
        elif w == 0.0 and cur > 0.0:                    # full exit
            if cur >= MIN_TRADE_GBP:
                action, diff_exec = "SELL", diff
            else:
                action, reason = "HOLD", "residual below min trade size"
        elif cur == 0.0 and tgt > 0.0:                  # new position
            if tgt >= MIN_TRADE_GBP:
                action, diff_exec = "BUY", diff
            else:
                action, reason = "HOLD", "target below min trade size"
        elif abs(cur_w - w) > REBALANCE_BAND and abs(diff) >= MIN_TRADE_GBP:
            action = "BUY" if diff > 0 else "SELL"      # adjust existing
            diff_exec = diff
        else:
            action = "HOLD"
            if w > 0.0 and cur > 0.0:
                reason = f"within {100 * REBALANCE_BAND:.0f}pp rebalance band"

        if action in ("BUY", "SELL"):
            shares = abs(diff_exec) * fx / price
        fee = abs(diff_exec) * FX_FEE_RATE

        notes = []
        if w == 0.0 and cur > 0 and action != "NO DATA":
            if not diag["liq_ok"].get(t, True):
                notes.append("below liquidity floor")
            elif not diag["trend_ok"].get(t, False):
                if np.isfinite(diag["dist"].get(t, np.nan)):
                    notes.append("below trend SMA")
                else:
                    notes.append("insufficient history for trend filter")
            elif t in diag["skipped_area"]:
                notes.append(f"area cap: already {MAX_PER_AREA} picks in "
                             f"'{diag['skipped_area'][t]}'")
            elif t in diag["rank"]:
                notes.append(f"momentum rank {diag['rank'][t]}/{diag['n_ranked']}")
            else:
                notes.append("insufficient history for momentum")
        if bool(diag["fallback"].get(t, False)):
            notes.append(f"short history: {FALLBACK_SMA}d SMA used")
        if reason:
            notes.insert(0, reason)

        rows.append({
            "ticker": t, "action": action,
            "current_gbp": round(cur, 2),
            "target_pct": round(100 * w, 2),
            "target_gbp": round(tgt, 2),
            "trade_gbp": round(diff_exec, 2),
            "fx_fee_gbp": round(fee, 2),
            "shares": round(shares, 4) if np.isfinite(shares) else "",
            "last_usd": round(price, 2) if np.isfinite(price) else "",
            "mom_z": round(mz, 2) if np.isfinite(mz) else "",
            "trend": "Y" if diag["trend_ok"].get(t, False) else "N",
            "area": SECTOR_OF.get(t, "other"),
            "note": "; ".join(notes),
        })

    df = pd.DataFrame(rows)
    order = pd.CategoricalIndex(df["action"],
                                categories=["BUY", "SELL", "HOLD", "NO DATA"])
    df = df.assign(_o=order.codes).sort_values(["_o", "ticker"]).drop(columns="_o")

    cash_target = (1.0 - float(weights.sum())) * total

    sector_w: dict[str, float] = {}
    for t, wt in weights.items():
        s = SECTOR_OF.get(t, "other")
        sector_w[s] = sector_w.get(s, 0.0) + float(wt)
    sector_target = pd.Series(sector_w).sort_values(ascending=False)

    flags = []
    picks = list(weights.index)
    if len(picks) >= 2:
        rc = close[picks].pct_change().tail(CORR_WINDOW).corr()
        for i, a_ in enumerate(picks):
            for b_ in picks[i + 1:]:
                cv = float(rc.loc[a_, b_])
                if np.isfinite(cv) and cv > CORR_FLAG:
                    flags.append(f"{a_}-{b_}: {CORR_WINDOW}d return correlation "
                                 f"{cv:.2f} -- these are close to one bet")
    for s, wt in sector_target.items():
        if wt > SECTOR_FLAG:
            flags.append(f"'{s}' target exposure {wt:.0%} -- concentrated area")
    if not picks:
        flags.append("no name passes the filters today; target is 100% cash")

    uni = [t for t in close.columns if t != BENCHMARK]
    meta = {
        "n_universe": len(uni),
        "n_liquid": int(diag["liq_ok"].reindex(uni).fillna(False).sum()),
        "n_trend": int((diag["liq_ok"].reindex(uni).fillna(False)
                        & diag["trend_ok"].reindex(uni).fillna(False)).sum()),
        "n_ranked": diag["n_ranked"],
        "fees_gbp": float(df["fx_fee_gbp"].sum()),
    }
    return df, weights, cash_target, flags, sector_target, meta


def print_report(df, weights, cash_target, flags, sector_target, meta, fx,
                 close, holdings_gbp, cash_gbp, source):
    last_date = close.index[-1].date()
    total = float(sum(holdings_gbp.values()) + cash_gbp)
    print("=" * 78)
    print(f"Daily signal report | data through {last_date} | GBPUSD = {fx:.4f}")
    print(f"Portfolio value used: GBP {total:,.2f} "
          f"(holdings {sum(holdings_gbp.values()):,.2f} + cash {cash_gbp:,.2f})"
          f" | source: {source}")
    print(f"Universe: {meta['n_universe']} names | liquid: {meta['n_liquid']} "
          f"| liquid & above trend: {meta['n_trend']} "
          f"| ranked: {meta['n_ranked']} | held: top {TOP_N}")
    print("=" * 78)

    age = (datetime.now(timezone.utc).date() - last_date).days
    if age > 4:
        print(f"WARNING: last price bar is {age} days old; data may be stale.\n")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.to_string(index=False))

    print(f"\nTarget cash buffer: GBP {cash_target:,.2f} "
          f"({100 * cash_target / total:.1f}% of portfolio)")
    print(f"Estimated FX fees for today's orders: GBP {meta['fees_gbp']:,.2f} "
          f"({FX_FEE_RATE:.2%} per side)")

    if len(sector_target):
        print("\nTarget exposure by area:")
        for s, wt in sector_target.items():
            print(f"  {wt:6.1%}  {s}")

    if flags:
        print("\nRisk flags:")
        for f in flags:
            print(f"  - {f}")

    print("\nReminders: execute as market orders by the GBP amount in trade_gbp"
          "\n(the shares column is indicative, based on the last close);"
          "\ncheck earnings dates before trading (the script does not);"
          "\nupdate HOLDINGS_GBP after fills if not using the Trading 212 API."
          "\nMechanical rule output only -- not investment advice.")

# ---------------------------------------------------------------------------
# Execution via the Trading 212 API (opt-in, --execute)
# ---------------------------------------------------------------------------

def plan_orders(df, last_usd, fx, state):
    """Turn the day's BUY/SELL rows into a market-order plan, sells first.
    Full exits sell the exact tradable quantity reported by the API; all
    other orders use quantities computed from the last close."""
    plan = []
    for _, r in df.iterrows():
        if r["action"] not in ("BUY", "SELL"):
            continue
        t = str(r["ticker"])
        gbp = float(r["trade_gbp"])
        px = float(last_usd.get(t, np.nan))
        if not np.isfinite(px) or px <= 0:
            print(f"skip {t}: no price available to compute a quantity")
            continue
        full_exit = (r["action"] == "SELL" and float(r["target_pct"]) == 0.0)
        if full_exit and t in state["sellable_qty"]:
            q = -round(float(state["sellable_qty"][t]), 8)
            in_pies = float(state["pie_qty"].get(t, 0.0))
            if in_pies:
                print(f"note: {t}: {in_pies} shares sit inside pies and "
                      f"cannot be sold by this order")
        else:
            q = round(abs(gbp) * fx / px, EXECUTE_QTY_DECIMALS)
            if q <= 0:
                continue
            if gbp < 0:
                # never sell more than the API says is tradable
                q = -min(q, float(state["sellable_qty"].get(t, q)))
        if q == 0:
            continue
        plan.append({"ticker": t, "t212_ticker": _yahoo_to_t212(t, state),
                     "quantity": q, "est_gbp": round(abs(gbp), 2)})
    plan.sort(key=lambda o: o["quantity"] >= 0)   # sells before buys
    return plan


def execute_orders(plan, assume_yes=False):
    """Place the planned market orders. Safety properties: hard per-order
    and per-run GBP caps; an explicit typed confirmation gate; sells are
    placed first, then cash is re-checked and buys scaled down to fit;
    every request/response is appended to an audit log; and nothing is
    ever retried, because the order endpoint is not idempotent."""
    if not plan:
        print("\nNothing to execute today.")
        return
    total = sum(o["est_gbp"] for o in plan)
    if any(o["est_gbp"] > MAX_ORDER_GBP for o in plan) or total > MAX_RUN_GBP:
        print(f"\nEXECUTION REFUSED: caps exceeded (per-order cap "
              f"{MAX_ORDER_GBP:,.0f}, run cap {MAX_RUN_GBP:,.0f}, planned "
              f"total {total:,.2f}). Raise the caps in CONFIG deliberately "
              f"if this is intended. No orders were placed.")
        return

    print(f"\nPlanned market orders ({T212_ENV} environment, sells first):")
    for o in plan:
        side = "SELL" if o["quantity"] < 0 else "BUY"
        print(f"  {side:4s} {o['ticker']:6s} qty {abs(o['quantity']):>12.4f}"
              f"  (~GBP {o['est_gbp']:,.2f})")
    print(f"  planned total ~GBP {total:,.2f}")
    print("  Market orders can slip from the last close; orders placed "
          "while the market is closed queue for the next open.")
    if not assume_yes:
        answer = input("Type EXECUTE to place these orders "
                       "(anything else aborts): ")
        if answer.strip() != "EXECUTE":
            print("Aborted; no orders placed.")
            return

    import json as _json
    os.makedirs(EXEC_DIR, exist_ok=True)
    log_path = os.path.join(
        EXEC_DIR, f"executions_{datetime.now(timezone.utc).date()}.jsonl")

    def _log(entry):
        with open(log_path, "a") as fh:
            fh.write(_json.dumps(entry) + "\n")

    def _place(order):
        payload = {"ticker": order["t212_ticker"],
                   "quantity": order["quantity"],
                   "extendedHours": EXTENDED_HOURS}
        try:
            resp = _t212_call("POST", "/api/v0/equity/orders/market", payload)
            print(f"  placed {order['ticker']}: id {resp.get('id')}, "
                  f"status {resp.get('status')}")
            _log({"ts": datetime.now(timezone.utc).isoformat(),
                  "yahoo_ticker": order["ticker"],
                  "est_gbp": order["est_gbp"],
                  "request": payload, "response": resp})
            return resp
        except Exception as exc:
            print(f"  FAILED {order['ticker']}: {exc}\n"
                  f"    not retried (endpoint is not idempotent) -- check "
                  f"pending orders in the app before running again")
            _log({"ts": datetime.now(timezone.utc).isoformat(),
                  "yahoo_ticker": order["ticker"],
                  "est_gbp": order["est_gbp"],
                  "request": payload, "error": str(exc)})
            return None

    sells = [o for o in plan if o["quantity"] < 0]
    buys = [o for o in plan if o["quantity"] > 0]

    if buys and sells and not _us_session_open_now():
        window = ("extended session (09:00 UK to 01:00 UK)" if EXTENDED_HOURS
                  else "regular session (14:30-21:00 UK)")
        print(f"\nUS session is closed: SELL orders are placed now and will "
              f"queue, but the BUY orders are deferred because sale "
              f"proceeds do not exist yet. Run --execute again during the "
              f"{window} to place the buys with actual proceeds.")
        buys = []

    sell_ids = []
    for o in sells:
        resp = _place(o)
        if resp and resp.get("id") is not None:
            sell_ids.append(resp["id"])
        time.sleep(ORDER_PAUSE)

    if buys:
        if sell_ids:
            # wait until the sells have left the pending queue (max ~30s)
            for _ in range(5):
                time.sleep(6.0)      # pending-orders endpoint: 1 req/5s
                try:
                    pending = _t212_call("GET", "/api/v0/equity/orders")
                    left = ({o.get("id") for o in pending or []}
                            & set(sell_ids))
                except Exception:
                    break
                if not left:
                    break
            else:
                print("note: some sells are still pending; buys will be "
                      "scaled against currently available cash")
        avail = None
        try:
            summary = _t212_call("GET", "/api/v0/equity/account/summary")
            avail = float(((summary.get("cash") or {})
                           .get("availableToTrade")) or 0.0)
        except Exception as exc:
            print(f"warning: could not re-check cash ({exc}); "
                  f"buys proceed as planned and may be rejected if short")
        planned = sum(o["est_gbp"] for o in buys)
        if avail is not None and planned > avail * 0.995 and planned > 0:
            scale = max(0.0, avail * 0.995 / planned)
            print(f"note: scaling buys to {scale:.1%} of plan to fit "
                  f"available cash GBP {avail:,.2f}")
            for o in buys:
                o["quantity"] = round(o["quantity"] * scale,
                                      EXECUTE_QTY_DECIMALS)
        for o in buys:
            if o["quantity"] > 0:
                _place(o)
                time.sleep(ORDER_PAUSE)

    print(f"\nExecution attempt finished. Audit log: {log_path}"
          f"\nVerify fills in the Trading 212 app before trusting "
          f"tomorrow's report.")

# ---------------------------------------------------------------------------
# Rough backtest (monthly rebalance, same rules)
# ---------------------------------------------------------------------------

def run_backtest(close, dollar_volume):
    idx = close.index
    warm = max(TREND_SMA + 30, MOM_LONG[0] + MOM_LONG[1] + 5)
    if len(idx) <= warm + REBALANCE_EVERY:
        raise SystemExit("not enough history for a backtest; increase HISTORY")

    uni = [c for c in close.columns if c != BENCHMARK]
    t = warm
    prev = pd.Series(dtype=float)
    eq, ew, bm, dts = [1.0], [1.0], [1.0], [idx[t]]

    while t < len(idx) - 1:
        dv = dollar_volume.iloc[: t + 1] if dollar_volume is not None else None
        w, _ = select_portfolio(close.iloc[: t + 1], dv)
        nxt = min(t + REBALANCE_EVERY, len(idx) - 1)
        pr = close.iloc[nxt] / close.iloc[t] - 1.0

        gross = float((w * pr.reindex(w.index)).fillna(0.0).sum()) if len(w) else 0.0
        turnover = float(w.subtract(prev, fill_value=0.0).abs().sum())
        eq.append(eq[-1] * (1.0 + gross - turnover * COST_BPS_PER_SIDE / 1e4))

        ew.append(ew[-1] * (1.0 + float(pr[uni].mean())))
        b = float(pr.get(BENCHMARK, np.nan))
        bm.append(bm[-1] * (1.0 + (b if np.isfinite(b) else 0.0)))

        dts.append(idx[nxt])
        prev = w
        t = nxt

    per_year = 252.0 / REBALANCE_EVERY

    def stats(e):
        e = np.asarray(e, dtype=float)
        yrs = (dts[-1] - dts[0]).days / 365.25
        r = e[1:] / e[:-1] - 1.0
        cagr = e[-1] ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan
        vol = r.std(ddof=0) * np.sqrt(per_year)
        sharpe = (r.mean() * per_year) / vol if vol > 0 else np.nan
        maxdd = float((1.0 - e / np.maximum.accumulate(e)).max())
        return cagr, vol, sharpe, maxdd

    print("=" * 78)
    print(f"Backtest {dts[0].date()} -> {dts[-1].date()} | rebalance every "
          f"{REBALANCE_EVERY} trading days | costs {COST_BPS_PER_SIDE:.0f} bps/side "
          f"({FX_FEE_RATE:.2%} FX + {SPREAD_BPS:.0f} bps spread)")
    print("=" * 78)
    for name, series in [("Strategy", eq),
                         ("Equal-weight universe", ew),
                         (f"{BENCHMARK} buy & hold", bm)]:
        c, v, s, d = stats(series)
        print(f"{name:24s}  CAGR {c:7.2%}  vol {v:6.2%}  "
              f"Sharpe {s:5.2f}  maxDD {d:6.2%}")
    print("\nCaveats: survivorship/selection bias (watchlist chosen with"
          "\nhindsight), no dividends withheld/taxes, coarse cost model, and"
          "\nnames are dropped from returns while their data is missing."
          "\nTreat this as a sanity check, not an expected return.")

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backtest", action="store_true",
                    help="run a rough monthly-rebalance backtest instead of "
                         "the daily report")
    ap.add_argument("--execute", action="store_true",
                    help="after the report, place the BUY/SELL orders as "
                         "market orders via the Trading 212 API (asks for "
                         "typed confirmation; requires an API key with the "
                         "orders:execute scope)")
    ap.add_argument("--t212-test", action="store_true",
                    help="only test Trading 212 API connectivity: fetch "
                         "positions and cash, print them, and exit")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive confirmation when executing "
                         "(not recommended)")
    args = ap.parse_args()

    if args.t212_test:
        key = os.environ.get(T212_KEY_ENV, "")
        secret = os.environ.get(T212_SECRET_ENV, "")
        print(f"T212_ENV = '{T212_ENV}' | {T212_KEY_ENV}: "
              f"{'set (' + key[:4] + '..., ' + str(len(key)) + ' chars)' if key else 'NOT SET'}"
              f" | {T212_SECRET_ENV}: "
              f"{'set (' + str(len(secret)) + ' chars)' if secret else 'NOT SET'}")
        if not key or not secret:
            raise SystemExit("export both variables in this shell, then "
                             "rerun: export T212_API_KEY=... ; "
                             "export T212_API_SECRET=...")
        fx = fetch_fx()
        state = fetch_t212_state(fx)
        if state is None:
            raise SystemExit("connectivity test failed -- see the warning "
                             "above")
        print(f"\nGBPUSD = {fx:.4f}")
        for t in sorted(state["holdings_gbp"]):
            print(f"  {t:6s} qty {state['qty'][t]:>12.6f}  "
                  f"~GBP {state['holdings_gbp'][t]:>10,.2f}")
        total = sum(state["holdings_gbp"].values())
        print(f"  cash available to trade: GBP {state['cash_gbp']:,.2f}"
              f"\n  holdings total ~GBP {total:,.2f} | account total "
              f"~GBP {total + state['cash_gbp']:,.2f}")
        try:
            time.sleep(1.0)
            pending = _t212_call("GET", "/api/v0/equity/orders")
            if pending:
                print("\nPending orders (queued or working):")
                for o in pending:
                    print(f"  id {o.get('id')}  {str(o.get('type', '?')):8s}"
                          f"{str(o.get('ticker', '?')):16s}"
                          f"qty {o.get('quantity')}  "
                          f"status {o.get('status')}")
            else:
                print("\nPending orders: none")
        except Exception as exc:
            print(f"\n(could not read pending orders: {exc})")
        print("\nCompare these against the app; if they match, the API "
              "is working and daily runs will use it automatically.")
        return

    close, dollar_volume, fx = fetch_prices()

    if args.backtest:
        run_backtest(close, dollar_volume)
        return

    state = fetch_t212_state(fx)
    if state is not None:
        holdings_gbp, cash_gbp = state["holdings_gbp"], state["cash_gbp"]
        source = f"Trading 212 API ({T212_ENV})"
    else:
        holdings_gbp, cash_gbp = dict(HOLDINGS_GBP), float(CASH_GBP)
        source = "config constants"

    df, weights, cash_target, flags, sector_target, meta = build_report(
        close, dollar_volume, fx, holdings_gbp, cash_gbp)
    print_report(df, weights, cash_target, flags, sector_target, meta, fx,
                 close, holdings_gbp, cash_gbp, source)
    os.makedirs(DAILY_DIR, exist_ok=True)
    fname = os.path.join(DAILY_DIR, f"signals_{close.index[-1].date()}.csv")
    total_val = float(sum(holdings_gbp.values()) + cash_gbp)
    cash_row = pd.DataFrame([{
        "ticker": "CASH", "action": "CASH",
        "current_gbp": round(cash_gbp, 2),
        "target_pct": (round(100 * cash_target / total_val, 2)
                       if total_val else 0.0),
        "target_gbp": round(cash_target, 2), "trade_gbp": 0.0,
        "fx_fee_gbp": 0.0, "shares": "", "last_usd": "", "mom_z": "",
        "trend": "", "area": "cash",
        "note": "account cash (recorded for monthly reporting)",
    }])
    pd.concat([df, cash_row], ignore_index=True).to_csv(fname, index=False)
    print(f"\nSaved: {fname}")

    if args.execute:
        if state is None:
            print("\n--execute needs live account state from the Trading "
                  "212 API; no orders placed.")
            return
        age = (datetime.now(timezone.utc).date()
               - close.index[-1].date()).days
        if age > 4:
            print("\n--execute refused: price data is stale.")
            return
        plan = plan_orders(df, close.iloc[-1], fx, state)
        execute_orders(plan, assume_yes=args.yes)


if __name__ == "__main__":
    main()