#!/usr/bin/env python3
"""
monthly_report.py -- build monthly account reports from the daily
Reads the daily files written by quant_signals.py under a Report/ folder:
  Report/Daily/     signals_YYYY-MM-DD.csv
  Report/Execution/ executions_YYYY-MM-DD.jsonl
and writes each monthly report to:
  Report/Monthly/   report_YYYY-MM.md

Each report contains: account value at the start and end of the month
(including cash when the signal files record a CASH row -- files written
by older versions of quant_signals.py contain holdings only, and the
report says so), every position held during the month with start/end
values and an approximate P&L, and a chronological log of BUY/SELL
actions marking which were actually placed via the API.

Honest limitations, stated up front:
  * Deposits and withdrawals are not tracked; they appear inside the
    month's value change.
  * "Actions" in the signal files are PROPOSED orders. An action is
    marked [executed] only when a matching entry exists in that day's
    executions_*.jsonl; otherwise it is marked [proposed].
  * Approximate per-position P&L = end value - start value - net
    proposed flow. It is exact only if every proposed order filled at
    the signal price; treat it as indicative.

Usage:
  python monthly_report.py               # latest month, prints + saves .md
  python monthly_report.py --months 3    # last three months
  python monthly_report.py --base PATH   # Report/ folder location
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import date

import pandas as pd

SIGNAL_RE = re.compile(r"signals_(\d{4}-\d{2}-\d{2})\.csv$")
EXEC_RE = re.compile(r"executions_(\d{4}-\d{2}-\d{2})\.jsonl$")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.join(_BASE_DIR, "Report")


def daily_dir(base):
    return os.path.join(base, "Daily")


def exec_dir(base):
    return os.path.join(base, "Execution")


def monthly_dir(base):
    return os.path.join(base, "Monthly")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_signal_files(folder="."):
    """Return {date: DataFrame} for every readable signals_*.csv."""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "signals_*.csv"))):
        m = SIGNAL_RE.search(os.path.basename(path))
        if not m:
            continue
        try:
            out[date.fromisoformat(m.group(1))] = pd.read_csv(path)
        except Exception as exc:
            print(f"warning: could not read {path}: {exc}")
    return out


def load_executions(folder="."):
    """Return {date: [entries]} for every executions_*.jsonl."""
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "executions_*.jsonl"))):
        m = EXEC_RE.search(os.path.basename(path))
        if not m:
            continue
        d = date.fromisoformat(m.group(1))
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                out[d].append(json.loads(line))
            except Exception:
                print(f"warning: bad line in {path}")
    return out


def list_months(signals):
    return sorted({(d.year, d.month) for d in signals})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(x):
    try:
        v = float(x)
        return v if v == v else 0.0            # NaN -> 0
    except (TypeError, ValueError):
        return 0.0


def day_state(df):
    """(positions {ticker: value_gbp}, cash_gbp or None) for one day."""
    positions, cash = {}, None
    for _, r in df.iterrows():
        t = str(r["ticker"])
        v = _num(r.get("current_gbp"))
        if t == "CASH":
            cash = v
        elif v > 0:
            positions[t] = v
    return positions, cash


def _exec_matches(entries, ticker, sell):
    """True if an executions entry that day matches ticker and direction."""
    for e in entries:
        req = e.get("request") or {}
        qty = _num(req.get("quantity"))
        if (qty < 0) != sell or "response" not in e:
            continue
        y = e.get("yahoo_ticker")
        if y == ticker:
            return True
        if y is None:                          # older logs: heuristic match
            base = str(req.get("ticker", "")).split("_")[0]
            if base == ticker or base.rstrip("0123456789") == ticker:
                return True
    return False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_month_report(month, signals, execs):
    """Return the markdown report for (year, month)."""
    year, mon = month
    days = sorted(d for d in signals if (d.year, d.month) == month)
    if not days:
        return f"# Monthly report {year}-{mon:02d}\n\nNo signal files.\n"

    first, last = days[0], days[-1]
    pos0, cash0 = day_state(signals[first])
    pos1, cash1 = day_state(signals[last])
    inv0, inv1 = sum(pos0.values()), sum(pos1.values())

    lines = [f"# Monthly report {year}-{mon:02d}", ""]
    lines.append(f"Coverage: {len(days)} report days, {first} to {last}.")

    # ---- account value ----
    lines.append("")
    lines.append("## Account value")
    if cash0 is not None and cash1 is not None:
        t0, t1 = inv0 + cash0, inv1 + cash1
        pct = (t1 / t0 - 1) * 100 if t0 else float("nan")
        lines.append(f"- Start ({first}): GBP {t0:,.2f} "
                     f"(invested {inv0:,.2f} + cash {cash0:,.2f})")
        lines.append(f"- End   ({last}): GBP {t1:,.2f} "
                     f"(invested {inv1:,.2f} + cash {cash1:,.2f})")
        lines.append(f"- Change: {t1 - t0:+,.2f} GBP ({pct:+.2f}%). Deposits/"
                     f"withdrawals are not tracked and are included here.")
    else:
        lines.append(f"- Invested value: GBP {inv0:,.2f} -> GBP {inv1:,.2f} "
                     f"({inv1 - inv0:+,.2f}).")
        lines.append("- Cash is not recorded in these files (older format), "
                     "so totals exclude it.")

    # ---- gather actions per ticker ----
    flows = defaultdict(float)                 # net proposed flow, GBP
    actions = []                               # (date, action, ticker, gbp, executed)
    fees = 0.0
    for d in days:
        entries = execs.get(d, [])
        for _, r in signals[d].iterrows():
            act = str(r.get("action", ""))
            if act not in ("BUY", "SELL"):
                continue
            t = str(r["ticker"])
            gbp = _num(r.get("trade_gbp"))
            fees += _num(r.get("fx_fee_gbp"))
            flows[t] += gbp
            actions.append((d, act, t, gbp,
                            _exec_matches(entries, t, sell=(act == "SELL"))))

    # ---- positions table ----
    lines.append("")
    lines.append("## Positions (GBP)")
    lines.append("| ticker | start | end | net proposed flow | approx P&L* "
                 "| actions | status |")
    lines.append("|---|---|---|---|---|---|---|")
    tickers = sorted(set(pos0) | set(pos1) | set(flows),
                     key=lambda t: -(pos1.get(t, 0.0) or pos0.get(t, 0.0)))
    for t in tickers:
        s, e = pos0.get(t, 0.0), pos1.get(t, 0.0)
        fl = flows.get(t, 0.0)
        pnl = e - s - fl
        n_act = sum(1 for a in actions if a[2] == t)
        if s == 0 and e > 0:
            status = "opened"
        elif s > 0 and e == 0:
            status = "closed"
        elif s > 0:
            status = "held"
        else:
            status = "touched"
        lines.append(f"| {t} | {s:,.2f} | {e:,.2f} | {fl:+,.2f} "
                     f"| {pnl:+,.2f} | {n_act} | {status} |")
    lines.append("")
    lines.append("*approx P&L = end - start - net proposed flow; exact only "
                 "if every proposed order filled at the signal price.")

    # ---- action log ----
    lines.append("")
    lines.append("## Action log")
    if actions:
        for d, act, t, gbp, done in actions:
            tag = "executed" if done else "proposed only"
            lines.append(f"- {d}  {act:4s} {t:6s} {gbp:+10,.2f} GBP  [{tag}]")
    else:
        lines.append("- No BUY/SELL signals this month.")

    # ---- totals ----
    buys = sum(g for _, a, _, g, _ in actions if a == "BUY")
    sells = sum(-g for _, a, _, g, _ in actions if a == "SELL")
    placed = sum(1 for d in days for e in execs.get(d, []) if "response" in e)
    lines.append("")
    lines.append("## Totals")
    lines.append(f"- Proposed buys GBP {buys:,.2f}; proposed sells "
                 f"GBP {sells:,.2f}; estimated FX fees GBP {fees:,.2f}.")
    lines.append(f"- Orders actually placed via the API this month: {placed} "
                 f"(from executions logs).")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=1,
                    help="how many recent months to report (default 1)")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="Report/ folder holding Daily/, Execution/, "
                         "Monthly/ (default: ./Report next to this script)")
    args = ap.parse_args()

    signals = load_signal_files(daily_dir(args.base))
    if not signals:
        raise SystemExit(
            f"no signals_*.csv files found in '{daily_dir(args.base)}'")
    execs = load_executions(exec_dir(args.base))

    out_dir = monthly_dir(args.base)
    os.makedirs(out_dir, exist_ok=True)
    for month in list_months(signals)[-args.months:]:
        report = build_month_report(month, signals, execs)
        fname = os.path.join(out_dir, f"report_{month[0]}-{month[1]:02d}.md")
        with open(fname, "w") as fh:
            fh.write(report)
        print(report)
        print(f"saved: {fname}\n" + "=" * 78)


if __name__ == "__main__":
    main()