#!/usr/bin/env python3
"""
strategy_review.py -- prepare an OFFLINE strategy-review pack. This program
does not contact any cloud service. It does two things:

  1. builds a combined report for the recent months (default 3) from the
     daily files under Report/, and

  2. writes a single self-contained prompt file that you can paste into any
     LLM (ChatGPT, Claude, Gemini, a local model, ...). That file already
     contains the review instructions, the recent report, and the full
     current quant_signals.py, so one paste is enough. The LLM is asked to
     return either "NO CHANGE RECOMMENDED" or a replacement for the strategy
     section, which you then paste into quant_signals.py YOURSELF.

Nothing is ever written to quant_signals.py by this program. An optional
--check mode lets you validate a candidate strategy section (the code you
got back) against the same static and smoke-test checks before you paste
it in; it still does not modify quant_signals.py.

Outputs (under Report/Review/):
  review_report_<from>_<to>.md   the combined recent-month report
  llm_review_prompt_<date>.md    the ready-to-paste prompt bundle

Usage:
  python strategy_review.py                 # last 3 months
  python strategy_review.py --months 6
  python strategy_review.py --base PATH      # Report/ folder location
  python strategy_review.py --check FILE      # validate a candidate section
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "quant_signals.py")
BEGIN = "# ========================= STRATEGY SECTION -- BEGIN ========================="
END = "# ========================== STRATEGY SECTION -- END =========================="
FENCE_RE = re.compile(r"```python-strategy\s*\n(.*?)```", re.DOTALL)

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "statistics"}
BANNED_NAMES = {
    "os", "sys", "subprocess", "socket", "urllib", "requests", "shutil",
    "pathlib", "importlib", "builtins", "eval", "exec", "open",
    "__import__", "compile", "input", "breakpoint", "globals", "locals",
    "getattr", "setattr", "vars",
    "_t212_call", "fetch_t212_state", "execute_orders", "plan_orders",
    "fetch_prices", "fetch_fx", "_T212_TRANSPORT", "T212_KEY_ENV",
    "T212_SECRET_ENV",
}

# The instruction block that goes at the top of the prompt bundle.
PROMPT_INSTRUCTIONS = f"""You are reviewing a small personal quantitative
trading system for US-listed shares held in a GBP Trading 212 account. Below
this instruction block you will find (a) recent monthly performance reports
and (b) the full source of quant_signals.py.

Your task: analyse the results honestly, then either recommend no change or
propose a replacement for the strategy section only.

Hard rules:
- You may propose changes to ONLY the code between the marker lines
  '{BEGIN}'
  and
  '{END}'.
  Everything else (data download, the Trading 212 API, order execution,
  reporting, the backtest) must not be changed.
- A replacement must satisfy the contract stated at the top of that section:
  define every constant the section currently defines (other code reads
  them), define select_portfolio(close, dollar_volume) with the same return
  structure (weights: a pandas Series of non-negative target weights over
  tickers in close.columns excluding BENCHMARK, summing to <= 1.0; diag: a
  dict with keys mom_z, trend_ok, dist, vol, liq_ok, rank, skipped_area,
  n_ranked, fallback), import only numpy / pandas / math / statistics, and
  perform no file, network, or system access.
- Be conservative. A few months of reports is weak evidence. Prefer NO
  CHANGE over fitting to one period. Any parameter change needs a rationale
  that would have been reasonable before seeing these results.
- Do not suggest raising the execution caps, changing the fee assumptions,
  or automating order execution.

Output format:
1. A plain analysis of the reports: performance, turnover, fees,
   concentration, and anything unusual.
2. Then EITHER the single line
     NO CHANGE RECOMMENDED
   followed by your reasons, OR the complete replacement strategy section
   (the code that goes BETWEEN the two marker lines, without the marker
   lines themselves) inside one fenced block that begins with
   ```python-strategy
   and ends with ```
   , followed by a short justification and what to watch for after the
   change.

The person running this will read your output and update quant_signals.py
themselves. Do not assume any change is applied automatically."""


# ---------------------------------------------------------------------------
# quant_signals.py marker handling (used by --check only)
# ---------------------------------------------------------------------------

def split_target(path=TARGET):
    text = open(path).read()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise SystemExit(f"{path} must contain exactly one strategy-section "
                         f"marker pair; found {text.count(BEGIN)}/"
                         f"{text.count(END)}")
    pre, rest = text.split(BEGIN, 1)
    section, post = rest.split(END, 1)
    return pre, section, post


def splice(pre, section_body, post):
    """Exact reconstruction: everything outside the markers is preserved
    byte-for-byte. Candidate bodies are normalised by the caller."""
    return pre + BEGIN + section_body + END + post


def validate_section(code):
    """Static checks on a candidate section. Returns a list of problems."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in ALLOWED_IMPORTS:
                problems.append(f"from '{node.module}' import not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"reference to banned name '{node.id}'")
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in BANNED_NAMES:
                problems.append(f"attribute access on banned '{root.id}'")
    if "def select_portfolio(" not in code:
        problems.append("select_portfolio is not defined")
    return sorted(set(problems))


SMOKE_SNIPPET = r"""
import importlib.util, sys
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("qs_candidate", sys.argv[1])
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)
rng = np.random.default_rng(0); n = 560
dates = pd.bdate_range("2024-04-01", periods=n)
close = pd.DataFrame({t: 100*np.exp(np.cumsum(rng.normal(5e-4, 0.02, n)))
                      for t in qs.UNIVERSE}, index=dates)
vol = pd.DataFrame(rng.integers(500_000, 5_000_000, size=close.shape),
                   index=dates, columns=close.columns).astype(float)
w, diag = qs.select_portfolio(close, close * vol)
assert float(w.sum()) <= 1.0 + 1e-9, "weights sum above 1"
assert (w >= -1e-12).all(), "negative weight"
assert set(w.index) <= set(close.columns), "weight on unknown ticker"
assert qs.BENCHMARK not in w.index, "benchmark selected"
for key in ("mom_z", "trend_ok", "dist", "vol", "liq_ok", "rank",
            "skipped_area", "n_ranked", "fallback"):
    assert key in diag, f"diag missing '{key}'"
df, w2, cash_t, flags, sect, meta = qs.build_report(
    close, close * vol, 1.33, {"LITE": 500.0}, 1000.0)
assert len(df) > 0
print("SMOKE OK: invested", round(float(w.sum()), 3), "picks", len(w))
"""


def smoke_test(candidate_text):
    """Compile and run the spliced candidate on synthetic data in a
    subprocess. Returns (ok, output)."""
    with tempfile.TemporaryDirectory() as td:
        cand = os.path.join(td, "qs_candidate.py")
        runner = os.path.join(td, "run_smoke.py")
        open(cand, "w").write(candidate_text)
        open(runner, "w").write(SMOKE_SNIPPET)
        try:
            r = subprocess.run([sys.executable, runner, cand],
                               capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            return False, "smoke test timed out"
        return r.returncode == 0, (r.stdout + r.stderr).strip()


# ---------------------------------------------------------------------------
# Report gathering
# ---------------------------------------------------------------------------

def gather_month_reports(months, base):
    """Return (list_of_month_report_strings, (first_date, last_date))."""
    import monthly_report as mr
    signals = mr.load_signal_files(mr.daily_dir(base))
    if not signals:
        return ([], None)
    execs = mr.load_executions(mr.exec_dir(base))
    chosen = mr.list_months(signals)[-months:]
    reports = [mr.build_month_report(m, signals, execs) for m in chosen]
    dates = sorted(signals)
    return (reports, (dates[0], dates[-1]))


def combined_report(reports, span, months):
    header = [f"# Strategy review report -- last {months} month(s)",
              f"Generated {date.today().isoformat()}."]
    if span:
        header.append(f"Underlying daily files span {span[0]} to {span[1]}.")
    header.append("")
    if not reports:
        header.append("No signal files were found under Report/Daily, so "
                      "there is nothing to report yet. Run quant_signals.py "
                      "for a few days first.")
    return "\n".join(header) + "\n" + "\n\n---\n\n".join(reports) + "\n"


def build_prompt_bundle(report_text, code_text):
    return (
        "# LLM strategy-review prompt (self-contained)\n\n"
        "How to use: copy everything below the line of dashes and paste it "
        "into any LLM in a single message. It already includes the recent "
        "report and the full quant_signals.py. Read the reply, then edit the "
        "strategy section of quant_signals.py yourself. This file was "
        "generated offline; sending it to an LLM is your own manual step.\n\n"
        "------------------------------------------------------------------"
        "----------\n\n"
        + PROMPT_INSTRUCTIONS
        + "\n\n<monthly_reports>\n" + report_text.strip()
        + "\n</monthly_reports>\n\n<quant_signals_py>\n" + code_text.strip()
        + "\n</quant_signals_py>\n")


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------

def run_check(path):
    raw = open(path).read()
    m = FENCE_RE.search(raw)
    candidate = m.group(1) if m else raw
    print(f"Checking candidate strategy section from {path} "
          f"({'fenced block' if m else 'raw file'}) ...\n")

    problems = validate_section(candidate)
    if problems:
        print("STATIC CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("\nThis section would be unsafe or would break the contract. "
              "Do not paste it in as-is.")
        return 1
    print("static check passed (imports, no I/O, select_portfolio present).")

    pre, _old, post = split_target()
    spliced = splice(pre, "\n" + candidate.strip("\n") + "\n\n", post)
    try:
        compile(spliced, TARGET, "exec")
    except SyntaxError as exc:
        print(f"COMPILE FAILED once spliced: {exc}")
        return 1
    print("spliced file compiles.")

    print("running synthetic smoke test ...")
    ok, out = smoke_test(spliced)
    print("  " + out.replace("\n", "\n  "))
    if not ok:
        print("SMOKE TEST FAILED. Do not use this section.")
        return 1
    print("\nAll checks passed. You can paste this code between the two "
          "STRATEGY SECTION markers in quant_signals.py. Back up the file "
          "first, then run 'python quant_signals.py --backtest'.")
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=3,
                    help="how many recent months to include (default 3)")
    ap.add_argument("--base", default=None,
                    help="Report/ folder holding Daily/ and Execution/ "
                         "(default: ./Report next to the scripts)")
    ap.add_argument("--check", metavar="FILE", default=None,
                    help="validate a candidate strategy section (raw code or "
                         "a ```python-strategy block) without writing")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        raise SystemExit(f"cannot find {TARGET}")

    if args.check:
        raise SystemExit(run_check(args.check))

    import monthly_report as mr
    base = args.base or mr.DEFAULT_BASE
    reports, span = gather_month_reports(args.months, base)

    review_dir = os.path.join(base, "Review")
    os.makedirs(review_dir, exist_ok=True)

    report_text = combined_report(reports, span, args.months)
    if span:
        rname = os.path.join(review_dir,
                             f"review_report_{span[0]}_{span[1]}.md")
    else:
        rname = os.path.join(review_dir,
                             f"review_report_{date.today().isoformat()}.md")
    with open(rname, "w") as fh:
        fh.write(report_text)

    code_text = open(TARGET).read()
    bundle = build_prompt_bundle(report_text, code_text)
    pname = os.path.join(review_dir,
                         f"llm_review_prompt_{date.today().isoformat()}.md")
    with open(pname, "w") as fh:
        fh.write(bundle)

    print("Offline review pack written:")
    print(f"  report: {rname}")
    print(f"  prompt: {pname}")
    if not reports:
        print("\nNote: no daily files were found yet, so the report is "
              "empty. The prompt bundle still contains quant_signals.py, but "
              "run the daily script for a while before asking for a review.")
    else:
        print(f"\nPaste the contents of the prompt file into any LLM. It "
              f"includes the last {len(reports)} month(s) of reports and the "
              f"current quant_signals.py. Apply any suggested change "
              f"yourself, then optionally verify it with:\n"
              f"  python strategy_review.py --check <saved_section_file>")


if __name__ == "__main__":
    main()