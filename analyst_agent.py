"""
Analyst Agent — fundamental + price-action snapshot from yfinance.

Pulls valuation, growth, profitability, and recent price metrics, then asks
Claude to produce a structured investment view (BUY / HOLD / SELL with rationale).
"""

import numpy as np

from common import (
    claude_call,
    extract_text,
    fetch_company_snapshot,
    fmt_num,
    fmt_pct,
)


def build_analyst_report(snapshot: dict) -> str:
    info = snapshot["info"]
    closes = snapshot["closes"]
    returns = snapshot["returns"]
    ticker = snapshot["ticker"]

    last_close = closes[-1]
    pct_1m = (closes[-1] / closes[-21] - 1) if len(closes) > 21 else None
    pct_6m = (closes[-1] / closes[-126] - 1) if len(closes) > 126 else None
    pct_1y = (closes[-1] / closes[-252] - 1) if len(closes) > 252 else None
    max_dd = float((closes / np.maximum.accumulate(closes) - 1).min())
    ann_vol = float(returns.std() * np.sqrt(252))

    rows = [
        f"ANALYST DATA — {ticker}",
        f"Company        : {info.get('longName', 'N/A')}",
        f"Sector / Industry : {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}",
        f"Country        : {info.get('country', 'N/A')}",
        f"Employees      : {fmt_num(info.get('fullTimeEmployees'), decimals=0)}",
        "",
        "PRICE ACTION",
        f"  Last close   : ${last_close:.2f}",
        f"  1-month      : {fmt_pct(pct_1m) if pct_1m is not None else 'N/A'}",
        f"  6-month      : {fmt_pct(pct_6m) if pct_6m is not None else 'N/A'}",
        f"  1-year       : {fmt_pct(pct_1y) if pct_1y is not None else 'N/A'}",
        f"  Max drawdown : {fmt_pct(max_dd)}",
        f"  Ann. vol     : {fmt_pct(ann_vol)}",
        "",
        "VALUATION",
        f"  Market cap   : {fmt_num(info.get('marketCap'), prefix='$')}",
        f"  Trailing P/E : {fmt_num(info.get('trailingPE'))}",
        f"  Forward P/E  : {fmt_num(info.get('forwardPE'))}",
        f"  PEG ratio    : {fmt_num(info.get('pegRatio'))}",
        f"  Price/Book   : {fmt_num(info.get('priceToBook'))}",
        f"  EV/EBITDA    : {fmt_num(info.get('enterpriseToEbitda'))}",
        f"  EV/Revenue   : {fmt_num(info.get('enterpriseToRevenue'))}",
        "",
        "PROFITABILITY",
        f"  Gross margin    : {fmt_pct(info.get('grossMargins'))}",
        f"  Operating margin: {fmt_pct(info.get('operatingMargins'))}",
        f"  Profit margin   : {fmt_pct(info.get('profitMargins'))}",
        f"  ROE             : {fmt_pct(info.get('returnOnEquity'))}",
        f"  ROA             : {fmt_pct(info.get('returnOnAssets'))}",
        "",
        "GROWTH",
        f"  Revenue growth (YoY) : {fmt_pct(info.get('revenueGrowth'))}",
        f"  Earnings growth (YoY): {fmt_pct(info.get('earningsGrowth'))}",
        "",
        "BALANCE SHEET",
        f"  Total cash       : {fmt_num(info.get('totalCash'), prefix='$')}",
        f"  Total debt       : {fmt_num(info.get('totalDebt'), prefix='$')}",
        f"  Debt/Equity      : {fmt_num(info.get('debtToEquity'))}",
        f"  Current ratio    : {fmt_num(info.get('currentRatio'))}",
        f"  Free cash flow   : {fmt_num(info.get('freeCashflow'), prefix='$')}",
        "",
        "SHAREHOLDER RETURNS",
        f"  Dividend yield : {fmt_pct(info.get('dividendYield'))}",
        f"  Payout ratio   : {fmt_pct(info.get('payoutRatio'))}",
        f"  Beta           : {fmt_num(info.get('beta'))}",
    ]
    return "\n".join(rows)


ANALYST_SYSTEM = """\
You are a senior equity analyst. Based ONLY on the numerical data below,
produce a structured investment view in this format:

## Investment Signal: BUY | HOLD | SELL
### Conviction: Strong / Moderate / Weak

### Valuation Assessment
(Cite multiples — cheap, fair, or expensive vs sector norms?)

### Financial Health
(D/E, current ratio, FCF, cash position)

### Growth & Profitability
(Margins, ROE, growth trajectory)

### Price Action
(Trend, drawdown, volatility — anything notable?)

### Key Risks From The Numbers
(Red flags only — be specific, cite numbers)

Rules: every claim must reference a specific number from the data.
No qualitative speculation. Under 500 words.
"""


def run_analyst(client, snapshot: dict) -> dict:
    """Returns dict with 'data' (the raw report) and 'analysis' (Claude's view)."""
    report = build_analyst_report(snapshot)
    user_msg = f"Data for {snapshot['ticker']}:\n\n{report}\n\nProvide your analysis."
    response = claude_call(client, ANALYST_SYSTEM, user_msg, max_tokens=4000)
    return {"data": report, "analysis": extract_text(response)}
