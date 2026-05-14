"""
Market Research Agent — deep competitive analysis with live web search.

Three-phase approach
────────────────────
Phase 1    LLM identifies the top 4-5 PRODUCT-LINE competitors (not just
           sector peers) and the subject's primary revenue segments.

Phase 2    yfinance pulls live fundamentals for the subject + every
           competitor and builds a per-company text scorecard.

Phase 2.5  Python fires targeted DuckDuckGo searches covering:
             • per-segment market share with source + date
             • product pricing and features
             • customer reviews & brand perception
             • analyst ratings & price targets
             • one search per competitor
             • industry tailwinds & headwinds
           Results are labelled by query and injected verbatim.

Phase 3    LLM synthesises the scorecard + web snippets into a structured
           8-section report. No tables — structured text only.
           Every market-share claim must cite source, month, and year.
"""

import json
import re
import time

import numpy as np
import yfinance as yf

from common import claude_call, extract_text, fmt_num, fmt_pct

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
# PHASE 1 — Identify competitor tickers (product-line focused)
# ════════════════════════════════════════════════════════════════════

_COMPETITOR_ID_SYSTEM = """\
You are a financial data analyst. Given a company, identify 4-5 companies
that compete DIRECTLY for the same customers in the same PRODUCT CATEGORIES.

CRITICAL: Base competitors on ACTUAL PRODUCT LINES, not the sector label.

Wrong approach for Apple: Sony, Panasonic, LG
  — they do not sell smartphones or PCs, so they are NOT direct competitors.
Right approach for Apple: Samsung (smartphones + tablets), Google/Pixel
  (smartphones), Microsoft (Surface tablets + PC software), Dell/Lenovo (PCs).

Return ONLY a valid JSON object — no markdown fences, no explanation:

{
  "competitors": [
    {
      "ticker": "SSNLF",
      "name": "Samsung Electronics",
      "reason": "direct competitor in smartphones (Galaxy vs iPhone) and tablets (Galaxy Tab vs iPad)",
      "competing_segments": ["smartphones", "tablets"]
    }
  ],
  "primary_segments": ["Smartphones (iPhone)", "Personal Computers (Mac)", "Tablets (iPad)", "Services"],
  "sector": "Technology",
  "industry": "Consumer Electronics / Smartphones",
  "price_segment": "Premium (iPhone: $799-$1599, Mac: $999-$3499)"
}

Rules:
- 4-5 competitors ONLY, ordered by directness of competition
- For multi-product companies focus on the 2-3 LARGEST revenue segments
- Prefer NYSE/NASDAQ tickers so yfinance can pull data; use null if unlisted
- "primary_segments": list the subject's actual revenue product lines
- Return ONLY the JSON object
"""


def get_competitor_tickers(client, ticker: str, company_name: str,
                           sector: str, industry: str) -> dict:
    """Phase 1 — identify product-line competitors. Returns parsed JSON."""
    user_msg = (
        f"Company : {company_name} ({ticker})\n"
        f"Sector  : {sector}\n"
        f"Industry: {industry}\n\n"
        f"Identify 4-5 companies that compete directly in the same product "
        f"categories as {company_name}'s primary revenue-generating segments. "
        f"Return the JSON."
    )
    response = claude_call(client, _COMPETITOR_ID_SYSTEM, user_msg, max_tokens=1000)
    raw = extract_text(response)
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"competitors": [], "primary_segments": [], "sector": sector,
                "industry": industry, "price_segment": "N/A"}


# ════════════════════════════════════════════════════════════════════
# PHASE 2 — Financial scorecard via yfinance (text format, no table)
# ════════════════════════════════════════════════════════════════════


def _fmt_cell(val, fmt: str) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return str(val)
    if fmt == "pct":
        return fmt_pct(v)
    if fmt == "dollar":
        return fmt_num(v, prefix="$")
    if fmt == "price":
        return f"${v:.2f}"
    return fmt_num(v)


def _analyst_consensus(info: dict) -> str:
    key = info.get("recommendationKey", "")
    n   = info.get("numberOfAnalystOpinions") or 0
    if key:
        label = key.replace("_", " ").title()
        return f"{label} ({n} analysts)" if n else label
    rec = info.get("recommendationMean")
    if rec:
        label = {1: "Strong Buy", 2: "Buy", 3: "Hold",
                 4: "Underperform", 5: "Sell"}.get(round(rec), f"{rec:.1f}")
        return f"{label} ({n} analysts)" if n else label
    return "N/A"


def _fetch_info(t: str) -> dict:
    try:
        return yf.Ticker(t).info or {}
    except Exception:
        return {}


def build_financial_scorecard(subject_ticker: str, subject_info: dict,
                               competitors: list[dict]) -> str:
    """Per-company text blocks — no markdown table."""
    entries = [{
        "ticker": subject_ticker,
        "name":   subject_info.get("longName", subject_ticker),
        "info":   subject_info,
        "is_subject": True,
    }]
    for c in competitors:
        t = c.get("ticker")
        entries.append({
            "ticker":   t or "—",
            "name":     c.get("name", t or "Unknown"),
            "info":     _fetch_info(t) if t else {},
            "segments": c.get("competing_segments", []),
            "is_subject": False,
        })

    blocks = []
    for e in entries:
        info = e["info"]
        tag  = " ★ SUBJECT" if e["is_subject"] else ""
        seg  = f"  Competing segments: {', '.join(e.get('segments', []))}" if e.get("segments") else ""
        lines = [
            f"--- {e['name']} ({e['ticker']}){tag} ---",
        ]
        if seg:
            lines.append(seg)
        lines += [
            f"  Revenue (TTM)      : {_fmt_cell(info.get('totalRevenue'), 'dollar')}",
            f"  Revenue Growth YoY : {_fmt_cell(info.get('revenueGrowth'), 'pct')}",
            f"  Earnings Growth YoY: {_fmt_cell(info.get('earningsGrowth'), 'pct')}",
            f"  Gross Margin       : {_fmt_cell(info.get('grossMargins'), 'pct')}",
            f"  Operating Margin   : {_fmt_cell(info.get('operatingMargins'), 'pct')}",
            f"  Net Margin         : {_fmt_cell(info.get('profitMargins'), 'pct')}",
            f"  ROE                : {_fmt_cell(info.get('returnOnEquity'), 'pct')}",
            f"  Market Cap         : {_fmt_cell(info.get('marketCap'), 'dollar')}",
            f"  P/E (Trail / Fwd)  : {_fmt_cell(info.get('trailingPE'), 'raw')} / {_fmt_cell(info.get('forwardPE'), 'raw')}",
            f"  EV / EBITDA        : {_fmt_cell(info.get('enterpriseToEbitda'), 'raw')}",
            f"  PEG Ratio          : {_fmt_cell(info.get('pegRatio'), 'raw')}",
            f"  Free Cash Flow     : {_fmt_cell(info.get('freeCashflow'), 'dollar')}",
            f"  Debt / Equity      : {_fmt_cell(info.get('debtToEquity'), 'raw')}",
            f"  Current Ratio      : {_fmt_cell(info.get('currentRatio'), 'raw')}",
            f"  52-Week Return     : {_fmt_cell(info.get('52WeekChange'), 'pct')}",
            f"  Analyst Consensus  : {_analyst_consensus(info)}",
            f"  Analyst Price Tgt  : {_fmt_cell(info.get('targetMeanPrice'), 'price')}",
        ]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_product_context(subject_info: dict, comp_meta: dict) -> str:
    summary = subject_info.get("longBusinessSummary", "N/A")
    if len(summary) > 700:
        summary = summary[:700] + "..."

    segments = comp_meta.get("primary_segments", [])
    lines = [
        "PRODUCT & SEGMENT CONTEXT",
        f"  Business summary  : {summary}",
        f"  Price segment     : {comp_meta.get('price_segment', 'N/A')}",
        f"  Sector / Industry : {comp_meta.get('sector', subject_info.get('sector', 'N/A'))} / "
        f"{comp_meta.get('industry', subject_info.get('industry', 'N/A'))}",
        f"  Country           : {subject_info.get('country', 'N/A')}",
        f"  Employees         : {fmt_num(subject_info.get('fullTimeEmployees'), decimals=0)}",
    ]
    if segments:
        lines.append("  Primary segments  : " + " | ".join(segments))
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# PHASE 2.5 — Web search
# ════════════════════════════════════════════════════════════════════

_SLEEP_BETWEEN_QUERIES = 1.5


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    if not _DDGS_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def _format_snippets(query: str, results: list[dict]) -> str:
    if not results:
        return f"[SEARCH: {query}]\n  (no results returned)\n"
    lines = [f"[SEARCH: {query}]"]
    for i, r in enumerate(results, 1):
        lines.append(f"  {i}. {r.get('title', 'No title')}")
        lines.append(f"     URL    : {r.get('href', '')}")
        lines.append(f"     Snippet: {r.get('body', '')[:350]}")
    lines.append("")
    return "\n".join(lines)


def run_web_searches(company_name: str, ticker: str, comp_meta: dict) -> str:
    """
    Phase 2.5 — targeted DuckDuckGo searches, results injected into prompt.
    comp_meta is the full dict from get_competitor_tickers().
    """
    if not _DDGS_AVAILABLE:
        return "WARNING: duckduckgo_search not installed — web search skipped.\n"

    competitors = comp_meta.get("competitors", [])
    segments    = comp_meta.get("primary_segments", [])
    sector      = comp_meta.get("sector", "")
    industry    = comp_meta.get("industry", "")

    # Core queries
    queries = [
        f"{company_name} {ticker} analyst price target rating buy hold sell 2025",
        f"{company_name} customer satisfaction review NPS score 2024 2025",
        f"{sector} {industry} market trends tailwinds headwinds 2024 2025",
        f"{industry} total addressable market size CAGR forecast 2025 2026",
        f"{company_name} product pricing latest lineup comparison 2025",
    ]

    # Per-segment market share (up to 2 largest segments)
    for seg in segments[:2]:
        queries.append(
            f"{company_name} {seg} market share percentage 2024 2025 IDC Counterpoint Statista"
        )

    # Per-competitor: market share in the specific segments they compete in
    for c in competitors[:4]:
        name = c.get("name") or c.get("ticker", "")
        segs = c.get("competing_segments", [])
        seg_str = segs[0] if segs else industry
        if name:
            queries.append(
                f"{name} {seg_str} market share revenue 2024 2025"
            )

    total = len(queries)
    blocks = []
    for idx, q in enumerate(queries, 1):
        print(f"    [market] web search {idx}/{total}: {q[:72]}")
        results = _ddg_search(q, max_results=5)
        blocks.append(_format_snippets(q, results))
        if idx < total:
            time.sleep(_SLEEP_BETWEEN_QUERIES)

    return "\n".join(blocks)


# ════════════════════════════════════════════════════════════════════
# PHASE 3 — LLM synthesis
# ════════════════════════════════════════════════════════════════════

_MARKET_RESEARCH_SYSTEM = """\
You are a senior market research analyst. You have three data sources:

  SOURCE A — Financial Scorecard  [yfinance, live]
    Per-company text blocks with live fundamentals.
    Tag every claim from this source as [yfinance, live].

  SOURCE B — Web Search Snippets  [DuckDuckGo, collected now]
    Labelled [SEARCH: <query>]. Contains title, URL, and snippet.
    Tag claims as (Article Title, URL, Month Year).
    Month and Year are REQUIRED for every market-share figure.

  SOURCE C — Training knowledge
    ONLY for gaps not in A or B. Tag as [training knowledge — verify].

CITATION RULE: every factual claim must carry one of the three tags.
A claim with no tag = hallucination. Reject it yourself.

DO NOT USE MARKDOWN TABLES anywhere in your output.
Use structured text (labelled paragraphs, bullet lists) instead.

─────────────────────────────────────────────────────────────────────
Produce these eight sections:

## 1. Company Overview
One paragraph: what they do, primary revenue segments, lifecycle stage.
Cite revenue and growth figures from SOURCE A [yfinance, live].

## 2. Financial Head-to-Head  [all metrics tagged yfinance, live]

For EACH company in SOURCE A, write one block in this exact format:

Company Name (TICKER) [★ SUBJECT / competitor]
  Competing segments: [from scorecard]
  Revenue: $X  |  Growth YoY: X%  |  Gross Margin: X%  |  Op. Margin: X%
  P/E (T/F): X / X  |  EV/EBITDA: X  |  Market Cap: $X
  FCF: $X  |  Debt/Equity: X  |  52-Week Return: X%
  Analyst Consensus: [from scorecard]

After all company blocks, write a "So What?" section with 5 bullets:
  • Who leads on revenue growth and by how much? (cite exact %)
  • Who is most and least profitable (gross margin, operating margin)?
  • Cheapest and most expensive on P/E and EV/EBITDA?
  • Strongest balance sheet (FCF, D/E, current ratio)?
  • What does analyst consensus say about the subject vs. peers?

## 3. Product & Price Positioning

For the SUBJECT and EACH competitor, write one labelled block:

Company Name
  Competing segment(s): [list]
  Price range: $X - $Y
  Key product(s): [top 2-3 products]
  Market share in [segment]: X% — (Source Name, Month Year)
    If not found in search results: "[not found in search results — verify]"
  Core differentiator vs. [subject]: [one sentence]
  Main weakness vs. [subject]: [one sentence]

End with 2-3 sentences: does the subject have a direct rival at every price
tier? Where is the biggest gap?

## 4. Customer & Market Perception

For the subject AND each major competitor, write:
  Company Name
    Review score: [platform, score, date — from search results only]
    NPS: [if publicly reported, cite source] or "[not found]"
    Brand perception: [premium / value / trusted / disrupted / loved / hated]
    Top complaint: [one line — cite source] or "[not found]"
    Top praise: [one line — cite source] or "[not found]"

## 5. Value Proposition & Moat
  Core problem solved: [one sentence]
  Unique mechanism: [technology / network effects / brand / distribution / IP]
  Switching cost: what does a customer lose by leaving?
  Moat rating: WIDE / NARROW / NONE — two-sentence justification with evidence

## 6. Market Share & Growth Dynamics

For EACH primary segment of the subject:
  Segment: [name]
  Subject share: X% — (Source Name, Month Year)
    If not in search results: "[not found in search results — verify]"
  Top rival shares: [Company: X% — (Source, Month Year)] for top 2-3 rivals
  Market growth: CAGR X% through YYYY — (Source Name, Month Year)
  Share trend: gaining / holding / losing — cite evidence from search results

## 7. Industry Tailwinds & Headwinds

Three tailwinds and three headwinds. For each:
  [Name of trend] — quantify where possible (cite source + date)
  Impact: benefits subject [more / less / equally] than peers — reason

## 8. Market Position Verdict
STRONG / DEFENSIBLE / VULNERABLE — one line.

Biggest competitive threat: [one sentence, cite a specific number or source]
Biggest competitive advantage: [one sentence, cite a specific number or source]

─────────────────────────────────────────────────────────────────────
Rules:
  • No vague claims ("strong brand", "great products") without evidence
  • If web search returned nothing for a query, say so — do not fill gaps
    with uncited assertions
  • NO TABLES — use the structured text format above
  • Word count: 1,200-1,800 words
"""


def run_market_research(client, ticker: str, company_name: str,
                        snapshot: dict | None = None) -> str:
    if snapshot is None:
        info = yf.Ticker(ticker).info or {}
    else:
        info = snapshot.get("info", {})

    sector   = info.get("sector",   "Unknown")
    industry = info.get("industry", "Unknown")

    # Phase 1
    print(f"    [market] Phase 1 — identifying product-line competitors for {ticker}...")
    comp_meta   = get_competitor_tickers(client, ticker, company_name, sector, industry)
    competitors = comp_meta.get("competitors", [])
    segments    = comp_meta.get("primary_segments", [])
    print(f"    [market] Segments : {segments}")
    print(f"    [market] Competitors: {[c.get('ticker') or c.get('name') for c in competitors]}")

    # Phase 2
    print(f"    [market] Phase 2 — building financial scorecard...")
    scorecard       = build_financial_scorecard(ticker, info, competitors)
    product_context = build_product_context(info, comp_meta)

    # Phase 2.5
    print(f"    [market] Phase 2.5 — running web searches...")
    search_block = run_web_searches(company_name, ticker, comp_meta)

    # Phase 3
    print(f"    [market] Phase 3 — composing analysis...")

    competitor_list = "\n".join(
        f"  - {c.get('name', '?')} ({c.get('ticker', 'unlisted')}): "
        f"{c.get('reason', '')} "
        f"[segments: {', '.join(c.get('competing_segments', []))}]"
        for c in competitors
    )

    user_msg = f"""\
Subject company  : {company_name} ({ticker})
Primary segments : {' | '.join(segments) if segments else 'see scorecard'}
Price segment    : {comp_meta.get('price_segment', 'N/A')}

COMPETITORS
{competitor_list}

{'=' * 72}
SOURCE A — FINANCIAL SCORECARD  [yfinance, live]
{'=' * 72}
{scorecard}

{'=' * 72}
SOURCE B — WEB SEARCH SNIPPETS  [DuckDuckGo, collected now]
{'=' * 72}
{search_block}
{'=' * 72}
ADDITIONAL PRODUCT CONTEXT
{'=' * 72}
{product_context}

Produce the full 8-section market research analysis. NO TABLES.
"""

    response = claude_call(client, _MARKET_RESEARCH_SYSTEM, user_msg, max_tokens=8000)
    return extract_text(response)
