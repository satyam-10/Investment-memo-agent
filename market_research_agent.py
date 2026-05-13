"""
Market Research Agent — competitive landscape and value proposition.

Uses Claude with server-side web search to investigate:
  - Who are the main competitors?
  - What is the company's unique value proposition?
  - How does the product/service differentiate?
  - Market position and share, where reported.
"""

from common import WEB_SEARCH_TOOL, claude_call, extract_text


MARKET_RESEARCH_SYSTEM = """\
You are a market research analyst. Use web search to investigate the company
and produce a structured competitive analysis. Cite sources inline (e.g.,
"according to [Source, Date]"). Prefer reporting from the last 12 months.

Produce the following sections:

## Company Overview
(One paragraph — what they do, primary revenue lines)

## Value Proposition
(What problem do they solve? Who is the customer?)

## Competitive Landscape
- Top 3-5 direct competitors with brief positioning
- Where the company sits — leader / challenger / niche

## Differentiation & Moat
(What makes the product/service stand out? Brand, technology, network effects,
switching costs, scale?)

## Market Trends Affecting The Company
(Tailwinds and headwinds in the industry)

## Market Position Verdict
(Strong / Defensible / Vulnerable — one line, then justify)

Rules: every factual claim needs a source. If a source is older than 12 months,
flag it. Under 800 words.
"""


def run_market_research(client, ticker: str, company_name: str) -> str:
    user_msg = (
        f"Research {company_name} ({ticker}). Search the web for current "
        f"information about competitors, value proposition, and market positioning. "
        f"Then produce the structured analysis."
    )
    response = claude_call(
        client,
        MARKET_RESEARCH_SYSTEM,
        user_msg,
        tools=WEB_SEARCH_TOOL,
        max_tokens=8000,
    )
    return extract_text(response)
