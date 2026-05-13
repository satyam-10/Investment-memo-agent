"""
Leadership Analysis Agent — strategy and management quality.

Uses Claude with web search to investigate the CEO and senior leadership:
  - Background and track record
  - Strategic direction and recent communications
  - Capital allocation decisions
  - Any governance flags
"""

from common import WEB_SEARCH_TOOL, claude_call, extract_text


LEADERSHIP_SYSTEM = """\
You are a corporate governance and leadership analyst. Use web search to
investigate the company's leadership and produce a structured analysis.
Cite sources inline. Prefer information from the last 18 months.

Produce the following sections:

## CEO Profile
- Name, tenure, prior roles
- Track record at this company (revenue, margin, stock performance under tenure)

## Senior Leadership
- Key C-suite (CFO, COO, CTO where applicable) and notable changes

## Strategic Direction
(What is leadership saying about the next 1-3 years? Reference recent earnings
calls, investor days, or interviews.)

## Capital Allocation Track Record
(Buybacks, dividends, M&A, R&D spend — has it created or destroyed value?)

## Governance & Red Flags
(Board independence, insider selling, recent departures, lawsuits, restatements)

## Leadership Verdict
(Strong / Adequate / Concerning — one line, then justify)

Rules: every factual claim needs a source. Flag anything older than 18 months.
Under 800 words.
"""


def run_leadership(client, ticker: str, company_name: str) -> str:
    user_msg = (
        f"Investigate the leadership and strategy of {company_name} ({ticker}). "
        f"Use web search to find current information about the CEO, key executives, "
        f"recent strategic announcements, and any governance concerns. Then produce "
        f"the structured analysis."
    )
    response = claude_call(
        client,
        LEADERSHIP_SYSTEM,
        user_msg,
        tools=WEB_SEARCH_TOOL,
        max_tokens=8000,
    )
    return extract_text(response)
