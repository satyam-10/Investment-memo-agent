"""
Leadership Analysis Agent — strategy and management quality.

Pre-fetches targeted DuckDuckGo searches in Python and injects the results
directly into the prompt so the LLM synthesises from live data, not training
knowledge.
"""

from common import claude_call, extract_text, run_searches


LEADERSHIP_SYSTEM = """\
You are a corporate governance and leadership analyst.

You have been given real web search results collected moments ago (labelled
[SEARCH: <query>]). Use these as your PRIMARY source for all factual claims.

Citation rules (every factual claim needs one):
  • (Article Title, URL) — for anything found in the search results
  • [yfinance, live]    — for financial metrics passed in the prompt
  • [training knowledge — verify] — ONLY for gaps not covered by search results;
    flag these explicitly so the reader knows to verify

Produce these sections:

## CEO Profile
- Full name, tenure start date, prior roles before this company
- Track record: cite at least one revenue or margin figure from his/her tenure
- Link CEO tenure to stock performance if mentioned in search results

## Senior Leadership
- Key C-suite (CFO, COO, CTO where relevant) — name + one notable fact each
- Any executive appointments or departures in the last 18 months (cite source)

## Strategic Direction
What is leadership saying about the next 1–3 years?
Reference recent earnings calls, investor days, or interviews found in search
results. Quote directly where possible.

## Capital Allocation Track Record
Buybacks, dividends, M&A, R&D spend — has it created or destroyed value?
Cite specific dollar amounts and dates from search results.

## Governance & Red Flags
Board independence, insider selling patterns, recent departures, lawsuits,
or restatements. If nothing found, state "no red flags found in search results."

## Leadership Verdict
STRONG / ADEQUATE / CONCERNING — one line verdict.
Then 2–3 sentences justifying it, each referencing a specific cited fact.

Rules:
  • Do NOT use training knowledge for facts that should be in search results
    (CEO name, recent announcements, current strategy).
  • If search results returned nothing on a topic, say so rather than inventing.
  • Under 900 words.
"""


def run_leadership(client, ticker: str, company_name: str) -> str:
    queries = [
        f"{company_name} CEO name tenure background 2024 2025",
        f"{company_name} {ticker} earnings call strategy outlook 2025",
        f"{company_name} executive leadership team CFO COO appointment 2024 2025",
        f"{company_name} capital allocation buyback dividend M&A 2024 2025",
        f"{company_name} {ticker} board governance insider selling lawsuit 2024 2025",
    ]
    search_block = run_searches(queries, label="leadership")

    user_msg = f"""\
Investigate the leadership and strategy of {company_name} ({ticker}).

{'=' * 72}
WEB SEARCH RESULTS  [DuckDuckGo, collected now]
{'=' * 72}
{search_block}
{'=' * 72}

Using the search results above as your primary source, produce the structured
leadership analysis. Cite every factual claim.
"""
    response = claude_call(client, LEADERSHIP_SYSTEM, user_msg, max_tokens=8000)
    return extract_text(response)
