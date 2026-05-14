"""
Orchestrator + Critic + Human-in-the-loop.

Flow:
  1. Run Analyst, Market Research, Leadership, and Risk agents.
  2. Memo Writer composes a single institutional-grade investment memo.
  3. Critic reviews the memo and produces a structured critique.
  4. Show memo + critique to the human.
       - approve  -> write final memo to disk and exit
       - revise   -> Memo Writer applies the critique, loop back to step 3
       - quit     -> bail out
"""

from common import claude_call, extract_text, section


MEMO_WRITER_SYSTEM = """\
You are the chief investment officer's writer. You take research from four
specialist analysts and produce a single investment memo. The memo must be
suitable for an institutional investment committee.

Structure:

# Investment Memo: {COMPANY} ({TICKER})

## Recommendation
**BUY / HOLD / SELL** — Conviction: Strong / Moderate / Weak
Target horizon: (state — e.g. 12-18 months)

## Executive Summary
(3-5 bullets. The TL;DR for the committee.)

## Investment Thesis
(2-3 paragraphs. Why this works — or why it doesn't.)

## Financial Snapshot
(Cite specific numbers from the Analyst's report.)

## Competitive Analysis
This section MUST contain three sub-parts drawn directly from the Market
Research report. Do NOT use markdown tables — structured text only.

### Financial Head-to-Head
Reproduce the per-company financial blocks from the Market Research report
(Revenue, Margins, P/E, FCF, Debt/Equity, Analyst Consensus — one block per
company). Then write 4-5 "So What?" bullets: who leads on growth, margins,
cheapest valuation, strongest balance sheet, analyst consensus vs. peers.
Tag all figures [yfinance, live].

### Product & Price Positioning
For each competitor: competing segment(s), price range, key products, market
share with source and month/year, core differentiator vs. the subject, main
weakness. If market share was not found in search results say so explicitly.

### Market Position Verdict
State the verdict word (STRONG / DEFENSIBLE / VULNERABLE) from the research.
Give the single biggest competitive threat and the single biggest advantage —
each backed by a cited number or source from the Market Research report.

## Leadership Assessment
(From the Leadership findings — include CEO name, tenure, and one specific
 metric that proves or disproves their track record.)

## Risk Assessment
Quote the exact 95% and 99% 1-day VaR and CVaR figures from the Risk report.
Then summarise the top 3 qualitative risks with severity tags.

## Key Catalysts
(Upside drivers in the next 6-18 months — be specific, no vague assertions.)

## Bear Case
(What would make this thesis wrong? Name at least one specific risk from the
 Risk report and one competitive threat from the Market Research report.)

## Decision Summary
(One paragraph. State the recommendation and the single most important reason.)

Rules:
  - Synthesise across the four reports. Do not just paste them EXCEPT for
    the financial scorecard table in Competitive Analysis — reproduce that verbatim.
  - Reconcile contradictions explicitly (e.g. "Analyst flags a stretched
    multiple but Market Research shows expanding moat — net call is...").
  - Cite specific numbers wherever possible.
  - Length: 1000-1800 words (scorecard table excluded from word count).
"""


CRITIC_SYSTEM = """\
You are a skeptical investment committee chair. Critique the memo below.

Evaluate it on:
  1. Internal consistency — does the recommendation match the evidence?
  2. Use of data — are numbers cited specifically and correctly?
  3. Competitive analysis — does the memo include the full financial scorecard
     table, named competitors with market share figures, and a product/price
     comparison? Generic phrases like "faces competition from X" are failures.
  4. Source integration — are findings from all four research streams
     (Analyst, Market Research, Leadership, Risk) reflected? Any gaps?
  5. Contradictions — are tensions in the underlying research surfaced
     and resolved, or papered over?
  6. Bear case quality — is the downside taken seriously, or strawmanned?
  7. Actionability — could a PM act on this, or is it hedged into mush?

Output format:

## Verdict
APPROVE / REVISE / REJECT

## Critical Issues
(Numbered list. Each issue = a concrete problem the writer must fix.)

## Strengths
(What works — keep these in any revision.)

## Required Changes
(If REVISE: a checklist the writer should action. Be specific —
"add the 99% CVaR figure to the Risk section" beats "improve risk section".)

Be direct. Do not be polite at the cost of accuracy. Under 500 words.
"""


def run_memo_writer(client, ticker: str, company_name: str, research: dict) -> str:
    user_msg = f"""\
Compose the investment memo for {company_name} ({ticker}).

═══════════════════════════════════════════════════════════════════════
ANALYST REPORT
═══════════════════════════════════════════════════════════════════════
{research['analyst']['data']}

ANALYST'S VIEW:
{research['analyst']['analysis']}

═══════════════════════════════════════════════════════════════════════
MARKET RESEARCH
═══════════════════════════════════════════════════════════════════════
{research['market']}

═══════════════════════════════════════════════════════════════════════
LEADERSHIP ANALYSIS
═══════════════════════════════════════════════════════════════════════
{research['leadership']}

═══════════════════════════════════════════════════════════════════════
RISK ANALYSIS
═══════════════════════════════════════════════════════════════════════
{research['risk']['var_report']}

RISK ANALYST'S VIEW:
{research['risk']['analysis']}
═══════════════════════════════════════════════════════════════════════

Now write the memo.
"""
    response = claude_call(
        client,
        MEMO_WRITER_SYSTEM.replace("{COMPANY}", company_name).replace("{TICKER}", ticker),
        user_msg,
        max_tokens=8000,
    )
    return extract_text(response)


def run_critic(client, memo: str, research: dict) -> str:
    user_msg = f"""\
Here is the memo to critique:

{memo}

─── For your reference, the underlying research the memo was built from: ───

ANALYST: {research['analyst']['analysis']}

MARKET RESEARCH: {research['market']}

LEADERSHIP: {research['leadership']}

RISK: {research['risk']['analysis']}
"""
    response = claude_call(client, CRITIC_SYSTEM, user_msg, max_tokens=3000)
    return extract_text(response)


def run_reviser(client, memo: str, critique: str, ticker: str, company_name: str) -> str:
    """Memo writer applies the critic's required changes."""
    system = MEMO_WRITER_SYSTEM.replace("{COMPANY}", company_name).replace("{TICKER}", ticker)
    user_msg = f"""\
The investment committee chair has reviewed your memo and returned the
following critique. Produce a REVISED memo that addresses every required
change. Keep the strengths the critic identified.

CRITIQUE:
{critique}

CURRENT MEMO:
{memo}

Now produce the revised memo. Use the same structure.
"""
    response = claude_call(client, system, user_msg, max_tokens=8000)
    return extract_text(response)


def get_human_decision() -> str:
    """Returns one of: approve / revise / quit."""
    print("\n" + "─" * 70)
    print("  HUMAN REVIEW")
    print("─" * 70)
    print("  [a]pprove  — accept the memo, write to disk")
    print("  [r]evise   — send back to the writer with the critique")
    print("  [q]uit     — abandon")
    while True:
        choice = input("  Decision: ").strip().lower()
        if choice in ("a", "approve"):
            return "approve"
        if choice in ("r", "revise"):
            return "revise"
        if choice in ("q", "quit"):
            return "quit"
        print("  Please enter a, r, or q.")


def critic_loop(client, memo: str, research: dict, ticker: str, company_name: str,
                max_iterations: int = 5) -> tuple[str, bool]:
    """Returns (final_memo, approved)."""
    for i in range(1, max_iterations + 1):
        print(section(f"ITERATION {i} — MEMO"))
        print(memo)

        print(section(f"ITERATION {i} — CRITIC"))
        critique = run_critic(client, memo, research)
        print(critique)

        decision = get_human_decision()

        if decision == "approve":
            return memo, True
        if decision == "quit":
            return memo, False

        # revise
        print(section(f"ITERATION {i} — REVISING"))
        memo = run_reviser(client, memo, critique, ticker, company_name)

    print(f"\n[Reached max iterations = {max_iterations}.]")
    final_decision = get_human_decision()
    return memo, final_decision == "approve"
