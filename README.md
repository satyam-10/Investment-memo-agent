# Investment Memo Writer

A multi-agent system that produces institutional-grade investment memos for any publicly listed ticker. Four specialist agents run in parallel to gather research, a memo writer synthesises their output, a critic reviews it, and a human approves or sends it back for revision.

## How it works

```
            ┌─────────────────────────────────────────┐
            │  Fetch market data (yfinance, 2y)       │
            └────────────────────┬────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │            (4 agents run in parallel)           │
        ▼            ▼                       ▼            ▼
   ┌────────┐  ┌──────────┐         ┌──────────────┐  ┌──────┐
   │Analyst │  │  Market  │         │  Leadership  │  │ Risk │
   │  (#s)  │  │ Research │         │  & Strategy  │  │ VaR  │
   └────┬───┘  └─────┬────┘         └──────┬───────┘  └──┬───┘
        └────────────┴────────────┬────────┴─────────────┘
                                  ▼
                         ┌─────────────────┐
                         │   Memo Writer   │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │     Critic      │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Human review   │  ── revise ──► back to Critic
                         │  approve / quit │
                         └─────────────────┘
```

## The agents

| Agent | What it does | Inputs |
|---|---|---|
| **Analyst** (`analyst_agent.py`) | Fundamental + price-action snapshot. Builds a deterministic data block (valuation multiples, margins, growth, balance sheet, drawdown, volatility) and asks the LLM for a BUY/HOLD/SELL view grounded in those numbers. | `yfinance` |
| **Market Research** (`market_research_agent.py`) | Competitive landscape, value proposition, differentiation, market position. | LLM (web search if available) |
| **Leadership** (`leadership_agent.py`) | CEO and senior-leadership profile, strategic direction, capital allocation track record, governance flags. | LLM (web search if available) |
| **Risk** (`risk_agent.py`) | Deterministic VaR/CVaR at 95% and 99% (historical + parametric, 1-day horizon) **plus** an LLM-driven qualitative risk briefing that interprets the numbers alongside regulatory/litigation/supply-chain factors. | `yfinance` + LLM |

The **Memo Writer** (`orchestrator.py`) takes all four outputs and composes a single memo with the standard sections an IC would expect: Recommendation, Executive Summary, Thesis, Financials, Market Position, Leadership, Risk, Catalysts, Bear Case, Decision Summary.

The **Critic** (`orchestrator.py`) reviews the memo on internal consistency, use of data, source integration, contradictions surfaced, bear-case quality, and actionability — and returns APPROVE / REVISE / REJECT with a required-changes checklist.

The **Human** (you) decides whether to approve, revise, or quit. Revising re-runs the writer with the critique and loops back to the critic.

## Installation

Requires Python 3.10+.

```bash
git clone <this repo>
cd <this repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` (or create `.env` directly) and fill in:

```env
API_KEY="your-api-key-here"
API_BASE_URL="https://your-llm-endpoint/v1"
MODEL="your-model-name"
```

The client uses the OpenAI-compatible Chat Completions interface, so any OpenAI-compatible endpoint works.

> **Note on web search.** The Market Research, Leadership, and Risk agents are designed to use Claude-native web search. Through an OpenAI-compatible proxy that tool isn't available, so `WEB_SEARCH_TOOL` in `common.py` is set to `None` and those agents currently rely on the model's parametric knowledge. Swap in your provider's tool spec to re-enable live web search.

## Usage

```bash
# Prompted for a ticker
python main.py

# Or pass it directly
python main.py AAPL
```

Run looks like this:

```
======================================================================
  Fetching market data for AAPL
======================================================================
  Apple Inc. — 502 trading days of history

======================================================================
  RESEARCH PHASE — running 4 agents in parallel
======================================================================
  ...
  [✓] analyst agent done
  [✓] market agent done
  [✓] leadership agent done
  [✓] risk agent done

======================================================================
  MEMO COMPOSITION
======================================================================
...

======================================================================
  CRITIC LOOP (human-in-the-loop)
======================================================================
...

──────────────────────────────────────────────────────────────────────
  HUMAN REVIEW
──────────────────────────────────────────────────────────────────────
  [a]pprove  — accept the memo, write to disk
  [r]evise   — send back to the writer with the critique
  [q]uit     — abandon
  Decision:
```

Approved memos are written to `memos/<TICKER>_memo_<YYYY-MM-DD_HHMM>.md`.

## Project structure

```
.
├── main.py                      # Entry point — orchestrates the run
├── common.py                    # Client setup, yfinance fetch, formatting helpers
├── analyst_agent.py             # Fundamental + price-action analysis
├── market_research_agent.py     # Competitive landscape
├── leadership_agent.py          # CEO / strategy / governance
├── risk_agent.py                # VaR/CVaR + qualitative risk
├── orchestrator.py              # Memo writer, critic, human-in-the-loop
├── requirements.txt
├── .env                         # API_KEY, API_BASE_URL, MODEL  (gitignored)
└── memos/                       # Approved memos land here
```

## Design notes

- **Deterministic where it matters.** Anything numerical — multiples, returns, drawdown, VaR, CVaR — is computed in Python from `yfinance` and passed to the LLM as a fixed data block. The model is instructed to quote those numbers, not estimate them. This keeps the financial claims auditable.
- **Parallel research.** The four research agents are independent and run on a `ThreadPoolExecutor` to keep wall time down.
- **Critic loop, not autonomy.** The critic does not get to approve the memo on its own. The human is the final arbiter; the critic's job is to surface what the human should be looking at.
- **Iteration cap.** The critic loop runs for up to 5 rounds before forcing a final decision, so a stubborn disagreement between writer and critic can't run forever.

## Limitations

- `yfinance` data is best-effort scraping of Yahoo Finance — fields can be missing, especially for non-US tickers. The formatters handle `None` gracefully but expect occasional `N/A` cells.
- This is research tooling, not investment advice. Memos are a starting point for human analysis, not a substitute for it.

## License

Internal project. Add a license before distribution.
