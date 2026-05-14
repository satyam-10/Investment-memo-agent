"""
Investment Memo Writer — entry point.
 
Usage:
    python main.py
    python main.py AAPL
"""
 
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
 
from common import fetch_company_snapshot, get_client, section
from analyst_agent import run_analyst
from market_research_agent import run_market_research
from leadership_agent import run_leadership
from risk_agent import run_risk
from orchestrator import run_memo_writer, critic_loop
 
OUTPUT_DIR = Path(__file__).parent / "memos"
 
 
def gather_research(client, snapshot: dict) -> dict:
    """Run the four research agents in parallel."""
    ticker       = snapshot["ticker"]
    company_name = snapshot["info"].get("longName", ticker)
 
    print(section("RESEARCH PHASE — running 4 agents in parallel"))
    print(f"  Ticker  : {ticker}")
    print(f"  Company : {company_name}")
    print("  Agents  : Analyst | Market Research | Leadership | Risk")
    print("  This typically takes 2–4 minutes (market agent runs web searches)...\n")
 
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_analyst    = pool.submit(run_analyst,        client, snapshot)
        f_market     = pool.submit(run_market_research, client, ticker, company_name, snapshot)
        f_leadership = pool.submit(run_leadership,     client, ticker, company_name)
        f_risk       = pool.submit(run_risk,           client, snapshot)
 
        results = {}
        for name, future in [
            ("analyst",    f_analyst),
            ("market",     f_market),
            ("leadership", f_leadership),
            ("risk",       f_risk),
        ]:
            try:
                results[name] = future.result()
                print(f"  [✓] {name} agent done")
            except Exception as e:
                print(f"  [✗] {name} agent FAILED: {type(e).__name__}: {e}")
                raise
 
    return results
 
 
def save_memo(ticker: str, memo: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    path  = OUTPUT_DIR / f"{ticker}_memo_{stamp}.md"
    path.write_text(memo, encoding="utf-8")
    return path
 
 
def main():
    ticker = sys.argv[1].strip().upper() if len(sys.argv) > 1 else ""
    if not ticker:
        ticker = input("Enter ticker symbol: ").strip().upper()
    if not ticker:
        print("No ticker provided.")
        return
 
    client = get_client()
 
    print(section(f"Fetching market data for {ticker}"))
    snapshot     = fetch_company_snapshot(ticker)
    company_name = snapshot["info"].get("longName", ticker)
    print(f"  {company_name} — {len(snapshot['closes'])} trading days of history")
 
    research = gather_research(client, snapshot)
 
    print(section("MEMO COMPOSITION"))
    memo = run_memo_writer(client, ticker, company_name, research)
 
    print(section("CRITIC LOOP (human-in-the-loop)"))
    final_memo, approved = critic_loop(client, memo, research, ticker, company_name)
 
    if approved:
        path = save_memo(ticker, final_memo)
        print(section("APPROVED"))
        print(f"  Memo written to: {path}")
    else:
        print(section("ABANDONED"))
        print("  Memo not saved.")
 
 
if __name__ == "__main__":
    main()
 