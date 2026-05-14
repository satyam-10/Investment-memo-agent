"""
Shared utilities for the Investment Memo multi-agent system.

Centralizes:
  - OpenAI-compatible client setup
  - yfinance data fetching (price history + fundamentals)
  - Web search via DuckDuckGo (no API key required)
  - Formatting helpers
"""

import json
import os
import sys
import io
import time
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL = os.getenv("MODEL")


def get_client() -> OpenAI:
    if not API_KEY:
        raise SystemExit("API_KEY is not set (check your .env).")
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ── yfinance data ───────────────────────────────────────────────────
def fetch_company_snapshot(ticker: str) -> dict:
    """One-shot fetch of everything the agents will need from yfinance."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    hist = t.history(period="2y", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price history for '{ticker}'")

    closes = hist["Close"].values.astype(float)
    returns = np.diff(np.log(closes))

    return {
        "ticker": ticker.upper(),
        "info": info,
        "price_history": hist,
        "closes": closes,
        "returns": returns,
    }


def fmt_num(val, prefix="", suffix="", decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return str(val)
    if abs(v) >= 1e12:
        return f"{prefix}{v/1e12:.{decimals}f}T{suffix}"
    if abs(v) >= 1e9:
        return f"{prefix}{v/1e9:.{decimals}f}B{suffix}"
    if abs(v) >= 1e6:
        return f"{prefix}{v/1e6:.{decimals}f}M{suffix}"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n  {title}\n{bar}"


def claude_call(
    client: OpenAI,
    system: str,
    user: str,
    *,
    tools: list | None = None,
    max_tokens: int = 8000,
):
    """Single API call via OpenAI-compatible client. Returns the raw ChatCompletion."""
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)


def extract_text(response) -> str:
    """Extract text content from an OpenAI-format chat completion response."""
    return (response.choices[0].message.content or "").strip()


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return formatted results."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    Source: {r['href']}")
            lines.append(f"    {r['body']}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed for '{query}': {e}"


def run_searches(queries: list[str], label: str = "", sleep: float = 1.5) -> str:
    """Run multiple queries and return results as labelled blocks for prompt injection."""
    blocks = []
    total = len(queries)
    for i, q in enumerate(queries, 1):
        if label:
            print(f"    [{label}] web search {i}/{total}: {q[:72]}")
        result = web_search(q)
        blocks.append(f"[SEARCH: {q}]\n{result}")
        if i < total:
            time.sleep(sleep)
    return "\n\n".join(blocks)


WEB_SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information about companies, people, or events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    }
]


def claude_call_with_search(
    client: OpenAI,
    system: str,
    user: str,
    *,
    max_rounds: int = 6,
    max_tokens: int = 8000,
) -> str:
    """Agentic loop: model can call web_search repeatedly until it produces a final answer."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=messages,
            tools=WEB_SEARCH_TOOL,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return (msg.content or "").strip()

        # Append assistant turn with its tool-call requests
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            if tc.function.name == "web_search":
                args = json.loads(tc.function.arguments)
                result = web_search(args.get("query", ""))
            else:
                result = f"Unknown tool: {tc.function.name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Exhausted rounds — get a final response without offering tools
    final = client.chat.completions.create(model=MODEL, max_tokens=max_tokens, messages=messages)
    return (final.choices[0].message.content or "").strip()
