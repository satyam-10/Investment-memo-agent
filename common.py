"""
Shared utilities for the Investment Memo multi-agent system.

Centralizes:
  - Claude client setup
  - yfinance data fetching (price history + fundamentals)
  - Formatting helpers
"""

import os
import sys
import io
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


WEB_SEARCH_TOOL = None  # web_search_20260209 is Claude-native; not supported by this proxy
