"""
Risk Agent — quantitative VaR/CVaR plus web-sourced risk factors.

Two parts:
  1. Deterministic VaR/CVaR computation from yfinance return history
     (historical and parametric, multiple confidence levels)
  2. Web-search-driven narrative risk analysis (regulatory, litigation,
     supply chain, geopolitical, sector-specific)

The narrative analysis sees the VaR numbers and synthesises them with
qualitative findings into a single risk briefing.
"""

import numpy as np

from common import claude_call, extract_text, fmt_pct, run_searches


def compute_var_cvar(returns: np.ndarray, last_close: float) -> dict:
    """Historical + parametric VaR/CVaR at 95% and 99%, 1-day horizon."""
    if len(returns) < 30:
        raise ValueError("Not enough return history for VaR (need >= 30 days).")

    sorted_r = np.sort(returns)
    n = len(sorted_r)

    # Historical VaR (left-tail quantile of returns)
    hvar_95 = -sorted_r[int(n * 0.05)]
    hvar_99 = -sorted_r[int(n * 0.01)]
    # CVaR = mean loss beyond the VaR cut
    hcvar_95 = -sorted_r[: int(n * 0.05)].mean() if int(n * 0.05) > 0 else hvar_95
    hcvar_99 = -sorted_r[: int(n * 0.01)].mean() if int(n * 0.01) > 0 else hvar_99

    mu = returns.mean()
    sigma = returns.std()
    # Parametric (normal) VaR — z-scores at 95 / 99
    pvar_95 = -(mu - 1.645 * sigma)
    pvar_99 = -(mu - 2.326 * sigma)
    # Parametric CVaR (normal): sigma * phi(z) / (1 - alpha) - mu
    from math import sqrt, pi, exp
    def cvar_normal(alpha_tail):
        z = {0.05: 1.645, 0.01: 2.326}[alpha_tail]
        phi_z = exp(-0.5 * z * z) / sqrt(2 * pi)
        return sigma * phi_z / alpha_tail - mu

    pcvar_95 = cvar_normal(0.05)
    pcvar_99 = cvar_normal(0.01)

    return {
        "n_obs": n,
        "daily_mean": mu,
        "daily_vol": sigma,
        "ann_vol": sigma * np.sqrt(252),
        "hist_var_95": hvar_95,
        "hist_var_99": hvar_99,
        "hist_cvar_95": hcvar_95,
        "hist_cvar_99": hcvar_99,
        "param_var_95": pvar_95,
        "param_var_99": pvar_99,
        "param_cvar_95": pcvar_95,
        "param_cvar_99": pcvar_99,
        "last_close": last_close,
    }


def format_var_report(ticker: str, v: dict) -> str:
    lc = v["last_close"]
    lines = [
        f"VaR / CVaR ANALYSIS — {ticker}",
        f"  Observations    : {v['n_obs']} trading days",
        f"  Daily mean ret  : {fmt_pct(v['daily_mean'], 4)}",
        f"  Daily vol       : {fmt_pct(v['daily_vol'], 4)}",
        f"  Annualised vol  : {fmt_pct(v['ann_vol'])}",
        "",
        "  Historical (empirical left tail):",
        f"    95% 1-day VaR  : {fmt_pct(v['hist_var_95'])}   (loss per share: ${lc * v['hist_var_95']:.2f})",
        f"    95% 1-day CVaR : {fmt_pct(v['hist_cvar_95'])}   (loss per share: ${lc * v['hist_cvar_95']:.2f})",
        f"    99% 1-day VaR  : {fmt_pct(v['hist_var_99'])}   (loss per share: ${lc * v['hist_var_99']:.2f})",
        f"    99% 1-day CVaR : {fmt_pct(v['hist_cvar_99'])}   (loss per share: ${lc * v['hist_cvar_99']:.2f})",
        "",
        "  Parametric (normal assumption):",
        f"    95% 1-day VaR  : {fmt_pct(v['param_var_95'])}",
        f"    95% 1-day CVaR : {fmt_pct(v['param_cvar_95'])}",
        f"    99% 1-day VaR  : {fmt_pct(v['param_var_99'])}",
        f"    99% 1-day CVaR : {fmt_pct(v['param_cvar_99'])}",
        "",
        "  Note: VaR is the threshold loss; CVaR (Expected Shortfall) is the",
        "  average loss conditional on breaching VaR. Use historical for",
        "  fat-tailed names where the normal assumption underestimates risk.",
    ]
    return "\n".join(lines)


RISK_SYSTEM = """\
You are a senior risk analyst. You have two inputs:

  SOURCE A — Quantitative VaR/CVaR block
    Computed from the past two years of actual return history.
    Quote these numbers exactly — do not estimate or round them.
    Tag as [yfinance, live].

  SOURCE B — Web Search Results
    Real DuckDuckGo results collected moments ago, labelled [SEARCH: <query>].
    Use these as your PRIMARY source for all qualitative risk claims.
    Tag each claim with (Article Title, URL).
    If a search returned nothing on a topic, state that explicitly.

  [training knowledge — verify]
    Use ONLY for genuine gaps not covered by B. Flag every such claim.

Produce this structured briefing:

## Quantitative Risk Profile
Interpret the VaR numbers from SOURCE A. Is volatility high or low for the
sector? Does historical VaR materially exceed parametric (fat tails)?
Quote the exact 95% and 99% VaR and CVaR figures.

## Qualitative Risk Factors
Bullet list from SOURCE B. Each bullet: risk name, evidence from search
result (cite title + URL), severity tag HIGH / MEDIUM / LOW.
Do not invent risks not found in the search results.

## Concentration & Dependency Risks
Customer, geography, supplier, product concentration — cite search sources.

## Risk Verdict
ROUTINE / WATCH / ELEVATED — one line.
Then 2–3 sentences justifying it, each referencing a specific cited fact.

Rules: every qualitative claim needs a source. Under 900 words.
"""


def run_risk(client, snapshot: dict) -> dict:
    var_metrics = compute_var_cvar(snapshot["returns"], snapshot["closes"][-1])
    var_report = format_var_report(snapshot["ticker"], var_metrics)
    ticker = snapshot["ticker"]
    company_name = snapshot["info"].get("longName", ticker)

    queries = [
        f"{company_name} {ticker} regulatory fine lawsuit 2024 2025",
        f"{company_name} supply chain risk disruption 2024 2025",
        f"{company_name} {ticker} geopolitical risk tariff 2024 2025",
        f"{company_name} litigation settlement antitrust 2024 2025",
        f"{company_name} {ticker} revenue concentration customer risk 2024",
    ]
    search_block = run_searches(queries, label="risk")

    user_msg = f"""\
Risk analysis for {company_name} ({ticker}).

{'=' * 72}
SOURCE A — QUANTITATIVE VAR/CVAR  [yfinance, live]
{'=' * 72}
{var_report}

{'=' * 72}
SOURCE B — WEB SEARCH RESULTS  [DuckDuckGo, collected now]
{'=' * 72}
{search_block}
{'=' * 72}

Produce the structured risk briefing using both sources above.
"""
    response = claude_call(client, RISK_SYSTEM, user_msg, max_tokens=8000)
    return {"var_metrics": var_metrics, "var_report": var_report, "analysis": extract_text(response)}
