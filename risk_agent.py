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

from common import WEB_SEARCH_TOOL, claude_call, extract_text, fmt_pct


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

  1. Quantitative VaR/CVaR computed from the past two years of returns
  2. Your own web search results on the company's qualitative risk factors

Use web search to investigate (last 12 months preferred):
  - Regulatory / compliance issues
  - Open litigation or settlements
  - Supply chain or input-cost exposures
  - Customer or geographic concentration
  - Geopolitical exposure
  - Sector-specific risks (cyclicality, tech disruption, etc.)

Then produce this structured briefing:

## Quantitative Risk Profile
(Interpret the VaR numbers. Is the company high or low volatility for its
sector? Does historical VaR materially exceed parametric — i.e. fat tails?)

## Qualitative Risk Factors
(Bullet list, each with a source citation and a severity tag: HIGH / MEDIUM / LOW)

## Concentration & Dependency Risks
(Customer, geography, supplier, single-product?)

## Risk Verdict
ROUTINE / WATCH / ELEVATED — one line, then justify in 2-3 sentences.

Rules: cite every qualitative claim. Quote VaR numbers from the data block,
do not estimate. Under 800 words.
"""


def run_risk(client, snapshot: dict) -> dict:
    var_metrics = compute_var_cvar(snapshot["returns"], snapshot["closes"][-1])
    var_report = format_var_report(snapshot["ticker"], var_metrics)
    company_name = snapshot["info"].get("longName", snapshot["ticker"])

    user_msg = (
        f"Risk analysis for {company_name} ({snapshot['ticker']}).\n\n"
        f"{var_report}\n\n"
        f"Now use web search to find qualitative risk factors from the last 12 months "
        f"and produce the structured risk briefing."
    )

    response = claude_call(
        client,
        RISK_SYSTEM,
        user_msg,
        tools=WEB_SEARCH_TOOL,
        max_tokens=8000,
    )
    return {
        "var_metrics": var_metrics,
        "var_report": var_report,
        "analysis": extract_text(response),
    }
