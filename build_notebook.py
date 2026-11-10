"""Builds and executes portfolio_risk_report.ipynb. Run: python build_notebook.py"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Portfolio Risk Report: Model Portfolio vs. Policy Benchmark

A risk review of a diversified **model portfolio** (illustrative weights, public ETFs only) against a **60/40 policy benchmark**. It answers four questions an investment committee asks:

1. **How much could we lose?** Value at risk (VaR) and conditional VaR (CVaR), parametric and historical.
2. **Where does the risk come from?** Each holding's contribution to total volatility.
3. **How far are we from the policy portfolio?** Tracking error, and which active bets drive it.
4. **Are we inside our allocation bands?** Drift check and the trades to get back to target.

Data: daily adjusted closes from Yahoo Finance. The functions live in `risk.py` and are tested in `test_risk.py`."""),
code("""import numpy as np
import pandas as pd
import risk

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

PORTFOLIO = pd.Series({
    "SPY": 0.30,  # US large cap
    "IWM": 0.05,  # US small cap
    "EFA": 0.15,  # developed international
    "AGG": 0.25,  # US core bonds
    "TIP": 0.05,  # inflation-protected Treasuries
    "VNQ": 0.05,  # REITs
    "GLD": 0.05,  # gold
    "BIL": 0.10,  # T-bills (cash)
})
# 60/40 policy benchmark, built from the same index funds so active weights read cleanly:
# 60% equity split 2:1 US/international, 40% core bonds
BENCHMARK = pd.Series({"SPY": 0.40, "EFA": 0.20, "AGG": 0.40})
assert np.isclose(PORTFOLIO.sum(), 1) and np.isclose(BENCHMARK.sum(), 1)

prices = risk.load_prices(PORTFOLIO.index.union(BENCHMARK.index), years=5)
rets = risk.daily_returns(prices).dropna()
print(f"{len(rets)} trading days, {rets.index[0].date()} to {rets.index[-1].date()}")"""),
md("""## 1. Summary statistics"""),
code("""port = risk.portfolio_returns(PORTFOLIO, rets)
bench = risk.portfolio_returns(BENCHMARK, rets)

def summary(r):
    years = len(r) / risk.TRADING_DAYS
    growth = (1 + r).prod()
    dd = (1 + r).cumprod() / (1 + r).cumprod().cummax() - 1
    return {"CAGR": growth ** (1 / years) - 1,
            "Volatility": r.std() * np.sqrt(risk.TRADING_DAYS),
            "Max Drawdown": dd.min()}

pd.DataFrame({"Portfolio": summary(port), "Benchmark": summary(bench)})"""),
md("""## 2. VaR and CVaR

**VaR** is the loss we shouldn't exceed on, say, 95% of days. **CVaR** is the average loss on the days that are worse than VaR, so it describes the tail. Parametric assumes normal returns; historical uses the actual return history, so it picks up fat tails. Monthly figures scale the daily ones by the square root of 21 trading days."""),
code("""rows = {}
for conf in (0.95, 0.99):
    for label, h in (("1-day", 1), ("1-month", 21)):
        rows[f"{conf:.0%} {label}"] = risk.var_cvar(port, conf, h)
var_table = pd.DataFrame(rows).T
var_table"""),
code("""ACCOUNT = 1_000_000
print(f"On a ${ACCOUNT:,.0f} account, 95% 1-month historical VaR = "
      f"${var_table.loc['95% 1-month', 'Historical VaR'] * ACCOUNT:,.0f}")"""),
md("""## 3. Where the risk comes from

**MCTR** (marginal contribution to risk) is how much portfolio volatility rises per unit of added weight. Weight x MCTR gives each holding's **contribution**, and the contributions add up exactly to portfolio volatility. The usual finding: equities carry far more of the risk than their weight."""),
code("""rc, sigma_p = risk.risk_contribution(PORTFOLIO, rets)
print(f"Portfolio volatility: {sigma_p:.2%}")
rc.sort_values("% of Risk", ascending=False)"""),
md("""## 4. Tracking error vs. the policy benchmark

**Tracking error** is the volatility of (portfolio return minus benchmark return): the size of the active bets. The table splits it by holding, using **active weight** = portfolio weight minus benchmark weight. A negative contribution means that bet *reduces* tracking error, usually because it offsets another active bet."""),
code("""ar, te = risk.active_risk(PORTFOLIO, BENCHMARK, rets)
te_check = risk.realized_tracking_error(port, bench)
print(f"Ex-ante tracking error: {te:.2%}   (realized check: {te_check:.2%})")
ar.sort_values("% of Active Risk", ascending=False)"""),
md("""## 5. Drift check and rebalance trades

Suppose the account was set at target weights three years ago and never rebalanced. Market moves change the weights. Anything more than **+/- 5 percentage points** from target is out of band, and the trades below return every holding to target (they net to zero)."""),
code("""last_3y = rets.iloc[-756:]
start_value = 1_000_000
current_values = PORTFOLIO * start_value * (1 + last_3y[PORTFOLIO.index]).prod()
drift = risk.drift_check(current_values, PORTFOLIO, band=0.05)
print(f"Account value now: ${current_values.sum():,.0f}; trades net to ${drift['Trade ($)'].sum():,.2f}")
drift"""),
code("""# Tighter 2-point band, to show which holdings move first
risk.drift_check(current_values, PORTFOLIO, band=0.02)[["Drift", "Out of Band"]]"""),
md("""## Limits of this analysis

- **Backward-looking.** Every number is what these weights *would have* done over the sample. It's a risk estimate, not a forecast.
- **Normal and square-root-of-time assumptions** understate tail risk when volatility clusters; the historical columns are the check on that.
- **Sample choice matters.** A five-year window includes the 2022 bond sell-off, which raised the correlation between stocks and bonds."""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "portfolio_risk_report.ipynb")
print("wrote portfolio_risk_report.ipynb")
