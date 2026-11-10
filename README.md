# Portfolio Risk Analytics

A risk review of a diversified model portfolio against a 60/40 policy benchmark. It covers the four questions an investment committee asks about a portfolio:

1. **How much could we lose?** Value at risk (VaR) and conditional VaR (CVaR), both parametric and historical.
2. **Where does the risk come from?** Each holding's contribution to total volatility.
3. **How far are we from the policy portfolio?** Tracking error, and which active bets drive it.
4. **Are we inside our allocation bands?** A drift check, and the trades to get back to target.

**Open [`portfolio_risk_report.ipynb`](portfolio_risk_report.ipynb)** to see the full report with outputs.

The portfolio weights are illustrative, not a client account. All holdings are public index ETFs, and the price data comes from Yahoo Finance (5 years of daily data, Nov 2021 to Nov 2026).

## Key results

| | Portfolio | 60/40 benchmark |
|---|---|---|
| Annual return (CAGR) | 6.4% | 6.3% |
| Volatility | 10.0% | 10.7% |
| Max drawdown | -19.2% | -21.2% |
| 95% one-month VaR (historical) | 4.3% (about $43k on $1M) | |
| Tracking error | 1.56% | |

- **Equities hold far more of the risk than their weight.** US large cap (SPY) is 30% of the portfolio but **49% of its risk**. Core bonds are 25% of the weight and only 7% of the risk.
- **The largest source of active risk is the US equity underweight**, at 57% of tracking error. Small caps (IWM) *reduce* tracking error, because they partly refill that underweight.
- **Three years without rebalancing** pushes US large cap 5.8 points over target and bonds 5.8 points under. Both are outside a 5-point band, and the notebook sizes the cash-neutral trades to fix it.

## How the numbers are calculated

| Metric | Formula | In plain words |
|---|---|---|
| Parametric VaR | -(mean + z x stdev) x sqrt(days) | The loss you shouldn't exceed at the chosen confidence, if returns are normal |
| Historical VaR | The 5th (or 1st) percentile of actual returns | The same idea using what actually happened, fat tails included |
| CVaR | The average loss beyond VaR | How bad the bad days are |
| MCTR | (Cov x w) / portfolio volatility | Extra portfolio risk per unit of added weight |
| Contribution to risk | weight x MCTR | Adds up exactly to portfolio volatility |
| Tracking error | sqrt(a' Cov a), where a = active weights | The volatility of the return gap versus the benchmark |

Two checks are built in. The risk contributions must add up to total volatility. The covariance-based tracking error must match the realized standard deviation of the return gap.

## Files

| File | What it is |
|---|---|
| `portfolio_risk_report.ipynb` | The report, with outputs |
| `risk.py` | The risk functions, kept short so each can be checked by hand |
| `test_risk.py` | Tests: contributions sum to the total, tracking error matches the realized figure, VaR scales correctly, rebalance trades net to zero (`python -m pytest`) |
| `build_notebook.py` | Regenerates and re-runs the notebook with fresh data |

## Limits

- **The numbers look backward.** They show what these weights *would have* done over the sample. They're a risk estimate, not a forecast.
- **Normal returns and square-root-of-time scaling understate tail risk when volatility clusters.** The historical VaR columns are the check on that.
- **The sample choice matters.** This window includes 2022, when stocks and bonds fell together.

## Background

This is a public version of risk analysis I built during a portfolio management internship at a registered investment adviser. That work compared a client's current and proposed portfolios on these same measures. The client data stays private, so this version uses a model portfolio.

---
Built with AI coding assistance under my direction. I designed the analysis, checked the math, and verified every output.
