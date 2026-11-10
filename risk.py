"""Portfolio risk toolkit: VaR/CVaR, risk contribution, tracking error, drift.

All functions take plain pandas objects so they can be checked by hand.
Returns are daily simple returns; annualization uses 252 trading days.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def load_prices(tickers, years=5):
    """Daily adjusted closes from Yahoo Finance (free, public data)."""
    import yfinance as yf

    px = yf.download(sorted(set(tickers)), period=f"{years}y",
                     auto_adjust=True, progress=False)["Close"]
    return px.dropna(axis=1, how="all")


def daily_returns(prices):
    # Drop rows only after selecting the columns in use, so one short-history
    # ticker can't silently shorten every other ticker's sample.
    return prices.pct_change(fill_method=None).dropna(how="all")


def portfolio_returns(weights, returns):
    r = returns[weights.index].dropna()
    return r @ weights


def var_cvar(port_ret, conf=0.95, horizon_days=1):
    """Parametric (normal) and historical VaR/CVaR, as positive loss fractions.

    Multi-day figures use square-root-of-time scaling, which assumes returns
    are independent from day to day.
    """
    scale = np.sqrt(horizon_days)
    mu, sd = port_ret.mean(), port_ret.std()
    z = norm.ppf(1 - conf)
    p_var = -(mu + z * sd) * scale
    p_cvar = -(mu - sd * norm.pdf(z) / (1 - conf)) * scale

    cutoff = port_ret.quantile(1 - conf)
    h_var = -cutoff * scale
    h_cvar = -port_ret[port_ret <= cutoff].mean() * scale
    return {"Parametric VaR": p_var, "Parametric CVaR": p_cvar,
            "Historical VaR": h_var, "Historical CVaR": h_cvar}


def risk_contribution(weights, returns):
    """Each holding's contribution to total portfolio volatility (annualized).

    MCTR_i = (Cov @ w)_i / sigma_p      marginal: extra vol per unit of weight
    CTR_i  = w_i * MCTR_i               contributions sum to sigma_p
    """
    cov = returns[weights.index].dropna().cov() * TRADING_DAYS
    w = weights.values
    sigma_p = np.sqrt(w @ cov.values @ w)
    mctr = cov.values @ w / sigma_p
    ctr = w * mctr
    out = pd.DataFrame({"Weight": w, "MCTR": mctr, "Contribution": ctr,
                        "% of Risk": ctr / sigma_p}, index=weights.index)
    return out, sigma_p


def active_risk(weights, bench_weights, returns):
    """Tracking error and each holding's contribution to it.

    Active weights a = w - b. Tracking error = sqrt(a' Cov a).
    Contribution to active risk_i = a_i * (Cov @ a)_i / TE; sums to TE.
    """
    names = weights.index.union(bench_weights.index)
    w = weights.reindex(names, fill_value=0.0)
    b = bench_weights.reindex(names, fill_value=0.0)
    a = (w - b).values
    cov = returns[names].dropna().cov() * TRADING_DAYS
    te = np.sqrt(a @ cov.values @ a)
    contrib = a * (cov.values @ a) / te
    out = pd.DataFrame({"Active Weight": a, "Contribution to TE": contrib,
                        "% of Active Risk": contrib / te}, index=names)
    return out, te


def realized_tracking_error(port_ret, bench_ret):
    """Ex-post check: stdev of the return difference, annualized."""
    diff = (port_ret - bench_ret).dropna()
    return diff.std() * np.sqrt(TRADING_DAYS)


def drift_check(current_values, targets, band=0.05):
    """Flag holdings outside target +/- band and size the trades back to target.

    band is absolute (0.05 = 5 percentage points). Trade ($) is the dollar
    amount that returns each holding to target (positive = buy). The trades
    sum to zero, so a full rebalance is cash-neutral.
    """
    total = current_values.sum()
    cur = current_values / total
    df = pd.DataFrame({"Target": targets, "Current": cur})
    df["Drift"] = df["Current"] - df["Target"]
    df["Out of Band"] = df["Drift"].abs() > band
    df["Trade ($)"] = (df["Target"] - df["Current"]) * total
    return df
