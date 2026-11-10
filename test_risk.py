"""Hand-checkable tests: run with `python -m pytest`."""
import numpy as np
import pandas as pd

import risk

rng = np.random.default_rng(0)
RET = pd.DataFrame(rng.normal(0.0003, 0.01, size=(1000, 3)) @ np.array(
    [[1.0, 0.3, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 0.8]]), columns=["A", "B", "C"])
W = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


def test_contributions_sum_to_portfolio_vol():
    table, sigma_p = risk.risk_contribution(W, RET)
    direct = risk.portfolio_returns(W, RET).std() * np.sqrt(252)
    assert np.isclose(table["Contribution"].sum(), sigma_p)
    assert np.isclose(sigma_p, direct)
    assert np.isclose(table["% of Risk"].sum(), 1.0)


def test_active_risk_matches_realized_te():
    bench = pd.Series({"A": 0.6, "B": 0.4})
    table, te = risk.active_risk(W, bench, RET)
    realized = risk.realized_tracking_error(risk.portfolio_returns(W, RET),
                                            risk.portfolio_returns(bench, RET))
    assert np.isclose(table["Contribution to TE"].sum(), te)
    assert np.isclose(te, realized)


def test_var_ordering_and_scaling():
    pr = risk.portfolio_returns(W, RET)
    v95, v99 = risk.var_cvar(pr, 0.95), risk.var_cvar(pr, 0.99)
    assert v99["Parametric VaR"] > v95["Parametric VaR"] > 0
    assert v95["Historical CVaR"] >= v95["Historical VaR"]
    v95_21 = risk.var_cvar(pr, 0.95, horizon_days=21)
    assert np.isclose(v95_21["Historical VaR"], v95["Historical VaR"] * np.sqrt(21))


def test_drift_trades_are_cash_neutral():
    df = risk.drift_check(pd.Series({"A": 70.0, "B": 20.0, "C": 10.0}),
                          pd.Series({"A": 0.6, "B": 0.3, "C": 0.1}))
    assert np.isclose(df["Trade ($)"].sum(), 0.0)
    assert list(df["Out of Band"]) == [True, True, False]
