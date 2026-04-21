"""
market_neutral.py -- CAPM market-neutrality check for strategy backtests.

Runs CAPM regressions against Nifty 50 daily returns:
    (r_p - r_f) = alpha + beta * (r_m - r_f) + eps

Strategies checked:
  - MPHDRL
  - Benchmark RL
  - Traditional pairs

Data sources (defaults):
  - data/backtest/{mphdrl,benchmark,traditional}.csv
  - data/trading/nifty50.csv

Usage:
  python market_neutral.py
  python market_neutral.py --risk-free-daily 0.0001
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


STRATEGY_FILES = (
    ("MPHDRL", "mphdrl.csv"),
    ("Benchmark RL", "benchmark.csv"),
    ("Traditional pairs", "traditional.csv"),
)

BACKTEST_DIR_CANDIDATES = (
    os.path.join("data", "backtest"),
    os.path.join("data", "backtesting"),
)


@dataclass
class CapmResult:
    strategy: str
    n_obs: int
    alpha_daily: float
    beta: float
    alpha_pvalue: float
    beta_pvalue: float
    r_squared: float
    market_neutral_5pct: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAPM market-neutrality check vs Nifty 50.")
    p.add_argument(
        "--backtest-dir",
        type=str,
        default="",
        help="Directory with strategy CSVs (default auto-detect data/backtest*).",
    )
    p.add_argument(
        "--nifty-path",
        type=str,
        default=os.path.join("data", "trading", "nifty50.csv"),
        help="Path to Nifty 50 OHLCV CSV.",
    )
    p.add_argument(
        "--nifty-price-col",
        type=str,
        default="Adj Close",
        help="Price column for Nifty market return (fallback to Close if unavailable).",
    )
    p.add_argument(
        "--risk-free-daily",
        type=float,
        default=0.0,
        help="Daily risk-free rate used in CAPM excess return computation.",
    )
    return p.parse_args()


def resolve_backtest_dir(explicit: str) -> str:
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        raise FileNotFoundError(f"Backtest directory not found: {explicit}")
    for d in BACKTEST_DIR_CANDIDATES:
        if os.path.isdir(d):
            return d
    tried = ", ".join(BACKTEST_DIR_CANDIDATES)
    raise FileNotFoundError(f"No backtest directory found. Tried: {tried}")


def load_nifty_returns(path: str, price_col: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Nifty file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    if price_col not in df.columns:
        if "Close" in df.columns:
            price_col = "Close"
        else:
            raise ValueError(f"Nifty file missing both '{price_col}' and 'Close'.")
    out = df[["Date", price_col]].copy()
    out = out.rename(columns={"Date": "date", price_col: "market_price"})
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out["market_return"] = out["market_price"].pct_change()
    return out[["date", "market_return"]].dropna().reset_index(drop=True)


def load_strategy_returns(backtest_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(backtest_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["date"])
    if "net_portfolio_return" not in df.columns:
        raise ValueError(f"{path} missing net_portfolio_return column.")
    out = df[["date", "net_portfolio_return"]].copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


def run_capm(strategy: str, merged: pd.DataFrame, rf_daily: float) -> CapmResult:
    y = merged["net_portfolio_return"].to_numpy(dtype=np.float64) - float(rf_daily)
    x = merged["market_return"].to_numpy(dtype=np.float64) - float(rf_daily)

    if HAS_STATSMODELS:
        x_with_const = sm.add_constant(x, has_constant="add")
        model = sm.OLS(y, x_with_const, missing="drop")
        res = model.fit()
        alpha = float(res.params[0])
        beta = float(res.params[1])
        alpha_p = float(res.pvalues[0]) if len(res.pvalues) > 0 else float("nan")
        beta_p = float(res.pvalues[1]) if len(res.pvalues) > 1 else float("nan")
        r2 = float(res.rsquared)
    else:
        # Fallback: estimate alpha/beta with least squares; p-values unavailable.
        x_mat = np.column_stack([np.ones_like(x), x])
        coeffs, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
        alpha = float(coeffs[0])
        beta = float(coeffs[1])
        alpha_p = float("nan")
        beta_p = float("nan")
        y_hat = x_mat @ coeffs
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-16 else float("nan")

    neutral = bool(abs(beta) < 0.1 and (np.isnan(beta_p) or beta_p >= 0.05))
    return CapmResult(
        strategy=strategy,
        n_obs=int(len(merged)),
        alpha_daily=alpha,
        beta=beta,
        alpha_pvalue=alpha_p,
        beta_pvalue=beta_p,
        r_squared=r2,
        market_neutral_5pct=neutral,
    )


def format_results_table(results: List[CapmResult]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in results:
        rows.append(
            {
                "strategy": r.strategy,
                "n_obs": r.n_obs,
                "alpha_daily": r.alpha_daily,
                "beta": r.beta,
                "alpha_pvalue": r.alpha_pvalue,
                "beta_pvalue": r.beta_pvalue,
                "r_squared": r.r_squared,
                "market_neutral_5pct": r.market_neutral_5pct,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    try:
        backtest_dir = resolve_backtest_dir(args.backtest_dir)
        mkt = load_nifty_returns(args.nifty_path, args.nifty_price_col)
    except Exception as e:
        print(f"Setup error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 88)
    print("CAPM Market Neutrality Check vs Nifty 50")
    print("=" * 88)
    print(f"Backtest dir: {os.path.abspath(backtest_dir)}")
    print(f"Nifty path:   {os.path.abspath(args.nifty_path)}")
    print(f"Risk-free:    {args.risk_free_daily:.8f} (daily)")
    print(f"statsmodels:  {'yes' if HAS_STATSMODELS else 'no (using numpy fallback)'}")
    print()

    results: List[CapmResult] = []
    skipped = []
    for name, fname in STRATEGY_FILES:
        try:
            s = load_strategy_returns(backtest_dir, fname)
            merged = s.merge(mkt, on="date", how="inner").dropna()
            if merged.empty:
                raise ValueError("no overlapping dates with market returns")
            results.append(run_capm(name, merged, args.risk_free_daily))
        except Exception as e:
            skipped.append((name, str(e)))

    if not results:
        print("No strategy regressions could be run.", file=sys.stderr)
        for name, msg in skipped:
            print(f"  - {name}: {msg}", file=sys.stderr)
        sys.exit(1)

    table = format_results_table(results)
    print("CAPM regression results")
    print("-" * 88)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(table.round(6).to_string(index=False))

    if skipped:
        print("\nSkipped strategies")
        print("-" * 88)
        for name, msg in skipped:
            print(f"  - {name}: {msg}")


if __name__ == "__main__":
    main()
