"""
backtest.py -- Walk-forward backtesting for MPHDRL, Benchmark RL, Traditional pairs (benchmark3), and Nifty 50 buy-and-hold.

Produces daily gross/net PnL, returns, costs, taxes for each strategy and exports
CSV files to data/backtest/ for downstream performance comparison.

Usage:
    python backtest.py
    python backtest.py --device cpu
    python backtest.py --mphdrl-checkpoint models/MPHDRL/final.pt \
                       --benchmark-checkpoint models/benchmark/final.pt
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from MPHDRL import (
    HPARAMS,
    MPHDRL_MODEL_DIR,
    MPHDRLTrader,
    build_pair_ticker_mapping,
)
from benchmark import BENCHMARK_MODEL_DIR, H_BENCHMARK, BenchmarkDDPG
from traditional import (
    TRADITIONAL_PARAMS_PATH,
    compute_traditional_weights_by_date,
    load_traditional_params,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
INITIAL_CASH = 10_000_000
TXN_COST_RATE = 0.0000307
SHORT_COST_ANNUAL = 0.0657
SHORT_COST_DAILY = SHORT_COST_ANNUAL / 252
STCG_RATE = 0.20
LTCG_RATE = 0.125
MGMT_FEE_ANNUAL = 0.0022
MGMT_FEE_DAILY = MGMT_FEE_ANNUAL / 252
FISCAL_YEAR_END = (3, 31)

BACKTEST_DIR = os.path.join("data", "backtest")

RESULT_COLUMNS = [
    "date",
    "gross_portfolio_value",
    "gross_long_pnl",
    "gross_short_pnl",
    "gross_long_return",
    "gross_short_return",
    "gross_portfolio_return",
    "transaction_cost",
    "shorting_cost",
    "tax_flow",
    "tax_carryforward",
    "net_long_pnl",
    "net_short_pnl",
    "net_long_return",
    "net_short_return",
    "net_portfolio_value",
    "net_portfolio_return",
]

# ---------------------------------------------------------------------------
# Data loading (mirrors test.py load_test_bundle + price pivot)
# ---------------------------------------------------------------------------

def load_test_bundle():
    base = os.path.join("data", "pickle")
    spread_dir = os.path.join("data", "spread")
    paths = {
        "X_test": os.path.join(base, "spread_X_test.pkl"),
        "y_test": os.path.join(base, "spread_y_test.pkl"),
        "pairs": os.path.join(base, "cointegrated_pairs.pkl"),
        "meta": os.path.join(spread_dir, "sequence_xy.csv"),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        print("Missing files:", [paths[m] for m in missing])
        print("Run preprocessing first.")
        sys.exit(1)

    with open(paths["X_test"], "rb") as f:
        x_test = pickle.load(f)
    with open(paths["y_test"], "rb") as f:
        y_test = pickle.load(f)
    with open(paths["pairs"], "rb") as f:
        pairs = pickle.load(f)

    meta = pd.read_csv(paths["meta"], parse_dates=["target_date"])
    if "Unnamed: 0" in meta.columns:
        meta = meta.drop(columns=["Unnamed: 0"])

    split_date = pd.to_datetime("2024-01-01")
    train_mask = meta["target_date"] < split_date
    test_idx = np.where(~train_mask.values)[0]
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    n = min(len(meta_test), len(x_test))
    meta_test = meta_test.iloc[:n].reset_index(drop=True)
    x_test = x_test[:n]
    y_test = y_test[:n]

    return x_test, y_test, pairs, meta_test


def load_price_matrix(tickers):
    """Pivot raw.csv to wide close-price matrix indexed by Date."""
    raw_path = os.path.join("data", "trading", "raw.csv")
    if not os.path.isfile(raw_path):
        print(f"Missing {raw_path}. Run collection.py first.")
        sys.exit(1)
    raw = pd.read_csv(raw_path, parse_dates=["Date"])
    wide = raw.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    missing_t = [t for t in tickers if t not in wide.columns]
    if missing_t:
        print(f"WARNING: {len(missing_t)} tickers missing from raw.csv: {missing_t[:5]}...")
    return wide


def load_nifty50():
    path = os.path.join("data", "trading", "nifty50.csv")
    if not os.path.isfile(path):
        print(f"Missing {path}. Run collection.py first.")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _default_ckpt(model_dir):
    for name in ("final.pt", "checkpoint.pt"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    return None

# ---------------------------------------------------------------------------
# Model weight extraction for all test dates
# ---------------------------------------------------------------------------

def _windows_for_date(meta_test, x_test, pair_key_to_idx, n_pairs, F_dim, d):
    sub = meta_test[meta_test["target_date"] == d]
    windows = np.zeros((n_pairs, x_test.shape[1], F_dim), dtype=np.float32)
    present = []
    for _, row in sub.iterrows():
        iloc = row.name
        pkey = str(row["Pair"])
        if pkey not in pair_key_to_idx:
            continue
        p_idx = pair_key_to_idx[pkey]
        windows[p_idx] = x_test[iloc]
        present.append(pkey)
    if not present:
        return None
    return windows


def get_all_test_weights(model, meta_test, x_test, pairs, pair_key_to_idx, n_pairs, F_dim, dates):
    """Run model.forward_step(explore=False) for every test date; return {date: w_np}."""
    weights_by_date = {}
    model.eval()
    for d in dates:
        windows = _windows_for_date(meta_test, x_test, pair_key_to_idx, n_pairs, F_dim, d)
        if windows is None:
            continue
        with torch.no_grad():
            out = model.forward_step(windows, explore=False)
        w = out["weights"].detach().cpu().numpy().reshape(-1)
        weights_by_date[d] = w
    return weights_by_date

# ---------------------------------------------------------------------------
# Fiscal year boundary helper
# ---------------------------------------------------------------------------

def _crosses_fiscal_year(d_prev, d_curr):
    """True when d_prev and d_curr straddle a March 31 boundary."""
    fy_mm, fy_dd = FISCAL_YEAR_END
    def fy_of(dt):
        if (dt.month, dt.day) > (fy_mm, fy_dd):
            return dt.year + 1
        return dt.year
    return fy_of(d_prev) != fy_of(d_curr)

# ---------------------------------------------------------------------------
# Realized PnL (for tax): only closed/reduced positions
# ---------------------------------------------------------------------------

def _compute_realized_pnl(w_prev, w_new, r, portfolio_value):
    """
    Turnover profit on the portion of each position that was closed or reduced.

    Returns scalar INR value (can be negative = realized loss).
    """
    closed = np.zeros_like(w_prev)
    for i in range(len(w_prev)):
        wp, wn = w_prev[i], w_new[i]
        if wp == 0.0:
            continue
        same_sign = (np.sign(wp) == np.sign(wn))
        if same_sign and abs(wn) < abs(wp):
            closed[i] = abs(wp) - abs(wn)
        elif not same_sign:
            closed[i] = abs(wp)
    turnover_profit = portfolio_value * np.nansum(closed * r * np.sign(w_prev))
    return turnover_profit

# ---------------------------------------------------------------------------
# Walk-forward loop for MPHDRL / Benchmark RL
# ---------------------------------------------------------------------------

def run_strategy_backtest(strategy_name, weights_by_date, price_wide, tickers, test_dates):
    """
    Daily walk-forward simulation.

    weights_by_date: {pd.Timestamp: np.array(n_tickers)}
    price_wide:      DataFrame indexed by Date with columns per ticker
    tickers:         ordered list matching weight vector columns
    test_dates:      sorted list of target_dates in the test set

    Returns a DataFrame with RESULT_COLUMNS.
    """
    price_dates = price_wide.index
    n_tickers = len(tickers)

    net_pv = float(INITIAL_CASH)
    gross_pv = float(INITIAL_CASH)
    prev_w = np.zeros(n_tickers, dtype=np.float64)
    tax_carryforward = 0.0
    prev_date = None
    rows = []

    for idx in range(len(test_dates) - 1):
        d_t = test_dates[idx]
        d_t1 = test_dates[idx + 1]

        if d_t not in price_dates or d_t1 not in price_dates:
            continue
        w = weights_by_date.get(d_t)
        if w is None:
            continue
        w = w.astype(np.float64)

        p_t = price_wide.loc[d_t, tickers].values.astype(np.float64)
        p_t1 = price_wide.loc[d_t1, tickers].values.astype(np.float64)
        safe_p = np.where(np.abs(p_t) < 1e-12, 1.0, p_t)
        r = (p_t1 - p_t) / safe_p

        w_long = np.maximum(w, 0.0)
        w_short = np.minimum(w, 0.0)
        prev_w_long = np.maximum(prev_w, 0.0)
        prev_w_short = np.minimum(prev_w, 0.0)

        gross_long_ret = np.nansum(w_long * r)
        gross_short_ret = np.nansum(w_short * r)
        gross_portfolio_ret = gross_long_ret + gross_short_ret
        gross_long_pnl = net_pv * gross_long_ret
        gross_short_pnl = net_pv * gross_short_ret
        gross_pnl = gross_long_pnl + gross_short_pnl
        gross_pv += gross_pnl

        turnover_long = np.nansum(np.abs(w_long - prev_w_long))
        turnover_short = np.nansum(np.abs(w_short - prev_w_short))
        total_turnover = turnover_long + turnover_short

        txn_cost = net_pv * total_turnover * TXN_COST_RATE
        if total_turnover > 0:
            txn_cost_long = txn_cost * (turnover_long / total_turnover)
            txn_cost_short = txn_cost * (turnover_short / total_turnover)
        else:
            txn_cost_long = 0.0
            txn_cost_short = 0.0

        gross_short_exposure = np.nansum(np.abs(w_short))
        shorting_cost = net_pv * gross_short_exposure * SHORT_COST_DAILY

        realized_gross = _compute_realized_pnl(prev_w, w, r, net_pv)
        net_realized_pnl = realized_gross - txn_cost - shorting_cost

        # tax_carryforward = non-negative pool of STCG already paid on realized gains
        # that loss-days can rebate against (never rebate without prior payments).
        if net_realized_pnl > 0:
            payment = net_realized_pnl * STCG_RATE
            tax_flow = -payment
            tax_carryforward += payment
        elif net_realized_pnl < 0:
            raw_rebate = abs(net_realized_pnl) * STCG_RATE
            rebate = min(raw_rebate, tax_carryforward)
            tax_flow = rebate
            tax_carryforward -= rebate
        else:
            tax_flow = 0.0

        if prev_date is not None and _crosses_fiscal_year(prev_date, d_t):
            if tax_carryforward > 0:
                tax_carryforward = 0.0

        tax_per_leg = tax_flow / 2.0

        net_long_pnl = gross_long_pnl - txn_cost_long - tax_per_leg
        net_short_pnl = gross_short_pnl - txn_cost_short - shorting_cost - tax_per_leg
        net_pnl = net_long_pnl + net_short_pnl

        net_long_ret = net_long_pnl / net_pv if net_pv else 0.0
        net_short_ret = net_short_pnl / net_pv if net_pv else 0.0
        net_portfolio_ret = net_pnl / net_pv if net_pv else 0.0

        net_pv += net_pnl

        rows.append({
            "date": d_t,
            "gross_portfolio_value": gross_pv,
            "gross_long_pnl": gross_long_pnl,
            "gross_short_pnl": gross_short_pnl,
            "gross_long_return": gross_long_ret,
            "gross_short_return": gross_short_ret,
            "gross_portfolio_return": gross_portfolio_ret,
            "transaction_cost": txn_cost,
            "shorting_cost": shorting_cost,
            "tax_flow": tax_flow,
            "tax_carryforward": tax_carryforward,
            "net_long_pnl": net_long_pnl,
            "net_short_pnl": net_short_pnl,
            "net_long_return": net_long_ret,
            "net_short_return": net_short_ret,
            "net_portfolio_value": net_pv,
            "net_portfolio_return": net_portfolio_ret,
        })

        prev_w = w.copy()
        prev_date = d_t

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    print(f"  {strategy_name}: {len(df)} trading days simulated")
    return df

# ---------------------------------------------------------------------------
# Nifty 50 buy-and-hold
# ---------------------------------------------------------------------------

def run_nifty_backtest(nifty_df, test_start, test_end):
    """
    Buy at first close in test window, hold daily with mgmt fee,
    sell at last close with txn cost, LTCG tax retroactively amortized.
    """
    mask = (nifty_df["Date"] >= test_start) & (nifty_df["Date"] <= test_end)
    nf = nifty_df.loc[mask].sort_values("Date").reset_index(drop=True)
    if len(nf) < 2:
        print("  Nifty 50: insufficient data in test window.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    closes = nf["Close"].values.astype(np.float64)
    dates = nf["Date"].values

    buy_cost = INITIAL_CASH * TXN_COST_RATE
    invested = INITIAL_CASH - buy_cost
    units = invested / closes[0]

    rows = []
    net_pv = invested

    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        curr_close = closes[i]
        gross_ret = (curr_close - prev_close) / prev_close if prev_close else 0.0
        gross_pnl = net_pv * gross_ret
        gross_pv = net_pv + gross_pnl

        mgmt_cost = gross_pv * MGMT_FEE_DAILY
        net_pnl = gross_pnl - mgmt_cost
        net_ret = net_pnl / net_pv if net_pv else 0.0
        net_pv += net_pnl

        txn_cost_day = buy_cost if i == 1 else 0.0

        rows.append({
            "date": pd.Timestamp(dates[i]),
            "gross_portfolio_value": gross_pv,
            "gross_long_pnl": gross_pnl,
            "gross_short_pnl": 0.0,
            "gross_long_return": gross_ret,
            "gross_short_return": 0.0,
            "gross_portfolio_return": gross_ret,
            "transaction_cost": txn_cost_day + mgmt_cost,
            "shorting_cost": 0.0,
            "tax_flow": 0.0,
            "tax_carryforward": 0.0,
            "net_long_pnl": net_pnl,
            "net_short_pnl": 0.0,
            "net_long_return": net_ret,
            "net_short_return": 0.0,
            "net_portfolio_value": net_pv,
            "net_portfolio_return": net_ret,
        })

    sell_cost = net_pv * TXN_COST_RATE
    pv_before_sell = net_pv
    net_pv -= sell_cost
    if rows:
        rows[-1]["transaction_cost"] += sell_cost
        rows[-1]["net_long_pnl"] -= sell_cost
        rows[-1]["net_portfolio_value"] = net_pv
        rows[-1]["net_long_return"] = rows[-1]["net_long_pnl"] / pv_before_sell if pv_before_sell else 0.0
        rows[-1]["net_portfolio_return"] = rows[-1]["net_long_return"]

    total_profit = net_pv - INITIAL_CASH
    n_days = len(rows)
    if total_profit > 0 and n_days > 0:
        ltcg_tax = total_profit * LTCG_RATE
        daily_tax = ltcg_tax / n_days
        for row in rows:
            row["tax_flow"] = -daily_tax
            row["net_long_pnl"] -= daily_tax
        running_pv = INITIAL_CASH - buy_cost
        for row in rows:
            running_pv += row["net_long_pnl"]
            row["net_portfolio_value"] = running_pv
            row["net_portfolio_return"] = row["net_long_pnl"] / max(running_pv - row["net_long_pnl"], 1.0)
            row["net_long_return"] = row["net_portfolio_return"]

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    print(f"  Nifty 50 B&H: {len(df)} trading days simulated")
    return df

# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward backtest for RL Pairs Trading")
    p.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu)")
    p.add_argument(
        "--mphdrl-checkpoint", type=str, default=None, dest="mphdrl_ckpt",
        help="MPHDRL .pt checkpoint (default: models/MPHDRL/final.pt or checkpoint.pt)",
    )
    p.add_argument(
        "--benchmark-checkpoint", type=str, default=None, dest="bench_ckpt",
        help="Benchmark .pt checkpoint (default: models/benchmark/final.pt or checkpoint.pt)",
    )
    p.add_argument(
        "--traditional-params",
        type=str,
        default=None,
        help=f"Traditional pairs params pickle (default: {TRADITIONAL_PARAMS_PATH})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    dev = torch.device(args.device)

    print("=" * 60)
    print("Walk-Forward Backtest")
    print("=" * 60)

    # --- Load test spread data ---
    x_test, y_test, pairs, meta_test = load_test_bundle()
    M, tickers, ticker_to_idx = build_pair_ticker_mapping(pairs)
    n_pairs = len(pairs)
    n_tickers = len(tickers)
    F_dim = x_test.shape[2]
    pair_key_to_idx = {f"{a}|{b}": i for i, (a, b) in enumerate(pairs)}

    test_dates = sorted(meta_test["target_date"].unique())
    print(f"Test dates: {len(test_dates)}  |  Pairs: {n_pairs}  |  Tickers: {n_tickers}")

    # --- Load price matrix ---
    price_wide = load_price_matrix(tickers)
    valid_dates = [d for d in test_dates if d in price_wide.index]
    print(f"Test dates with price data: {len(valid_dates)}")

    # --- Load Nifty 50 ---
    nifty_df = load_nifty50()

    # --- Load models ---
    mphdrl_ckpt = args.mphdrl_ckpt or _default_ckpt(MPHDRL_MODEL_DIR)
    bench_ckpt = args.bench_ckpt or _default_ckpt(BENCHMARK_MODEL_DIR)

    if mphdrl_ckpt is None or not os.path.isfile(mphdrl_ckpt):
        print(f"ERROR: No MPHDRL checkpoint found under {MPHDRL_MODEL_DIR}. Train first.")
        sys.exit(1)
    if bench_ckpt is None or not os.path.isfile(bench_ckpt):
        print(f"ERROR: No Benchmark checkpoint found under {BENCHMARK_MODEL_DIR}. Train first.")
        sys.exit(1)

    mphdrl_model = MPHDRLTrader(F_dim, n_pairs, n_tickers, M, device=str(dev))
    mphdrl_model.load_checkpoint(mphdrl_ckpt)
    mphdrl_model.eval()

    try:
        raw_bench = torch.load(bench_ckpt, map_location=str(dev), weights_only=False)
    except TypeError:
        raw_bench = torch.load(bench_ckpt, map_location=str(dev))
    hidden = int(raw_bench.get("meta", {}).get("hidden_size", H_BENCHMARK))
    bench_model = BenchmarkDDPG(F_dim, n_pairs, n_tickers, M, hidden_size=hidden, device=str(dev))
    bench_model.load_checkpoint(bench_ckpt)
    bench_model.eval()

    print(f"\nMPHDRL checkpoint:    {os.path.abspath(mphdrl_ckpt)}")
    print(f"Benchmark checkpoint: {os.path.abspath(bench_ckpt)}")
    print(f"Initial cash: {INITIAL_CASH:,.0f} INR")
    print(f"Txn cost: {TXN_COST_RATE*100:.5f}%  |  Short cost: {SHORT_COST_ANNUAL*100:.2f}% ann.")
    print(f"STCG: {STCG_RATE*100:.0f}%  |  LTCG: {LTCG_RATE*100:.1f}%  |  Mgmt fee: {MGMT_FEE_ANNUAL*100:.2f}% ann.")
    print()

    # --- Extract weights for all test dates ---
    print("Extracting model weights for all test dates...")
    mphdrl_weights = get_all_test_weights(
        mphdrl_model, meta_test, x_test, pairs, pair_key_to_idx, n_pairs, F_dim, test_dates,
    )
    bench_weights = get_all_test_weights(
        bench_model, meta_test, x_test, pairs, pair_key_to_idx, n_pairs, F_dim, test_dates,
    )
    print(f"  MPHDRL dates with weights: {len(mphdrl_weights)}")
    print(f"  Benchmark dates with weights: {len(bench_weights)}")

    trad_params_path = args.traditional_params or TRADITIONAL_PARAMS_PATH
    trad_weights: dict = {}
    if os.path.isfile(trad_params_path):
        trad_params = load_traditional_params(trad_params_path)
        hedge_path = os.path.join("data", "pickle", "hedge_ratios.pkl")
        with open(hedge_path, "rb") as f:
            hedge_ratios = pickle.load(f)
        spread_raw_path = os.path.join("data", "spread", "raw.csv")
        spread_wide = None
        if os.path.isfile(spread_raw_path):
            raw_sp = pd.read_csv(spread_raw_path, parse_dates=["Date"])
            spread_wide = raw_sp.pivot(index="Date", columns="Pair", values="spread").sort_index()
        trad_weights = compute_traditional_weights_by_date(
            valid_dates, pairs, hedge_ratios, trad_params, spread_wide=spread_wide,
        )
        print(f"  Traditional (benchmark3) dates with weights: {len(trad_weights)}")
    else:
        print(
            f"\n  SKIP Traditional pairs (benchmark3): missing {trad_params_path}\n"
            f"    Run:  python traditional.py"
        )

    # --- Run backtests ---
    print("\nRunning walk-forward backtests...")
    df_mphdrl = run_strategy_backtest("MPHDRL", mphdrl_weights, price_wide, tickers, valid_dates)
    df_bench = run_strategy_backtest("Benchmark", bench_weights, price_wide, tickers, valid_dates)
    df_trad = run_strategy_backtest(
        "Traditional pairs", trad_weights, price_wide, tickers, valid_dates,
    )

    test_start = pd.Timestamp(valid_dates[0]) if valid_dates else pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp(valid_dates[-1]) if valid_dates else pd.Timestamp("2025-12-31")
    df_nifty = run_nifty_backtest(nifty_df, test_start, test_end)

    # --- Export ---
    for name, df in [
        ("mphdrl", df_mphdrl),
        ("benchmark", df_bench),
        ("benchmark3", df_trad),
        ("nifty50", df_nifty),
    ]:
        out_path = os.path.join(BACKTEST_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("BACKTEST SUMMARY")
    print(f"{'=' * 60}")
    for label, df in [
        ("MPHDRL", df_mphdrl),
        ("Benchmark RL", df_bench),
        ("Traditional pairs (benchmark3)", df_trad),
        ("Nifty 50 B&H", df_nifty),
    ]:
        if df.empty:
            print(f"  {label}: no data")
            continue
        start_val = INITIAL_CASH
        end_val = df["net_portfolio_value"].iloc[-1]
        total_ret = (end_val - start_val) / start_val * 100
        total_txn = df["transaction_cost"].sum()
        total_short = df["shorting_cost"].sum()
        total_tax = -df["tax_flow"].sum()
        print(f"  {label}:")
        print(f"    Period:       {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
        print(f"    Final value:  {end_val:>14,.2f} INR")
        print(f"    Total return: {total_ret:>+10.4f}%")
        print(f"    Total txn:    {total_txn:>14,.2f}")
        print(f"    Total short:  {total_short:>14,.2f}")
        print(f"    Total tax:    {total_tax:>14,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
