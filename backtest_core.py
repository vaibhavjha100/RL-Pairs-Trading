from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from MPHDRL import HPARAMS, TradingEnvironment

INITIAL_CASH = 10_000_000
TXN_COST_RATE = 0.0000307
SHORT_COST_ANNUAL = 0.0657
SHORT_COST_DAILY = SHORT_COST_ANNUAL / 252
STCG_RATE = 0.20
LTCG_RATE = 0.125
FISCAL_YEAR_END = (3, 31)

# Mean-variance utility U = μ_ann - 0.5 * γ * σ²_ann (used in backtest summaries & comparison.py).
BACKTEST_RISK_AVERSION = 2.0

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
    "mean_abs_weight",
    "l1_turnover",
]


def load_sequence_bundle(split: str = "test") -> Tuple[np.ndarray, np.ndarray, list, pd.DataFrame]:
    base = os.path.join("data", "pickle")
    spread_dir = os.path.join("data", "spread")
    split_name = split.strip().lower()
    if split_name not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    paths = {
        "X": os.path.join(base, f"spread_X_{split_name}.pkl"),
        "y": os.path.join(base, f"spread_y_{split_name}.pkl"),
        "pairs": os.path.join(base, "cointegrated_pairs.pkl"),
        "meta": os.path.join(spread_dir, "sequence_xy.csv"),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        print("Missing files:", [paths[m] for m in missing])
        print("Run preprocessing first.")
        sys.exit(1)

    with open(paths["X"], "rb") as f:
        x_split = pickle.load(f)
    with open(paths["y"], "rb") as f:
        y_split = pickle.load(f)
    with open(paths["pairs"], "rb") as f:
        pairs = pickle.load(f)

    meta = pd.read_csv(paths["meta"], parse_dates=["target_date"])
    if "Unnamed: 0" in meta.columns:
        meta = meta.drop(columns=["Unnamed: 0"])

    split_date = pd.to_datetime("2024-01-01")
    use_mask = meta["target_date"] < split_date if split_name == "train" else meta["target_date"] >= split_date
    idx = np.where(use_mask.values)[0]
    meta_split = meta.iloc[idx].reset_index(drop=True)

    n = min(len(meta_split), len(x_split))
    meta_split = meta_split.iloc[:n].reset_index(drop=True)
    x_split = x_split[:n]
    y_split = y_split[:n]
    return x_split, y_split, pairs, meta_split


def load_price_matrix(tickers):
    raw_path = os.path.join("data", "trading", "raw.csv")
    if not os.path.isfile(raw_path):
        print(f"Missing {raw_path}. Run collection.py first.")
        sys.exit(1)
    raw = pd.read_csv(
        raw_path,
        parse_dates=["Date"],
        usecols=["Date", "Ticker", "Close"],
    )
    need = set(tickers)
    raw = raw.loc[raw["Ticker"].isin(need), ["Date", "Ticker", "Close"]]
    # Duplicate Date×Ticker rows break pivot/unstack and can explode memory; collapse explicitly.
    raw = raw.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last")
    wide = raw.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    missing_t = [t for t in tickers if t not in wide.columns]
    if missing_t:
        print(f"WARNING: {len(missing_t)} tickers missing from raw.csv: {missing_t[:5]}...")
    return wide


def windows_and_mask_for_date(meta_df, x_arr, pair_key_to_idx, n_pairs, f_dim, d):
    sub = meta_df[meta_df["target_date"] == d]
    windows = np.zeros((n_pairs, x_arr.shape[1], f_dim), dtype=np.float32)
    mask = np.zeros(n_pairs, dtype=np.float32)
    present = []
    for _, row in sub.iterrows():
        iloc = row.name
        pkey = str(row["Pair"])
        if pkey not in pair_key_to_idx:
            continue
        p_idx = pair_key_to_idx[pkey]
        windows[p_idx] = x_arr[iloc]
        mask[p_idx] = 1.0
        present.append(pkey)
    if not present:
        return None, None
    return windows, mask


def get_all_weights_by_date(model, meta_df, x_arr, pair_key_to_idx, n_pairs, f_dim, dates):
    weights_by_date = {}
    model.eval()
    for d in dates:
        windows, mask = windows_and_mask_for_date(meta_df, x_arr, pair_key_to_idx, n_pairs, f_dim, d)
        if windows is None:
            continue
        with torch.no_grad():
            out = model.forward_step(windows, explore=False, pair_mask=mask)
        w = out["weights"].detach().cpu().numpy().reshape(-1)
        weights_by_date[d] = w
    return weights_by_date


def get_mphdrl_weights_by_env(model, meta_df, x_arr, y_arr, pairs, tickers, ticker_to_idx, spread_wide=None, split="test"):
    env = TradingEnvironment(
        pairs=pairs,
        tickers=tickers,
        ticker_to_idx=ticker_to_idx,
        trading_raw_path=os.path.join("data", "trading", "raw.csv"),
        sequence_meta=meta_df,
        X_train=x_arr,
        y_train=y_arr,
        zeta=HPARAMS["zeta"],
        gamma=HPARAMS["gamma"],
        risk_lambda=HPARAMS["risk_lambda"],
        var_window=HPARAMS["var_window"],
        terminal_utility_weight=HPARAMS["terminal_utility_weight"],
        spread_pivot=spread_wide,
        split=split,
        txn_cost_rate=TXN_COST_RATE,
        short_cost_annual=SHORT_COST_ANNUAL,
        stcg_rate=STCG_RATE,
        fiscal_year_end=FISCAL_YEAR_END,
    )
    weights_by_date = {}
    state = env.reset()
    model.eval()
    while state is not None and env.t < len(env.unique_dates):
        d_t = env.unique_dates[env.t]
        windows, mask, y_spread = state
        with torch.no_grad():
            out = model.forward_step(windows, explore=False, pair_mask=mask)

        raw_w = out["weights"].detach().cpu().numpy().reshape(-1)
        sl_actions = out["sl_actions"].detach().cpu().numpy().reshape(-1)
        spread_values = np.where(mask, y_spread, np.nan).astype(np.float64)
        w_adj = env._apply_stop_outs(raw_w, sl_actions=sl_actions, spread_values=spread_values)
        weights_by_date[d_t] = w_adj.astype(np.float64)

        state, _, done = env.step(raw_w, sl_actions=sl_actions, spread_values=spread_values)
        if done:
            break
    return weights_by_date


def _crosses_fiscal_year(d_prev, d_curr):
    fy_mm, fy_dd = FISCAL_YEAR_END

    def fy_of(dt):
        if (dt.month, dt.day) > (fy_mm, fy_dd):
            return dt.year + 1
        return dt.year

    return fy_of(d_prev) != fy_of(d_curr)


def _compute_realized_pnl(w_prev, w_new, r, portfolio_value):
    closed = np.zeros_like(w_prev)
    for i in range(len(w_prev)):
        wp, wn = w_prev[i], w_new[i]
        if wp == 0.0:
            continue
        same_sign = np.sign(wp) == np.sign(wn)
        if same_sign and abs(wn) < abs(wp):
            closed[i] = abs(wp) - abs(wn)
        elif not same_sign:
            closed[i] = abs(wp)
    turnover_profit = portfolio_value * np.nansum(closed * r * np.sign(w_prev))
    return turnover_profit


def run_strategy_backtest(strategy_name, weights_by_date, price_wide, tickers, eval_dates):
    price_dates = price_wide.index
    n_tickers = len(tickers)
    net_pv = float(INITIAL_CASH)
    gross_pv = float(INITIAL_CASH)
    prev_w = np.zeros(n_tickers, dtype=np.float64)
    tax_carryforward = 0.0
    prev_date = None
    rows = []

    for idx in range(len(eval_dates) - 1):
        d_t = eval_dates[idx]
        d_t1 = eval_dates[idx + 1]
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

        if prev_date is not None and _crosses_fiscal_year(prev_date, d_t) and tax_carryforward > 0:
            tax_carryforward = 0.0

        tax_per_leg = tax_flow / 2.0
        net_long_pnl = gross_long_pnl - txn_cost_long - tax_per_leg
        net_short_pnl = gross_short_pnl - txn_cost_short - shorting_cost - tax_per_leg
        net_pnl = net_long_pnl + net_short_pnl
        net_long_ret = net_long_pnl / net_pv if net_pv else 0.0
        net_short_ret = net_short_pnl / net_pv if net_pv else 0.0
        net_portfolio_ret = net_pnl / net_pv if net_pv else 0.0
        net_pv += net_pnl

        rows.append(
            {
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
                "mean_abs_weight": float(np.nanmean(np.abs(w))),
                "l1_turnover": float(total_turnover),
            }
        )
        prev_w = w.copy()
        prev_date = d_t

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    print(f"  {strategy_name}: {len(df)} trading days simulated")
    return df


def run_nifty50_buy_hold_backtest(
    strategy_name: str,
    eval_dates: List[Any],
    nifty_path: str | None = None,
    price_col: str = "Adj Close",
    initial_cash: float = INITIAL_CASH,
    stcg_rate: float = STCG_RATE,
    txn_cost_rate: float = TXN_COST_RATE,
) -> pd.DataFrame:
    """
    Nifty 50 index buy-and-hold on ``eval_dates`` (same calendar as other strategies).

    Gross path: fully invested in the index from ``initial_cash`` at the first usable
    close; daily gross return follows index levels from ``nifty50.csv``.

    Costs: one buy-side transaction charge on ``initial_cash`` and a simplified STCG
    tax on the **total** index capital gain to the final mark (no daily mark-to-market
    tax). Both are **split evenly across every simulated day** in INR so
    ``transaction_cost`` and ``tax_flow`` are constant per row (shorting cost is zero).

    This is a smoothed accounting benchmark, not identical to realized Indian tax lots.
    """
    path = nifty_path or os.path.join("data", "trading", "nifty50.csv")
    if not os.path.isfile(path):
        print(f"  {strategy_name}: missing {path} (skip)")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    raw = pd.read_csv(path, parse_dates=["Date"])
    if price_col not in raw.columns:
        if "Close" in raw.columns:
            price_col = "Close"
        else:
            print(f"  {strategy_name}: no price column in {path} (skip)")
            return pd.DataFrame(columns=RESULT_COLUMNS)

    raw = raw.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    series = raw.set_index("Date")[price_col].astype(float).sort_index()

    steps: list[tuple[pd.Timestamp, float, float, float]] = []
    for idx in range(len(eval_dates) - 1):
        d0 = pd.Timestamp(eval_dates[idx]).normalize()
        d1 = pd.Timestamp(eval_dates[idx + 1]).normalize()
        if d0 not in series.index or d1 not in series.index:
            continue
        p0 = float(series.loc[d0])
        p1 = float(series.loc[d1])
        if not (np.isfinite(p0) and np.isfinite(p1)) or abs(p0) < 1e-12:
            continue
        gret = p1 / p0 - 1.0
        steps.append((d0, gret, p0, p1))

    if not steps:
        print(f"  {strategy_name}: no overlapping Nifty prices for eval window (skip)")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    n = len(steps)
    gross_end = float(initial_cash)
    for _, gret, _, _ in steps:
        gross_end *= 1.0 + gret
    gain = gross_end - float(initial_cash)
    tax_total = max(0.0, gain) * float(stcg_rate)
    txn_total = float(initial_cash) * float(txn_cost_rate)
    daily_txn = txn_total / n
    daily_tax_payment = tax_total / n

    gross_pv = float(initial_cash)
    net_pv = float(initial_cash)
    tax_carryforward = 0.0
    rows: list[dict[str, Any]] = []

    for d0, gret, _p0, _p1 in steps:
        gross_long_ret = gret
        gross_short_ret = 0.0
        gross_portfolio_ret = gret
        gross_long_pnl = gross_pv * gret
        gross_short_pnl = 0.0
        gross_pnl = gross_long_pnl
        gross_pv += gross_pnl

        txn_cost = daily_txn
        shorting_cost = 0.0
        tax_flow = -daily_tax_payment

        net_long_pnl = gross_long_pnl - txn_cost + tax_flow
        net_short_pnl = 0.0
        net_pnl = net_long_pnl + net_short_pnl
        net_long_ret = net_long_pnl / net_pv if net_pv else 0.0
        net_short_ret = 0.0
        net_portfolio_ret = net_pnl / net_pv if net_pv else 0.0
        net_pv += net_pnl

        rows.append(
            {
                "date": d0,
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
                "mean_abs_weight": 1.0,
                "l1_turnover": 0.0,
            }
        )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    print(
        f"  {strategy_name}: {len(df)} trading days (buy-hold; "
        f"txn amort {daily_txn:.2f}/day, tax amort {daily_tax_payment:.2f}/day on {n} days)"
    )
    return df


def compute_annualized_utility(
    net_returns, gamma: float = BACKTEST_RISK_AVERSION,
) -> Dict[str, float]:
    r = pd.to_numeric(pd.Series(net_returns), errors="coerce").fillna(0.0).to_numpy()
    mu = float(np.mean(r)) if len(r) else 0.0
    var = float(np.var(r, ddof=1)) if len(r) > 1 else 0.0
    ann_ret = mu * 252.0
    utility = ann_ret - 0.5 * gamma * var * 252.0
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(252.0) if sd > 1e-12 else float("nan")
    return {
        "utility": float(utility),
        "annual_return": float(ann_ret),
        "variance": float(var),
        "sharpe": float(sharpe),
    }


def summarize_backtest_dataframe(
    df: pd.DataFrame, gamma: float = BACKTEST_RISK_AVERSION,
) -> Dict[str, float]:
    if df is None or df.empty or "net_portfolio_return" not in df.columns:
        return {
            "utility": float("-inf"),
            "annual_return": float("nan"),
            "variance": float("nan"),
            "sharpe": float("nan"),
            "mean_abs_weight": float("nan"),
            "l1_turnover": float("nan"),
            "transaction_cost": float("nan"),
            "shorting_cost": float("nan"),
        }
    core = compute_annualized_utility(df["net_portfolio_return"].to_numpy(), gamma=gamma)
    core.update(
        {
            "mean_abs_weight": float(df["mean_abs_weight"].mean()),
            "l1_turnover": float(df["l1_turnover"].mean()),
            "transaction_cost": float(df["transaction_cost"].mean()),
            "shorting_cost": float(df["shorting_cost"].mean()),
        }
    )
    return core
