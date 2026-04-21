from __future__ import annotations

import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from rl_pairs_trading.mphdrl import HPARAMS, TradingEnvironment

INITIAL_CASH = 10_000_000
TXN_COST_RATE = 0.0000307
SHORT_COST_ANNUAL = 0.0657
SHORT_COST_DAILY = SHORT_COST_ANNUAL / 252
STCG_RATE = 0.20
LTCG_RATE = 0.125
FISCAL_YEAR_END = (3, 31)
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
    return x_split[:n], y_split[:n], pairs, meta_split.iloc[:n].reset_index(drop=True)


def load_price_matrix(tickers):
    raw_path = os.path.join("data", "trading", "raw.csv")
    if not os.path.isfile(raw_path):
        print(f"Missing {raw_path}. Run collection.py first.")
        sys.exit(1)
    raw = pd.read_csv(raw_path, parse_dates=["Date"], usecols=["Date", "Ticker", "Close"])
    raw = raw.loc[raw["Ticker"].isin(set(tickers)), ["Date", "Ticker", "Close"]]
    raw = raw.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last")
    wide = raw.pivot(index="Date", columns="Ticker", values="Close").sort_index()
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
        weights_by_date[d] = out["weights"].detach().cpu().numpy().reshape(-1)
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
    return (d_prev.year, d_prev.month, d_prev.day) != (d_curr.year, d_curr.month, d_curr.day) and (
        (d_prev.month, d_prev.day) <= (fy_mm, fy_dd) < (d_curr.month, d_curr.day)
        or (d_curr.year > d_prev.year and (fy_mm, fy_dd) >= (d_prev.month, d_prev.day))
    )


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
    return portfolio_value * np.nansum(closed * r * np.sign(w_prev))


def run_strategy_backtest(strategy_name, weights_by_date, price_wide, tickers, eval_dates):
    price_dates = price_wide.index
    net_pv = float(INITIAL_CASH)
    gross_pv = float(INITIAL_CASH)
    prev_w = np.zeros(len(tickers), dtype=np.float64)
    tax_carryforward = 0.0
    prev_date = None
    rows = []
    for idx in range(len(eval_dates) - 1):
        d_t, d_t1 = eval_dates[idx], eval_dates[idx + 1]
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
        w_long, w_short = np.maximum(w, 0.0), np.minimum(w, 0.0)
        prev_w_long, prev_w_short = np.maximum(prev_w, 0.0), np.minimum(prev_w, 0.0)
        gross_long_ret, gross_short_ret = np.nansum(w_long * r), np.nansum(w_short * r)
        gross_portfolio_ret = gross_long_ret + gross_short_ret
        gross_long_pnl, gross_short_pnl = net_pv * gross_long_ret, net_pv * gross_short_ret
        gross_pv += gross_long_pnl + gross_short_pnl
        turnover_long = np.nansum(np.abs(w_long - prev_w_long))
        turnover_short = np.nansum(np.abs(w_short - prev_w_short))
        total_turnover = turnover_long + turnover_short
        txn_cost = net_pv * total_turnover * TXN_COST_RATE
        txn_cost_long = txn_cost * (turnover_long / total_turnover) if total_turnover > 0 else 0.0
        txn_cost_short = txn_cost * (turnover_short / total_turnover) if total_turnover > 0 else 0.0
        shorting_cost = net_pv * np.nansum(np.abs(w_short)) * SHORT_COST_DAILY
        realized_gross = _compute_realized_pnl(prev_w, w, r, net_pv)
        net_realized_pnl = realized_gross - txn_cost - shorting_cost
        if net_realized_pnl > 0:
            payment = net_realized_pnl * STCG_RATE
            tax_flow = -payment
            tax_carryforward += payment
        elif net_realized_pnl < 0:
            rebate = min(abs(net_realized_pnl) * STCG_RATE, tax_carryforward)
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
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize_backtest_dataframe(df_bt: pd.DataFrame):
    if df_bt.empty:
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
    r = df_bt["net_portfolio_return"].astype(float).values
    mu_ann = float(np.nanmean(r)) * 252.0
    var_ann = float(np.nanvar(r, ddof=1)) * 252.0 if len(r) > 1 else 0.0
    utility = mu_ann - 0.5 * BACKTEST_RISK_AVERSION * var_ann
    sharpe = float(np.sqrt(252.0) * np.nanmean(r) / (np.nanstd(r, ddof=1) + 1e-12)) if len(r) > 1 else 0.0
    return {
        "utility": utility,
        "annual_return": mu_ann,
        "variance": var_ann,
        "sharpe": sharpe,
        "mean_abs_weight": float(df_bt["mean_abs_weight"].mean()),
        "l1_turnover": float(df_bt["l1_turnover"].mean()),
        "transaction_cost": float(df_bt["transaction_cost"].sum()),
        "shorting_cost": float(df_bt["shorting_cost"].sum()),
    }


def run_nifty50_buy_hold_backtest(strategy_name, eval_dates, nifty_path, initial_cash=INITIAL_CASH):
    if not os.path.isfile(nifty_path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.read_csv(nifty_path, parse_dates=["Date"])
    price_col = "Close" if "Close" in df.columns else df.columns[-1]
    p = df[["Date", price_col]].dropna().rename(columns={price_col: "Close"}).set_index("Date").sort_index()
    dates = [d for d in eval_dates if d in p.index]
    if len(dates) < 2:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    units = initial_cash / float(p.loc[dates[0], "Close"])
    rows = []
    for i in range(len(dates) - 1):
        d_t, d_t1 = dates[i], dates[i + 1]
        pv_t = units * float(p.loc[d_t, "Close"])
        pv_t1 = units * float(p.loc[d_t1, "Close"])
        ret = (pv_t1 - pv_t) / pv_t if pv_t else 0.0
        rows.append(
            {
                "date": d_t,
                "gross_portfolio_value": pv_t1,
                "gross_long_pnl": pv_t1 - pv_t,
                "gross_short_pnl": 0.0,
                "gross_long_return": ret,
                "gross_short_return": 0.0,
                "gross_portfolio_return": ret,
                "transaction_cost": 0.0,
                "shorting_cost": 0.0,
                "tax_flow": 0.0,
                "tax_carryforward": 0.0,
                "net_long_pnl": pv_t1 - pv_t,
                "net_short_pnl": 0.0,
                "net_long_return": ret,
                "net_short_return": 0.0,
                "net_portfolio_value": pv_t1,
                "net_portfolio_return": ret,
                "mean_abs_weight": 1.0,
                "l1_turnover": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)

