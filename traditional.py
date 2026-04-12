"""
traditional.py -- Classical mean-reversion pairs trading (beta-hedged) for benchmark3.

Uses close spread from data/spread/raw.csv and hedge ratios from data/pickle/hedge_ratios.pkl.
Fits entry/exit/lookback on the pre-split training period (same cut as backtest.load_test_bundle),
saves data/pickle/traditional_params.pkl for use in backtest.py.

Usage:
    python traditional.py
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

SPLIT_DATE = pd.Timestamp("2024-01-01")
PICKLE_DIR = os.path.join("data", "pickle")
SPREAD_DIR = os.path.join("data", "spread")
TRADITIONAL_PARAMS_PATH = os.path.join(PICKLE_DIR, "traditional_params.pkl")
SPREAD_RAW_PATH = os.path.join(SPREAD_DIR, "raw.csv")
HEDGE_PATH = os.path.join(PICKLE_DIR, "hedge_ratios.pkl")
PAIRS_PATH = os.path.join(PICKLE_DIR, "cointegrated_pairs.pkl")
PRICE_RAW_PATH = os.path.join("data", "trading", "raw.csv")


def build_pair_ticker_mapping(
    pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Same as MPHDRL.build_pair_ticker_mapping (+1/-1 M not used here; tickers order only)."""
    tickers = sorted({t for pair in pairs for t in pair})
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    n_pairs = len(pairs)
    n_tickers = len(tickers)
    M = np.zeros((n_pairs, n_tickers), dtype=np.float32)
    for p_idx, (a, b) in enumerate(pairs):
        M[p_idx, ticker_to_idx[a]] = 1.0
        M[p_idx, ticker_to_idx[b]] = -1.0
    return M, tickers, ticker_to_idx


def exposures_to_weights_np(E: np.ndarray, M_hedge: np.ndarray) -> np.ndarray:
    """
    Same geometry as benchmark.pair_exposures_to_weights: u = E @ M, L1 norm, demean.

    E: (n_pairs,)
    M_hedge: (n_pairs, n_tickers)
    """
    E = np.asarray(E, dtype=np.float64).ravel()
    u = E @ M_hedge
    s = np.abs(u).sum()
    if s < 1e-12:
        return np.zeros(M_hedge.shape[1], dtype=np.float64)
    w = u / s
    w = w - w.mean()
    return w.astype(np.float64)


def build_M_hedge(
    pairs: list[tuple[str, str]],
    hedge_ratios: dict[tuple[str, str], float],
    ticker_to_idx: dict[str, int],
) -> np.ndarray:
    """Row p: +1 on leg A, -beta on leg B for spread log(A) - beta*log(B)."""
    n_pairs = len(pairs)
    n_t = len(ticker_to_idx)
    M = np.zeros((n_pairs, n_t), dtype=np.float64)
    for p_idx, (a, b) in enumerate(pairs):
        key = (a, b)
        if key not in hedge_ratios:
            raise KeyError(f"Hedge ratio missing for pair {key}")
        beta = float(hedge_ratios[key])
        M[p_idx, ticker_to_idx[a]] = 1.0
        M[p_idx, ticker_to_idx[b]] = -beta
    return M


def _rolling_z(spread: np.ndarray, L: int) -> np.ndarray:
    """
    For each index i, z[i] uses mu,std from spread[i-L:i] (L past points, exclusive of i),
    then z[i] = (spread[i] - mu) / sig. First L indices are nan.
    """
    n = len(spread)
    z = np.full(n, np.nan, dtype=np.float64)
    if n <= L:
        return z
    s = np.asarray(spread, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(s)])
    c2 = np.concatenate([[0.0], np.cumsum(s * s)])
    idx = np.arange(L, n, dtype=np.int64)
    sum_x = c[idx] - c[idx - L]
    sum_xx = c2[idx] - c2[idx - L]
    mu = sum_x / L
    den = max(L - 1, 1)
    var = np.maximum((sum_xx - sum_x * sum_x / L) / den, 0.0)
    sig = np.sqrt(np.maximum(var, 1e-16))
    z[idx] = (s[idx] - mu) / sig
    return z


def _state_machine_exposures(
    z: np.ndarray,
    z_entry: float,
    z_exit: float,
) -> np.ndarray:
    """
    Per-pair discrete states: -1 short spread, 0 flat, +1 long spread.
    Hysteresis: z_exit < z_entry.
    """
    n = len(z)
    E = np.zeros(n, dtype=np.float64)
    state = 0  # -1, 0, +1
    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            E[i] = 0.0
            state = 0
            continue
        if state == 0:
            if zi > z_entry:
                state = -1
            elif zi < -z_entry:
                state = 1
        elif state == -1:
            if zi < z_exit:
                state = 0
        elif state == 1:
            if zi > -z_exit:
                state = 0
        E[i] = float(state)
    return E


def _pair_aligned_series(
    spread_wide: pd.DataFrame,
    pair_key: str,
    calendar: list[pd.Timestamp],
) -> np.ndarray:
    """Values aligned to calendar; forward-fill gaps for rolling stats."""
    if pair_key not in spread_wide.columns:
        return np.full(len(calendar), np.nan, dtype=np.float64)
    s = spread_wide[pair_key].reindex(calendar).ffill().bfill()
    v = s.values.astype(np.float64)
    v[~np.isfinite(v)] = 0.0
    return v


def compute_exposure_matrix(
    calendar: list[pd.Timestamp],
    spread_wide: pd.DataFrame,
    pair_keys: list[str],
    L: int,
    z_entry: float,
    z_exit: float,
) -> np.ndarray:
    """
    E_all: shape (len(calendar), n_pairs) exposures at each date (after state update that day).
    """
    n_d = len(calendar)
    n_p = len(pair_keys)
    E_all = np.zeros((n_d, n_p), dtype=np.float64)
    for j, pk in enumerate(pair_keys):
        sp = _pair_aligned_series(spread_wide, pk, calendar)
        z = _rolling_z(sp, L)
        E_all[:, j] = _state_machine_exposures(z, z_entry, z_exit)
    return E_all


def _train_sharpe_from_E(
    calendar: list[pd.Timestamp],
    E_all: np.ndarray,
    M_hedge: np.ndarray,
    price_wide: pd.DataFrame,
    tickers: list[str],
    split_date: pd.Timestamp,
) -> float:
    """Daily returns w_t * r_{t->t+1} on train dates only (both t and t+1 < split optional: t < split)."""
    rets: list[float] = []
    for i in range(len(calendar) - 1):
        d0 = calendar[i]
        d1 = calendar[i + 1]
        if d0 >= split_date:
            break
        if d1 >= split_date:
            break
        if d0 not in price_wide.index or d1 not in price_wide.index:
            continue
        w = exposures_to_weights_np(E_all[i], M_hedge)
        p0 = price_wide.loc[d0, tickers].values.astype(np.float64)
        p1 = price_wide.loc[d1, tickers].values.astype(np.float64)
        safe = np.where(np.abs(p0) < 1e-12, 1.0, p0)
        r = (p1 - p0) / safe
        if not np.all(np.isfinite(r)):
            continue
        rets.append(float(np.nansum(w * r)))
    if len(rets) < 3:
        return float("-inf")
    arr = np.asarray(rets, dtype=np.float64)
    mu = float(np.mean(arr))
    sig = float(np.std(arr, ddof=1))
    if sig < 1e-12:
        return float("-inf")
    return float(np.sqrt(252.0) * mu / sig)


def fit_traditional_params(
    split_date: pd.Timestamp | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """
    Grid search on training period; write pickle with best hyperparameters.
    """
    split_date = split_date or SPLIT_DATE
    out_path = out_path or TRADITIONAL_PARAMS_PATH

    with open(PAIRS_PATH, "rb") as f:
        pairs: list[tuple[str, str]] = pickle.load(f)
    with open(HEDGE_PATH, "rb") as f:
        hedge_ratios: dict[tuple[str, str], float] = pickle.load(f)

    _, tickers, ticker_to_idx = build_pair_ticker_mapping(pairs)
    M_hedge = build_M_hedge(pairs, hedge_ratios, ticker_to_idx)
    pair_keys = [f"{a}|{b}" for a, b in pairs]

    if not os.path.isfile(SPREAD_RAW_PATH):
        raise FileNotFoundError(SPREAD_RAW_PATH)
    raw_sp = pd.read_csv(SPREAD_RAW_PATH, parse_dates=["Date"])
    spread_wide = raw_sp.pivot(index="Date", columns="Pair", values="spread").sort_index()

    price_wide = pd.read_csv(PRICE_RAW_PATH, parse_dates=["Date"])
    price_wide = price_wide.pivot(index="Date", columns="Ticker", values="Close").sort_index()

    calendar = [pd.Timestamp(d) for d in spread_wide.index if d in price_wide.index]
    calendar = sorted(set(calendar))

    best: dict[str, Any] | None = None
    best_score = float("-inf")

    # Moderate grid: full factorial is slow on many pairs; extend here if needed.
    L_grid = [20, 40, 60, 90]
    z_entry_grid = [1.5, 2.0, 2.5]
    z_exit_grid = [0.0, 0.5, 1.0]

    z_by_L: dict[int, np.ndarray] = {}
    for L in L_grid:
        cols = []
        for pk in pair_keys:
            sp = _pair_aligned_series(spread_wide, pk, calendar)
            cols.append(_rolling_z(sp, L))
        z_by_L[L] = np.column_stack(cols) if cols else np.zeros((len(calendar), 0))

    for L in L_grid:
        z_mat = z_by_L[L]
        for z_e in z_entry_grid:
            for z_x in z_exit_grid:
                if z_x >= z_e:
                    continue
                n_d, n_p = z_mat.shape
                E_all = np.zeros((n_d, n_p), dtype=np.float64)
                for j in range(n_p):
                    E_all[:, j] = _state_machine_exposures(z_mat[:, j], z_e, z_x)
                score = _train_sharpe_from_E(calendar, E_all, M_hedge, price_wide, tickers, split_date)
                if score > best_score:
                    best_score = score
                    best = {
                        "lookback": L,
                        "z_entry": z_e,
                        "z_exit": z_x,
                        "split_date": split_date,
                        "train_sharpe": score,
                    }

    if best is None:
        best = {"lookback": 60, "z_entry": 2.0, "z_exit": 0.5, "split_date": split_date, "train_sharpe": 0.0}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(best, f)
    return best


def load_traditional_params(path: str | None = None) -> dict[str, Any]:
    path = path or TRADITIONAL_PARAMS_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_traditional_weights_by_date(
    dates: list[pd.Timestamp],
    pairs: list[tuple[str, str]],
    hedge_ratios: dict[tuple[str, str], float],
    params: dict[str, Any],
    spread_wide: pd.DataFrame | None = None,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Causal warmup: build exposure path over full spread calendar up to max(dates),
    then return weights for each d in dates (must exist in spread_wide index).

    dates: sorted unique trading dates (e.g. test_dates from meta).
    """
    _, tickers, ticker_to_idx = build_pair_ticker_mapping(pairs)
    M_hedge = build_M_hedge(pairs, hedge_ratios, ticker_to_idx)
    pair_keys = [f"{a}|{b}" for a, b in pairs]

    if spread_wide is None:
        raw_sp = pd.read_csv(SPREAD_RAW_PATH, parse_dates=["Date"])
        spread_wide = raw_sp.pivot(index="Date", columns="Pair", values="spread").sort_index()

    L = int(params["lookback"])
    z_e = float(params["z_entry"])
    z_x = float(params["z_exit"])

    max_d = max(dates) if dates else None
    if max_d is None:
        return {}
    calendar = [pd.Timestamp(d) for d in spread_wide.index if d <= max_d]
    calendar = sorted(set(calendar))

    E_all = compute_exposure_matrix(calendar, spread_wide, pair_keys, L, z_e, z_x)
    date_to_idx = {d: i for i, d in enumerate(calendar)}

    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    for d in dates:
        if d not in date_to_idx:
            continue
        i = date_to_idx[d]
        w = exposures_to_weights_np(E_all[i], M_hedge)
        weights_by_date[d] = w
    return weights_by_date


def parse_args():
    p = argparse.ArgumentParser(description="Fit traditional pairs params (grid search on train)")
    p.add_argument(
        "--out",
        type=str,
        default=TRADITIONAL_PARAMS_PATH,
        help=f"Output pickle path (default: {TRADITIONAL_PARAMS_PATH})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    best = fit_traditional_params(out_path=args.out)
    print("Traditional pairs — grid search (train Sharpe)")
    print(f"  Saved: {os.path.abspath(args.out)}")
    print(f"  Best: lookback={best['lookback']}, z_entry={best['z_entry']}, z_exit={best['z_exit']}")
    print(f"  Train Sharpe: {best['train_sharpe']:.4f}")


if __name__ == "__main__":
    main()
