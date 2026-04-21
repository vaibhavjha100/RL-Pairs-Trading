"""
Goal: Compute traditional mean-reversion pair-trading weights and parameter handling.

Inputs: Spread data, hedge ratios, optional parameter overrides, and evaluation dates.

Processing: Builds z-score style signals, applies entry/exit logic, and maps pair weights to ticker exposure.

Outputs: Weight-by-date mappings and optional parameter artifacts consumed by backtest flows.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

PICKLE_DIR = os.path.join("data", "pickle")
SPREAD_DIR = os.path.join("data", "spread")
TRADITIONAL_PARAMS_PATH = os.path.join(PICKLE_DIR, "traditional_params.pkl")
SPREAD_RAW_PATH = os.path.join(SPREAD_DIR, "raw.csv")


def load_precomputed_spread_wide(path: str | None = None) -> pd.DataFrame:
    """
    Load precomputed log-spreads from disk (wide: index Date, columns Pair).

    Expects pipeline output (e.g. `spread.py` -> `data/spread/raw.csv`); does not
    recompute spreads from equity prices.
    """
    path = path or SPREAD_RAW_PATH
    raw_sp = pd.read_csv(path, parse_dates=["Date"])
    return raw_sp.pivot(index="Date", columns="Pair", values="spread").sort_index()


# Fixed mean-reversion bands (rolling z on spread; hysteresis z_exit < z_entry).
DEFAULT_LOOKBACK = 60
DEFAULT_Z_ENTRY = 2.0
DEFAULT_Z_EXIT = 0


def default_traditional_params() -> dict[str, Any]:
    """Fixed strategy parameters (no optimization)."""
    return {
        "lookback": DEFAULT_LOOKBACK,
        "z_entry": DEFAULT_Z_ENTRY,
        "z_exit": DEFAULT_Z_EXIT,
    }


def resolve_traditional_params(path: str | None = None) -> dict[str, Any]:
    """
    Strategy params for weight construction: defaults, optionally overridden by a pickle
    with keys lookback, z_entry, z_exit (other keys ignored).
    """
    base = default_traditional_params()
    path = path or TRADITIONAL_PARAMS_PATH
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        for k in ("lookback", "z_entry", "z_exit"):
            if k in loaded:
                base[k] = loaded[k]
    return base


def save_traditional_params(out_path: str | None = None) -> dict[str, Any]:
    """Write fixed default parameters to pickle (optional convenience)."""
    out_path = out_path or TRADITIONAL_PARAMS_PATH
    params = default_traditional_params()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(params, f)
    return params


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
    """
    Row p: +1 on leg A, -beta on leg B (same beta as in precomputed spread log(A)-beta*log(B)).

    `hedge_ratios` must be the loaded `hedge_ratios.pkl` map; betas are not estimated here.
    """
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

    ``pairs`` / ``hedge_ratios`` / spread columns must match the RL pipeline (same universe,
    same `Pair` keys as `A|B` in spread files). If ``spread_wide`` is None, loads
    ``load_precomputed_spread_wide()`` (pipeline raw spreads only).
    """
    _, tickers, ticker_to_idx = build_pair_ticker_mapping(pairs)
    M_hedge = build_M_hedge(pairs, hedge_ratios, ticker_to_idx)
    pair_keys = [f"{a}|{b}" for a, b in pairs]

    if spread_wide is None:
        spread_wide = load_precomputed_spread_wide()

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
    p = argparse.ArgumentParser(description="Write fixed traditional pairs parameters to pickle")
    p.add_argument(
        "--out",
        type=str,
        default=TRADITIONAL_PARAMS_PATH,
        help=f"Output pickle path (default: {TRADITIONAL_PARAMS_PATH})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    params = save_traditional_params(out_path=args.out)
    print("Traditional pairs — fixed strategy parameters")
    print(f"  Saved: {os.path.abspath(args.out)}")
    print(
        f"  lookback={params['lookback']}, z_entry={params['z_entry']}, z_exit={params['z_exit']}"
    )


if __name__ == "__main__":
    main()
