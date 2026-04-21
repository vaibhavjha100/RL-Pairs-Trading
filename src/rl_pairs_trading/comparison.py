"""
comparison.py -- Formal backtest comparison: performance metrics, common-size costs,
and one-tailed paired Wilcoxon signed-rank tests on daily mean-variance utility (net).

Reads CSVs from backtest.py:
    data/backtest/mphdrl.csv, nifty50_buy_hold.csv, benchmark.csv, traditional.csv, srrl.csv

Writes:
    data/backtest/results/comparison_summary.txt
    data/backtest/results/comparison_metrics.csv
    data/backtest/results/wilcoxon_utility.json

Requires scipy (Wilcoxon test): pip install scipy

Usage:
    python comparison.py
    python comparison.py --backtest-dir data/backtest --results-dir data/backtest/results
    python comparison.py --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError as e:
    raise SystemExit(
        "comparison.py requires scipy for Wilcoxon signed-rank tests. "
        "Install with:  pip install scipy\n"
        f"Original error: {e}"
    ) from e

try:
    from rl_pairs_trading.backtest_core import INITIAL_CASH
except ImportError:
    INITIAL_CASH = 10_000_000.0
try:
    from rl_pairs_trading.backtest import BACKTEST_DIR
except ImportError:
    BACKTEST_DIR = os.path.join("data", "backtest")

try:
    from rl_pairs_trading.backtest_core import BACKTEST_RISK_AVERSION as UTILITY_GAMMA
except ImportError:
    UTILITY_GAMMA = 2.0

RESULTS_SUBDIR = "results"
DEFAULT_UTILITY_WINDOW = 60
DEFAULT_SIGNIFICANCE_ALPHA = 0.05

STRATEGY_FILES = (
    ("MPHDRL", "mphdrl.csv"),
    ("Nifty 50 buy-and-hold", "nifty50_buy_hold.csv"),
    ("Benchmark RL", "benchmark.csv"),
    ("Traditional pairs", "traditional.csv"),
    ("SRRL", "srrl.csv"),
)


def utility_definition_text(window: int, gamma: float) -> str:
    # Avoid f-string `{...}` in subscript notation (would parse as format fields).
    return (
        f"Daily net utility U_t on trailing {window} trading days (inclusive of t): "
        f"R_ann(t)=mean(r[t-{window}+1:t])*252, Var_ann(t)=var(r,ddof=1)*252 on same window, "
        f"U_t=R_ann(t)-0.5*{gamma}*Var_ann(t), with r_t=net_portfolio_return."
    )


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity.astype(float))
    peak = np.where(peak < 1e-12, np.nan, peak)
    dd = (peak - equity.astype(float)) / peak
    return float(np.nanmax(dd)) if np.any(np.isfinite(dd)) else 0.0


def annualized_sharpe(daily_returns: np.ndarray, rf_daily: float = 0.0) -> float:
    r = np.asarray(daily_returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    excess = r - rf_daily
    mu = float(np.nanmean(excess))
    sig = float(np.nanstd(excess, ddof=1))
    if sig < 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * mu / sig)


def annualized_mean_return(daily_returns: np.ndarray) -> float:
    r = np.asarray(daily_returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 1:
        return 0.0
    return float(np.nanmean(r)) * 252.0


def annualized_variance_returns(daily_returns: np.ndarray) -> float:
    r = np.asarray(daily_returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    return float(np.nanvar(r, ddof=1)) * 252.0


def mean_variance_utility_ann(
    daily_returns: np.ndarray, gamma: float = UTILITY_GAMMA
) -> tuple[float, float, float]:
    r_ann = annualized_mean_return(daily_returns)
    v_ann = annualized_variance_returns(daily_returns)
    u = r_ann - 0.5 * v_ann * gamma
    return u, r_ann, v_ann


def _normalize_date_column(s: pd.Series) -> pd.Series:
    """UTC-stripped, calendar-day timestamps for stable merges across CSVs."""
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()


def normalize_backtest_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Sort by date, drop duplicate days, validate required columns."""
    req = [
        "date",
        "gross_portfolio_value",
        "net_portfolio_value",
        "gross_portfolio_return",
        "net_portfolio_return",
        "transaction_cost",
        "shorting_cost",
        "tax_flow",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    out = df.copy()
    out["date"] = _normalize_date_column(out["date"])
    out = out.dropna(subset=["date"]).sort_values("date")
    if out["date"].duplicated().any():
        n_dup = int(out["date"].duplicated().sum())
        out = out.drop_duplicates(subset=["date"], keep="last")
        print(
            f"Warning: {name} had {n_dup} duplicate date row(s); kept last row per day.",
            file=sys.stderr,
        )
    return out.reset_index(drop=True)


def rolling_daily_net_utility(
    df: pd.DataFrame, window: int, gamma: float
) -> pd.DataFrame:
    """
    U_t from trailing net returns on aligned rows. NaN until min_periods met.
    Returns a two-column frame (date, rolling_u) for merge-safe alignment (no reindex drift).
    """
    d = df.sort_values("date").reset_index(drop=True)
    r = d["net_portfolio_return"].astype(float)
    r_ann = r.rolling(window=window, min_periods=2).mean() * 252.0
    v_ann = r.rolling(window=window, min_periods=2).var(ddof=1) * 252.0
    u = r_ann - 0.5 * gamma * v_ann
    out = pd.DataFrame({"date": _normalize_date_column(d["date"]), "rolling_u": u.values})
    return out


def load_strategy_csv(base_dir: str, filename: str) -> pd.DataFrame | None:
    path = os.path.join(base_dir, filename)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{path}: no 'date' column (got {list(df.columns)[:8]}...)")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def looks_like_flat_returns(df: pd.DataFrame, col: str = "net_portfolio_return") -> bool:
    r = pd.to_numeric(df[col], errors="coerce").astype(float).values
    r = r[np.isfinite(r)]
    if r.size < 2:
        return True
    if float(np.nanstd(r, ddof=1)) < 1e-14 and float(np.nanmax(np.abs(r))) < 1e-14:
        return True
    return False


def summarize(df: pd.DataFrame, name: str) -> dict[str, Any]:
    if df is None or df.empty:
        return {"strategy": name, "ok": False}

    if looks_like_flat_returns(df, "net_portfolio_return"):
        print(
            f"Warning: {name} net_portfolio_return is flat (~zero variance). "
            f"Re-run backtest after training; metrics may look like no trading.",
            file=sys.stderr,
        )

    g_end = float(df["gross_portfolio_value"].iloc[-1])
    n_end = float(df["net_portfolio_value"].iloc[-1])
    gross_ret_total = (g_end - INITIAL_CASH) / INITIAL_CASH * 100.0
    net_ret_total = (n_end - INITIAL_CASH) / INITIAL_CASH * 100.0
    gross_dd = max_drawdown(df["gross_portfolio_value"].values)
    net_dd = max_drawdown(df["net_portfolio_value"].values)

    txn = float(df["transaction_cost"].sum())
    short_c = float(df["shorting_cost"].sum())
    tax_net = float(-df["tax_flow"].sum())

    g_vol = (
        float(np.nanstd(df["gross_portfolio_return"].values) * np.sqrt(252))
        if len(df) > 1
        else 0.0
    )
    n_vol = (
        float(np.nanstd(df["net_portfolio_return"].values) * np.sqrt(252))
        if len(df) > 1
        else 0.0
    )

    g_sharpe = annualized_sharpe(df["gross_portfolio_return"].values)
    n_sharpe = annualized_sharpe(df["net_portfolio_return"].values)
    g_u, _, _ = mean_variance_utility_ann(df["gross_portfolio_return"].values)
    n_u, _, _ = mean_variance_utility_ann(df["net_portfolio_return"].values)

    mean_gpv = float(df["gross_portfolio_value"].mean())

    return {
        "strategy": name,
        "ok": True,
        "n_days": len(df),
        "start_date": df["date"].iloc[0].isoformat(),
        "end_date": df["date"].iloc[-1].isoformat(),
        "gross_ret_pct": gross_ret_total,
        "net_ret_pct": net_ret_total,
        "gross_dd_pct": gross_dd * 100.0,
        "net_dd_pct": net_dd * 100.0,
        "gross_ann_vol_pct": g_vol * 100.0,
        "net_ann_vol_pct": n_vol * 100.0,
        "gross_sharpe": g_sharpe,
        "net_sharpe": n_sharpe,
        "gross_end_inr": g_end,
        "net_end_inr": n_end,
        "sum_txn_inr": txn,
        "sum_short_inr": short_c,
        "sum_tax_inr": tax_net,
        "txn_pct_initial_cash": txn / INITIAL_CASH * 100.0,
        "short_pct_initial_cash": short_c / INITIAL_CASH * 100.0,
        "tax_pct_initial_cash": tax_net / INITIAL_CASH * 100.0,
        "total_costs_pct_initial_cash": (txn + short_c + tax_net) / INITIAL_CASH * 100.0,
        "costs_bps_per_year_of_mean_gross_pv": (
            (txn + short_c + tax_net) / max(mean_gpv, 1.0) / max(len(df) / 252.0, 1e-9) * 10000.0
        ),
        "scalar_gross_util_ann": g_u,
        "scalar_net_util_ann": n_u,
    }


def run_wilcoxon_paired(
    u_a: pd.Series,
    u_b: pd.Series,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank: H1 median(u_a - u_b) > 0."""
    j = pd.DataFrame({"a": u_a, "b": u_b}).dropna()
    if len(j) < 3:
        return {
            "strategy_a": label_a,
            "benchmark": label_b,
            "n_pairs": int(len(j)),
            "error": "insufficient paired observations (need >= 3)",
        }
    d = (j["a"] - j["b"]).values.astype(float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return {
            "strategy_a": label_a,
            "benchmark": label_b,
            "n_pairs": int(d.size),
            "error": "insufficient finite differences",
        }
    try:
        try:
            res = wilcoxon(d, alternative="greater", zero_method="wilcox", method="auto")
        except TypeError:
            res = wilcoxon(d, alternative="greater", zero_method="wilcox")
    except ValueError as e:
        return {
            "strategy_a": label_a,
            "benchmark": label_b,
            "n_pairs": int(d.size),
            "error": str(e),
        }
    stat = res.statistic
    pval = res.pvalue
    return {
        "strategy_a": label_a,
        "benchmark": label_b,
        "test": "Wilcoxon signed-rank",
        "pairing": "paired_calendar_dates",
        "difference": f"U_{label_a}(t) - U_{label_b}(t)",
        "n_pairs": int(d.size),
        "alternative": "greater",
        "zero_method": "wilcox",
        "statistic": float(stat) if stat is not None and np.isfinite(stat) else None,
        "pvalue": float(pval) if pval is not None and np.isfinite(pval) else None,
    }


def _wilcoxon_conclusion(pvalue: float | None, alpha: float) -> tuple[bool | None, str]:
    """(reject_h0, one-line conclusion). reject_h0 True => evidence MPHDRL median utility > benchmark."""
    if pvalue is None or not np.isfinite(pvalue):
        return None, "p-value unavailable; no decision."
    reject = pvalue < alpha
    if reject:
        return (
            True,
            f"Reject H0 at alpha={alpha:g} (p < {alpha:g}); evidence that median paired difference > 0.",
        )
    return (
        False,
        f"Fail to reject H0 at alpha={alpha:g} (p >= {alpha:g}); insufficient evidence that median difference > 0.",
    )


def build_report(
    summaries: list[dict[str, Any]],
    wilcox_rows: list[dict[str, Any]],
    window: int,
    gamma: float,
    alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
) -> str:
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("FORMAL BACKTEST COMPARISON")
    lines.append(f"Initial cash: {INITIAL_CASH:,.0f} INR")
    lines.append("=" * 88)
    lines.append("")
    lines.append("--- Summary performance ---")
    hdr = (
        f"{'Strategy':<18} {'Days':>5} {'Net ret%':>9} {'Gross vol%':>11} {'Net vol%':>9} "
        f"{'Net Sharpe':>10} {'Net end (M)':>12}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s in summaries:
        if not s.get("ok"):
            continue
        lines.append(
            f"{s['strategy']:<18} {s['n_days']:>5d} {s['net_ret_pct']:>+8.2f}% "
            f"{s['gross_ann_vol_pct']:>10.2f}% {s['net_ann_vol_pct']:>8.2f}% "
            f"{s['net_sharpe']:>10.3f} {s['net_end_inr']/1e6:>12.3f}"
        )
    lines.append("")
    lines.append("--- Scalar mean-variance utility (full sample, net), gamma={} ---".format(gamma))
    lines.append("U = R_ann - 0.5 * Var_ann * gamma on all net daily returns.")
    for s in summaries:
        if s.get("ok"):
            lines.append(f"  {s['strategy']}: U_net = {s['scalar_net_util_ann']:.6f}")
    lines.append("")
    lines.append("--- Common-size costs (% of initial cash) ---")
    hdr2 = f"{'Strategy':<18} {'Txn%':>8} {'Short%':>8} {'Tax%':>8} {'Total%':>8} {'Bps/yr*':>10}"
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    lines.append("* Total txn+short+tax relative to mean gross PV, annualized bps (see JSON).")
    for s in summaries:
        if not s.get("ok"):
            continue
        lines.append(
            f"{s['strategy']:<18} {s['txn_pct_initial_cash']:>7.3f}% "
            f"{s['short_pct_initial_cash']:>7.3f}% {s['tax_pct_initial_cash']:>7.3f}% "
            f"{s['total_costs_pct_initial_cash']:>7.3f}% {s['costs_bps_per_year_of_mean_gross_pv']:>10.1f}"
        )
    lines.append("")
    lines.append("--- Hypothesis tests: median daily rolling utility (paired Wilcoxon) ---")
    lines.append(utility_definition_text(window, gamma))
    lines.append("")
    lines.append("Specification (same for each test below):")
    lines.append("  Procedure : scipy.stats.wilcoxon (paired Wilcoxon signed-rank)")
    lines.append("  Pairing   : one observation per common calendar date t (inner join on date)")
    lines.append("  Alternative: 'greater' (one-tailed test on d_t = U_A(t) - U_B(t))")
    lines.append("  Zeros     : zero_method='wilcox' (zeros excluded from ranking per Wilcoxon)")
    lines.append(f"  Significance level: alpha = {alpha:g} (reject H0 if p-value < alpha)")
    lines.append("")
    for w in wilcox_rows:
        strat_a = w.get("strategy_a", "?")
        bench = w.get("benchmark", "?")
        lines.append(f"  [{strat_a} vs {bench}]")
        lines.append(f"    H0: median(U_{strat_a} - U_{bench}) <= 0")
        lines.append(f"    H1: median(U_{strat_a} - U_{bench}) > 0")
        if "error" in w:
            lines.append(f"    Status   : not run — {w['error']}")
            lines.append(f"    n (pairs): {w.get('n_pairs', 0)}")
        else:
            pv = w.get("pvalue")
            st = w.get("statistic")
            reject, verdict = _wilcoxon_conclusion(pv, alpha)
            lines.append(f"    n (paired days)     : {w['n_pairs']}")
            lines.append(f"    Wilcoxon statistic  : {st}")
            lines.append(f"    p-value (one-tailed): {pv}")
            lines.append(f"    Decision (alpha={alpha:g}) : {verdict}")
    lines.append("")
    lines.append("=" * 88)
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Formal backtest comparison and Wilcoxon tests")
    p.add_argument("--backtest-dir", type=str, default=BACKTEST_DIR, help="Directory with strategy CSVs")
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output directory (default: <backtest-dir>/results)",
    )
    p.add_argument("--utility-window", type=int, default=DEFAULT_UTILITY_WINDOW, help="Trailing days for U_t")
    p.add_argument("--gamma", type=float, default=UTILITY_GAMMA, help="Risk aversion in U_t")
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_SIGNIFICANCE_ALPHA,
        help="Significance level for Wilcoxon H0 rejection (default: 0.05)",
    )
    args = p.parse_args()

    base = args.backtest_dir
    results_dir = args.results_dir or os.path.join(base, RESULTS_SUBDIR)
    window = int(args.utility_window)
    gamma = float(args.gamma)
    alpha = float(args.alpha)
    if not (0.0 < alpha < 1.0):
        print("--alpha must be in (0, 1)", file=sys.stderr)
        sys.exit(1)

    print("Data sources (run `python backtest.py` before this if metrics look stale):\n")
    loaded: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    for name, fname in STRATEGY_FILES:
        path = os.path.join(base, fname)
        df = load_strategy_csv(base, fname)
        if df is None:
            print(f"Missing: {path}", file=sys.stderr)
            summaries.append({"strategy": name, "ok": False})
        else:
            try:
                df = normalize_backtest_df(df, name)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                summaries.append({"strategy": name, "ok": False})
                continue
            loaded[name] = df
            summaries.append(summarize(df, name))
            try:
                mtime = os.path.getmtime(path)
                ts = datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
                print(f"Loaded {name}: {os.path.abspath(path)} (modified {ts})")
            except OSError:
                print(f"Loaded {name}: {os.path.abspath(path)}")

    ok_names = [s["strategy"] for s in summaries if s.get("ok")]
    if len(ok_names) < 2:
        print(
            f"Need at least 2 strategies with valid CSVs for comparison. Found: {ok_names}. "
            "Run:  python backtest.py",
            file=sys.stderr,
        )
        sys.exit(1)
    missing_names = [s["strategy"] for s in summaries if not s.get("ok")]
    if missing_names:
        print(f"  Skipping strategies with missing/invalid CSVs: {missing_names}", file=sys.stderr)

    rolling_frames: dict[str, pd.DataFrame] = {}
    for name in ok_names:
        df_s = loaded[name].sort_values("date").reset_index(drop=True)
        udf = rolling_daily_net_utility(df_s, window, gamma).rename(
            columns={"rolling_u": f"u_{name}"}
        )
        rolling_frames[name] = udf

    j = None
    for name, udf in rolling_frames.items():
        if j is None:
            j = udf
        else:
            j = j.merge(udf, on="date", how="inner")
    if j is None or j.empty:
        print(
            "Error: no overlapping dates after merging rolling utilities. Check date formats in CSVs.",
            file=sys.stderr,
        )
        sys.exit(1)
    dates = j["date"].values
    util_series: dict[str, pd.Series] = {}
    for name in ok_names:
        col = f"u_{name}"
        if col in j.columns:
            util_series[name] = pd.Series(j[col].values, index=dates)

    wilcox_rows: list[dict[str, Any]] = []
    if "MPHDRL" in util_series:
        if "Nifty 50 buy-and-hold" in util_series:
            wilcox_rows.append(
                run_wilcoxon_paired(
                    util_series["MPHDRL"],
                    util_series["Nifty 50 buy-and-hold"],
                    "MPHDRL",
                    "Nifty 50 buy-and-hold",
                )
            )
        if "Benchmark RL" in util_series:
            wilcox_rows.append(
                run_wilcoxon_paired(util_series["MPHDRL"], util_series["Benchmark RL"], "MPHDRL", "Benchmark RL")
            )
        if "Traditional pairs" in util_series:
            wilcox_rows.append(
                run_wilcoxon_paired(util_series["MPHDRL"], util_series["Traditional pairs"], "MPHDRL", "Traditional pairs")
            )
    if "SRRL" in util_series:
        if "Traditional pairs" in util_series:
            wilcox_rows.append(
                run_wilcoxon_paired(util_series["SRRL"], util_series["Traditional pairs"], "SRRL", "Traditional pairs")
            )
        if "MPHDRL" in util_series:
            wilcox_rows.append(
                run_wilcoxon_paired(util_series["SRRL"], util_series["MPHDRL"], "SRRL", "MPHDRL")
            )

    tests_for_json: list[dict[str, Any]] = []
    for w in wilcox_rows:
        row: dict[str, Any] = dict(w)
        row["significance_alpha"] = alpha
        if "error" not in w and w.get("pvalue") is not None:
            rej, verdict = _wilcoxon_conclusion(float(w["pvalue"]), alpha)
            row["reject_h0"] = rej
            row["conclusion"] = verdict
        else:
            row["reject_h0"] = None
            row["conclusion"] = row.get("error", "not run")
        tests_for_json.append(row)

    util_def = utility_definition_text(window, gamma)
    wilcox_json = {
        "utility_definition": util_def,
        "gamma": gamma,
        "window": window,
        "significance_alpha": alpha,
        "hypotheses": {
            "H0": "median(U_A - U_B) <= 0",
            "H1": "median(U_A - U_B) > 0",
        },
        "test_specification": {
            "procedure": "scipy.stats.wilcoxon",
            "paired_on": "date",
            "difference": "U_A(t) - U_B(t)",
            "alternative": "greater",
            "zero_method": "wilcox",
        },
        "tests": tests_for_json,
    }

    report = build_report(summaries, wilcox_rows, window, gamma, alpha=alpha)
    print(report, end="")

    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote: {os.path.abspath(summary_path)}")

    metrics_path = os.path.join(results_dir, "comparison_metrics.csv")
    ok_sum = [s for s in summaries if s.get("ok")]
    metrics_df = pd.DataFrame(ok_sum)
    if "ok" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["ok"])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote: {os.path.abspath(metrics_path)}")

    json_path = os.path.join(results_dir, "wilcoxon_utility.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(wilcox_json, f, indent=2)
    print(f"Wrote: {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
