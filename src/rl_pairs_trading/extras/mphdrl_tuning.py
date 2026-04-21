"""
MPHDRL hyperparameter tuning runner.

Stages:
  - Stage 0 baseline
  - Stage 1 coarse random search
  - Stage 2 focused refinement around top configs
  - Stage 3 seed confirmation for finalists

Training is invoked via environment variables (see training.py): MPHDRL_HP_PATCH, etc.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pickle
import torch

from rl_pairs_trading.mphdrl import MPHDRLTrader, build_pair_ticker_mapping

from rl_pairs_trading.backtest_core import (
    load_price_matrix,
    load_sequence_bundle,
    get_mphdrl_weights_by_env,
    run_strategy_backtest,
    summarize_backtest_dataframe,
)

from rl_pairs_trading.training import merge_mphdrl_params_dict

# MPHDRL logs: Epoch .../... | ...s | trans=... | loss keys (no SRRL sigma/churn line)
EPOCH_RE = re.compile(
    r"Epoch\s+\d+/\d+\s+\|\s+[0-9.]+s\s+\|\s+trans=\d+\s+\|\s+(?P<losses>.+)"
)


@dataclass
class TrialResult:
    stage: str
    trial_id: str
    seed: int
    params: Dict[str, float]
    status: str
    utility: float
    annual_return: float
    sharpe: float
    mean_abs_weight: float
    l1_turnover: float
    transaction_cost: float
    shorting_cost: float
    churn: float
    abs_e: float
    p_rev: float
    checkpoint_path: str
    train_log_path: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune MPHDRL hyperparameters.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--stage1-trials", type=int, default=24)
    parser.add_argument("--stage2-trials", type=int, default=12)
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=4)
    parser.add_argument("--confirm-epochs", type=int, default=4)
    parser.add_argument("--confirm-seeds", type=int, default=2)
    parser.add_argument("--topk-stage1", type=int, default=6)
    parser.add_argument("--topk-stage2", type=int, default=3)
    parser.add_argument("--prune-churn", type=float, default=0.99)
    parser.add_argument("--outdir", type=str, default="artifacts/mphdrl_tuning")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def stable_id(stage: str, params: Dict[str, float], seed: int) -> str:
    raw = json.dumps({"stage": stage, "seed": seed, "params": params}, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def ensure_dirs(root: Path, outdir: str) -> Dict[str, Path]:
    base = root / outdir
    ckpt = base / "checkpoints"
    logs = base / "logs"
    for p in (base, ckpt, logs):
        p.mkdir(parents=True, exist_ok=True)
    return {"base": base, "ckpt": ckpt, "logs": logs}


def load_trials_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def append_trial(csv_path: Path, row: TrialResult):
    file_exists = csv_path.exists()
    payload = asdict(row)
    payload["params"] = json.dumps(payload["params"], sort_keys=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)


def sample_stage1_params(rng: random.Random) -> Dict[str, float]:
    lr = 10 ** rng.uniform(math.log10(5e-5), math.log10(5e-4))
    return {
        "lr": lr,
        "tau": rng.uniform(0.002, 0.012),
        "n_step": float(rng.choice([5, 10, 15, 20])),
        "sigma_explore": rng.uniform(0.4, 0.95),
        "sigma_smooth": rng.uniform(0.4, 0.95),
        "batch_size": float(rng.choice([16, 32, 64])),
        "discount_gamma": rng.uniform(0.97, 0.995),
        "per_alpha": rng.uniform(0.4, 0.75),
        "per_beta_start": rng.uniform(0.25, 0.55),
        "zeta": 10 ** rng.uniform(math.log10(1e-4), math.log10(0.02)),
        "gamma": rng.uniform(0.35, 0.65),
        "risk_lambda": rng.uniform(0.7, 1.4),
        "terminal_utility_weight": rng.uniform(0.5, 1.5),
        "delay_c": float(rng.choice([2, 4, 6, 8])),
        "delay_b": float(rng.choice([1, 2, 3, 4])),
    }


def sample_stage2_params(rng: random.Random, base_cfg: Dict[str, float]) -> Dict[str, float]:
    cfg = dict(base_cfg)
    cfg["lr"] = float(np.clip(cfg["lr"] * rng.uniform(0.75, 1.25), 5e-5, 5e-4))
    cfg["tau"] = float(np.clip(cfg["tau"] * rng.uniform(0.75, 1.25), 0.002, 0.015))
    cfg["sigma_explore"] = float(np.clip(cfg["sigma_explore"] + rng.uniform(-0.1, 0.1), 0.25, 1.0))
    cfg["sigma_smooth"] = float(np.clip(cfg["sigma_smooth"] + rng.uniform(-0.1, 0.1), 0.25, 1.0))
    cfg["batch_size"] = float(rng.choice([16, 32, 64]))
    cfg["discount_gamma"] = float(np.clip(cfg["discount_gamma"] + rng.uniform(-0.015, 0.015), 0.96, 0.999))
    cfg["per_alpha"] = float(np.clip(cfg["per_alpha"] + rng.uniform(-0.08, 0.08), 0.3, 0.85))
    cfg["per_beta_start"] = float(np.clip(cfg["per_beta_start"] + rng.uniform(-0.08, 0.08), 0.15, 0.65))
    cfg["zeta"] = float(np.clip(cfg["zeta"] * rng.uniform(0.5, 2.0), 1e-5, 0.05))
    cfg["gamma"] = float(np.clip(cfg["gamma"] + rng.uniform(-0.08, 0.08), 0.25, 0.75))
    cfg["risk_lambda"] = float(np.clip(cfg["risk_lambda"] + rng.uniform(-0.15, 0.15), 0.5, 2.0))
    cfg["terminal_utility_weight"] = float(
        np.clip(cfg["terminal_utility_weight"] + rng.uniform(-0.25, 0.25), 0.25, 2.0)
    )
    cfg["delay_c"] = float(rng.choice([2, 4, 6, 8]))
    cfg["delay_b"] = float(rng.choice([1, 2, 3, 4]))
    cfg["n_step"] = float(rng.choice([5, 10, 15, 20]))
    return cfg


def parse_last_epoch_metrics(train_output: str) -> Dict[str, float]:
    matches = list(EPOCH_RE.finditer(train_output))
    if not matches:
        return {"churn": float("nan"), "abs_e": float("nan"), "p_rev": float("nan")}
    last = matches[-1]
    loss_blob = last.group("losses") or ""
    churn = float("nan")
    if "churn=" in loss_blob:
        for part in loss_blob.replace("|", " ").split():
            if part.startswith("churn="):
                try:
                    churn = float(part.split("=", 1)[1])
                except ValueError:
                    pass
                break
    return {"churn": churn, "abs_e": float("nan"), "p_rev": float("nan")}


def run_cmd(
    cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    kw: Dict = dict(cwd=str(cwd), text=True, capture_output=True)
    if env is not None:
        kw["env"] = env
    return subprocess.run(cmd, **kw)


def resolve_device(preference: str) -> str:
    name = (preference or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def compute_utility_from_mphdrl_csv(csv_path: Path, gamma: float = 0.5) -> Dict[str, float]:
    if not csv_path.exists():
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
    df = pd.read_csv(csv_path)
    return summarize_backtest_dataframe(df, gamma=gamma)


def score_mphdrl_checkpoint(root: Path, ckpt_path: Path, params: Dict[str, float], device_pref: str) -> Dict[str, float]:
    """
    Walk-forward utility on the test split (same gamma as training in-sample selection).
    Merges trial params into HPARAMS so env reward matches the trained run.
    """
    merge_mphdrl_params_dict(params)
    dev_str = resolve_device(device_pref)
    device = torch.device(dev_str)

    x_test, y_test, pairs, meta_test = load_sequence_bundle(split="test")
    n_pairs = len(pairs)
    f_dim = x_test.shape[2]

    hedge_path = root / "data" / "pickle" / "hedge_ratios.pkl"
    hedge_ratios = None
    if hedge_path.is_file():
        with open(hedge_path, "rb") as f:
            hedge_ratios = pickle.load(f)
    M, tickers, ticker_to_idx = build_pair_ticker_mapping(pairs, hedge_ratios=hedge_ratios)

    model = MPHDRLTrader(f_dim, n_pairs, len(tickers), M, device=str(device))
    model.load_checkpoint(str(ckpt_path))
    model.eval()

    spread_raw_path = root / "data" / "spread" / "raw.csv"
    spread_wide = None
    if spread_raw_path.is_file():
        from rl_pairs_trading.traditional import load_precomputed_spread_wide

        spread_wide = load_precomputed_spread_wide(str(spread_raw_path))

    weights_by_date = get_mphdrl_weights_by_env(
        model,
        meta_test,
        x_test,
        y_test,
        pairs,
        tickers,
        ticker_to_idx,
        spread_wide=spread_wide,
        split="test",
    )
    price_wide = load_price_matrix(tickers)
    test_dates = sorted(meta_test["target_date"].unique())
    valid_dates = [d for d in test_dates if d in price_wide.index]
    if len(valid_dates) < 2:
        return compute_utility_from_mphdrl_csv(Path())

    df_bt = run_strategy_backtest("MPHDRL tuning", weights_by_date, price_wide, tickers, valid_dates)
    return summarize_backtest_dataframe(df_bt)


def run_trial(
    root: Path,
    paths: Dict[str, Path],
    pybin: str,
    device: str,
    stage: str,
    params: Dict[str, float],
    seed: int,
    epochs: int,
    prune_churn: float,
) -> TrialResult:
    trial_id = stable_id(stage, params, seed)
    log_path = paths["logs"] / f"{stage}_{trial_id}.log"
    ckpt_path = paths["ckpt"] / f"{stage}_{trial_id}.pt"

    patch_path = paths["base"] / f"mphdrl_hp_patch_{trial_id}.json"
    patch_path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    trial_env = os.environ.copy()
    trial_env["MPHDRL_HP_PATCH"] = str(patch_path.resolve())
    trial_env["MPHDRL_TRAIN_EPOCHS"] = str(epochs)
    trial_env["MPHDRL_SAVE_EVERY"] = str(epochs)
    trial_env["MPHDRL_DEVICE"] = device
    trial_env["MPHDRL_SEED"] = str(seed)
    trial_env["TRAINING_AGENT"] = "MPHDRL"

    train_cmd = [pybin, "-m", "rl_pairs_trading.training"]

    train_proc = run_cmd(train_cmd, root, env=trial_env)
    log_path.write_text(train_proc.stdout + "\n\nSTDERR:\n" + train_proc.stderr, encoding="utf-8")
    m = parse_last_epoch_metrics(train_proc.stdout)
    if train_proc.returncode != 0:
        return TrialResult(
            stage=stage,
            trial_id=trial_id,
            seed=seed,
            params=params,
            status="train_failed",
            utility=float("-inf"),
            annual_return=float("nan"),
            sharpe=float("nan"),
            mean_abs_weight=float("nan"),
            l1_turnover=float("nan"),
            transaction_cost=float("nan"),
            shorting_cost=float("nan"),
            churn=m["churn"],
            abs_e=m["abs_e"],
            p_rev=m["p_rev"],
            checkpoint_path="",
            train_log_path=str(log_path),
            note=f"training return code {train_proc.returncode}",
        )
    if np.isfinite(m["churn"]) and m["churn"] > prune_churn:
        return TrialResult(
            stage=stage,
            trial_id=trial_id,
            seed=seed,
            params=params,
            status="pruned",
            utility=float("-inf"),
            annual_return=float("nan"),
            sharpe=float("nan"),
            mean_abs_weight=float("nan"),
            l1_turnover=float("nan"),
            transaction_cost=float("nan"),
            shorting_cost=float("nan"),
            churn=m["churn"],
            abs_e=m["abs_e"],
            p_rev=m["p_rev"],
            checkpoint_path="",
            train_log_path=str(log_path),
            note=f"churn {m['churn']:.4f} > prune_churn {prune_churn:.4f}",
        )

    final_ckpt = root / "models" / "MPHDRL" / "final.pt"
    if final_ckpt.exists():
        shutil.copy2(final_ckpt, ckpt_path)
    else:
        return TrialResult(
            stage=stage,
            trial_id=trial_id,
            seed=seed,
            params=params,
            status="missing_ckpt",
            utility=float("-inf"),
            annual_return=float("nan"),
            sharpe=float("nan"),
            mean_abs_weight=float("nan"),
            l1_turnover=float("nan"),
            transaction_cost=float("nan"),
            shorting_cost=float("nan"),
            churn=m["churn"],
            abs_e=m["abs_e"],
            p_rev=m["p_rev"],
            checkpoint_path="",
            train_log_path=str(log_path),
            note="models/MPHDRL/final.pt not found",
        )

    try:
        metrics = score_mphdrl_checkpoint(root, ckpt_path, params, device)
    except Exception as e:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n\nSCORE EXCEPTION:\n{e}\n")
        return TrialResult(
            stage=stage,
            trial_id=trial_id,
            seed=seed,
            params=params,
            status="backtest_failed",
            utility=float("-inf"),
            annual_return=float("nan"),
            sharpe=float("nan"),
            mean_abs_weight=float("nan"),
            l1_turnover=float("nan"),
            transaction_cost=float("nan"),
            shorting_cost=float("nan"),
            churn=m["churn"],
            abs_e=m["abs_e"],
            p_rev=m["p_rev"],
            checkpoint_path=str(ckpt_path),
            train_log_path=str(log_path),
            note=f"score_mphdrl_checkpoint failed: {e}",
        )
    return TrialResult(
        stage=stage,
        trial_id=trial_id,
        seed=seed,
        params=params,
        status="ok",
        utility=metrics["utility"],
        annual_return=metrics["annual_return"],
        sharpe=metrics["sharpe"],
        mean_abs_weight=metrics["mean_abs_weight"],
        l1_turnover=metrics["l1_turnover"],
        transaction_cost=metrics["transaction_cost"],
        shorting_cost=metrics["shorting_cost"],
        churn=m["churn"],
        abs_e=m["abs_e"],
        p_rev=m["p_rev"],
        checkpoint_path=str(ckpt_path),
        train_log_path=str(log_path),
        note="",
    )


def top_configs(df: pd.DataFrame, stage: str, k: int) -> List[Dict[str, float]]:
    sub = df[(df["stage"] == stage) & (df["status"] == "ok")].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("utility", ascending=False).head(k)
    return [json.loads(p) if isinstance(p, str) else p for p in sub["params"].tolist()]


def summarize(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    out = {
        "num_trials": int(len(df)),
        "num_ok": int((df["status"] == "ok").sum()),
        "best_utility": float(df.loc[df["utility"].idxmax(), "utility"]),
        "best_trial_id": str(df.loc[df["utility"].idxmax(), "trial_id"]),
    }
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    paths = ensure_dirs(root, args.outdir)
    trials_csv = paths["base"] / "trials.csv"
    summary_json = paths["base"] / "summary.json"
    rng = random.Random(args.base_seed)
    device = resolve_device(args.device)

    existing = load_trials_csv(trials_csv) if args.resume else pd.DataFrame()
    if not existing.empty and "status" in existing.columns:
        resumable_statuses = {"ok", "pruned"}
        seen_ids = set(existing.loc[existing["status"].isin(resumable_statuses), "trial_id"].tolist())
    else:
        seen_ids = set()

    def maybe_run(stage: str, params: Dict[str, float], seed: int, epochs: int):
        tid = stable_id(stage, params, seed)
        if tid in seen_ids:
            return
        result = run_trial(
            root=root,
            paths=paths,
            pybin=args.python_bin,
            device=device,
            stage=stage,
            params=params,
            seed=seed,
            epochs=epochs,
            prune_churn=args.prune_churn,
        )
        append_trial(trials_csv, result)
        seen_ids.add(tid)
        print(
            f"[{stage}] {tid} status={result.status} "
            f"utility={result.utility:.6f} churn={result.churn:.4f}"
        )

    baseline_params = {
        "lr": 1e-4,
        "tau": 0.005,
        "n_step": 10.0,
        "sigma_explore": 0.7,
        "sigma_smooth": 0.7,
        "batch_size": 32.0,
        "discount_gamma": 0.99,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "zeta": 0.003,
        "gamma": 0.5,
        "risk_lambda": 1.0,
        "terminal_utility_weight": 1.0,
        "delay_c": 4.0,
        "delay_b": 2.0,
    }

    maybe_run("stage0_baseline", baseline_params, args.base_seed, args.stage1_epochs)

    for _ in range(args.stage1_trials):
        maybe_run("stage1", sample_stage1_params(rng), args.base_seed, args.stage1_epochs)

    df = pd.read_csv(trials_csv)
    s1_top = top_configs(df, "stage1", args.topk_stage1)
    if not s1_top:
        print("No successful Stage 1 trials found; stopping.")
        summarize(df, summary_json)
        return

    for i in range(args.stage2_trials):
        base_cfg = s1_top[i % len(s1_top)]
        maybe_run("stage2", sample_stage2_params(rng, base_cfg), args.base_seed, args.stage2_epochs)

    df = pd.read_csv(trials_csv)
    s2_top = top_configs(df, "stage2", args.topk_stage2)
    if not s2_top:
        print("No successful Stage 2 trials found; stopping.")
        summarize(df, summary_json)
        return

    confirm_seeds = [args.base_seed + i + 1 for i in range(args.confirm_seeds)]
    for rank, cfg in enumerate(s2_top, start=1):
        for sd in confirm_seeds:
            maybe_run(f"stage3_rank{rank}", cfg, sd, args.confirm_epochs)

    df = pd.read_csv(trials_csv)
    summarize(df, summary_json)
    print(f"Tuning complete. Results: {trials_csv}")
    print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
