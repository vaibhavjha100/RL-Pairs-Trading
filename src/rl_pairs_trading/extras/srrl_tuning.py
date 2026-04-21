"""
SRRL hyperparameter tuning runner.

Primary project workflow uses MPHDRL — see mphdrl_tuning.py for the default tuner.

Stages:
  - Stage 0 baseline
  - Stage 1 coarse random search
  - Stage 2 focused refinement around top configs
  - Stage 3 seed confirmation for finalists
"""

from __future__ import annotations

import argparse
import csv
import os
import hashlib
import json
import math
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
import torch

from rl_pairs_trading.backtest_core import summarize_backtest_dataframe

EPOCH_RE = re.compile(
    r"Epoch\s+\d+/\d+\s+\|\s+[0-9.]+s\s+\|\s+(?:(?P<phase>[a-zA-Z_]+)\s+\|\s+)?trans=\d+\s+\|\s+(?P<losses>.*?)\s+\|\s+"
    r"sigma=(?P<sigma>[0-9.\-nan]+)\s+\|E\|=(?P<abs_e>[0-9.\-nan]+)\s+"
    r"churn=(?P<churn>[0-9.\-nan]+)\s+p_rev=(?P<p_rev>[0-9.\-nan]+)"
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
    parser = argparse.ArgumentParser(description="Tune SRRL hyperparameters.")
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
    parser.add_argument("--prune-churn", type=float, default=0.20)
    parser.add_argument("--outdir", type=str, default="artifacts/srrl_tuning")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def stable_id(stage: str, params: Dict[str, float], seed: int) -> str:
    raw = json.dumps({"stage": stage, "seed": seed, "params": params}, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def cap_cls_warmup_epochs(params: Dict[str, float], epochs: int) -> Dict[str, float]:
    """Copy params for trial identity; training.py splits epochs half cls / half RL automatically."""
    return dict(params)


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
        "n_step": rng.choice([5, 10, 15, 20]),
        "turnover_penalty": rng.choice([0.0, 0.01, 0.03, 0.05, 0.1]),
        "sigma_explore": rng.uniform(0.15, 0.35),
        "sigma_explore_min": rng.uniform(0.03, 0.15),
        "tau": rng.uniform(0.002, 0.01),
        "dropout": rng.uniform(0.0, 0.2),
        "weight_decay": 10 ** rng.uniform(-5.0, -3.0),
        "cls_label_smoothing": rng.uniform(0.0, 0.14),
        "cls_warmup_epochs": rng.choice([0, 0, 2, 4, 8, 12, 16, 24, 32]),
    }


def sample_stage2_params(rng: random.Random, base_cfg: Dict[str, float]) -> Dict[str, float]:
    cfg = dict(base_cfg)
    cfg["lr"] = float(np.clip(cfg["lr"] * rng.uniform(0.7, 1.3), 5e-5, 5e-4))
    cfg["tau"] = float(np.clip(cfg["tau"] * rng.uniform(0.7, 1.3), 0.002, 0.01))
    cfg["turnover_penalty"] = rng.choice([0.0, 0.01, 0.03, 0.05, 0.1])
    cfg["sigma_explore"] = float(np.clip(cfg["sigma_explore"] + rng.uniform(-0.05, 0.05), 0.15, 0.35))
    cfg["sigma_explore_min"] = float(np.clip(cfg["sigma_explore_min"] + rng.uniform(-0.03, 0.03), 0.03, 0.15))
    cfg["sigma_explore_min"] = min(cfg["sigma_explore_min"], cfg["sigma_explore"])
    cfg["batch_size"] = rng.choice([32, 64])
    cfg["discount_gamma"] = rng.uniform(0.97, 0.995)
    cfg["gamma_risk"] = rng.choice([0.4, 0.5, 0.6])
    cfg["risk_lambda"] = rng.choice([0.8, 1.0, 1.2])
    d0 = float(cfg.get("dropout", 0.12))
    wd0 = float(cfg.get("weight_decay", 1e-4))
    ls0 = float(cfg.get("cls_label_smoothing", 0.08))
    cfg["dropout"] = float(np.clip(d0 + rng.uniform(-0.05, 0.05), 0.0, 0.25))
    cfg["weight_decay"] = float(np.clip(wd0 * rng.uniform(0.5, 2.0), 1e-6, 5e-3))
    cfg["cls_label_smoothing"] = float(np.clip(ls0 + rng.uniform(-0.04, 0.04), 0.0, 0.2))
    cw0 = int(round(float(cfg.get("cls_warmup_epochs", 0))))
    cfg["cls_warmup_epochs"] = int(
        np.clip(cw0 + rng.choice([-8, -4, -2, 0, 2, 4, 8]), 0, 48)
    )
    return cfg


def parse_last_epoch_metrics(train_output: str) -> Dict[str, float]:
    matches = list(EPOCH_RE.finditer(train_output))
    if not matches:
        return {"churn": float("nan"), "abs_e": float("nan"), "p_rev": float("nan")}
    last = matches[-1]
    sigma = last.group("sigma")
    abs_e = last.group("abs_e")
    churn = last.group("churn")
    p_rev = last.group("p_rev")
    return {
        "sigma": float(sigma),
        "abs_e": float(abs_e),
        "churn": float(churn),
        "p_rev": float(p_rev),
    }


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


def compute_utility_from_srrl_csv(csv_path: Path, gamma: float = 0.5) -> Dict[str, float]:
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
    params = cap_cls_warmup_epochs(params, epochs)
    trial_id = stable_id(stage, params, seed)
    log_path = paths["logs"] / f"{stage}_{trial_id}.log"
    ckpt_path = paths["ckpt"] / f"{stage}_{trial_id}.pt"

    patch_path = paths["base"] / f"srrl_hp_patch_{trial_id}.json"
    patch_path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    trial_env = os.environ.copy()
    trial_env["SRRL_HP_PATCH"] = str(patch_path.resolve())
    trial_env["SRRL_TRAIN_EPOCHS"] = str(epochs)
    trial_env["SRRL_SAVE_EVERY"] = str(epochs)
    trial_env["SRRL_DEVICE"] = device
    trial_env["SRRL_SEED"] = str(seed)
    trial_env["TRAINING_AGENT"] = "SRRL"

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

    final_ckpt = root / "models" / "srrl" / "final.pt"
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
            note="models/srrl/final.pt not found",
        )

    bt_cmd = [
        pybin,
        "backtest.py",
        "--device",
        device,
        "--srrl-checkpoint",
        str(ckpt_path),
    ]
    bt_proc = run_cmd(bt_cmd, root)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n\nBACKTEST STDOUT:\n")
        f.write(bt_proc.stdout)
        f.write("\n\nBACKTEST STDERR:\n")
        f.write(bt_proc.stderr)
    if bt_proc.returncode != 0:
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
            note=f"backtest return code {bt_proc.returncode}",
        )

    metrics = compute_utility_from_srrl_csv(root / "data" / "backtest" / "srrl.csv")
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
        params = cap_cls_warmup_epochs(params, epochs)
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
        "n_step": 10,
        "turnover_penalty": 0.05,
        "sigma_explore": 0.3,
        "sigma_explore_min": 0.10,
        "dropout": 0.12,
        "weight_decay": 1e-4,
        "cls_label_smoothing": 0.08,
        "cls_warmup_epochs": 0,
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
