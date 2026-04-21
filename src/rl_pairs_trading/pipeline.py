"""
Ordered pipeline: collection → preprocessing ×2 (after spread creates raw.csv) → optional EDA →
train MPHDRL + Benchmark → backtest → comparison → market neutrality.

Run from repo root (with pip install -e .):
  python -m rl_pairs_trading.pipeline
  rl-pairs-pipeline
Or: python run_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    """Assume cwd is the project root (data/, models/ live here)."""
    return Path.cwd().resolve()


def _append_log(log_path: Path | None, text: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text)


def _run_step(
    name: str,
    argv: list[str],
    *,
    cwd: Path,
    extra_env: dict[str, str] | None,
    dry_run: bool,
    log_path: Path | None,
) -> int:
    cmd = [sys.executable, *argv]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    line = f"\n{'=' * 60}\nSTEP: {name}\n$ {' '.join(cmd)}\n"
    print(line, end="")
    _append_log(log_path, line)
    if dry_run:
        msg = "(dry-run: not executed)\n"
        print(msg, end="")
        _append_log(log_path, msg)
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    out = f"exit_code={proc.returncode}\n"
    _append_log(log_path, out)
    return int(proc.returncode)


def _eda_step(repo: Path, dry_run: bool, log_path: Path | None) -> int:
    nb = repo / "eda.ipynb"
    line = f"\n{'=' * 60}\nSTEP: eda (notebook)\nTarget: {nb}\n"
    print(line, end="")
    _append_log(log_path, line)
    if not nb.is_file():
        msg = f"eda.ipynb not found at {nb}; skip.\n"
        print(msg, end="")
        _append_log(log_path, msg)
        return 0
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(nb),
        "--output",
        str(nb.with_name("eda_executed.ipynb")),
    ]
    if dry_run:
        print("(dry-run: not executed)\n")
        _append_log(log_path, "(dry-run: not executed)\n")
        return 0
    proc = subprocess.run(cmd, cwd=str(repo))
    _append_log(log_path, f"exit_code={proc.returncode}\n")
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the RL pairs trading pipeline from repo root.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps without running subprocesses (EDA dry-run skips nbconvert).",
    )
    p.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip optional EDA notebook execution after the second preprocessing pass.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Append step transcript to this file (default: artifacts/pipeline_logs/<timestamp>.log).",
    )
    p.add_argument(
        "--no-default-log",
        action="store_true",
        help="Do not write the default log under artifacts/pipeline_logs/ unless --log-file is set.",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of step names to run (see --list-steps).",
    )
    p.add_argument(
        "--list-steps",
        action="store_true",
        help="Print step names and exit.",
    )
    return p


STEP_SPECS: list[tuple[str, list[str], dict[str, str] | None]] = [
    ("collection", ["-m", "rl_pairs_trading.collection"], None),
    ("preprocessing_pass1", ["-m", "rl_pairs_trading.preprocessing"], None),
    ("pairs", ["-m", "rl_pairs_trading.pairs"], None),
    ("spread", ["-m", "rl_pairs_trading.spread"], None),
    ("preprocessing_pass2", ["-m", "rl_pairs_trading.preprocessing"], None),
    ("train_mphdrl", ["-m", "rl_pairs_trading.training"], {"TRAINING_AGENT": "MPHDRL"}),
    ("train_benchmark", ["-m", "rl_pairs_trading.training"], {"TRAINING_AGENT": "Benchmark"}),
    ("backtest", ["-m", "rl_pairs_trading.backtest"], None),
    ("comparison", ["-m", "rl_pairs_trading.comparison"], None),
    ("market_neutral", ["-m", "rl_pairs_trading.market_neutral"], None),
]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_steps:
        print("Steps:", ", ".join([s[0] for s in STEP_SPECS]))
        print("Optional: eda (not listed; use --only eda or omit --skip-eda)")
        return 0

    repo = _repo_root()
    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    log_path: Path | None = None
    if args.log_file:
        log_path = Path(args.log_file)
    elif not args.no_default_log:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = repo / "artifacts" / "pipeline_logs" / f"pipeline_{stamp}.log"

    header = f"Pipeline start {datetime.now().isoformat()}\ncwd={repo}\n"
    _append_log(log_path, header)
    print(header, end="")

    steps_to_run: list[tuple[str, list[str], dict[str, str] | None]] = []
    for spec in STEP_SPECS:
        name = spec[0]
        if only is not None and name not in only:
            continue
        steps_to_run.append(spec)

    run_eda = not args.skip_eda and (only is None or "eda" in only)

    for name, argv_m, env_extra in steps_to_run:
        code = _run_step(name, argv_m, cwd=repo, extra_env=env_extra, dry_run=args.dry_run, log_path=log_path)
        if code != 0:
            print(f"Step {name} failed with exit code {code}.")
            return code
        if name == "preprocessing_pass2" and run_eda:
            code = _eda_step(repo, args.dry_run, log_path)
            if code != 0:
                print(f"EDA step failed with exit code {code}.")
                return code

    if only is not None and "eda" in only and "preprocessing_pass2" not in only:
        # User asked only for eda
        code = _eda_step(repo, args.dry_run, log_path)
        if code != 0:
            return code

    tail = f"\nPipeline finished OK {datetime.now().isoformat()}\n"
    _append_log(log_path, tail)
    print(tail, end="")
    if log_path:
        print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
