from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_pairs_trading.mphdrl import MPHDRL_MODEL_DIR, MPHDRLTrader, build_pair_ticker_mapping
from rl_pairs_trading.backtest_core import load_sequence_bundle

try:
    from captum.attr import IntegratedGradients
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Captum is required for xai_mphdrl.py. Install with: pip install captum"
    ) from e


@dataclass
class StateSelection:
    date: pd.Timestamp
    windows: np.ndarray
    pair_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference-time explainability for a frozen MPHDRL checkpoint using Captum IG."
    )
    p.add_argument("--device", type=str, default="cpu", help="Torch device (cpu/cuda/mps/auto).")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional explicit MPHDRL checkpoint path (defaults to final.pt or checkpoint.pt).",
    )
    p.add_argument(
        "--date",
        type=str,
        default="",
        help="Optional target_date (YYYY-MM-DD) from sequence_xy test split.",
    )
    p.add_argument(
        "--state-ordinal",
        type=int,
        default=0,
        help="If --date omitted, choose this ordinal date from sorted test dates.",
    )
    p.add_argument("--pair-index", type=int, default=-1, help="Pair index for actor/stop attribution.")
    p.add_argument("--action-id", type=int, default=0, help="Actor action id in {0,1,2}.")
    p.add_argument("--sl-idx", type=int, default=0, help="Stop-loss action index in {0..4}.")
    p.add_argument("--ticker-index", type=int, default=0, help="Ticker index for portfolio attribution.")
    p.add_argument("--ig-steps", type=int, default=64, help="Integrated gradients n_steps.")
    p.add_argument(
        "--pca-path",
        type=str,
        default=os.path.join("data", "pickle", "spread_pca.pkl"),
        help="Path to saved PCA pickle for PC->original feature mapping.",
    )
    p.add_argument(
        "--preprocessed-path",
        type=str,
        default=os.path.join("data", "spread", "preprocessed.csv"),
        help="Path to spread preprocessed CSV for original feature names.",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join("data", "backtest", "results"),
        help="Directory for XAI output files.",
    )
    return p.parse_args()


def resolve_device(preference: str) -> torch.device:
    name = (preference or "cpu").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def default_checkpoint_path(explicit: str) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for name in ("final.pt", "checkpoint.pt"):
        p = os.path.join(MPHDRL_MODEL_DIR, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint found. Tried explicit path and {MPHDRL_MODEL_DIR}/final.pt or checkpoint.pt"
    )


def get_continuous_feature_names(preprocessed_path: str) -> List[str]:
    if not os.path.isfile(preprocessed_path):
        raise FileNotFoundError(f"Missing spread preprocessed CSV: {preprocessed_path}")
    df = pd.read_csv(preprocessed_path, nrows=1)
    id_cols = {"Date", "Pair"}
    feature_cols = [c for c in df.columns if c not in id_cols]
    continuous_cols = [c for c in feature_cols if c != "body_color"]
    if not continuous_cols:
        raise ValueError("No continuous spread features found in preprocessed CSV.")
    return continuous_cols


def load_pca_mapping(pca_path: str, preprocessed_path: str) -> Tuple[np.ndarray, List[str]]:
    if not os.path.isfile(pca_path):
        raise FileNotFoundError(f"Missing PCA pickle: {pca_path}")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    if not hasattr(pca, "components_"):
        raise ValueError(f"PCA object at {pca_path} has no components_.")
    comps = np.asarray(pca.components_, dtype=np.float64)
    names = get_continuous_feature_names(preprocessed_path)
    if comps.shape[1] != len(names):
        raise ValueError(
            f"PCA component width ({comps.shape[1]}) != continuous feature count ({len(names)})."
        )
    return comps, names


def build_state_for_date(
    x_test: np.ndarray,
    pairs: list,
    meta_test: pd.DataFrame,
    chosen_date: pd.Timestamp,
) -> StateSelection:
    pair_key_to_idx = {f"{a}|{b}": i for i, (a, b) in enumerate(pairs)}
    n_pairs = len(pairs)
    f_dim = x_test.shape[2]
    sub = meta_test[meta_test["target_date"] == chosen_date]
    windows = np.zeros((n_pairs, x_test.shape[1], f_dim), dtype=np.float32)
    mask = np.zeros(n_pairs, dtype=np.float32)
    for _, row in sub.iterrows():
        iloc = row.name
        pkey = str(row["Pair"])
        if pkey in pair_key_to_idx:
            p_idx = pair_key_to_idx[pkey]
            windows[p_idx] = x_test[iloc]
            mask[p_idx] = 1.0
    if mask.sum() < 1:
        raise ValueError(f"No pair windows found for selected date: {chosen_date.date()}")
    return StateSelection(date=chosen_date, windows=windows, pair_mask=mask)


def choose_date(meta_test: pd.DataFrame, date_arg: str, ordinal: int) -> pd.Timestamp:
    dates = sorted(pd.to_datetime(meta_test["target_date"].unique()))
    if not dates:
        raise ValueError("No test dates available in sequence metadata.")
    if date_arg.strip():
        d = pd.Timestamp(date_arg).normalize()
        for cand in dates:
            if pd.Timestamp(cand).normalize() == d:
                return pd.Timestamp(cand)
        raise ValueError(f"Date {date_arg} not found in test split dates.")
    idx = max(0, min(int(ordinal), len(dates) - 1))
    return pd.Timestamp(dates[idx])


class _FrozenHeadBase(nn.Module):
    def __init__(self, model: MPHDRLTrader, pair_mask: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("pair_mask", pair_mask.float())

    def _pair_mask_batch(self, batch_size: int) -> torch.Tensor:
        if self.pair_mask.dim() == 1:
            pm = self.pair_mask.unsqueeze(0)
        else:
            pm = self.pair_mask
        return pm.expand(batch_size, -1)

    def _forward_components(self, windows: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_actor = self.model.encode_all_pairs(windows, self.model.srl_actor)
        h_stop = self.model.encode_all_pairs(windows, self.model.srl_stop)
        b, p, h = h_actor.shape

        actor_logits = self.model.actor(h_actor.reshape(b * p, h)).reshape(b, p, 3)
        actor_probs = F.softmax(actor_logits, dim=-1)
        actions = actor_logits.argmax(dim=-1)

        stop_logits = self.model.stop_loss(h_stop.reshape(b * p, h)).reshape(b, p, -1)
        sl_actions = stop_logits.argmax(dim=-1)

        pm = self._pair_mask_batch(b)
        weights = self.model.portfolio(
            h_actor,
            actor_probs,
            sl_actions,
            self.model.M,
            actions=actions,
            training=False,
            pair_mask=pm,
        )
        return {
            "actor_probs": actor_probs,
            "stop_logits": stop_logits,
            "weights": weights,
        }


class MPHDRLActorHead(_FrozenHeadBase):
    def __init__(self, model: MPHDRLTrader, pair_mask: torch.Tensor, pair_idx: int, action_id: int):
        super().__init__(model, pair_mask)
        self.pair_idx = int(pair_idx)
        self.action_id = int(action_id)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        out = self._forward_components(windows)
        return out["actor_probs"][:, self.pair_idx, self.action_id]


class MPHDRLStopLossHead(_FrozenHeadBase):
    def __init__(self, model: MPHDRLTrader, pair_mask: torch.Tensor, pair_idx: int, sl_idx: int):
        super().__init__(model, pair_mask)
        self.pair_idx = int(pair_idx)
        self.sl_idx = int(sl_idx)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        out = self._forward_components(windows)
        return out["stop_logits"][:, self.pair_idx, self.sl_idx]


class MPHDRLPortfolioHead(_FrozenHeadBase):
    def __init__(self, model: MPHDRLTrader, pair_mask: torch.Tensor, ticker_idx: int):
        super().__init__(model, pair_mask)
        self.ticker_idx = int(ticker_idx)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        out = self._forward_components(windows)
        return out["weights"][:, self.ticker_idx]


def run_ig(
    head: nn.Module,
    windows_t: torch.Tensor,
    baseline_t: torch.Tensor,
    n_steps: int,
) -> Dict[str, Any]:
    ig = IntegratedGradients(head)
    attr, delta = ig.attribute(
        inputs=windows_t,
        baselines=baseline_t,
        n_steps=max(8, int(n_steps)),
        return_convergence_delta=True,
    )
    return {"attributions": attr.detach(), "delta": delta.detach()}


def aggregate_feature_attribution(
    attr_t: torch.Tensor,
    pair_mask: np.ndarray,
    pair_idx: int,
    feature_names: List[str],
) -> Dict[str, Any]:
    arr = attr_t.detach().cpu().numpy()
    # arr shape: (1, N_pairs, lookback, F)
    pair_feature_signed = arr.sum(axis=2)[0]  # (N_pairs, F)
    pair_feature_abs = np.abs(arr).sum(axis=2)[0]
    active = pair_mask > 0.5
    active_idx = np.where(active)[0]
    if len(active_idx) == 0:
        active_idx = np.arange(pair_feature_signed.shape[0])
    global_signed = pair_feature_signed[active_idx].sum(axis=0)
    global_abs = pair_feature_abs[active_idx].sum(axis=0)
    selected_signed = pair_feature_signed[pair_idx]
    selected_abs = pair_feature_abs[pair_idx]

    def _to_rows(v_signed: np.ndarray, v_abs: np.ndarray) -> List[Dict[str, float]]:
        idx = np.argsort(v_abs)[::-1]
        out = []
        for j in idx:
            out.append(
                {
                    "feature": feature_names[j],
                    "signed_attr": float(v_signed[j]),
                    "abs_attr": float(v_abs[j]),
                }
            )
        return out

    return {
        "selected_pair_feature_rows": _to_rows(selected_signed, selected_abs),
        "global_feature_rows": _to_rows(global_signed, global_abs),
    }


def map_pc_to_original(
    feature_rows: List[Dict[str, float]],
    pca_components: np.ndarray,
    original_names: List[str],
) -> Dict[str, Any]:
    feature_to_signed = {r["feature"]: float(r["signed_attr"]) for r in feature_rows}
    pc_count = pca_components.shape[0]
    pc_vec = np.array([feature_to_signed.get(f"PC{i + 1}", 0.0) for i in range(pc_count)], dtype=np.float64)
    orig_signed = pc_vec @ pca_components
    orig_abs = np.abs(orig_signed)
    idx = np.argsort(orig_abs)[::-1]
    rows = [
        {
            "original_feature": original_names[j],
            "signed_attr": float(orig_signed[j]),
            "abs_attr": float(orig_abs[j]),
        }
        for j in idx
    ]
    extra = {
        "body_color_signed_attr": float(feature_to_signed.get("body_color", 0.0)),
        "spread_signed_attr": float(feature_to_signed.get("spread", 0.0)),
    }
    return {
        "pc_signed_vector": pc_vec.tolist(),
        "original_feature_rows": rows,
        "non_pc_inputs": extra,
    }


def clamp_index(idx: int, upper_exclusive: int) -> int:
    if upper_exclusive <= 0:
        return 0
    return max(0, min(int(idx), upper_exclusive - 1))


def print_top_rows(title: str, rows: List[Dict[str, Any]], n: int = 8, key_name: str = "feature") -> None:
    print(title)
    print("-" * 88)
    print(f"{'rank':>4}  {key_name:<36}  {'signed_attr':>14}  {'abs_attr':>14}")
    for i, row in enumerate(rows[:n], start=1):
        print(
            f"{i:>4}  {str(row[key_name]):<36}  "
            f"{float(row['signed_attr']):>+14.6f}  {float(row['abs_attr']):>14.6f}"
        )
    print()


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    dev = resolve_device(args.device)
    x_test, _y_test, pairs, meta_test = load_sequence_bundle(split="test")

    hedge_path = os.path.join("data", "pickle", "hedge_ratios.pkl")
    hedge_ratios = None
    if os.path.isfile(hedge_path):
        with open(hedge_path, "rb") as f:
            hedge_ratios = pickle.load(f)
    m_np, tickers, _ticker_to_idx = build_pair_ticker_mapping(pairs, hedge_ratios=hedge_ratios)

    ckpt_path = default_checkpoint_path(args.checkpoint)
    model = MPHDRLTrader(x_test.shape[2], len(pairs), len(tickers), m_np, device=str(dev))
    model.load_checkpoint(ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    chosen_date = choose_date(meta_test, args.date, args.state_ordinal)
    selected = build_state_for_date(x_test, pairs, meta_test, chosen_date)
    pair_idx = args.pair_index
    if pair_idx < 0:
        active = np.where(selected.pair_mask > 0.5)[0]
        pair_idx = int(active[0]) if len(active) else 0
    pair_idx = clamp_index(pair_idx, len(pairs))
    action_id = clamp_index(args.action_id, 3)
    sl_idx = clamp_index(args.sl_idx, 5)
    ticker_idx = clamp_index(args.ticker_index, len(tickers))

    input_feature_names = [f"PC{i + 1}" for i in range(8)] + ["body_color", "spread"]
    if len(input_feature_names) != x_test.shape[2]:
        # Keep robust if dimensionality changes in future preprocessing.
        input_feature_names = [f"f{i}" for i in range(x_test.shape[2])]

    pca_components, original_feature_names = load_pca_mapping(args.pca_path, args.preprocessed_path)

    windows_t = torch.as_tensor(selected.windows, dtype=torch.float32, device=dev).unsqueeze(0)
    windows_t.requires_grad_(True)
    baseline_t = torch.zeros_like(windows_t)
    pair_mask_t = torch.as_tensor(selected.pair_mask, dtype=torch.float32, device=dev).unsqueeze(0)

    actor_head = MPHDRLActorHead(model, pair_mask_t, pair_idx=pair_idx, action_id=action_id).to(dev).eval()
    stop_head = MPHDRLStopLossHead(model, pair_mask_t, pair_idx=pair_idx, sl_idx=sl_idx).to(dev).eval()
    port_head = MPHDRLPortfolioHead(model, pair_mask_t, ticker_idx=ticker_idx).to(dev).eval()

    actor_res = run_ig(actor_head, windows_t, baseline_t, n_steps=args.ig_steps)
    stop_res = run_ig(stop_head, windows_t, baseline_t, n_steps=args.ig_steps)
    port_res = run_ig(port_head, windows_t, baseline_t, n_steps=args.ig_steps)

    actor_agg = aggregate_feature_attribution(
        actor_res["attributions"], selected.pair_mask, pair_idx, input_feature_names
    )
    stop_agg = aggregate_feature_attribution(
        stop_res["attributions"], selected.pair_mask, pair_idx, input_feature_names
    )
    port_agg = aggregate_feature_attribution(
        port_res["attributions"], selected.pair_mask, pair_idx, input_feature_names
    )

    actor_orig = map_pc_to_original(actor_agg["selected_pair_feature_rows"], pca_components, original_feature_names)
    stop_orig = map_pc_to_original(stop_agg["selected_pair_feature_rows"], pca_components, original_feature_names)
    port_orig = map_pc_to_original(port_agg["global_feature_rows"], pca_components, original_feature_names)

    selected_pair = pairs[pair_idx]
    selected_pair_key = f"{selected_pair[0]}|{selected_pair[1]}"
    selected_ticker = tickers[ticker_idx]

    print("=" * 88)
    print("MPHDRL XAI SUMMARY (INFERENCE-ONLY, FROZEN CHECKPOINT)")
    print("=" * 88)
    print(f"checkpoint:     {os.path.abspath(ckpt_path)}")
    print(f"device:         {dev}")
    print(f"date:           {selected.date.date()}")
    print(f"selected pair:  idx={pair_idx}  pair={selected_pair_key}")
    print(f"action id:      {action_id}")
    print(f"stop-loss idx:  {sl_idx}")
    print(f"ticker:         idx={ticker_idx}  ticker={selected_ticker}")
    print(f"active pairs:   {int((selected.pair_mask > 0.5).sum())}/{len(selected.pair_mask)}")
    print(f"input order:    {input_feature_names}")
    print()
    print(f"IG delta actor:     {float(actor_res['delta'].cpu().item()):+.6e}")
    print(f"IG delta stop-loss: {float(stop_res['delta'].cpu().item()):+.6e}")
    print(f"IG delta portfolio: {float(port_res['delta'].cpu().item()):+.6e}")
    print()

    print_top_rows(
        "Actor head (selected pair) - top input features",
        actor_agg["selected_pair_feature_rows"],
        n=8,
        key_name="feature",
    )
    print_top_rows(
        "Stop-loss head (selected pair) - top input features",
        stop_agg["selected_pair_feature_rows"],
        n=8,
        key_name="feature",
    )
    print_top_rows(
        "Portfolio head (global over active pairs) - top input features",
        port_agg["global_feature_rows"],
        n=8,
        key_name="feature",
    )

    print_top_rows(
        "Actor head mapped to original continuous spread features",
        actor_orig["original_feature_rows"],
        n=10,
        key_name="original_feature",
    )
    print_top_rows(
        "Stop-loss head mapped to original continuous spread features",
        stop_orig["original_feature_rows"],
        n=10,
        key_name="original_feature",
    )
    print_top_rows(
        "Portfolio head mapped to original continuous spread features",
        port_orig["original_feature_rows"],
        n=10,
        key_name="original_feature",
    )

    print("Non-PC inputs (signed attribution)")
    print("-" * 88)
    print(
        f"actor: body_color={actor_orig['non_pc_inputs']['body_color_signed_attr']:+.6f}, "
        f"spread={actor_orig['non_pc_inputs']['spread_signed_attr']:+.6f}"
    )
    print(
        f"stop:  body_color={stop_orig['non_pc_inputs']['body_color_signed_attr']:+.6f}, "
        f"spread={stop_orig['non_pc_inputs']['spread_signed_attr']:+.6f}"
    )
    print(
        f"port:  body_color={port_orig['non_pc_inputs']['body_color_signed_attr']:+.6f}, "
        f"spread={port_orig['non_pc_inputs']['spread_signed_attr']:+.6f}"
    )
    print()

    summary = {
        "meta": {
            "checkpoint_path": os.path.abspath(ckpt_path),
            "device": str(dev),
            "date": str(selected.date.date()),
            "pair_idx": pair_idx,
            "pair": selected_pair_key,
            "action_id": action_id,
            "sl_idx": sl_idx,
            "ticker_idx": ticker_idx,
            "ticker": selected_ticker,
            "input_feature_order": input_feature_names,
            "active_pairs": int((selected.pair_mask > 0.5).sum()),
            "n_pairs": len(selected.pair_mask),
            "ig_steps": int(args.ig_steps),
        },
        "actor_head": {
            "ig_delta": float(actor_res["delta"].cpu().item()),
            "selected_pair_input_features": actor_agg["selected_pair_feature_rows"],
            "global_input_features": actor_agg["global_feature_rows"],
            "mapped_original_features": actor_orig,
        },
        "stop_loss_head": {
            "ig_delta": float(stop_res["delta"].cpu().item()),
            "selected_pair_input_features": stop_agg["selected_pair_feature_rows"],
            "global_input_features": stop_agg["global_feature_rows"],
            "mapped_original_features": stop_orig,
        },
        "portfolio_head": {
            "ig_delta": float(port_res["delta"].cpu().item()),
            "selected_pair_input_features": port_agg["selected_pair_feature_rows"],
            "global_input_features": port_agg["global_feature_rows"],
            "mapped_original_features": port_orig,
        },
        "inference_only_guardrails": {
            "model_eval_mode": bool(not model.training),
            "all_params_require_grad_false": bool(all(not p.requires_grad for p in model.parameters())),
            "checkpoint_mutation": "none",
            "training_loop_called": False,
        },
    }

    out_json = os.path.join(args.results_dir, "xai_mphdrl_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Flat CSV for quick scan.
    flat_rows: List[Dict[str, Any]] = []
    for head_name, block in [
        ("actor", summary["actor_head"]),
        ("stop_loss", summary["stop_loss_head"]),
        ("portfolio", summary["portfolio_head"]),
    ]:
        for i, row in enumerate(block["mapped_original_features"]["original_feature_rows"], start=1):
            flat_rows.append(
                {
                    "head": head_name,
                    "rank": i,
                    "original_feature": row["original_feature"],
                    "signed_attr": row["signed_attr"],
                    "abs_attr": row["abs_attr"],
                    "ig_delta": block["ig_delta"],
                }
            )
    out_csv = os.path.join(args.results_dir, "xai_mphdrl_original_feature_attribution.csv")
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print(f"Saved summary JSON: {os.path.abspath(out_json)}")
    print(f"Saved mapping CSV:  {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
