"""
training.py -- MPHDRL training harness for RL pairs trading.

Run with no arguments:
    python training.py

Hyperparameters start from MPHDRL.py defaults each run, then optionally merge the best row from
artifacts/mphdrl_tuning/trials.csv if tuning has been run; otherwise defaults are used unchanged.
Single continuous RL schedule (no classification phase).

Optional environment overrides (for automation / mphdrl_tuning.py):
    MPHDRL_TRAIN_EPOCHS, MPHDRL_DEVICE, MPHDRL_SEED, MPHDRL_SAVE_EVERY, MPHDRL_TUNING_DIR,
    MPHDRL_HP_PATCH (path to JSON hyperparameter patch), MPHDRL_NO_AMP, MPHDRL_NO_COMPILE,
    MPHDRL_DIAG_NO_RISK_TAX.

SRRL trainer code is retained below but commented out of the registry.

On CUDA (default): cudnn benchmark, TF32, mixed precision, torch.compile unless disabled.
"""

import contextlib
import copy
import os
import sys
import time
import random
import pickle
import json
import shutil
from types import SimpleNamespace
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# from MPHDRL import (
#     HPARAMS,
#     MPHDRL_MODEL_DIR,
#     MPHDRLTrader,
#     TradingEnvironment,
#     PrioritizedReplayBuffer,
#     build_pair_ticker_mapping,
#     check_data_readiness,
# )
# from benchmark import BENCHMARK_MODEL_DIR, BenchmarkDDPG
from MPHDRL import (
    HPARAMS,
    MPHDRL_MODEL_DIR,
    MPHDRLTrader,
    TradingEnvironment,
    PrioritizedReplayBuffer,
    build_pair_ticker_mapping,
    check_data_readiness,
)
from benchmark import BENCHMARK_MODEL_DIR, BenchmarkDDPG

# Frozen copy of MPHDRL.py defaults for reset before optional tuning merge.
_DEFAULT_MPHDRL_HPARAMS = copy.deepcopy(dict(HPARAMS))
# SRRL trainer disabled by default; keep imports so SRRL class bodies parse if restored.
from SRRL import SRRL_HPARAMS, SRRL_MODEL_DIR, SRRLTrader
from backtest_core import (
    load_sequence_bundle,
    load_price_matrix,
    get_all_weights_by_date,
    get_mphdrl_weights_by_env,
    run_strategy_backtest,
    summarize_backtest_dataframe,
)

# Defaults when running `python training.py` with no CLI (override via MPHDRL_* env vars).
DEFAULT_TRAINING_EPOCHS = 100
DEFAULT_SAVE_EVERY = 10
DEFAULT_DEVICE = "auto"
DEFAULT_MPHDRL_TUNING_DIR = os.path.join("artifacts", "mphdrl_tuning")


def reset_mphdrl_hparams_to_defaults() -> None:
    """Reset MPHDRL.HPARAMS to built-in defaults (before loading tuning or patch)."""
    HPARAMS.clear()
    HPARAMS.update(copy.deepcopy(_DEFAULT_MPHDRL_HPARAMS))


def set_global_seed(seed: int | None):
    """Best-effort deterministic setup for reproducible tuning runs."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_best_mphdrl_params_from_tuning(tuning_dir: str) -> bool:
    """
    Merge the best status=ok row (max utility) from tuning trials.csv into HPARAMS.
    Returns True if hyperparameters were loaded from disk.
    """
    trials_csv = os.path.join(tuning_dir, "trials.csv")
    if not os.path.isfile(trials_csv):
        print(f"MPHDRL tuning: no {trials_csv} — using defaults from MPHDRL.py")
        return False
    try:
        df = pd.read_csv(trials_csv)
    except Exception as e:
        print(f"MPHDRL tuning: could not read {trials_csv}: {e}")
        return False
    if df.empty or "status" not in df.columns or "params" not in df.columns:
        print(f"MPHDRL tuning: unexpected CSV shape in {trials_csv}")
        return False
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("MPHDRL tuning: no status=ok rows — using defaults from MPHDRL.py")
        return False
    if "utility" in ok.columns:
        ok["utility"] = pd.to_numeric(ok["utility"], errors="coerce")
        ok = ok.dropna(subset=["utility"])
    if ok.empty:
        print("MPHDRL tuning: no usable utility values — using defaults from MPHDRL.py")
        return False
    ok = ok.sort_values("utility", ascending=False)
    row = ok.iloc[0]
    try:
        raw_p = row["params"]
        params = json.loads(raw_p) if isinstance(raw_p, str) else dict(raw_p)
    except Exception as e:
        print(f"MPHDRL tuning: could not parse params JSON: {e}")
        return False
    merge_mphdrl_params_dict(params)
    tid = row.get("trial_id", "?")
    util = float(row["utility"]) if "utility" in row else float("nan")
    print(f"MPHDRL tuning: applied best trial_id={tid} utility={util:.6f} from {trials_csv}")
    return True


def merge_mphdrl_params_dict(params: dict) -> None:
    """Merge tuning-style hyperparameter keys into MPHDRL.HPARAMS in place."""
    int_keys = {
        "n_step",
        "batch_size",
        "var_window",
        "delay_c",
        "delay_b",
        "H_srl",
        "N_w",
        "TW",
        "stop_loss_embed_dim",
    }
    for key, val in params.items():
        if key not in HPARAMS:
            continue
        cur = HPARAMS[key]
        if isinstance(cur, list):
            if isinstance(val, list):
                HPARAMS[key] = val
            continue
        if key in int_keys:
            HPARAMS[key] = int(round(float(val)))
        elif isinstance(cur, (int, float)):
            HPARAMS[key] = float(val)


def merge_mphdrl_hp_patch_from_env() -> None:
    """If MPHDRL_HP_PATCH points to a JSON file, merge into HPARAMS (used by mphdrl_tuning trials)."""
    path = os.environ.get("MPHDRL_HP_PATCH", "").strip()
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            patch = json.load(f)
    except Exception as e:
        print(f"MPHDRL_HP_PATCH: failed to load {path}: {e}")
        return
    if isinstance(patch, dict):
        merge_mphdrl_params_dict(patch)
        print(f"MPHDRL: merged hyperparameter patch from {path}")


def load_training_config() -> SimpleNamespace:
    """No CLI: defaults below; mphdrl_tuning sets MPHDRL_* environment variables."""
    raw_epochs = os.environ.get("MPHDRL_TRAIN_EPOCHS", "").strip()
    epochs = int(raw_epochs) if raw_epochs else DEFAULT_TRAINING_EPOCHS

    raw_save = os.environ.get("MPHDRL_SAVE_EVERY", "").strip()
    save_every = int(raw_save) if raw_save else DEFAULT_SAVE_EVERY

    device = os.environ.get("MPHDRL_DEVICE", "").strip() or DEFAULT_DEVICE

    tuning_dir = os.environ.get("MPHDRL_TUNING_DIR", "").strip() or DEFAULT_MPHDRL_TUNING_DIR

    seed_raw = os.environ.get("MPHDRL_SEED", "").strip()
    seed = int(seed_raw) if seed_raw else None

    no_amp = os.environ.get("MPHDRL_NO_AMP", "").strip().lower() in ("1", "true", "yes")
    no_compile = os.environ.get("MPHDRL_NO_COMPILE", "").strip().lower() in ("1", "true", "yes")
    env_diagnostic_no_risk_tax = os.environ.get("MPHDRL_DIAG_NO_RISK_TAX", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    return SimpleNamespace(
        agent="MPHDRL",
        epochs=epochs,
        save_every=max(1, save_every),
        device=device,
        tuning_dir=tuning_dir,
        seed=seed,
        no_amp=no_amp,
        no_compile=no_compile,
        env_diagnostic_no_risk_tax=env_diagnostic_no_risk_tax,
        max_grad_norm=1.0,
        resolved_device=None,
    )


def print_mphdrl_hparams():
    keys = [
        "lr",
        "tau",
        "batch_size",
        "discount_gamma",
        "n_step",
        "sigma_explore",
        "sigma_smooth",
        "gamma",
        "risk_lambda",
        "var_window",
        "terminal_utility_weight",
        "per_alpha",
        "per_beta_start",
        "per_xi",
        "zeta",
        "delay_c",
        "delay_b",
    ]
    print("MPHDRL hyperparameters:")
    for k in keys:
        print(f"  {k}: {HPARAMS[k]}")


def resolve_training_device(preference: str) -> torch.device:
    """
    Map CLI device string to a torch.device.

    - auto: prefer CUDA, then Apple MPS, else CPU.
    - cuda / mps / cpu: fixed choice; cuda/mps fail fast if unavailable.
    """
    name = (preference or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception:
                pass
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA was requested but is not available. Install a CUDA-enabled PyTorch build\n"
                "from https://pytorch.org/get-started/locally/ (Windows: match your driver; "
                "use `nvidia-smi` to check the driver CUDA capability)."
            )
            sys.exit(1)
        return torch.device("cuda")

    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            print("MPS was requested but is not available on this system.")
            sys.exit(1)
        return torch.device("mps")

    if name == "cpu":
        return torch.device("cpu")

    print(f"Unknown device '{preference}'. Use auto, cuda, mps, or cpu.")
    sys.exit(1)


def configure_accelerator(device: torch.device) -> None:
    """GPU-oriented runtime tuning (CUDA TF32 / cudnn; MPS matmul precision)."""
    if device.type == "cuda":
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def maybe_compile_forward_step(model, device: torch.device, args, sample_windows, tag: str) -> bool:
    """
    Try torch.compile on model.forward_step and eagerly run a smoke call.
    If backend deps (e.g. Triton) are missing, gracefully fall back to eager mode.
    """
    if device.type != "cuda" or getattr(args, "no_compile", False) or not hasattr(torch, "compile"):
        return False
    eager_forward = model.forward_step
    try:
        compiled = torch.compile(eager_forward, mode="default", fullgraph=False)
        model.forward_step = compiled
        with torch.no_grad():
            _ = model.forward_step(sample_windows, explore=True)
        return True
    except Exception as e:
        model.forward_step = eager_forward
        print(f"[training] torch.compile disabled for {tag}: {e}")
        return False


# ============================================================================
# Agent trainer registry
# ============================================================================

AGENT_REGISTRY = {}


def register_agent(name):
    def decorator(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator


class BaseTrainer:
    """Interface that every agent trainer must implement."""

    def __init__(self, args, data):
        self.args = args
        self.data = data

    def train(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


# ============================================================================
# MPHDRL Trainer
# ============================================================================

@register_agent("MPHDRL")
class MPHDRLTrainer(BaseTrainer):

    def __init__(self, args, data):
        super().__init__(args, data)
        self.device = args.resolved_device
        self.epochs = args.epochs
        self.save_every = args.save_every
        self.model_dir = MPHDRL_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        F_dim = data["X_train"].shape[2]
        n_pairs = len(data["pairs"])
        n_tickers = len(data["tickers"])
        self.n_pairs = n_pairs

        self.model = MPHDRLTrader(F_dim, n_pairs, n_tickers, data["M"], device=str(self.device))

        # --- Speed (CUDA): AMP + optional torch.compile; same algorithm / epoch count ---
        self.use_amp = self.device.type == "cuda" and not getattr(args, "no_amp", False)
        if (
            self.use_amp
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        ):
            self._amp_dtype = torch.bfloat16
            self._scaler = None  # bf16 rarely needs GradScaler on recent CUDA
        elif self.use_amp:
            self._amp_dtype = torch.float16
            self._scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self._amp_dtype = torch.float32
            self._scaler = None

        sample_windows = data["X_train"][0]
        self._forward_compiled = maybe_compile_forward_step(
            self.model, self.device, args, sample_windows, tag="MPHDRL"
        )

        _diag = getattr(args, "env_diagnostic_no_risk_tax", False)
        self.env = TradingEnvironment(
            pairs=data["pairs"],
            tickers=data["tickers"],
            ticker_to_idx=data["ticker_to_idx"],
            trading_raw_path=os.path.join("data", "trading", "raw.csv"),
            sequence_meta=data["sequence_meta"],
            X_train=data["X_train"],
            y_train=data["y_train"],
            zeta=HPARAMS["zeta"],
            gamma=HPARAMS["gamma"],
            risk_lambda=HPARAMS["risk_lambda"],
            var_window=HPARAMS["var_window"],
            terminal_utility_weight=0.0 if _diag else HPARAMS["terminal_utility_weight"],
            use_stop_loss=True,
            diagnostic_no_risk_tax=_diag,
        )
        self.n_step = HPARAMS["n_step"]

        buffer_cap = max(10000, self.env.num_steps * 2)
        self.replay = PrioritizedReplayBuffer(
            capacity=buffer_cap,
            alpha=HPARAMS["per_alpha"],
            beta_start=HPARAMS["per_beta_start"],
            xi=HPARAMS["per_xi"],
        )

        self._build_optimizers()

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def _h2d(self, arr, dtype=torch.float32):
        """Numpy (or array-like) → model device; pin_memory + non_blocking on CUDA only."""
        dev = self.model.device
        if dev.type == "cuda":
            return torch.as_tensor(np.asarray(arr), dtype=dtype).pin_memory().to(
                dev, non_blocking=True
            )
        return torch.tensor(np.asarray(arr), dtype=dtype, device=dev)

    def _build_optimizers(self):
        lr = HPARAMS["lr"]
        self.opt_critic1 = torch.optim.Adam(
            list(self.model.srl_critic1.parameters()) + list(self.model.critic1.parameters()),
            lr=lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.99 * 0,
        )
        self.opt_critic2 = torch.optim.Adam(
            list(self.model.srl_critic2.parameters()) + list(self.model.critic2.parameters()),
            lr=lr, betas=(0.9, 0.999), eps=1e-7,
        )
        self.opt_actor = torch.optim.Adam(
            list(self.model.srl_actor.parameters())
            + list(self.model.actor.parameters())
            + list(self.model.regression_head.parameters())
            + list(self.model.portfolio.parameters()),
            lr=lr, betas=(0.9, 0.999), eps=1e-7,
        )
        self.opt_stop = torch.optim.Adam(
            list(self.model.srl_stop.parameters()) + list(self.model.stop_loss.parameters()),
            lr=lr, betas=(0.9, 0.999), eps=1e-7,
        )

    def _collect_episode(self):
        state_info = self.env.reset()
        if state_info is None:
            return 0
        transitions_collected = 0
        n_step = self.n_step
        discount = HPARAMS["discount_gamma"]

        nstep_buf: list[dict] = []

        def _flush_nstep(extra_bonus=0.0):
            """Emit n-step transition from head of buffer."""
            nonlocal transitions_collected
            if not nstep_buf:
                return
            G = 0.0
            for k in range(len(nstep_buf) - 1, -1, -1):
                G = nstep_buf[k]["reward"] + discount * G
            G += float(extra_bonus)
            head = nstep_buf[0]
            tail = nstep_buf[-1]
            transition = {
                "state_windows": head["state_windows"],
                "actions": head["actions"],
                "stop_loss": head["stop_loss"],
                "weights": head["weights"],
                "reward": np.float32(G),
                "y_spread": head["y_spread"],
                "next_state_windows": tail["next_state_windows"],
                "n_step": len(nstep_buf),
                "done": np.float32(1.0 if tail["done"] else 0.0),
            }
            self.replay.add(transition)
            transitions_collected += 1

        while True:
            windows, mask, y_spread = state_info
            with torch.no_grad():
                with self._autocast():
                    step_out = self.model.forward_step(windows, explore=True, pair_mask=mask)

            w_np = step_out["weights"].detach().cpu().numpy().flatten()
            actions_np = step_out["actions"].detach().cpu().numpy().flatten()
            sl_np = step_out["sl_actions"].detach().cpu().numpy().flatten()

            spread_values = np.where(mask, y_spread, np.nan).astype(np.float64)
            next_info, reward, done = self.env.step(
                w_np, sl_actions=sl_np, spread_values=spread_values,
            )

            if next_info is not None:
                next_windows, next_mask, _next_y = next_info
            else:
                next_windows = np.zeros_like(windows)

            nstep_buf.append({
                "state_windows": windows.astype(np.float32),
                "actions": actions_np.astype(np.int64),
                "stop_loss": sl_np.astype(np.int64),
                "weights": w_np.astype(np.float32),
                "reward": float(reward),
                "y_spread": y_spread.astype(np.float32),
                "next_state_windows": next_windows.astype(np.float32),
                "done": bool(done),
            })

            if len(nstep_buf) >= n_step:
                _flush_nstep()
                nstep_buf.pop(0)

            if done:
                terminal_bonus = float(self.env.episode_utility_bonus())
                while nstep_buf:
                    _flush_nstep(extra_bonus=terminal_bonus if len(nstep_buf) == 1 else 0.0)
                    nstep_buf.pop(0)
                break
            state_info = next_info

        return transitions_collected

    def _update_step(self, epoch, step_in_epoch):
        B = HPARAMS["batch_size"]
        discount_gamma = HPARAMS["discount_gamma"]
        dev = self.model.device

        if len(self.replay) < B:
            return {}

        batch, indices, is_weights = self.replay.sample(B)
        nb = dev.type == "cuda"
        is_weights = is_weights.to(dev, non_blocking=nb)

        sw = self._h2d(np.stack([t["state_windows"] for t in batch]))
        acts = self._h2d(np.stack([t["actions"] for t in batch]), dtype=torch.int64)
        sls = self._h2d(np.stack([t["stop_loss"] for t in batch]), dtype=torch.int64)
        rews = self._h2d(np.array([t["reward"] for t in batch]))
        nsw = self._h2d(np.stack([t["next_state_windows"] for t in batch]))
        dones = self._h2d(np.array([t.get("done", 0.0) for t in batch]))
        ns = self._h2d(np.array([t.get("n_step", 1) for t in batch]))

        acts_onehot = F.one_hot(acts, num_classes=3).float()

        # -- Target actions (smoothed) with n-step discount --
        with torch.no_grad():
            with self._autocast():
                h_next_actor = self.model.encode_all_pairs(nsw, self.model.srl_actor_target)
                Bp, P, H = h_next_actor.shape
                next_logits = self.model.actor_target(h_next_actor.reshape(Bp * P, H))
                noise = torch.randn_like(next_logits) * HPARAMS["sigma_smooth"]
                noise = noise.clamp(-HPARAMS["clip_smooth"], HPARAMS["clip_smooth"])
                next_probs = F.softmax(next_logits + noise, dim=-1)
                next_acts = next_probs.argmax(dim=-1).reshape(Bp, P)
                next_acts_oh = F.one_hot(next_acts, num_classes=3).float()

                h_nc1 = self.model.encode_all_pairs(nsw, self.model.srl_critic1_target)
                h_nc2 = self.model.encode_all_pairs(nsw, self.model.srl_critic2_target)
                q1_next = self.model.critic1_target(h_nc1.mean(dim=1), next_acts_oh.reshape(Bp, -1))
                q2_next = self.model.critic2_target(h_nc2.mean(dim=1), next_acts_oh.reshape(Bp, -1))
                gamma_n = discount_gamma ** ns
                y_critic = rews + (1.0 - dones) * gamma_n * torch.min(q1_next, q2_next)

        # -- Critic 1 update --
        with self._autocast():
            h_c1 = self.model.encode_all_pairs(sw, self.model.srl_critic1)
            q1 = self.model.critic1(h_c1.mean(dim=1), acts_onehot.reshape(B, -1))
            td1 = y_critic - q1
            loss_c1 = (is_weights * td1.pow(2)).mean()
        self.opt_critic1.zero_grad()
        if self._scaler is not None:
            self._scaler.scale(loss_c1).backward()
            self._scaler.step(self.opt_critic1)
        else:
            loss_c1.backward()
            self.opt_critic1.step()

        # -- Critic 2 update --
        with self._autocast():
            h_c2 = self.model.encode_all_pairs(sw, self.model.srl_critic2)
            q2 = self.model.critic2(h_c2.mean(dim=1), acts_onehot.reshape(B, -1))
            td2 = y_critic - q2
            loss_c2 = (is_weights * td2.pow(2)).mean()
        self.opt_critic2.zero_grad()
        if self._scaler is not None:
            self._scaler.scale(loss_c2).backward()
            self._scaler.step(self.opt_critic2)
        else:
            loss_c2.backward()
            self.opt_critic2.step()

        min_td = torch.min(td1.abs(), td2.abs()).detach().cpu().numpy()
        self.replay.update_priorities(indices, min_td)

        # -- Stop-loss update (DDQN) --
        with torch.no_grad():
            with self._autocast():
                h_ns_stop = self.model.encode_all_pairs(nsw, self.model.srl_stop_target)
                h_ns_stop_online = self.model.encode_all_pairs(nsw, self.model.srl_stop)
                Bp2, P2, Hs = h_ns_stop.shape
                q_next_online = self.model.stop_loss(h_ns_stop_online.reshape(Bp2 * P2, Hs))
                best_sl_next = q_next_online.argmax(dim=-1)
                q_next_target = self.model.stop_loss_target(h_ns_stop.reshape(Bp2 * P2, Hs))
                q_sl_target_vals = q_next_target.gather(1, best_sl_next.unsqueeze(1)).squeeze(1)
                q_sl_target_vals = q_sl_target_vals.reshape(Bp2, P2)
                rews_expanded = rews.unsqueeze(1).expand_as(q_sl_target_vals)
                dones_expanded = dones.unsqueeze(1).expand_as(q_sl_target_vals)
                gamma_n_expanded = gamma_n.unsqueeze(1).expand_as(q_sl_target_vals)
                y_sl = rews_expanded + (1.0 - dones_expanded) * gamma_n_expanded * q_sl_target_vals

        with self._autocast():
            h_s_stop = self.model.encode_all_pairs(sw, self.model.srl_stop)
            q_sl = self.model.stop_loss(h_s_stop.reshape(B * self.n_pairs, -1))
            sls_flat = sls.reshape(B * self.n_pairs)
            q_sl_selected = q_sl.gather(1, sls_flat.unsqueeze(1)).squeeze(1).reshape(B, self.n_pairs)
            loss_sl = F.mse_loss(q_sl_selected, y_sl)
        self.opt_stop.zero_grad()
        if self._scaler is not None:
            self._scaler.scale(loss_sl).backward()
            self._scaler.step(self.opt_stop)
        else:
            loss_sl.backward()
            self.opt_stop.step()

        losses = {"critic1": loss_c1.item(), "critic2": loss_c2.item(), "stop_loss": loss_sl.item()}

        # -- Delayed actor + portfolio + regression update --
        delay = MPHDRLTrader.dynamic_delay(epoch)
        if step_in_epoch % delay == 0:
            with self._autocast():
                h_a = self.model.encode_all_pairs(sw, self.model.srl_actor)
                Ba, Pa, Ha = h_a.shape
                logits_a = self.model.actor(h_a.reshape(Ba * Pa, Ha))
                probs_a = F.softmax(logits_a, dim=-1).reshape(Ba, Pa, 3)
                actions_a = probs_a.argmax(dim=-1)

                # Straight-through hard actions in critic input (one-hot)
                oh_a = F.one_hot(actions_a, num_classes=3).float()
                soft_flat = probs_a.reshape(Ba, Pa * 3)
                hard_flat = oh_a.reshape(Ba, Pa * 3)
                st_actions = (hard_flat - soft_flat).detach() + soft_flat

                h_c1_for_actor = self.model.encode_all_pairs(sw, self.model.srl_critic1)
                q_for_actor = self.model.critic1(h_c1_for_actor.mean(dim=1), st_actions)
                loss_actor_pg = -q_for_actor.mean()

                # Regression loss (§1.4, §8.5): per-pair label from same row as state window.
                spread_pred = self.model.regression_head(h_a.reshape(Ba * Pa, Ha)).reshape(Ba, Pa)
                y_batch = self._h2d(
                    np.stack(
                        [t.get("y_spread", np.zeros(self.n_pairs, dtype=np.float32)) for t in batch]
                    )
                )
                loss_reg = F.mse_loss(spread_pred, y_batch)

                loss_actor_total = loss_actor_pg + loss_reg

            self.opt_actor.zero_grad()
            if self._scaler is not None:
                self._scaler.scale(loss_actor_total).backward()
                self._scaler.step(self.opt_actor)
            else:
                loss_actor_total.backward()
                self.opt_actor.step()

            self.model.soft_update()

            losses["actor_pg"] = loss_actor_pg.item()
            losses["regression"] = loss_reg.item()

        if self._scaler is not None:
            self._scaler.update()

        return losses

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"Training MPHDRL agent for {self.epochs} epochs")
        speed = []
        if self.device.type == "cuda":
            speed.append(f"AMP={self._amp_dtype}" if self.use_amp else "AMP=off")
            speed.append("compile=on" if getattr(self, "_forward_compiled", False) else "compile=off")
        speed_s = (" | " + ", ".join(speed)) if speed else ""
        print(f"Device: {self.device} ({self.device.type}) | Pairs: {self.n_pairs}{speed_s}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            n_trans = self._collect_episode()
            self.replay.anneal_beta(epoch / self.epochs)

            epoch_losses = {}
            n_updates = min(n_trans, self.env.num_steps)
            for step in range(n_updates):
                step_losses = self._update_step(epoch, step)
                for k, v in step_losses.items():
                    epoch_losses.setdefault(k, []).append(v)

            elapsed = time.time() - t0
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"Epoch {epoch:4d}/{self.epochs} | {elapsed:5.1f}s | trans={n_trans} | {loss_str}")

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self.save(tag=f"epoch_{epoch}")

        self.save(tag="final")
        print(f"\nTraining complete. Final model saved to {self.model_dir}")

    def save(self, tag="checkpoint"):
        path = os.path.join(self.model_dir, f"{tag}.pt")
        self.model.save_checkpoint(path)


# ============================================================================
# Uniform replay (for Benchmark agent)
# ============================================================================

class UniformReplayBuffer:
    """Ring buffer with uniform random sampling (no prioritization)."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.buffer: list = []
        self.pos = 0

    def add(self, transition: dict) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        n = len(self.buffer)
        if n == 0:
            raise ValueError("UniformReplayBuffer is empty.")
        replace = batch_size > n
        idx = np.random.choice(n, size=batch_size, replace=replace)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# Benchmark (DDPG-style) trainer
# ============================================================================

# @register_agent("Benchmark")
class BenchmarkTrainer(BaseTrainer):
    """Plain actor–critic: Gaussian E, same env reward as MPHDRL, uniform replay."""

    def __init__(self, args, data):
        super().__init__(args, data)
        self.device = args.resolved_device
        self.epochs = args.epochs
        self.save_every = args.save_every
        self.model_dir = BENCHMARK_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        F_dim = data["X_train"].shape[2]
        n_pairs = len(data["pairs"])
        n_tickers = len(data["tickers"])
        self.n_pairs = n_pairs

        self.model = BenchmarkDDPG(
            F_dim, n_pairs, n_tickers, data["M"], device=str(self.device)
        )
        self.use_amp = self.device.type == "cuda" and not getattr(args, "no_amp", False)
        if (
            self.use_amp
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        ):
            self._amp_dtype = torch.bfloat16
            self._scaler = None
        elif self.use_amp:
            self._amp_dtype = torch.float16
            self._scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self._amp_dtype = torch.float32
            self._scaler = None

        sample_windows = data["X_train"][0]
        self._forward_compiled = maybe_compile_forward_step(
            self.model, self.device, args, sample_windows, tag="Benchmark"
        )

        _diag = getattr(args, "env_diagnostic_no_risk_tax", False)
        self.env = TradingEnvironment(
            pairs=data["pairs"],
            tickers=data["tickers"],
            ticker_to_idx=data["ticker_to_idx"],
            trading_raw_path=os.path.join("data", "trading", "raw.csv"),
            sequence_meta=data["sequence_meta"],
            X_train=data["X_train"],
            y_train=data["y_train"],
            zeta=HPARAMS["zeta"],
            gamma=HPARAMS["gamma"],
            risk_lambda=HPARAMS["risk_lambda"],
            var_window=HPARAMS["var_window"],
            terminal_utility_weight=0.0 if _diag else HPARAMS["terminal_utility_weight"],
            use_stop_loss=False,
            diagnostic_no_risk_tax=_diag,
        )
        self.n_step = HPARAMS["n_step"]

        buffer_cap = max(10000, self.env.num_steps * 2)
        self.replay = UniformReplayBuffer(buffer_cap)

        lr = HPARAMS["lr"]
        self.opt_actor = torch.optim.Adam(
            list(self.model.actor_encoder.parameters())
            + list(self.model.actor_mu.parameters())
            + [self.model.log_std],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        self.opt_critic = torch.optim.Adam(
            list(self.model.critic_encoder.parameters()) + list(self.model.critic_head.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        self.max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
        self.nan_batches = 0
        self.skipped_updates = 0

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    @staticmethod
    def _all_finite(*tensors) -> bool:
        for t in tensors:
            if t is None:
                continue
            if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                return False
        return True

    def _h2d(self, arr, dtype=torch.float32):
        dev = self.model.device
        if dev.type == "cuda":
            return torch.as_tensor(np.asarray(arr), dtype=dtype).pin_memory().to(
                dev, non_blocking=True
            )
        return torch.tensor(np.asarray(arr), dtype=dtype, device=dev)

    def _collect_episode(self):
        state_info = self.env.reset()
        if state_info is None:
            return 0
        n = 0
        n_step = self.n_step
        discount = HPARAMS["discount_gamma"]
        nstep_buf: list[dict] = []

        def _flush(extra_bonus=0.0):
            nonlocal n
            if not nstep_buf:
                return
            G = 0.0
            for k in range(len(nstep_buf) - 1, -1, -1):
                G = nstep_buf[k]["reward"] + discount * G
            G += float(extra_bonus)
            head = nstep_buf[0]
            tail = nstep_buf[-1]
            self.replay.add({
                "state_windows": head["state_windows"],
                "E": head["E"],
                "reward": np.float32(G),
                "next_state_windows": tail["next_state_windows"],
                "done": np.float32(1.0 if tail["done"] else 0.0),
                "n_step": len(nstep_buf),
            })
            n += 1

        while True:
            windows, mask, y_spread = state_info
            with torch.no_grad():
                step_out = self.model.forward_step(windows, explore=True, pair_mask=mask)

            w_np = step_out["weights"].detach().cpu().numpy().reshape(-1)
            E_np = step_out["pair_exposures"].detach().cpu().numpy().reshape(self.n_pairs)

            spread_values = np.where(mask, y_spread, np.nan).astype(np.float64)
            next_info, reward, done = self.env.step(
                w_np, sl_actions=None, spread_values=None,
            )
            if next_info is not None:
                next_windows, _, _ = next_info
            else:
                next_windows = np.zeros_like(windows)

            nstep_buf.append({
                "state_windows": windows.astype(np.float32),
                "E": E_np.astype(np.float32),
                "reward": float(reward),
                "next_state_windows": next_windows.astype(np.float32),
                "done": bool(done),
            })

            if len(nstep_buf) >= n_step:
                _flush()
                nstep_buf.pop(0)

            if done:
                terminal_bonus = float(self.env.episode_utility_bonus())
                while nstep_buf:
                    _flush(extra_bonus=terminal_bonus if len(nstep_buf) == 1 else 0.0)
                    nstep_buf.pop(0)
                break
            state_info = next_info
        return n

    def _update_step(self):
        B = HPARAMS["batch_size"]
        gamma = HPARAMS["discount_gamma"]
        if len(self.replay) < B:
            return {}

        batch = self.replay.sample(B)

        sw = self._h2d(np.stack([t["state_windows"] for t in batch]))
        E = self._h2d(np.stack([t["E"] for t in batch]))
        rews = self._h2d(np.array([t["reward"] for t in batch]))
        nsw = self._h2d(np.stack([t["next_state_windows"] for t in batch]))
        dones = self._h2d(np.array([t["done"] for t in batch]))
        ns = self._h2d(np.array([t.get("n_step", 1) for t in batch]))
        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        rews = torch.nan_to_num(rews, nan=0.0, posinf=0.0, neginf=0.0)
        dones = torch.nan_to_num(dones, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)
        ns = torch.nan_to_num(ns, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0)

        if not self._all_finite(sw, E, rews, nsw, dones, ns):
            self.nan_batches += 1
            self.skipped_updates += 1
            self.model.soft_update(tau=1.0)
            return {"skipped_nan": 1.0}

        with torch.no_grad():
            with self._autocast():
                mu_next = self.model.actor_mean_target(nsw)
                q_next = self.model.critic_q(
                    nsw, mu_next, self.model.critic_encoder_t, self.model.critic_head_t
                )
                gamma_n = gamma ** ns
                y = rews + (1.0 - dones) * gamma_n * q_next
                y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                q_next = torch.nan_to_num(q_next, nan=0.0, posinf=0.0, neginf=0.0)

        if not self._all_finite(y):
            self.nan_batches += 1
            self.skipped_updates += 1
            self.model.soft_update(tau=1.0)
            return {"skipped_nan": 1.0}

        with self._autocast():
            q = self.model.critic_q(sw, E, self.model.critic_encoder, self.model.critic_head)
            loss_c = F.mse_loss(q, y)
            q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
            loss_c = torch.nan_to_num(loss_c, nan=0.0, posinf=0.0, neginf=0.0)
        if not self._all_finite(loss_c):
            self.nan_batches += 1
            self.skipped_updates += 1
            self.model.soft_update(tau=1.0)
            return {"skipped_nan": 1.0}
        self.opt_critic.zero_grad()
        if self._scaler is not None:
            self._scaler.scale(loss_c).backward()
            self._scaler.unscale_(self.opt_critic)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.critic_encoder.parameters()) + list(self.model.critic_head.parameters()),
                self.max_grad_norm,
            )
            self._scaler.step(self.opt_critic)
        else:
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.critic_encoder.parameters()) + list(self.model.critic_head.parameters()),
                self.max_grad_norm,
            )
            self.opt_critic.step()

        for p in self.model.critic_encoder.parameters():
            p.requires_grad = False
        for p in self.model.critic_head.parameters():
            p.requires_grad = False

        with self._autocast():
            mu = self.model.actor_mean(sw)
            q_a = self.model.critic_q(sw, mu, self.model.critic_encoder, self.model.critic_head)
            loss_a = -q_a.mean()
            q_a = torch.nan_to_num(q_a, nan=0.0, posinf=0.0, neginf=0.0)
            loss_a = torch.nan_to_num(loss_a, nan=0.0, posinf=0.0, neginf=0.0)
        if not self._all_finite(loss_a):
            self.nan_batches += 1
            self.skipped_updates += 1
            for p in self.model.critic_encoder.parameters():
                p.requires_grad = True
            for p in self.model.critic_head.parameters():
                p.requires_grad = True
            self.model.soft_update(tau=1.0)
            return {"skipped_nan": 1.0}

        self.opt_actor.zero_grad()
        if self._scaler is not None:
            self._scaler.scale(loss_a).backward()
            self._scaler.unscale_(self.opt_actor)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.actor_encoder.parameters())
                + list(self.model.actor_mu.parameters())
                + [self.model.log_std],
                self.max_grad_norm,
            )
            self._scaler.step(self.opt_actor)
        else:
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.actor_encoder.parameters())
                + list(self.model.actor_mu.parameters())
                + [self.model.log_std],
                self.max_grad_norm,
            )
            self.opt_actor.step()
        with torch.no_grad():
            self.model.log_std.data.clamp_(-5.0, 2.0)

        for p in self.model.critic_encoder.parameters():
            p.requires_grad = True
        for p in self.model.critic_head.parameters():
            p.requires_grad = True

        if self._scaler is not None:
            self._scaler.update()

        self.model.soft_update()

        return {"critic": loss_c.item(), "actor": loss_a.item()}

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"Training Benchmark (DDPG-style) agent for {self.epochs} epochs")
        speed = []
        if self.device.type == "cuda":
            speed.append(f"AMP={self._amp_dtype}" if self.use_amp else "AMP=off")
            speed.append("compile=on" if getattr(self, "_forward_compiled", False) else "compile=off")
        speed_s = (" | " + ", ".join(speed)) if speed else ""
        print(f"Device: {self.device} ({self.device.type}) | Pairs: {self.n_pairs}{speed_s}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            n_trans = self._collect_episode()

            epoch_losses = {}
            epoch_nan = 0
            epoch_skips = 0
            n_updates = min(n_trans, self.env.num_steps)
            for _ in range(n_updates):
                step_losses = self._update_step()
                for k, v in step_losses.items():
                    epoch_losses.setdefault(k, []).append(v)
                if "skipped_nan" in step_losses:
                    epoch_nan += 1
                    epoch_skips += 1

            elapsed = time.time() - t0
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg.items()) if avg else "(no updates)"
            print(
                f"Epoch {epoch:4d}/{self.epochs} | {elapsed:5.1f}s | trans={n_trans} | "
                f"{loss_str} | nan_batches={epoch_nan} skipped={epoch_skips}"
            )

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self.save(tag=f"epoch_{epoch}")

        self.save(tag="final")
        print(f"\nTraining complete. Final model saved to {self.model_dir}")

    def save(self, tag="checkpoint"):
        path = os.path.join(self.model_dir, f"{tag}.pt")
        self.model.save_checkpoint(path)


# ============================================================================
# SRRL (Supervised-RL Hybrid) Trainer
# ============================================================================

class SRRLUniformReplay:
    """Simple ring buffer with uniform sampling for SRRL transitions."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.buffer: list = []
        self.pos = 0

    def add(self, transition: dict) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Replay is empty.")
        replace = batch_size > n
        idx = np.random.choice(n, size=batch_size, replace=replace)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


# @register_agent("SRRL")
class SRRLTrainer(BaseTrainer):
    """Supervised-RL Hybrid: classification gates DDPG actor for per-pair lever."""

    def __init__(self, args, data):
        super().__init__(args, data)
        self.device = args.resolved_device
        self.epochs = args.epochs
        self.save_every = args.save_every
        self.model_dir = SRRL_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        F_dim = data["X_train"].shape[2]
        n_pairs = len(data["pairs"])
        n_tickers = len(data["tickers"])
        self.n_pairs = n_pairs

        self.model = SRRLTrader(F_dim, n_pairs, n_tickers, data["M"], device=str(self.device))

        if "y_bin32_train" not in data:
            raise RuntimeError(
                "SRRL requires binary labels. Run preprocessing.py first "
                "(produces spread_y_bin32_train.pkl)."
            )
        self.y_bin32_train = data["y_bin32_train"]
        base_rate = float(self.y_bin32_train.mean()) if len(self.y_bin32_train) > 0 else 0.5
        self.cls_pos_weight = max((1.0 - base_rate) / max(base_rate, 1e-8), 1.0)
        print(f"  SRRL cls pos_weight: {self.cls_pos_weight:.3f} (base_rate={base_rate:.4f})")

        meta = data["sequence_meta"].copy()
        if "Unnamed: 0" in meta.columns:
            meta = meta.drop(columns=["Unnamed: 0"])
        meta["target_date"] = pd.to_datetime(meta["target_date"])
        train_meta = meta[meta["split"] == "train"].reset_index(drop=True)
        pair_key_to_idx = {f"{a}|{b}": i for i, (a, b) in enumerate(data["pairs"])}
        self._date_pair_label: dict = {}
        for iloc_idx in range(len(train_meta)):
            row = train_meta.iloc[iloc_idx]
            d = row["target_date"]
            pk = str(row["Pair"])
            if pk in pair_key_to_idx:
                p_idx = pair_key_to_idx[pk]
                self._date_pair_label.setdefault(d, {})[p_idx] = int(self.y_bin32_train[iloc_idx])

        _diag = getattr(args, "env_diagnostic_no_risk_tax", False)
        self.env = TradingEnvironment(
            pairs=data["pairs"],
            tickers=data["tickers"],
            ticker_to_idx=data["ticker_to_idx"],
            trading_raw_path=os.path.join("data", "trading", "raw.csv"),
            sequence_meta=data["sequence_meta"],
            X_train=data["X_train"],
            y_train=data["y_train"],
            zeta=SRRL_HPARAMS.get("zeta", HPARAMS["zeta"]),
            gamma=SRRL_HPARAMS["gamma_risk"],
            risk_lambda=SRRL_HPARAMS["risk_lambda"],
            var_window=SRRL_HPARAMS["var_window"],
            terminal_utility_weight=0.0 if _diag else SRRL_HPARAMS["terminal_utility_weight"],
            use_stop_loss=False,
            diagnostic_no_risk_tax=_diag,
        )
        self.n_step = SRRL_HPARAMS["n_step"]
        total = max(1, int(args.epochs))
        self.epochs = total
        self.cls_warmup_epochs = total // 2  # first half cls-only; second half RL-only (no CLI)
        self._cls_frozen_for_rl = False
        self._episode_diag = {}

        buffer_cap = max(10000, self.env.num_steps * 2)
        self.replay = SRRLUniformReplay(buffer_cap)

        lr = SRRL_HPARAMS["lr"]
        wd = float(SRRL_HPARAMS.get("weight_decay", 0.0) or 0.0)
        self.opt_cls = torch.optim.Adam(
            list(self.model.srl_cls.parameters()) + list(self.model.cls_head.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        self.opt_actor = torch.optim.Adam(
            list(self.model.srl_actor.parameters()) + list(self.model.actor.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        self.opt_critic = torch.optim.Adam(
            list(self.model.srl_critic.parameters()) + list(self.model.critic.parameters()),
            lr=lr,
            weight_decay=wd,
        )

    def _h2d(self, arr, dtype=torch.float32):
        dev = self.model.device
        if dev.type == "cuda":
            return torch.as_tensor(np.asarray(arr), dtype=dtype).pin_memory().to(
                dev, non_blocking=True
            )
        return torch.tensor(np.asarray(arr), dtype=dtype, device=dev)

    def _get_bin32_labels(self, date):
        """Per-pair binary labels for a given training date; 0 where unknown."""
        labels = np.zeros(self.n_pairs, dtype=np.float32)
        mapping = self._date_pair_label.get(date, {})
        for p_idx, lbl in mapping.items():
            labels[p_idx] = float(lbl)
        return labels

    def _explore_sigma(self, epoch: int) -> float:
        sigma_hi = float(SRRL_HPARAMS.get("sigma_explore", 0.3))
        sigma_lo = float(SRRL_HPARAMS.get("sigma_explore_min", sigma_hi))
        if self.epochs <= 1:
            return sigma_lo
        K = self.cls_warmup_epochs
        rl_total = max(1, self.epochs - K)
        if K > 0:
            # Anneal exploration across the RL phase only; hold max sigma during cls rollouts.
            if epoch <= K:
                return sigma_hi
            rl_epoch = epoch - K
            rl_span = max(1, rl_total - 1)
            frac = float(max(0, rl_epoch - 1)) / float(rl_span)
            return sigma_hi + (sigma_lo - sigma_hi) * frac
        frac = float(max(0, epoch - 1)) / float(max(1, self.epochs - 1))
        return sigma_hi + (sigma_lo - sigma_hi) * frac

    def _collect_episode(self, epoch: int):
        self.model.eval()
        state_info = self.env.reset()
        if state_info is None:
            return 0
        n = 0
        n_step = self.n_step
        discount = SRRL_HPARAMS["discount_gamma"]
        nstep_buf: list[dict] = []
        sigma_now = self._explore_sigma(epoch)
        exposure_abs_sum = 0.0
        p_revert_sum = 0.0
        n_steps = 0
        churn_sum = 0.0
        churn_count = 0
        prev_e_gated = None

        def _flush(extra_bonus=0.0):
            nonlocal n
            if not nstep_buf:
                return
            G = 0.0
            for k in range(len(nstep_buf) - 1, -1, -1):
                G = nstep_buf[k]["reward"] + discount * G
            G += float(extra_bonus)
            head = nstep_buf[0]
            tail = nstep_buf[-1]
            self.replay.add({
                "state_windows": head["state_windows"],
                "E_gated": head["E_gated"],
                "y_bin32": head["y_bin32"],
                "pair_mask": head["pair_mask"],
                "next_pair_mask": tail["next_pair_mask"],
                "reward": np.float32(G),
                "next_state_windows": tail["next_state_windows"],
                "done": np.float32(1.0 if tail["done"] else 0.0),
                "n_step": len(nstep_buf),
            })
            n += 1

        while True:
            windows, mask, y_spread = state_info
            date_t = self.env.unique_dates[self.env.t]
            bin32 = self._get_bin32_labels(date_t)

            with torch.no_grad():
                step_out = self.model.forward_step(
                    windows, explore=True, pair_mask=mask, explore_sigma=sigma_now
                )

            w_np = step_out["weights"].detach().cpu().numpy().reshape(-1)
            E_gated_np = step_out["pair_exposures"].detach().cpu().numpy().reshape(self.n_pairs)
            p_revert_np = step_out["p_revert"].detach().cpu().numpy().reshape(self.n_pairs)
            exposure_abs_sum += float(np.mean(np.abs(E_gated_np)))
            p_revert_sum += float(np.mean(p_revert_np))
            if prev_e_gated is not None:
                churn_sum += float(np.mean(np.abs(E_gated_np - prev_e_gated)))
                churn_count += 1
            prev_e_gated = E_gated_np.copy()
            n_steps += 1

            next_info, reward, done = self.env.step(w_np, sl_actions=None, spread_values=None)
            if next_info is not None:
                next_windows, next_mask, _ = next_info
            else:
                next_windows = np.zeros_like(windows)
                next_mask = np.zeros_like(mask)

            nstep_buf.append({
                "state_windows": windows.astype(np.float32),
                "E_gated": E_gated_np.astype(np.float32),
                "y_bin32": bin32.astype(np.float32),
                "pair_mask": mask.astype(np.float32),
                "next_pair_mask": next_mask.astype(np.float32),
                "reward": float(reward),
                "next_state_windows": next_windows.astype(np.float32),
                "done": bool(done),
            })

            if len(nstep_buf) >= n_step:
                _flush()
                nstep_buf.pop(0)

            if done:
                terminal_bonus = float(self.env.episode_utility_bonus())
                while nstep_buf:
                    _flush(extra_bonus=terminal_bonus if len(nstep_buf) == 1 else 0.0)
                    nstep_buf.pop(0)
                break
            state_info = next_info
        self._episode_diag = {
            "sigma": sigma_now,
            "mean_abs_Eg": (exposure_abs_sum / max(n_steps, 1)),
            "churn_proxy": (churn_sum / max(churn_count, 1)),
            "mean_p_revert": (p_revert_sum / max(n_steps, 1)),
        }
        return n

    def _set_cls_trainable(self, trainable: bool):
        for p in self.model.srl_cls.parameters():
            p.requires_grad = trainable
        for p in self.model.cls_head.parameters():
            p.requires_grad = trainable

    def _update_step(self, epoch: int):
        B = SRRL_HPARAMS["batch_size"]
        gamma = SRRL_HPARAMS["discount_gamma"]
        if len(self.replay) < B:
            return {}

        self.model.train()
        for m in (
            self.model.srl_actor_target,
            self.model.actor_target,
            self.model.srl_critic_target,
            self.model.critic_target,
        ):
            m.eval()

        batch = self.replay.sample(B)

        # Classification warm-up only needs state windows + labels + mask. Building full RL batches
        # (next_state_windows, critic targets, etc.) here wastes RAM/GPU and can OOM when many
        # update steps run per epoch (~num_steps).
        if epoch <= self.cls_warmup_epochs:
            sw = self._h2d(np.stack([t["state_windows"] for t in batch]))
            y_cls = self._h2d(np.stack([t["y_bin32"] for t in batch]))
            pmask = self._h2d(np.stack([t["pair_mask"] for t in batch]))
            # --- Phase 1: classification-only warm-up (no RL updates) ---
            h_cls = self.model.encode_all_pairs(sw, self.model.srl_cls)
            Bc, Pc, Hc = h_cls.shape
            p_pred = self.model.cls_head(h_cls.reshape(Bc * Pc, Hc)).reshape(Bc, Pc)
            pos_w = torch.tensor(self.cls_pos_weight, device=sw.device, dtype=sw.dtype)
            weight_per = pmask * (y_cls * pos_w + (1.0 - y_cls) * 1.0)
            ls = float(SRRL_HPARAMS.get("cls_label_smoothing", 0.0) or 0.0)
            if ls > 0.0:
                y_tgt = (y_cls * (1.0 - 2.0 * ls) + ls).clamp(0.0, 1.0)
            else:
                y_tgt = y_cls
            bce = F.binary_cross_entropy(p_pred, y_tgt, weight=weight_per)
            self.opt_cls.zero_grad()
            bce.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.srl_cls.parameters()) + list(self.model.cls_head.parameters()), 1.0
            )
            self.opt_cls.step()
            self.model.eval()
            return {"bce": bce.item()}

        sw = self._h2d(np.stack([t["state_windows"] for t in batch]))
        E_g = self._h2d(np.stack([t["E_gated"] for t in batch]))
        rews = self._h2d(np.array([t["reward"] for t in batch]))
        nsw = self._h2d(np.stack([t["next_state_windows"] for t in batch]))
        dones = self._h2d(np.array([t["done"] for t in batch]))
        ns = self._h2d(np.array([t.get("n_step", 1) for t in batch]))
        pmask = self._h2d(np.stack([t["pair_mask"] for t in batch]))
        npmask = self._h2d(np.stack([t["next_pair_mask"] for t in batch]))

        # --- Phase 2: RL-only updates (classification frozen) ---
        # --- Critic loss (DDPG-style, n-step) ---
        with torch.no_grad():
            mu_next = self.model.actor_mean_target(nsw)
            h_cls_next = self.model.encode_all_pairs(nsw, self.model.srl_cls)
            Bn, Pn, Hn = h_cls_next.shape
            p_next = self.model.cls_head(h_cls_next.reshape(Bn * Pn, Hn)).reshape(Bn, Pn)
            E_gated_next = p_next * mu_next
            E_gated_next = E_gated_next * npmask
            q_next = self.model.critic_q_target(nsw, E_gated_next)
            gamma_n = gamma ** ns
            y_q = rews + (1.0 - dones) * gamma_n * q_next
            y_q = torch.nan_to_num(y_q, nan=0.0)

        q = self.model.critic_q(sw, E_g, self.model.srl_critic, self.model.critic)
        loss_c = F.mse_loss(q, y_q)
        self.opt_critic.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.srl_critic.parameters()) + list(self.model.critic.parameters()), 1.0
        )
        self.opt_critic.step()

        # --- Actor loss: -Q(s, E_gated) ---
        for p in self.model.srl_critic.parameters():
            p.requires_grad = False
        for p in self.model.critic.parameters():
            p.requires_grad = False

        h_a = self.model.encode_all_pairs(sw, self.model.srl_actor)
        Ba, Pa, Ha = h_a.shape
        mu_a = self.model.actor(h_a.reshape(Ba * Pa, Ha)).reshape(Ba, Pa)
        h_c_detach = self.model.encode_all_pairs(sw, self.model.srl_cls).detach()
        Bcd, Pcd, Hcd = h_c_detach.shape
        p_rev_det = self.model.cls_head(h_c_detach.reshape(Bcd * Pcd, Hcd)).reshape(Bcd, Pcd).detach()
        E_gated_actor = p_rev_det * mu_a
        if pmask is not None:
            E_gated_actor = E_gated_actor * pmask
        q_a = self.model.critic_q(sw, E_gated_actor, self.model.srl_critic, self.model.critic)
        turnover_lambda = float(SRRL_HPARAMS.get("turnover_penalty", 0.0))
        if turnover_lambda > 0.0:
            turn_pen = (pmask * (E_gated_actor - E_g).abs()).sum() / pmask.sum().clamp_min(1.0)
        else:
            turn_pen = q_a.new_tensor(0.0)
        loss_a = -q_a.mean() + turnover_lambda * turn_pen

        self.opt_actor.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.srl_actor.parameters()) + list(self.model.actor.parameters()), 1.0
        )
        self.opt_actor.step()

        for p in self.model.srl_critic.parameters():
            p.requires_grad = True
        for p in self.model.critic.parameters():
            p.requires_grad = True

        self.model.soft_update()

        self.model.eval()
        return {"critic": loss_c.item(), "actor": loss_a.item(), "turn_pen": float(turn_pen.item())}

    def train(self):
        print(f"\n{'=' * 60}")
        K = self.cls_warmup_epochs
        rl_part = self.epochs - K
        if K > 0:
            print(
                f"Training SRRL for {self.epochs} epochs "
                f"({K} classification-only, then {rl_part} RL-only; first/second half split)"
            )
        else:
            print(f"Training SRRL for {self.epochs} epochs (RL-only; single epoch cannot split)")
        print(f"Device: {self.device} ({self.device.type}) | Pairs: {self.n_pairs}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.epochs + 1):
            if (
                self.cls_warmup_epochs > 0
                and epoch == self.cls_warmup_epochs + 1
                and not self._cls_frozen_for_rl
            ):
                self._set_cls_trainable(False)
                self._cls_frozen_for_rl = True
                print(
                    "\nClassification pretraining complete. "
                    "Switching to RL-only SRRL training (critic + actor + target soft updates).\n"
                )

            t0 = time.time()
            n_trans = self._collect_episode(epoch)

            epoch_losses = {}
            n_updates = min(n_trans, self.env.num_steps)
            for _ in range(n_updates):
                step_losses = self._update_step(epoch)
                for k, v in step_losses.items():
                    epoch_losses.setdefault(k, []).append(v)

            elapsed = time.time() - t0
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg.items()) if avg else "(no updates)"
            diag = self._episode_diag if isinstance(self._episode_diag, dict) else {}
            diag_str = (
                f"sigma={diag.get('sigma', float('nan')):.3f}  "
                f"|E|={diag.get('mean_abs_Eg', float('nan')):.4f}  "
                f"churn={diag.get('churn_proxy', float('nan')):.4f}  "
                f"p_rev={diag.get('mean_p_revert', float('nan')):.4f}"
            )
            phase = "cls_only" if epoch <= self.cls_warmup_epochs else "rl_only"
            print(
                f"Epoch {epoch:4d}/{self.epochs} | {elapsed:5.1f}s | "
                f"{phase:10s} | trans={n_trans} | {loss_str} | {diag_str}"
            )

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self.save(tag=f"epoch_{epoch}")

        self.save(tag="final")
        print(f"\nTraining complete. Final model saved to {self.model_dir}")

    def save(self, tag="checkpoint"):
        path = os.path.join(self.model_dir, f"{tag}.pt")
        self.model.save_checkpoint(path)


# ============================================================================
# Main entry point
# ============================================================================

def _checkpoint_rank_key(ckpt_path: str):
    base = os.path.basename(ckpt_path).lower()
    if base == "final.pt":
        return 10**9
    if base.startswith("epoch_") and base.endswith(".pt"):
        mid = base[len("epoch_") : -len(".pt")]
        try:
            return int(mid)
        except ValueError:
            return -1
    return -1


def _build_eval_model(agent_name: str, ckpt_path: str, f_dim: int, n_pairs: int, n_tickers: int, M, device: torch.device):
    if agent_name == "MPHDRL":
        model = MPHDRLTrader(f_dim, n_pairs, n_tickers, M, device=str(device))
        model.load_checkpoint(ckpt_path)
        model.eval()
        return model
    # if agent_name == "Benchmark":
    #     try:
    #         raw_bench = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    #     except TypeError:
    #         raw_bench = torch.load(ckpt_path, map_location=str(device))
    #     hidden = int(raw_bench.get("meta", {}).get("hidden_size", 64))
    #     model = BenchmarkDDPG(f_dim, n_pairs, n_tickers, M, hidden_size=hidden, device=str(device))
    #     model.load_checkpoint(ckpt_path)
    #     model.eval()
    #     return model
    # if agent_name == "SRRL":
    #     model = SRRLTrader(f_dim, n_pairs, n_tickers, M, device=str(device))
    #     model.load_checkpoint(ckpt_path)
    #     model.eval()
    #     return model
    raise ValueError(f"Unsupported agent_name: {agent_name}")


def evaluate_and_promote_best_insample_checkpoint(
    agent_name: str,
    model_dir: str,
    device: torch.device,
    trained_epochs: int,
    save_every: int,
):
    expected = {f"epoch_{e}.pt" for e in range(max(save_every, 1), trained_epochs + 1, max(save_every, 1))}
    expected.add("final.pt")
    ckpts = []
    if os.path.isdir(model_dir):
        for n in os.listdir(model_dir):
            if n in expected:
                ckpts.append(os.path.join(model_dir, n))
    if not ckpts:
        print(f"[in-sample] {agent_name}: no checkpoints found in {model_dir}, skipping promotion.")
        return

    x_train, y_train, pairs, meta_train = load_sequence_bundle(split="train")
    n_pairs = len(pairs)
    f_dim = x_train.shape[2]

    hedge_path = os.path.join("data", "pickle", "hedge_ratios.pkl")
    hedge_ratios = None
    if os.path.isfile(hedge_path):
        with open(hedge_path, "rb") as f:
            hedge_ratios = pickle.load(f)
    M, tickers, ticker_to_idx = build_pair_ticker_mapping(pairs, hedge_ratios=hedge_ratios)
    n_tickers = len(tickers)
    pair_key_to_idx = {f"{a}|{b}": i for i, (a, b) in enumerate(pairs)}

    price_wide = load_price_matrix(tickers)
    train_dates = sorted(meta_train["target_date"].unique())
    valid_dates = [d for d in train_dates if d in price_wide.index]
    if len(valid_dates) < 2:
        print(f"[in-sample] {agent_name}: insufficient train dates for evaluation, skipping promotion.")
        return

    spread_raw_path = os.path.join("data", "spread", "raw.csv")
    spread_wide = None
    if os.path.isfile(spread_raw_path):
        from traditional import load_precomputed_spread_wide

        spread_wide = load_precomputed_spread_wide(spread_raw_path)

    rows = []
    for ckpt in sorted(ckpts, key=lambda p: (_checkpoint_rank_key(p), p)):
        try:
            model = _build_eval_model(agent_name, ckpt, f_dim, n_pairs, n_tickers, M, device)
            if agent_name == "MPHDRL":
                weights_by_date = get_mphdrl_weights_by_env(
                    model,
                    meta_train,
                    x_train,
                    y_train,
                    pairs,
                    tickers,
                    ticker_to_idx,
                    spread_wide=spread_wide,
                    split="train",
                )
            else:
                weights_by_date = get_all_weights_by_date(
                    model, meta_train, x_train, pair_key_to_idx, n_pairs, f_dim, train_dates
                )
            df_bt = run_strategy_backtest(f"{agent_name} in-sample", weights_by_date, price_wide, tickers, valid_dates)
            m = summarize_backtest_dataframe(df_bt, gamma=0.5)
            rows.append(
                {
                    "checkpoint": os.path.abspath(ckpt),
                    "checkpoint_name": os.path.basename(ckpt),
                    "epoch_rank": _checkpoint_rank_key(ckpt),
                    "status": "ok",
                    **m,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "checkpoint": os.path.abspath(ckpt),
                    "checkpoint_name": os.path.basename(ckpt),
                    "epoch_rank": _checkpoint_rank_key(ckpt),
                    "status": "failed",
                    "utility": float("-inf"),
                    "annual_return": float("nan"),
                    "variance": float("nan"),
                    "sharpe": float("nan"),
                    "mean_abs_weight": float("nan"),
                    "l1_turnover": float("nan"),
                    "transaction_cost": float("nan"),
                    "shorting_cost": float("nan"),
                    "error": str(e),
                }
            )

    ok_rows = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("utility", float("-inf")))]
    if not ok_rows:
        print(f"[in-sample] {agent_name}: all checkpoint evaluations failed; keeping existing final.pt.")
        return

    best = sorted(ok_rows, key=lambda r: (r["utility"], r["epoch_rank"]), reverse=True)[0]
    best_ckpt = best["checkpoint"]
    final_path = os.path.join(model_dir, "final.pt")
    if os.path.abspath(best_ckpt) != os.path.abspath(final_path):
        shutil.copy2(best_ckpt, final_path)

    artifact_dir = os.path.join("artifacts", "insample_selection", agent_name.lower())
    os.makedirs(artifact_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(artifact_dir, f"{agent_name.lower()}_checkpoint_scores_{stamp}.csv")
    json_path = os.path.join(artifact_dir, f"{agent_name.lower()}_selection_{stamp}.json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent": agent_name,
                "timestamp": stamp,
                "selected_checkpoint": best_ckpt,
                "promoted_to": os.path.abspath(final_path),
                "best_metrics": {
                    "utility": best["utility"],
                    "annual_return": best["annual_return"],
                    "variance": best["variance"],
                    "mean_abs_weight": best["mean_abs_weight"],
                    "l1_turnover": best["l1_turnover"],
                },
                "num_evaluated": len(rows),
                "num_ok": len(ok_rows),
                "scores_csv": os.path.abspath(csv_path),
            },
            f,
            indent=2,
        )

    print(
        f"[in-sample] {agent_name}: promoted {os.path.basename(best_ckpt)} "
        f"to final.pt (utility={best['utility']:.6f})"
    )
    print(f"[in-sample] report: {csv_path}")


def main():
    args = load_training_config()
    set_global_seed(args.seed)
    reset_mphdrl_hparams_to_defaults()
    loaded_from_tuning = load_best_mphdrl_params_from_tuning(args.tuning_dir)
    merge_mphdrl_hp_patch_from_env()
    if not loaded_from_tuning and not os.environ.get("MPHDRL_HP_PATCH", "").strip():
        print(
            "Hyperparameters: built-in MPHDRL defaults (no trials.csv / patch). "
            "Run mphdrl_tuning.py to populate artifacts/mphdrl_tuning/trials.csv."
        )

    resolved = resolve_training_device(args.device)
    configure_accelerator(resolved)
    args.resolved_device = resolved

    print("=" * 60)
    print(f"Agent: {args.agent}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {resolved} (MPHDRL_DEVICE={args.device!r})")
    if resolved.type == "cuda":
        print(
            f"Speed opts: AMP={'off' if args.no_amp else 'on (bf16 or fp16+scaler)'}, "
            f"torch.compile={'off' if args.no_compile else 'on (forward_step)'}"
        )
    if getattr(args, "env_diagnostic_no_risk_tax", False):
        print("Env diagnostic: no variance penalty in reward, no taxes, no terminal bonus; costs on.")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("=" * 60)
    print_mphdrl_hparams()

    print("\nChecking data readiness...")
    ok, data = check_data_readiness()
    if not ok:
        print("Data readiness failed. Run the preprocessing pipeline first.")
        sys.exit(1)

    agent_order = [args.agent]

    for agent_name in agent_order:
        trainer_cls = AGENT_REGISTRY.get(agent_name)
        if trainer_cls is None:
            print(f"Unknown agent: {agent_name}. Available: {list(AGENT_REGISTRY.keys())}")
            sys.exit(1)
        print(f"\nStarting trainer: {agent_name}")
        trainer = trainer_cls(args, data)
        trainer.train()
        print(f"\nRunning in-sample checkpoint selection for {agent_name}...")
        evaluate_and_promote_best_insample_checkpoint(
            agent_name,
            trainer.model_dir,
            resolved,
            trained_epochs=getattr(trainer, "epochs", args.epochs),
            save_every=args.save_every,
        )


if __name__ == "__main__":
    main()
