"""
training.py -- Agent-agnostic training harness for RL pairs trading.

Supports multiple RL agent types via a registry pattern: MPHDRL (hybrid HDRL)
and Benchmark (plain GRU actor–critic, DDPG-style replay on pair exposures E).

Training logic follows multi_pair_hdrl_trader_architecture.md (TD3 critics,
DDQN stop-loss head, delayed actor + portfolio + auxiliary regression, PER).

Usage:
    python training.py --agent MPHDRL --epochs 100
    python training.py --agent MPHDRL --epochs 50 --device cuda
    python training.py --agent MPHDRL --epochs 50 --device auto
    python training.py --agent Benchmark --epochs 100 --device cuda

On CUDA (default): cudnn benchmark, TF32, fast H2D copies, mixed precision (bfloat16
if supported else float16 + GradScaler), and torch.compile on forward_step unless
disabled with --no-amp / --no-compile. Same training schedule (epochs, HPARAMS).
"""

import contextlib
import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from MPHDRL import (
    HPARAMS,
    MPHDRL_MODEL_DIR,
    MPHDRLTrader,
    TradingEnvironment,
    PrioritizedReplayBuffer,
    check_data_readiness,
)
from benchmark import BENCHMARK_MODEL_DIR, BenchmarkDDPG


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

        if self.device.type == "cuda" and not getattr(args, "no_compile", False) and hasattr(
            torch, "compile"
        ):
            try:
                self.model.forward_step = torch.compile(
                    self.model.forward_step,
                    mode="default",
                    fullgraph=False,
                )
                self._forward_compiled = True
            except Exception as e:
                print(f"[training] torch.compile skipped: {e}")
                self._forward_compiled = False
        else:
            self._forward_compiled = False

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
            terminal_utility_weight=HPARAMS["terminal_utility_weight"],
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
                    step_out = self.model.forward_step(windows, explore=True)

            w_np = step_out["weights"].detach().cpu().numpy().flatten()
            actions_np = step_out["actions"].detach().cpu().numpy().flatten()
            sl_np = step_out["sl_actions"].detach().cpu().numpy().flatten()

            next_info, reward, done = self.env.step(
                w_np, sl_actions=sl_np, spread_values=y_spread,
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
                bonus_applied = False
                while nstep_buf:
                    _flush_nstep(extra_bonus=terminal_bonus if not bonus_applied else 0.0)
                    bonus_applied = True
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
                y_sl = rews_expanded + (1.0 - dones_expanded) * discount_gamma * q_sl_target_vals

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

@register_agent("Benchmark")
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
            terminal_utility_weight=HPARAMS["terminal_utility_weight"],
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
            windows, _mask, y_spread = state_info
            with torch.no_grad():
                step_out = self.model.forward_step(windows, explore=True)

            w_np = step_out["weights"].detach().cpu().numpy().reshape(-1)
            E_np = step_out["pair_exposures"].detach().cpu().numpy().reshape(self.n_pairs)

            sl_zeros = np.zeros(self.n_pairs, dtype=np.int64)
            next_info, reward, done = self.env.step(
                w_np, sl_actions=sl_zeros, spread_values=y_spread,
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
                bonus_applied = False
                while nstep_buf:
                    _flush(extra_bonus=terminal_bonus if not bonus_applied else 0.0)
                    bonus_applied = True
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

        with torch.no_grad():
            mu_next = self.model.actor_mean_target(nsw)
            q_next = self.model.critic_q(
                nsw, mu_next, self.model.critic_encoder_t, self.model.critic_head_t
            )
            gamma_n = gamma ** ns
            y = rews + (1.0 - dones) * gamma_n * q_next

        q = self.model.critic_q(sw, E, self.model.critic_encoder, self.model.critic_head)
        loss_c = F.mse_loss(q, y)
        self.opt_critic.zero_grad()
        loss_c.backward()
        self.opt_critic.step()

        for p in self.model.critic_encoder.parameters():
            p.requires_grad = False
        for p in self.model.critic_head.parameters():
            p.requires_grad = False

        mu = self.model.actor_mean(sw)
        q_a = self.model.critic_q(sw, mu, self.model.critic_encoder, self.model.critic_head)
        loss_a = -q_a.mean()

        self.opt_actor.zero_grad()
        loss_a.backward()
        self.opt_actor.step()

        for p in self.model.critic_encoder.parameters():
            p.requires_grad = True
        for p in self.model.critic_head.parameters():
            p.requires_grad = True

        self.model.soft_update()

        return {"critic": loss_c.item(), "actor": loss_a.item()}

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"Training Benchmark (DDPG-style) agent for {self.epochs} epochs")
        print(f"Device: {self.device} ({self.device.type}) | Pairs: {self.n_pairs}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            n_trans = self._collect_episode()

            epoch_losses = {}
            n_updates = min(n_trans, self.env.num_steps)
            for _ in range(n_updates):
                step_losses = self._update_step()
                for k, v in step_losses.items():
                    epoch_losses.setdefault(k, []).append(v)

            elapsed = time.time() - t0
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg.items()) if avg else "(no updates)"
            print(f"Epoch {epoch:4d}/{self.epochs} | {elapsed:5.1f}s | trans={n_trans} | {loss_str}")

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self.save(tag=f"epoch_{epoch}")

        self.save(tag="final")
        print(f"\nTraining complete. Final model saved to {self.model_dir}")

    def save(self, tag="checkpoint"):
        path = os.path.join(self.model_dir, f"{tag}.pt")
        self.model.save_checkpoint(path)


# ============================================================================
# CLI entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RL Pairs Trading -- Training Harness")
    parser.add_argument("--agent", type=str, default="MPHDRL",
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Agent type to train (default: MPHDRL)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device: auto (CUDA>MPS>CPU), cuda, mps, or cpu (default: auto)")
    parser.add_argument("--save-every", type=int, default=10, dest="save_every",
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed precision (default: AMP on when using CUDA)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile on forward_step (default: compile on CUDA)")
    return parser.parse_args()


def main():
    args = parse_args()

    resolved = resolve_training_device(args.device)
    configure_accelerator(resolved)
    args.resolved_device = resolved

    print("=" * 60)
    print(f"Agent: {args.agent}")
    print(f"Device: {resolved} (from --device {args.device!r})")
    if resolved.type == "cuda":
        print(
            f"Speed opts: AMP={'off' if args.no_amp else 'on (bf16 or fp16+scaler)'}, "
            f"torch.compile={'off' if args.no_compile else 'on (forward_step)'}"
        )
    print("=" * 60)

    print("\nChecking data readiness...")
    ok, data = check_data_readiness()
    if not ok:
        print("Data readiness failed. Run the preprocessing pipeline first.")
        sys.exit(1)

    trainer_cls = AGENT_REGISTRY.get(args.agent)
    if trainer_cls is None:
        print(f"Unknown agent: {args.agent}. Available: {list(AGENT_REGISTRY.keys())}")
        sys.exit(1)

    trainer = trainer_cls(args, data)
    trainer.train()


if __name__ == "__main__":
    main()
