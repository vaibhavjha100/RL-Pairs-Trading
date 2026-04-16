"""
benchmark.py -- Plain-vanilla actor–critic (DDPG-style) baseline for pairs trading.

No SRL / gate: shared GRU per pair, Gaussian pair exposures E, fixed aggregation
u = E @ M and same L1 + market-neutral projection as MPHDRL's portfolio tail for w.

Checkpoints under models/benchmark/ (models/ is gitignored).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from MPHDRL import HPARAMS

BENCHMARK_MODEL_DIR = os.path.join("models", "benchmark")

# Hidden size for plain GRU encoders (deliberately separate from H_srl).
H_BENCHMARK = 128


def pair_exposures_to_weights(E: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Map per-pair exposures E to ticker weights w (same geometry as PortfolioWeightsNetwork tail).

    E:   (batch, n_pairs) or (n_pairs,) with batch implied
    M:   (n_pairs, n_tickers) on same device as E
    """
    if E.dim() == 1:
        E = E.unsqueeze(0)
    u = torch.matmul(E, M)
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    l1 = u.abs().sum(dim=-1, keepdim=True)
    fallback = M[0].unsqueeze(0).expand_as(u)
    fallback = fallback - fallback.mean(dim=-1, keepdim=True)
    fallback_l1 = torch.clamp(fallback.abs().sum(dim=-1, keepdim=True), min=1e-8)
    fallback = fallback / fallback_l1

    w = torch.where(l1 > 1e-8, u / torch.clamp(l1, min=1e-8), fallback)
    w = w - w.mean(dim=-1, keepdim=True)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    w_l1 = w.abs().sum(dim=-1, keepdim=True)
    w = torch.where(w_l1 > 1e-8, w / torch.clamp(w_l1, min=1e-8), fallback)
    return w


class PlainPairEncoder(nn.Module):
    """Single-layer GRU over each pair's window (no gate, no SRL)."""

    def __init__(self, feature_dim: int, hidden_size: int = H_BENCHMARK):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_size, num_layers=1, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, seq_len, F)
        _, h_n = self.gru(x)
        return h_n.squeeze(0)


class BenchmarkDDPG(nn.Module):
    """
    Actor: GRU per pair -> mu_E (and learned log_std per pair for exploration).
    Critic: separate GRU -> mean-pool state, concat E -> Q(s, E).
    Target copies with soft polyak (tau from HPARAMS).
    """

    def __init__(
        self,
        feature_dim: int,
        n_pairs: int,
        n_tickers: int,
        M_numpy: np.ndarray,
        hidden_size: int = H_BENCHMARK,
        device: str = "cpu",
    ):
        super().__init__()
        self._device = torch.device(device)
        self.feature_dim = feature_dim
        self.n_pairs = n_pairs
        self.n_tickers = n_tickers
        self.hidden_size = hidden_size

        self.register_buffer("M", torch.tensor(M_numpy, dtype=torch.float32, device=self._device))

        self.actor_encoder = PlainPairEncoder(feature_dim, hidden_size)
        self.actor_mu = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(n_pairs))

        self.critic_encoder = PlainPairEncoder(feature_dim, hidden_size)
        crit_in = hidden_size + n_pairs
        self.critic_head = nn.Sequential(
            nn.Linear(crit_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.actor_encoder_t = copy.deepcopy(self.actor_encoder)
        self.actor_mu_t = copy.deepcopy(self.actor_mu)
        self.critic_encoder_t = copy.deepcopy(self.critic_encoder)
        self.critic_head_t = copy.deepcopy(self.critic_head)

        self._freeze_targets()
        self.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def _freeze_targets(self) -> None:
        for m in (
            self.actor_encoder_t,
            self.actor_mu_t,
            self.critic_encoder_t,
            self.critic_head_t,
        ):
            for p in m.parameters():
                p.requires_grad = False

    def encode_actor(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (B, P, S, F) -> per-pair hidden (B, P, H)."""
        B, P, S, Fdim = windows.shape
        flat = windows.reshape(B * P, S, Fdim)
        h = self.actor_encoder(flat)
        return h.reshape(B, P, -1)

    def actor_mean(self, windows: torch.Tensor) -> torch.Tensor:
        h = self.encode_actor(windows)
        mu = self.actor_mu(h.reshape(-1, self.hidden_size)).reshape(windows.shape[0], self.n_pairs)
        return mu

    def actor_mean_target(self, windows: torch.Tensor) -> torch.Tensor:
        h = self.encode_windows_with(self.actor_encoder_t, windows)
        mu = self.actor_mu_t(h.reshape(-1, self.hidden_size)).reshape(windows.shape[0], self.n_pairs)
        return mu

    @staticmethod
    def encode_windows_with(enc: PlainPairEncoder, windows: torch.Tensor) -> torch.Tensor:
        B, P, S, Fdim = windows.shape
        flat = windows.reshape(B * P, S, Fdim)
        h = enc(flat)
        return h.reshape(B, P, -1)

    def critic_q(
        self,
        windows: torch.Tensor,
        E: torch.Tensor,
        encoder: PlainPairEncoder,
        head: nn.Sequential,
    ) -> torch.Tensor:
        h = self.encode_windows_with(encoder, windows)
        s_feat = h.mean(dim=1)
        x = torch.cat([s_feat, E], dim=-1)
        return head(x).squeeze(-1)

    def soft_update(self, tau: Optional[float] = None) -> None:
        if tau is None:
            tau = HPARAMS["tau"]
        for online, target in (
            (self.actor_encoder, self.actor_encoder_t),
            (self.actor_mu, self.actor_mu_t),
            (self.critic_encoder, self.critic_encoder_t),
            (self.critic_head, self.critic_head_t),
        ):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data.copy_(tau * po.data + (1.0 - tau) * pt.data)

    def sample_E(self, mu: torch.Tensor, explore: bool) -> torch.Tensor:
        if not explore:
            return mu
        std = torch.exp(self.log_std.clamp(-5.0, 2.0)).unsqueeze(0).expand_as(mu)
        return mu + std * torch.randn_like(mu)

    def forward_step(
        self,
        windows: np.ndarray | torch.Tensor,
        explore: bool = True,
        pair_mask: np.ndarray | torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        """
        MPHDRL-compatible keys: weights (B, n_tickers), plus pair_exposures for replay.

        Placeholder keys for scripts that expect MPHDRL dict shape.
        """
        if not isinstance(windows, torch.Tensor):
            w_t = torch.as_tensor(np.asarray(windows), dtype=torch.float32, device=self._device)
        else:
            w_t = windows.to(dtype=torch.float32, device=self._device)
        if w_t.dim() == 3:
            w_t = w_t.unsqueeze(0)
        B, P, S, Fdim = w_t.shape
        assert P == self.n_pairs and Fdim == self.feature_dim

        mu = self.actor_mean(w_t)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        # At eval, pure zeros in mu leave no gradient history for M @ w; nudge so pair weights can trade.
        if not explore and float(mu.detach().abs().max()) < 1e-8:
            k = torch.arange(self.n_pairs, device=self._device, dtype=mu.dtype)[None, :]
            mu = mu + 1e-5 * torch.sin(k * (np.pi / max(self.n_pairs, 1)))
        E = self.sample_E(mu, explore)
        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        if pair_mask is not None:
            pm = torch.as_tensor(np.asarray(pair_mask), dtype=E.dtype, device=self._device)
            if pm.dim() == 1:
                pm = pm.unsqueeze(0)
            E = E * pm
        w = pair_exposures_to_weights(E, self.M)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        probs = torch.full((B, P, 3), 1.0 / 3.0, device=self._device, dtype=w.dtype)
        return {
            "weights": w,
            "pair_exposures": E,
            "pair_means": mu,
            "actions": torch.zeros((B, P), dtype=torch.long, device=self._device),
            "probs": probs,
            "sl_actions": torch.zeros((B, P), dtype=torch.long, device=self._device),
        }

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(BENCHMARK_MODEL_DIR, "checkpoint.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "actor_encoder": self.actor_encoder.state_dict(),
            "actor_mu": self.actor_mu.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "critic_encoder": self.critic_encoder.state_dict(),
            "critic_head": self.critic_head.state_dict(),
            "actor_encoder_t": self.actor_encoder_t.state_dict(),
            "actor_mu_t": self.actor_mu_t.state_dict(),
            "critic_encoder_t": self.critic_encoder_t.state_dict(),
            "critic_head_t": self.critic_head_t.state_dict(),
            "meta": {
                "feature_dim": self.feature_dim,
                "n_pairs": self.n_pairs,
                "n_tickers": self.n_tickers,
                "hidden_size": self.hidden_size,
            },
        }
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(BENCHMARK_MODEL_DIR, "checkpoint.pt")
        try:
            state = torch.load(path, map_location=self._device, weights_only=False)
        except TypeError:
            state = torch.load(path, map_location=self._device)
        self.actor_encoder.load_state_dict(state["actor_encoder"])
        self.actor_mu.load_state_dict(state["actor_mu"])
        self.log_std.data.copy_(state["log_std"].to(self._device))
        self.critic_encoder.load_state_dict(state["critic_encoder"])
        self.critic_head.load_state_dict(state["critic_head"])
        self.actor_encoder_t.load_state_dict(state["actor_encoder_t"])
        self.actor_mu_t.load_state_dict(state["actor_mu_t"])
        self.critic_encoder_t.load_state_dict(state["critic_encoder_t"])
        self.critic_head_t.load_state_dict(state["critic_head_t"])
        print(f"Checkpoint loaded: {path}")
