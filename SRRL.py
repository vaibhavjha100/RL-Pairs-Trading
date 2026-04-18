"""
SRRL.py -- Supervised-RL Hybrid Pairs Trader.

Combines a spread-reversion binary classifier with a DDPG actor-critic,
sharing SRL (Gate+LSTM) encoders. Classification probability gates the
actor exposures before net-position portfolio mapping (E @ M, L1, de-mean).

No stop-loss head. No portfolio-weight network. Simple and direct.
"""

from __future__ import annotations

import copy
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SRRL_HPARAMS = {
    "H_srl": 128,
    "lr": 1e-4,
    "tau": 0.005,
    "batch_size": 32,
    "discount_gamma": 0.99,
    "n_step": 10,
    "gamma_risk": 0.5,
    "risk_lambda": 1.0,
    "var_window": 60,
    "terminal_utility_weight": 1.0,
    "sigma_explore": 0.3,
    "sigma_explore_min": 0.10,
    "turnover_penalty": 0.05,
    "cls_warmup_epochs": 0,
    # Anti–overfitting (train-time only; rollout uses eval mode)
    "dropout": 0.12,
    "weight_decay": 1e-4,
    "cls_label_smoothing": 0.08,
}

SRRL_MODEL_DIR = os.path.join("models", "srrl")

# ---------------------------------------------------------------------------
# SRL Encoder (Gate + LSTM) — same architecture as MPHDRL
# ---------------------------------------------------------------------------

class GateLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.W = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        return torch.sigmoid(self.W(x)) * x


class SRLEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_size=128, dropout: float = 0.0):
        super().__init__()
        self.gate = GateLayer(feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers=1, batch_first=True)
        self.hidden_size = hidden_size
        self.dropout_p = float(dropout or 0.0)

    def forward(self, x):
        gated = self.gate(x)
        _, (h_n, _) = self.lstm(gated)
        h = h_n.squeeze(0)
        if self.dropout_p > 0.0:
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h


# ---------------------------------------------------------------------------
# Classification Head — per-pair reversion probability
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=128, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout_p = float(dropout or 0.0)

    def forward(self, h):
        x = F.relu(self.fc1(h))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# DDPG Actor — per-pair continuous exposure in [-1, 1]
# ---------------------------------------------------------------------------

class DDPGActor(nn.Module):
    def __init__(self, hidden_size=128, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout_p = float(dropout or 0.0)

    def forward(self, h):
        x = F.relu(self.fc1(h))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return torch.tanh(self.fc2(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# DDPG Critic — Q(state, action)
# ---------------------------------------------------------------------------

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, dropout: float = 0.0):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)
        self.dropout_p = float(dropout or 0.0)

    def forward(self, state_pooled, actions_flat):
        x = torch.cat([state_pooled, actions_flat], dim=-1)
        x = F.relu(self.l1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.l2(x))
        return self.l3(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Portfolio mapping: E_gated @ M -> L1 normalise -> de-mean
# ---------------------------------------------------------------------------

def gated_exposures_to_weights(E_gated: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    if E_gated.dim() == 1:
        E_gated = E_gated.unsqueeze(0)
    E_gated = torch.nan_to_num(E_gated, nan=0.0, posinf=0.0, neginf=0.0)
    u = torch.matmul(E_gated, M)
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    l1 = u.abs().sum(dim=-1, keepdim=True)
    w = torch.where(l1 > 1e-8, u / torch.clamp(l1, min=1e-8), torch.zeros_like(u))
    w = w - w.mean(dim=-1, keepdim=True)
    return w


# ---------------------------------------------------------------------------
# SRRLTrader — full model
# ---------------------------------------------------------------------------

class SRRLTrader(nn.Module):
    def __init__(self, feature_dim, n_pairs, n_tickers, M_numpy, device="cpu"):
        super().__init__()
        self._device = torch.device(device)
        self.feature_dim = feature_dim
        self.n_pairs = n_pairs
        self.n_tickers = n_tickers
        H = SRRL_HPARAMS["H_srl"]
        do = float(SRRL_HPARAMS.get("dropout", 0.0) or 0.0)

        self.register_buffer("M", torch.tensor(M_numpy, dtype=torch.float32))

        self.srl_cls = SRLEncoder(feature_dim, H, dropout=do)
        self.srl_actor = SRLEncoder(feature_dim, H, dropout=do)
        self.srl_critic = SRLEncoder(feature_dim, H, dropout=do)

        self.cls_head = ClassificationHead(H, dropout=do)
        self.actor = DDPGActor(H, dropout=do)
        self.critic = DDPGCritic(H, n_pairs, dropout=do)

        self.srl_actor_target = copy.deepcopy(self.srl_actor)
        self.actor_target = copy.deepcopy(self.actor)
        self.srl_critic_target = copy.deepcopy(self.srl_critic)
        self.critic_target = copy.deepcopy(self.critic)
        self._freeze_targets()
        self.to(self._device)
        self.eval()

    @property
    def device(self) -> torch.device:
        return self._device

    def _freeze_targets(self):
        for m in (self.srl_actor_target, self.actor_target,
                  self.srl_critic_target, self.critic_target):
            for p in m.parameters():
                p.requires_grad = False

    def soft_update(self, tau=None):
        if tau is None:
            tau = SRRL_HPARAMS["tau"]
        for online, target in [
            (self.srl_actor, self.srl_actor_target),
            (self.actor, self.actor_target),
            (self.srl_critic, self.srl_critic_target),
            (self.critic, self.critic_target),
        ]:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data.copy_(tau * po.data + (1.0 - tau) * pt.data)

    def encode_all_pairs(self, windows, srl_module):
        B, P, S, Fdim = windows.shape
        flat = windows.reshape(B * P, S, Fdim)
        h = srl_module(flat)
        return h.reshape(B, P, -1)

    def forward_step(self, windows, explore=True, pair_mask=None, explore_sigma=None):
        if isinstance(windows, torch.Tensor):
            wt = windows.to(device=self._device, dtype=torch.float32)
        else:
            wt = torch.as_tensor(np.asarray(windows), dtype=torch.float32, device=self._device)
        if wt.dim() == 3:
            wt = wt.unsqueeze(0)

        h_cls = self.encode_all_pairs(wt, self.srl_cls)
        h_actor = self.encode_all_pairs(wt, self.srl_actor)

        B, P, H = h_cls.shape
        p_revert = self.cls_head(h_cls.reshape(B * P, H)).reshape(B, P)
        mu_E = self.actor(h_actor.reshape(B * P, H)).reshape(B, P)

        if explore:
            sigma = (
                SRRL_HPARAMS["sigma_explore"]
                if explore_sigma is None
                else float(explore_sigma)
            )
            noise = torch.randn_like(mu_E) * sigma
            E = (mu_E + noise).clamp(-1.0, 1.0)
        else:
            E = mu_E

        E_gated = p_revert * E

        if pair_mask is not None:
            pm = torch.as_tensor(np.asarray(pair_mask), dtype=torch.float32, device=self._device)
            if pm.dim() == 1:
                pm = pm.unsqueeze(0)
            E_gated = E_gated * pm

        w = gated_exposures_to_weights(E_gated, self.M)

        return {
            "weights": w,
            "pair_exposures": E_gated,
            "p_revert": p_revert,
            "mu_E": mu_E,
            "E": E,
            "h_cls": h_cls,
            "h_actor": h_actor,
        }

    def actor_mean_target(self, windows):
        if isinstance(windows, torch.Tensor):
            wt = windows
        else:
            wt = torch.as_tensor(np.asarray(windows), dtype=torch.float32, device=self._device)
        if wt.dim() == 3:
            wt = wt.unsqueeze(0)
        h = self.encode_all_pairs(wt, self.srl_actor_target)
        B, P, H = h.shape
        return self.actor_target(h.reshape(B * P, H)).reshape(B, P)

    def critic_q(self, windows, E, encoder, critic_net):
        if isinstance(windows, torch.Tensor):
            wt = windows
        else:
            wt = torch.as_tensor(np.asarray(windows), dtype=torch.float32, device=self._device)
        if wt.dim() == 3:
            wt = wt.unsqueeze(0)
        h = self.encode_all_pairs(wt, encoder)
        s_feat = h.mean(dim=1)
        return critic_net(s_feat, E)

    def critic_q_target(self, windows, E):
        return self.critic_q(windows, E, self.srl_critic_target, self.critic_target)

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(SRRL_MODEL_DIR, "checkpoint.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "srl_cls": self.srl_cls.state_dict(),
            "srl_actor": self.srl_actor.state_dict(),
            "srl_critic": self.srl_critic.state_dict(),
            "cls_head": self.cls_head.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "srl_actor_target": self.srl_actor_target.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "srl_critic_target": self.srl_critic_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "meta": {
                "feature_dim": self.feature_dim,
                "n_pairs": self.n_pairs,
                "n_tickers": self.n_tickers,
            },
        }
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(SRRL_MODEL_DIR, "checkpoint.pt")
        try:
            state = torch.load(path, map_location=self._device, weights_only=False)
        except TypeError:
            state = torch.load(path, map_location=self._device)

        def _remap_two_layer_fc(sd: dict) -> dict:
            """Old checkpoints: Sequential net.0 / net.2 -> fc1 / fc2."""
            if not sd or any(k.startswith("fc1.") for k in sd):
                return sd
            if not any(k.startswith("net.0.") for k in sd):
                return sd
            out = {}
            for k, v in sd.items():
                if k.startswith("net.0."):
                    out["fc1." + k[6:]] = v
                elif k.startswith("net.2."):
                    out["fc2." + k[6:]] = v
            return out

        def _remap_critic_sequential(sd: dict) -> dict:
            """Old checkpoints: net.0 / net.2 / net.4 -> l1 / l2 / l3."""
            if not sd or any(k.startswith("l1.") for k in sd):
                return sd
            if not any(k.startswith("net.0.") for k in sd):
                return sd
            out = {}
            for k, v in sd.items():
                if k.startswith("net.0."):
                    out["l1." + k[6:]] = v
                elif k.startswith("net.2."):
                    out["l2." + k[6:]] = v
                elif k.startswith("net.4."):
                    out["l3." + k[6:]] = v
            return out

        state = dict(state)
        if "cls_head" in state:
            state["cls_head"] = _remap_two_layer_fc(state["cls_head"])
        if "actor" in state:
            state["actor"] = _remap_two_layer_fc(state["actor"])
        if "critic" in state:
            state["critic"] = _remap_critic_sequential(state["critic"])
        if "actor_target" in state:
            state["actor_target"] = _remap_two_layer_fc(state["actor_target"])
        if "critic_target" in state:
            state["critic_target"] = _remap_critic_sequential(state["critic_target"])

        def _load(module, key):
            if key not in state:
                return
            missing, unexpected = module.load_state_dict(state[key], strict=False)
            if missing or unexpected:
                print(
                    f"  Note: partial load for {key}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

        _load(self.srl_cls, "srl_cls")
        _load(self.srl_actor, "srl_actor")
        _load(self.srl_critic, "srl_critic")
        _load(self.cls_head, "cls_head")
        _load(self.actor, "actor")
        _load(self.critic, "critic")
        _load(self.srl_actor_target, "srl_actor_target")
        _load(self.actor_target, "actor_target")
        _load(self.srl_critic_target, "srl_critic_target")
        _load(self.critic_target, "critic_target")
        print(f"Checkpoint loaded: {path}")
