"""
MPHDRL.py -- Multi-Pair Hybrid Deep Reinforcement Learning Trader

Full model architecture extending HDRL-Trader (Kim et al., 2022) to a multi-pair
portfolio setting. Contains all network modules, environment, replay buffer,
target-network wiring, and checkpoint utilities.

Design reference: multi_pair_hdrl_trader_architecture.md (repo root). Implemented
pieces match that document where noted below; deliberate or pending gaps:

- Stop-loss magnitudes are used in the portfolio network and stored in replay, but
  TradingEnvironment does not yet simulate spread paths against stop boundaries (§6.2).
- TD3 bootstrap uses next-state encodings and target-actor next discrete actions; it
  does not re-evaluate target portfolio weights w' in the critic target (§8.2 optional
  path); critics remain Q(S, A_pairs) as in §3.
- Portfolio aggregation uses per-pair MLP scalars E_p then u = E @ M (same as u = M^T E
  in column form); cross-pair attention from §5.4 is simplified to independent E_p heads.

training.py orchestrates the training loop and should stay consistent with this module.
"""

import os
import copy
import pickle
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Hyperparameters (paper defaults)
# ============================================================================

HPARAMS = {
    "H_srl": 128,
    "N_w": 60,
    "TW": 60,
    "sigma_explore": 0.7,
    "sigma_smooth": 0.7,
    "clip_smooth": 1.0,
    "tb": 1.0,
    "zeta": 0.003,
    "risk_lambda": 1.0,
    "gamma": 0.5,
    "var_window": 60,
    "delay_c": 4,
    "delay_b": 2,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_xi": 0.0001,
    "epsilon_sl": 0.3,
    "tau": 0.005,
    "batch_size": 32,
    "lr": 0.0001,
    "discount_gamma": 0.99,
    "stop_loss_magnitudes": [1.5, 2.0, 2.5, 3.0, 3.5],
    "stop_loss_embed_dim": 8,
}

# Trained weights live under models/<agent>/ (models/ is gitignored).
MPHDRL_MODEL_DIR = os.path.join("models", "MPHDRL")

# ============================================================================
# Segment 0: Data readiness check
# ============================================================================

def build_pair_ticker_mapping(pairs):
    tickers = sorted({t for pair in pairs for t in pair})
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    n_pairs = len(pairs)
    n_tickers = len(tickers)
    M = np.zeros((n_pairs, n_tickers), dtype=np.float32)
    for p_idx, (a, b) in enumerate(pairs):
        M[p_idx, ticker_to_idx[a]] = 1.0
        M[p_idx, ticker_to_idx[b]] = -1.0
    return M, tickers, ticker_to_idx


def check_data_readiness():
    base = os.path.join("data", "pickle")
    spread_dir = os.path.join("data", "spread")
    trading_dir = os.path.join("data", "trading")
    os.makedirs(MPHDRL_MODEL_DIR, exist_ok=True)

    artifacts = {
        "spread_X_train": os.path.join(base, "spread_X_train.pkl"),
        "spread_y_train": os.path.join(base, "spread_y_train.pkl"),
        "cointegrated_pairs": os.path.join(base, "cointegrated_pairs.pkl"),
        "hedge_ratios": os.path.join(base, "hedge_ratios.pkl"),
        "sequence_xy": os.path.join(spread_dir, "sequence_xy.csv"),
        "trading_raw": os.path.join(trading_dir, "raw.csv"),
    }

    loaded = {}
    all_ok = True
    for name, path in artifacts.items():
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            all_ok = False
            continue
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
        elif path.endswith(".csv"):
            loaded[name] = pd.read_csv(path)
        print(f"  OK: {name} ({path})")

    if not all_ok:
        print("\nDATA READINESS: FAIL -- missing artifacts above.")
        return False, {}

    X = loaded["spread_X_train"]
    y = loaded["spread_y_train"]
    pairs = loaded["cointegrated_pairs"]

    checks = [
        ("X_train is 3D", X.ndim == 3),
        ("y_train is 1D", y.ndim == 1),
        ("X/y sample count match", X.shape[0] == y.shape[0]),
        ("pairs list non-empty", len(pairs) > 0),
    ]
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {desc}")
        if not ok:
            all_ok = False

    M, tickers, ticker_to_idx = build_pair_ticker_mapping(pairs)
    print(f"\n  X_train shape      : {X.shape}")
    print(f"  y_train shape      : {y.shape}")
    print(f"  Pairs              : {len(pairs)}")
    print(f"  Unique tickers     : {len(tickers)}")
    print(f"  Mapping matrix M   : {M.shape}")
    print(f"  Model dir          : {os.path.abspath(MPHDRL_MODEL_DIR)}")

    if all_ok:
        print("\nDATA READINESS: PASS")
    else:
        print("\nDATA READINESS: FAIL")

    return all_ok, {
        "X_train": X, "y_train": y, "pairs": pairs,
        "hedge_ratios": loaded["hedge_ratios"],
        "sequence_meta": loaded["sequence_xy"],
        "M": M, "tickers": tickers, "ticker_to_idx": ticker_to_idx,
    }


# ============================================================================
# Segment 1: SRL Module (Gate + LSTM + Regression Head)
# ============================================================================

class GateLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.W = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        g = torch.sigmoid(self.W(x))
        return g * x


class SRLEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_size=128):
        super().__init__()
        self.gate = GateLayer(feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers=1, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        gated = self.gate(x)
        _, (h_n, _) = self.lstm(gated)
        return h_n.squeeze(0)  # (batch, hidden_size)


class RegressionHead(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


# ============================================================================
# Segment 2: Actor Network
# ============================================================================

class ActorNetwork(nn.Module):
    def __init__(self, hidden_size=128, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, h):
        # Returns raw logits (batch, n_actions)
        return self.net(h)

    def get_action(self, h, explore=False, sigma=0.7):
        logits = self.forward(h)
        if explore:
            noise = torch.randn_like(logits) * sigma
            logits = logits + noise
        probs = F.softmax(logits, dim=-1)
        actions = probs.argmax(dim=-1)
        return actions, probs

    def bc_loss(self, h, expert_actions):
        logits = self.forward(h)
        return F.cross_entropy(logits, expert_actions)


# ============================================================================
# Segment 3: Twin Critic Networks
# ============================================================================

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state_pooled, actions_flat):
        # state_pooled: (batch, state_dim)   actions_flat: (batch, action_dim)
        x = torch.cat([state_pooled, actions_flat], dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================================
# Segment 4: Stop-Loss Network (DDQN)
# ============================================================================

class StopLossNetwork(nn.Module):
    def __init__(self, hidden_size=128, n_sl_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_sl_actions),
        )

    def forward(self, h):
        return self.net(h)  # Q-values for each stop-loss magnitude

    def get_action(self, h, epsilon=0.3, training=True):
        q_vals = self.forward(h)
        if training:
            rand_mask = torch.rand(h.shape[0], device=h.device) < epsilon
            random_actions = torch.randint(0, q_vals.shape[-1], (h.shape[0],), device=h.device)
            greedy_actions = q_vals.argmax(dim=-1)
            actions = torch.where(rand_mask, random_actions, greedy_actions)
        else:
            actions = q_vals.argmax(dim=-1)
        return actions, q_vals


# ============================================================================
# Segment 5: Portfolio Weights Network
# ============================================================================

class PortfolioWeightsNetwork(nn.Module):
    def __init__(self, hidden_size=128, sl_embed_dim=8, n_sl_actions=5):
        super().__init__()
        self.sl_embedding = nn.Embedding(n_sl_actions, sl_embed_dim)
        pair_embed_dim = hidden_size + 1 + sl_embed_dim  # h_p + e_p + sl_emb
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_actor, actor_probs, sl_actions, M_tensor):
        # h_actor:     (batch, N_pairs, hidden_size)
        # actor_probs: (batch, N_pairs, 3)
        # sl_actions:  (batch, N_pairs) int
        # M_tensor:    (N_pairs, N_tickers) float
        e_p = (actor_probs[:, :, 0] - actor_probs[:, :, 2]).unsqueeze(-1)  # (batch, N_pairs, 1)
        sl_emb = self.sl_embedding(sl_actions)  # (batch, N_pairs, sl_embed_dim)
        z_p = torch.cat([h_actor, e_p, sl_emb], dim=-1)  # (batch, N_pairs, pair_embed_dim)

        E = self.pair_mlp(z_p).squeeze(-1)  # (batch, N_pairs)
        u = torch.matmul(E, M_tensor)  # (batch, N_tickers)

        # L1 normalization + market-neutral de-meaning
        w = u / torch.clamp(u.abs().sum(dim=-1, keepdim=True), min=1e-8)
        w = w - w.mean(dim=-1, keepdim=True)
        return w


# ============================================================================
# Segment 6: Environment, Reward, Replay Buffer
# ============================================================================

class TradingEnvironment:
    """
    Train-split daily rollouts: windows from spread_X_train, returns from trading
    closes, reward = net return minus HPARAMS-style risk term (§7).

    Pair-level stop-loss hits (§6.2) are not applied to returns yet; sl_actions are
    reserved for portfolio construction and learning only.
    """

    def __init__(self, pairs, tickers, ticker_to_idx, trading_raw_path,
                 sequence_meta, X_train, y_train, zeta=0.003, gamma=0.5,
                 risk_lambda=1.0, var_window=60):
        self.pairs = pairs
        self.tickers = tickers
        self.ticker_to_idx = ticker_to_idx
        self.n_tickers = len(tickers)
        self.n_pairs = len(pairs)
        self.zeta = zeta
        self.gamma = gamma
        self.risk_lambda = risk_lambda
        self.var_window = var_window

        raw_df = pd.read_csv(trading_raw_path, parse_dates=["Date"])
        price_wide = raw_df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
        self.price_dates = price_wide.index
        self.price_matrix = price_wide[tickers].values.astype(np.float64)

        meta = sequence_meta.copy()
        if "Unnamed: 0" in meta.columns:
            meta = meta.drop(columns=["Unnamed: 0"])
        meta["target_date"] = pd.to_datetime(meta["target_date"])
        self.train_meta = meta[meta["split"] == "train"].reset_index(drop=True)

        self.date_to_sample_indices = {}
        for idx, row in self.train_meta.iterrows():
            d = row["target_date"]
            self.date_to_sample_indices.setdefault(d, {})[row["Pair"]] = idx

        self.unique_dates = sorted(self.date_to_sample_indices.keys())
        self.X_train = X_train
        self.y_train = y_train

        self.stop_loss_magnitudes = np.array(HPARAMS["stop_loss_magnitudes"])
        self.return_history = []
        self.t = 0
        self.prev_w = np.zeros(self.n_tickers, dtype=np.float64)

    def reset(self):
        self.t = 0
        self.return_history = []
        self.prev_w = np.zeros(self.n_tickers, dtype=np.float64)
        return self._get_state_windows()

    def _get_state_windows(self):
        if self.t >= len(self.unique_dates):
            return None
        date = self.unique_dates[self.t]
        pair_indices = self.date_to_sample_indices.get(date, {})
        windows = np.zeros((self.n_pairs, self.X_train.shape[1], self.X_train.shape[2]),
                           dtype=np.float32)
        mask = np.zeros(self.n_pairs, dtype=bool)
        y_vec = np.zeros(self.n_pairs, dtype=np.float32)
        pair_key_to_idx = {f"{a}|{b}": i for i, (a, b) in enumerate(self.pairs)}
        for pair_str, sample_idx in pair_indices.items():
            if pair_str in pair_key_to_idx:
                p_idx = pair_key_to_idx[pair_str]
                windows[p_idx] = self.X_train[sample_idx]
                mask[p_idx] = True
                y_vec[p_idx] = float(self.y_train[sample_idx])
        return windows, mask, y_vec

    def step(self, w, sl_actions=None, spread_values=None):
        """
        sl_actions / spread_values are reserved for future §6.2 stop-loss simulation
        against realized spreads; portfolio weights w already reflect stop embeddings.
        """
        if self.t + 1 >= len(self.unique_dates):
            return None, 0.0, True

        date_t = self.unique_dates[self.t]
        date_t1 = self.unique_dates[self.t + 1]

        price_idx_t = self.price_dates.get_loc(date_t) if date_t in self.price_dates else None
        price_idx_t1 = self.price_dates.get_loc(date_t1) if date_t1 in self.price_dates else None

        if price_idx_t is None or price_idx_t1 is None:
            self.t += 1
            return self._get_state_windows(), 0.0, False

        p_t = self.price_matrix[price_idx_t]
        p_t1 = self.price_matrix[price_idx_t1]

        safe_p = np.where(np.abs(p_t) < 1e-12, 1.0, p_t)
        raw_returns = (p_t1 - p_t) / safe_p

        sell_mask = (np.sign(w) != np.sign(self.prev_w)) | (np.abs(w) < np.abs(self.prev_w))
        cost = np.where(sell_mask, np.abs(w) * self.zeta, 0.0)
        net_return = float(np.nansum(w * raw_returns - cost))

        self.return_history.append(net_return)
        if len(self.return_history) >= self.var_window:
            var_r = np.var(self.return_history[-self.var_window:])
        elif len(self.return_history) >= 2:
            var_r = np.var(self.return_history)
        else:
            var_r = 0.0

        reward = net_return - self.gamma * self.risk_lambda * var_r
        self.prev_w = w.copy()
        self.t += 1
        next_state = self._get_state_windows()
        done = next_state is None
        return next_state, reward, done

    @property
    def num_steps(self):
        return len(self.unique_dates)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, xi=0.0001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.xi = xi
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0

    def add(self, transition):
        max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.priorities[self.size] = max_p
            self.size += 1
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if self.size < 1:
            raise ValueError("Cannot sample from an empty replay buffer.")
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        # replace=False requires batch_size <= self.size; also batch_size can exceed capacity.
        replace = batch_size > self.size
        indices = np.random.choice(self.size, size=batch_size, replace=replace, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for i, idx in enumerate(indices):
            self.priorities[idx] = abs(td_errors[i]) + self.xi

    def anneal_beta(self, fraction):
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    def __len__(self):
        return self.size


# ============================================================================
# Segment 7: MPHDRLTrader -- full model wiring
# ============================================================================

class MPHDRLTrader(nn.Module):
    def __init__(self, feature_dim, n_pairs, n_tickers, M_numpy, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dim = feature_dim
        self.n_pairs = n_pairs
        self.n_tickers = n_tickers
        self.M = torch.tensor(M_numpy, dtype=torch.float32, device=self.device)

        H = HPARAMS["H_srl"]

        # Online networks
        self.srl_actor = SRLEncoder(feature_dim, H)
        self.srl_critic1 = SRLEncoder(feature_dim, H)
        self.srl_critic2 = SRLEncoder(feature_dim, H)
        self.srl_stop = SRLEncoder(feature_dim, H)

        self.regression_head = RegressionHead(H)
        self.actor = ActorNetwork(H, 3)
        self.critic1 = CriticNetwork(H, n_pairs * 3)
        self.critic2 = CriticNetwork(H, n_pairs * 3)
        self.stop_loss = StopLossNetwork(H, 5)
        self.portfolio = PortfolioWeightsNetwork(H, HPARAMS["stop_loss_embed_dim"], 5)

        # Target networks (deep copies)
        self.srl_actor_target = copy.deepcopy(self.srl_actor)
        self.srl_critic1_target = copy.deepcopy(self.srl_critic1)
        self.srl_critic2_target = copy.deepcopy(self.srl_critic2)
        self.srl_stop_target = copy.deepcopy(self.srl_stop)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.stop_loss_target = copy.deepcopy(self.stop_loss)
        self.portfolio_target = copy.deepcopy(self.portfolio)

        self._freeze_targets()
        self.to(self.device)

    def _freeze_targets(self):
        target_modules = [
            self.srl_actor_target, self.srl_critic1_target, self.srl_critic2_target,
            self.srl_stop_target, self.actor_target, self.critic1_target,
            self.critic2_target, self.stop_loss_target, self.portfolio_target,
        ]
        for m in target_modules:
            for p in m.parameters():
                p.requires_grad = False

    def soft_update(self, tau=None):
        if tau is None:
            tau = HPARAMS["tau"]
        pairs = [
            (self.srl_actor, self.srl_actor_target),
            (self.srl_critic1, self.srl_critic1_target),
            (self.srl_critic2, self.srl_critic2_target),
            (self.srl_stop, self.srl_stop_target),
            (self.actor, self.actor_target),
            (self.critic1, self.critic1_target),
            (self.critic2, self.critic2_target),
            (self.stop_loss, self.stop_loss_target),
            (self.portfolio, self.portfolio_target),
        ]
        for online, target in pairs:
            for p_online, p_target in zip(online.parameters(), target.parameters()):
                p_target.data.copy_(tau * p_online.data + (1.0 - tau) * p_target.data)

    @staticmethod
    def dynamic_delay(epoch):
        c, b = HPARAMS["delay_c"], HPARAMS["delay_b"]
        return (epoch % c) + b

    def encode_all_pairs(self, windows, srl_module):
        # windows: (batch, N_pairs, seq_len, F)
        B, P, S, Fdim = windows.shape
        flat = windows.reshape(B * P, S, Fdim)
        h = srl_module(flat)
        return h.reshape(B, P, -1)  # (batch, N_pairs, H)

    def forward_step(self, windows, explore=True):
        windows_t = torch.tensor(windows, dtype=torch.float32, device=self.device)
        if windows_t.dim() == 3:
            windows_t = windows_t.unsqueeze(0)

        h_actor = self.encode_all_pairs(windows_t, self.srl_actor)
        h_stop = self.encode_all_pairs(windows_t, self.srl_stop)

        B, P, H = h_actor.shape
        h_actor_flat = h_actor.reshape(B * P, H)
        h_stop_flat = h_stop.reshape(B * P, H)

        actions, probs = self.actor.get_action(
            h_actor_flat, explore=explore, sigma=HPARAMS["sigma_explore"]
        )
        sl_actions, _ = self.stop_loss.get_action(
            h_stop_flat, epsilon=HPARAMS["epsilon_sl"], training=explore
        )

        actions = actions.reshape(B, P)
        probs = probs.reshape(B, P, -1)
        sl_actions = sl_actions.reshape(B, P)

        w = self.portfolio(h_actor, probs, sl_actions, self.M)

        return {
            "actions": actions,
            "probs": probs,
            "sl_actions": sl_actions,
            "weights": w,
            "h_actor": h_actor,
            "h_stop": h_stop,
        }

    def compute_critic_values(self, windows, actions_onehot, critic_idx=1):
        windows_t = torch.tensor(windows, dtype=torch.float32, device=self.device) \
            if not isinstance(windows, torch.Tensor) else windows
        if windows_t.dim() == 3:
            windows_t = windows_t.unsqueeze(0)

        srl = self.srl_critic1 if critic_idx == 1 else self.srl_critic2
        critic = self.critic1 if critic_idx == 1 else self.critic2

        h = self.encode_all_pairs(windows_t, srl)
        state_pooled = h.mean(dim=1)

        B = windows_t.shape[0]
        actions_flat = actions_onehot.reshape(B, -1)
        return critic(state_pooled, actions_flat)

    def compute_regression(self, windows):
        windows_t = torch.tensor(windows, dtype=torch.float32, device=self.device) \
            if not isinstance(windows, torch.Tensor) else windows
        if windows_t.dim() == 3:
            windows_t = windows_t.unsqueeze(0)
        h = self.encode_all_pairs(windows_t, self.srl_actor)
        B, P, H = h.shape
        return self.regression_head(h.reshape(B * P, H)).reshape(B, P)

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(MPHDRL_MODEL_DIR, "checkpoint.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "srl_actor": self.srl_actor.state_dict(),
            "srl_critic1": self.srl_critic1.state_dict(),
            "srl_critic2": self.srl_critic2.state_dict(),
            "srl_stop": self.srl_stop.state_dict(),
            "regression_head": self.regression_head.state_dict(),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "stop_loss": self.stop_loss.state_dict(),
            "portfolio": self.portfolio.state_dict(),
            "srl_actor_target": self.srl_actor_target.state_dict(),
            "srl_critic1_target": self.srl_critic1_target.state_dict(),
            "srl_critic2_target": self.srl_critic2_target.state_dict(),
            "srl_stop_target": self.srl_stop_target.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "stop_loss_target": self.stop_loss_target.state_dict(),
            "portfolio_target": self.portfolio_target.state_dict(),
        }
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(MPHDRL_MODEL_DIR, "checkpoint.pt")
        state = torch.load(path, map_location=self.device)
        self.srl_actor.load_state_dict(state["srl_actor"])
        self.srl_critic1.load_state_dict(state["srl_critic1"])
        self.srl_critic2.load_state_dict(state["srl_critic2"])
        self.srl_stop.load_state_dict(state["srl_stop"])
        self.regression_head.load_state_dict(state["regression_head"])
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.stop_loss.load_state_dict(state["stop_loss"])
        self.portfolio.load_state_dict(state["portfolio"])
        self.srl_actor_target.load_state_dict(state["srl_actor_target"])
        self.srl_critic1_target.load_state_dict(state["srl_critic1_target"])
        self.srl_critic2_target.load_state_dict(state["srl_critic2_target"])
        self.srl_stop_target.load_state_dict(state["srl_stop_target"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic1_target.load_state_dict(state["critic1_target"])
        self.critic2_target.load_state_dict(state["critic2_target"])
        self.stop_loss_target.load_state_dict(state["stop_loss_target"])
        self.portfolio_target.load_state_dict(state["portfolio_target"])
        print(f"Checkpoint loaded: {path}")


# ============================================================================
# Standalone entry point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MPHDRL Data Readiness Check")
    print("=" * 60)
    ok, data = check_data_readiness()
    if ok:
        F_dim = data["X_train"].shape[2]
        n_pairs = len(data["pairs"])
        n_tickers = len(data["tickers"])
        print(f"\nInitializing MPHDRLTrader (F={F_dim}, pairs={n_pairs}, tickers={n_tickers})...")
        model = MPHDRLTrader(F_dim, n_pairs, n_tickers, data["M"])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters (online + target): {total_params:,}")
        print("Model instantiation: OK")
