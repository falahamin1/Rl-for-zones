"""DDPG agent for probabilistic job-shop scheduling.

Architecture
------------
Actor   : MLP  state_dim → 256 → 256 → action_dim  (tanh output → [-1, 1]^J)
Critic  : MLP  (state_dim + action_dim) → 256 → 256 → 1

The actor outputs a J-dimensional priority vector.  At each dispatch point
the environment dispatches the enabled job with the highest score.

Training follows standard DDPG:
  • Ornstein-Uhlenbeck exploration noise (σ decays linearly to σ_min)
  • Experience replay (uniform sampling)
  • Soft target-network updates (τ = 0.005)
  • Separate actor / critic learning rates
"""
from __future__ import annotations

import copy
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False

from .pta_env import PTADirectEnv


# ---------------------------------------------------------------------------
# Neural network modules (require PyTorch)
# ---------------------------------------------------------------------------

def _require_torch():
    if not _TORCH:
        raise ImportError(
            "PyTorch is required for DDPGAgent.  Install with: pip install torch"
        )


class _Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class _Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self._buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self._buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.float32),
            np.array(r,  dtype=np.float32).reshape(-1, 1),
            np.array(s2, dtype=np.float32),
            np.array(d,  dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self._buf)


# ---------------------------------------------------------------------------
# OU noise
# ---------------------------------------------------------------------------

class _OUNoise:
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.3):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self, sigma_override: float | None = None) -> np.ndarray:
        sig = sigma_override if sigma_override is not None else self.sigma
        dx = self.theta * (self.mu - self.state) + sig * np.random.randn(len(self.state))
        self.state += dx
        return self.state.copy()


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPGAgent:
    """DDPG agent for the job-shop scheduling problem.

    Parameters
    ----------
    env           DDPGSchedulingEnv to train on
    lr_actor      learning rate for the actor
    lr_critic     learning rate for the critic
    gamma         discount factor (1.0 = undiscounted makespan)
    tau           soft target-update coefficient
    buffer_size   replay buffer capacity
    batch_size    mini-batch size for gradient updates
    sigma_start   initial OU-noise standard deviation
    sigma_min     minimum OU-noise std (reached at end of training)
    seed          random seed
    """

    def __init__(
        self,
        env: PTADirectEnv,
        lr_actor:    float = 1e-3,
        lr_critic:   float = 1e-3,
        gamma:       float = 1.0,
        tau:         float = 0.005,
        buffer_size: int   = 100_000,
        batch_size:  int   = 128,
        sigma_start: float = 0.3,
        sigma_min:   float = 0.02,
        seed: Optional[int] = None,
    ):
        _require_torch()

        self.env        = env
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.sigma_start = sigma_start
        self.sigma_min   = sigma_min

        sd = env.state_dim
        ad = env.action_dim

        self.actor        = _Actor(sd, ad)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic        = _Critic(sd, ad)
        self.critic_target = copy.deepcopy(self.critic)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = _ReplayBuffer(buffer_size)
        self.noise  = _OUNoise(ad)

        self._episode_returns: List[float] = []

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    # ------------------------------------------------------------------
    def train(
        self,
        episodes: int,
        eval_interval: int = 0,
        eval_episodes: int = 20,
        eval_seed: Optional[int] = None,
    ) -> Dict:
        decay = (self.sigma_start - self.sigma_min) / max(1, episodes - 1)
        sigma = self.sigma_start
        checkpoint_evals: List[Tuple[int, float]] = []

        for ep in range(episodes):
            ep_return = self._run_episode(sigma)
            self._episode_returns.append(ep_return)
            sigma = max(self.sigma_min, sigma - decay)

            if eval_interval > 0 and (ep + 1) % eval_interval == 0:
                res = self.evaluate_real(n_episodes=eval_episodes, seed=eval_seed)
                checkpoint_evals.append((ep + 1, res["mean"]))

        return {
            "episode_returns":  self._episode_returns,
            "checkpoint_evals": checkpoint_evals,
        }

    def evaluate_real(self, n_episodes: int = 100,
                      seed: Optional[int] = None) -> Dict:
        """Run greedy policy on a fresh PTADirectEnv (independent from training env)."""
        eval_env = PTADirectEnv(self.env.instance, seed=seed)

        def _run_one() -> float:
            obs  = eval_env.reset()
            done = False
            makespan = 0.0
            while not done:
                with torch.no_grad():
                    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = self.actor(t).squeeze(0).numpy()
                obs, reward, done = eval_env.step(action)
                makespan += -reward   # reward = -elapsed → makespan = Σ elapsed
            return makespan

        from tqdm import tqdm
        makespans = [_run_one() for _ in tqdm(range(n_episodes), desc="  ddpg_eval", leave=False)]
        arr = np.array(makespans)
        return {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min":  float(arr.min()),  "max": float(arr.max()),
            "raw":  makespans,
        }

    def n_states(self) -> str:
        """DDPG uses continuous function approximation — return network size."""
        total = sum(p.numel() for p in self.actor.parameters())
        total += sum(p.numel() for p in self.critic.parameters())
        return f"neural ({total:,} params)"

    # ------------------------------------------------------------------
    def _run_episode(self, sigma: float) -> float:
        obs = self.env.reset()
        self.noise.reset()
        ep_return = 0.0

        done = False
        while not done:
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = self.actor(t).squeeze(0).numpy()
            action = np.clip(action + self.noise.sample(sigma), -1.0, 1.0)

            obs2, reward, done = self.env.step(action)
            self.buffer.push(obs, action, reward, obs2, float(done))
            ep_return += reward
            obs = obs2

            if len(self.buffer) >= self.batch_size:
                self._update()

        return ep_return

    def _update(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.tensor(s,  dtype=torch.float32)
        a  = torch.tensor(a,  dtype=torch.float32)
        r  = torch.tensor(r,  dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        d  = torch.tensor(d,  dtype=torch.float32)

        # ── Critic update ──
        with torch.no_grad():
            a2    = self.actor_target(s2)
            q_tgt = r + self.gamma * (1 - d) * self.critic_target(s2, a2)

        q_pred = self.critic(s, a)
        loss_c = nn.functional.mse_loss(q_pred, q_tgt)
        self.opt_critic.zero_grad()
        loss_c.backward()
        self.opt_critic.step()

        # ── Actor update ──
        loss_a = -self.critic(s, self.actor(s)).mean()
        self.opt_actor.zero_grad()
        loss_a.backward()
        self.opt_actor.step()

        # ── Soft target updates ──
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
