"""
PPO (Proximal Policy Optimization) agent for Orbit Wars.

Architecture:
  - StateEncoder: normalizes game observations into fixed-size feature vectors
  - ActorCritic: PyTorch network with shared backbone + multi-action policy/value heads
  - RolloutBuffer: stores trajectories and computes GAE advantages
  - PPOAgent: BaseAgent subclass for acting in the environment
  - PPOTrainer: orchestrates environment interaction and PPO weight updates

Usage:
    python train.py --agent ppo --opponent random --num_episodes 500 --ppo_train
    python train.py --agent ppo --opponent heuristic --num_episodes 200 --ppo_train
"""

import logging
import math
import os
import time
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from framework.agent import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PLANETS = 40
MAX_FLEETS = 80
MAX_SHIPS = 1000
BOARD_SIZE = 100.0
# Planet features: raw 8 + reaction(3) + static/comet(2) + indirect(3) + threat(1) + is_enemy(1) = 18
PP_FEAT_DIM = 18
FL_FEAT_DIM = 6    # x, y, sin(angle), cos(angle), ships, is_mine
# Global features: raw 11 + domination(1) + modes(3) + total_threat(1) = 16
GLOBAL_DIM = 16
STATE_DIM = MAX_PLANETS * PP_FEAT_DIM + MAX_FLEETS * FL_FEAT_DIM + GLOBAL_DIM
N_RATIOS = 8
N_ACTIONS = 5       # number of parallel moves per turn

# PPO defaults
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.03
LR = 2e-4
UPDATE_EPOCHS = 10
BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def compute_shaped_reward(obs_before: Dict, obs_after: Dict, player: int,
                          terminal_reward: float = 0.0) -> float:
    """Compute intermediate reward from game state change.

    Combines a state-value delta with explicit capture/loss bonuses and
    the terminal game outcome so PPO receives informative per-step signals.
    """
    if obs_before is None or obs_after is None:
        return terminal_reward

    try:
        pb = obs_before.get("planets", []) if isinstance(obs_before, dict) else getattr(obs_before, "planets", []) or []
        pa = obs_after.get("planets", []) if isinstance(obs_after, dict) else getattr(obs_after, "planets", []) or []
        fb = obs_before.get("fleets", []) if isinstance(obs_before, dict) else getattr(obs_before, "fleets", []) or []
        fa = obs_after.get("fleets", []) if isinstance(obs_after, dict) else getattr(obs_after, "fleets", []) or []

        # Build lookup tables keyed by planet id
        pb_map = {p[0]: p for p in pb}
        pa_map = {p[0]: p for p in pa}

        # --- planet capture / loss bonus (large, sparse) ---
        cap_bonus = 0.0
        for pid, pa_planet in pa_map.items():
            pb_planet = pb_map.get(pid)
            if pb_planet is None:
                continue
            was_mine = pb_planet[1] == player
            is_mine = pa_planet[1] == player
            was_enemy = pb_planet[1] not in (-1, player)
            is_enemy = pa_planet[1] not in (-1, player)
            if not was_mine and is_mine:
                cap_bonus += 1.0 + pa_planet[6] * 0.3    # captured! reward production
            elif was_mine and not is_mine:
                cap_bonus -= 1.0 + pb_planet[6] * 0.3    # lost planet
            elif was_enemy and not is_enemy:
                cap_bonus += 0.3                           # enemy lost a planet (neutralised)

        # --- continuous score (equal weight on garrison & fleet — no launch penalty) ---
        def score(planets, fleets, player):
            s = 0.0
            for p in planets:
                _, owner, _, _, _, ships, prod = p[:7]
                if owner == player:
                    s += ships * 1.0
                    s += prod * 5.0
                    s += 10.0                 # planet control
            for f in fleets:
                _, owner, _, _, _, _, ships = f[:7]
                if owner == player:
                    s += ships * 1.0          # fleet ships equal to garrison ships
            return s

        score_before = score(pb, fb, player)
        score_after = score(pa, fa, player)
        delta = (score_after - score_before) / 10.0

        # Fleet loss penalty: ships lost without gaining planets (sun/edge/failed attacks)
        my_fleet_before = sum(f[6] for f in fb if f[1] == player)
        my_fleet_after = sum(f[6] for f in fa if f[1] == player)
        fleet_loss = my_fleet_before - my_fleet_after
        gained_planets = sum(1 for pid, pa_p in pa_map.items()
                             if pb_map.get(pid) and pb_map[pid][1] != player and pa_p[1] == player)
        fleet_penalty = fleet_loss * 0.01 if fleet_loss > 0 and gained_planets == 0 else 0.0

        reward = delta + cap_bonus + terminal_reward - fleet_penalty
        return max(-2.0, min(2.0, reward))
    except Exception:
        return terminal_reward


# ---------------------------------------------------------------------------
# Strategic feature extraction (from WorldModel approach)
# ---------------------------------------------------------------------------

def _fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + 5.0 * (ratio ** 1.5)


def _is_static_planet(x: float, y: float, radius: float) -> bool:
    return math.hypot(x - 50.0, y - 50.0) + radius >= 50.0


def _estimate_travel_turns(sx, sy, tx, ty, ships):
    d = math.hypot(tx - sx, ty - sy)
    return max(1, int(math.ceil(d / _fleet_speed(max(1, ships)))))


def _compute_reaction_times(planets, my_planets, enemy_planets):
    """For each planet, compute the minimum turns until my / enemy forces can arrive."""
    my_times = {}
    enemy_times = {}
    for p in planets:
        pid, px, py = p[0], p[2], p[3]
        mt = 999
        for mp in my_planets:
            mt = min(mt, _estimate_travel_turns(mp[2], mp[3], px, py, max(1, mp[5] // 3)))
        my_times[pid] = mt

        et = 999
        for ep in enemy_planets:
            et = min(et, _estimate_travel_turns(ep[2], ep[3], px, py, max(1, ep[5] // 3)))
        enemy_times[pid] = et
    return my_times, enemy_times


def _compute_indirect_influence(planet, all_planets, player):
    """Friendly / neutral / enemy production influence weighted by distance."""
    px, py = planet[2], planet[3]
    friendly = neutral = enemy = 0.0
    for other in all_planets:
        if other[0] == planet[0]:
            continue
        d = math.hypot(other[2] - px, other[3] - py)
        if d < 1:
            continue
        factor = other[6] / (d + 12.0)
        if other[1] == player:
            friendly += factor
        elif other[1] == -1:
            neutral += factor
        else:
            enemy += factor
    return friendly, neutral, enemy


def _compute_incoming_threat(planet, fleets, player):
    """Ratio of enemy fleet ships heading roughly toward this planet to garrison."""
    px, py = planet[2], planet[3]
    incoming = 0
    for f in fleets:
        if f[1] == player or f[1] == -1:
            continue
        fx, fy, angle = f[2], f[3], f[4]
        to_planet = math.atan2(py - fy, px - fx)
        diff = abs(angle - to_planet)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        if diff < math.pi / 4 and math.hypot(px - fx, py - fy) < 50:
            incoming += f[6]
    return incoming / max(1, planet[5])


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

class StateEncoder:
    """Convert raw Orbit Wars observations to a fixed-size feature vector
    with strategic features (reaction times, indirect wealth, domination)."""

    @staticmethod
    def encode(obs: Union[Dict, Any], turn: int = 0) -> np.ndarray:
        if isinstance(obs, dict):
            player = obs.get("player", 0)
            raw_planets = obs.get("planets", []) or []
            raw_fleets = obs.get("fleets", []) or []
            comet_ids = set(obs.get("comet_planet_ids", []) or [])
            initial_planets = obs.get("initial_planets", []) or []
        else:
            player = obs.player
            raw_planets = obs.planets or []
            raw_fleets = obs.fleets or []
            comet_ids = set(getattr(obs, "comet_planet_ids", []) or [])
            initial_planets = getattr(obs, "initial_planets", []) or []

        my_planets = [p for p in raw_planets if p[1] == player]
        enemy_planets = [p for p in raw_planets if p[1] not in (-1, player)]
        neutral_planets = [p for p in raw_planets if p[1] == -1]

        # ---- strategic computations (once, shared across features) ----
        my_times, enemy_times = _compute_reaction_times(raw_planets, my_planets, enemy_planets)

        # Domination metrics
        my_ships = sum(p[5] for p in my_planets)
        enemy_ships = sum(p[5] for p in enemy_planets)
        my_fleet_ships = sum(f[6] for f in raw_fleets if f[1] == player)
        enemy_fleet_ships = sum(f[6] for f in raw_fleets if f[1] not in (-1, player))
        my_total = my_ships + my_fleet_ships
        enemy_total = enemy_ships + enemy_fleet_ships
        domination = (my_total - enemy_total) / max(1, my_total + enemy_total)
        my_prod = sum(p[6] for p in my_planets)
        enemy_prod = sum(p[6] for p in enemy_planets)
        is_ahead = float(domination > 0.18)
        is_behind = float(domination < -0.20)
        is_finishing = float(domination > 0.30 and my_prod > enemy_prod * 1.25)

        # ---- planet features (PP_FEAT_DIM = 18) ----
        p_arr = np.zeros((MAX_PLANETS, PP_FEAT_DIM), dtype=np.float32)
        for i, p in enumerate(raw_planets[:MAX_PLANETS]):
            pid, owner, x, y, radius, ships, prod = p[:7]
            center_dist = math.hypot(x - 50.0, y - 50.0) / 70.0
            mt = my_times.get(pid, 999)
            et = enemy_times.get(pid, 999)
            react_adv = np.clip((et - mt) / 30.0, -1.0, 1.0)  # +ve = I arrive first
            friendly, neutral, enemy_inf = _compute_indirect_influence(p, raw_planets, player)
            threat = _compute_incoming_threat(p, raw_fleets, player) if owner == player else 0.0
            p_arr[i] = [
                x / BOARD_SIZE,                    # 0
                y / BOARD_SIZE,                    # 1
                radius / 10.0,                     # 2
                ships / MAX_SHIPS,                 # 3
                prod / 5.0,                        # 4
                (owner + 1) / 4.0,                 # 5  owner proxy
                float(owner == player),            # 6  is_mine
                center_dist,                       # 7
                np.clip(mt / 100.0, 0.0, 1.0),    # 8  my reaction time
                np.clip(et / 100.0, 0.0, 1.0),    # 9  enemy reaction time
                react_adv,                         # 10 reaction advantage
                float(_is_static_planet(x, y, radius)),  # 11
                float(pid in comet_ids),           # 12 is_comet
                np.clip(friendly, 0.0, 1.0),      # 13 indirect friendly
                np.clip(neutral, 0.0, 1.0),       # 14 indirect neutral
                np.clip(enemy_inf, 0.0, 1.0),     # 15 indirect enemy
                np.clip(threat, 0.0, 1.0),        # 16 incoming threat ratio
                float(owner not in (-1, player)),  # 17 is_enemy
            ]

        # ---- fleet features (sin/cos for angle) ----
        f_arr = np.zeros((MAX_FLEETS, FL_FEAT_DIM), dtype=np.float32)
        for i, f in enumerate(raw_fleets[:MAX_FLEETS]):
            fid, owner, x, y, angle, from_pid, ships = f[:7]
            f_arr[i] = [
                x / BOARD_SIZE,
                y / BOARD_SIZE,
                math.sin(angle),
                math.cos(angle),
                ships / MAX_SHIPS,
                float(owner == player),
            ]

        # ---- global features (GLOBAL_DIM = 16) ----
        # Find best neutral target
        best_neutral_prod = 0.0
        best_neutral_ships = 0.0
        closest_neutral_dist = 1.0
        best_neutral_react = 0.0
        if my_planets and neutral_planets:
            nn = min(neutral_planets, key=lambda np_: min(
                math.hypot(np_[2] - mp[2], np_[3] - mp[3]) for mp in my_planets))
            best_neutral_prod = nn[6] / 5.0
            best_neutral_ships = nn[5] / MAX_SHIPS
            closest_neutral_dist = min(
                math.hypot(nn[2] - mp[2], nn[3] - mp[3]) / 70.0 for mp in my_planets)
            # Reaction advantage for the best neutral target
            nid = nn[0]
            best_neutral_react = np.clip((enemy_times.get(nid, 999) - my_times.get(nid, 999)) / 30.0, -1.0, 1.0)

        total_threat = sum(_compute_incoming_threat(p, raw_fleets, player) for p in my_planets)

        g_arr = np.array([
            player / 3.0,
            len(my_planets) / MAX_PLANETS,
            len(enemy_planets) / MAX_PLANETS,
            my_total / MAX_SHIPS,
            enemy_total / MAX_SHIPS,
            my_prod / 20.0,
            enemy_prod / 20.0,
            turn / 500.0,
            best_neutral_prod,
            best_neutral_ships,
            closest_neutral_dist,
            np.clip(domination, -1.0, 1.0),   # 11 domination ratio
            is_ahead,                           # 12
            is_behind,                          # 13
            is_finishing,                       # 14
            np.clip(total_threat, 0.0, 1.0),   # 15 total incoming threat
        ], dtype=np.float32)

        return np.concatenate([p_arr.ravel(), f_arr.ravel(), g_arr])

    @staticmethod
    def get_dim() -> int:
        return STATE_DIM


# ---------------------------------------------------------------------------
# Actor-Critic network (multi-action)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-backbone actor-critic with multi-action output.

    Outputs N_ACTIONS independent (source, target, ratio) selections,
    plus a scalar value estimate.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = N_ACTIONS

        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Multi-action policy heads
        self.source_heads = nn.ModuleList([nn.Linear(hidden_dim, MAX_PLANETS) for _ in range(N_ACTIONS)])
        self.target_heads = nn.ModuleList([nn.Linear(hidden_dim, MAX_PLANETS) for _ in range(N_ACTIONS)])
        self.ratio_heads = nn.ModuleList([nn.Linear(hidden_dim, N_RATIOS) for _ in range(N_ACTIONS)])

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.value_head:
                nn.init.orthogonal_(m.weight, gain=1.41)
                nn.init.constant_(m.bias, 0)
        if hasattr(self, 'value_head'):
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.constant_(self.value_head.bias, 0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.value_head(h)

    def reset_value_head(self):
        """Reinitialize the value head — use when switching opponents to prevent
        stale optimistic value predictions from corrupting GAE advantages."""
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)

    def _forward_heads(self, h: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """Return lists of logits per action head."""
        src_logits_list = [head(h) for head in self.source_heads]
        tgt_logits_list = [head(h) for head in self.target_heads]
        ratio_logits_list = [head(h) for head in self.ratio_heads]
        value = self.value_head(h)
        if src_mask is not None:
            src_logits_list = [sl.masked_fill(src_mask == 0, -1e9) for sl in src_logits_list]
        if tgt_mask is not None:
            tgt_logits_list = [tl.masked_fill(tgt_mask == 0, -1e9) for tl in tgt_logits_list]
        return src_logits_list, tgt_logits_list, ratio_logits_list, value

    def sample_actions(
        self, x: torch.Tensor,
        src_mask: torch.Tensor, tgt_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample N_ACTIONS independent moves.

        Returns list of (src_idx, tgt_idx, ratio_idx, log_prob) per action.
        """
        h = self.backbone(x)
        src_logits, tgt_logits, ratio_logits, _ = self._forward_heads(h, src_mask, tgt_mask)

        results = []
        for i in range(self.n_actions):
            sl, tl, rl = src_logits[i], tgt_logits[i], ratio_logits[i]
            if deterministic:
                si = sl.argmax(dim=-1)
                ti = tl.argmax(dim=-1)
                ri = rl.argmax(dim=-1)
                lp = torch.zeros(x.size(0), device=x.device)
            else:
                si = torch.distributions.Categorical(logits=sl).sample()
                ti = torch.distributions.Categorical(logits=tl).sample()
                ri = torch.distributions.Categorical(logits=rl).sample()
                # log prob for this action
                slp = F.log_softmax(sl, dim=-1).gather(1, si.unsqueeze(1)).squeeze(1)
                tlp = F.log_softmax(tl, dim=-1).gather(1, ti.unsqueeze(1)).squeeze(1)
                rlp = F.log_softmax(rl, dim=-1).gather(1, ri.unsqueeze(1)).squeeze(1)
                lp = slp + tlp + rlp
            results.append((si, ti, ri, lp))
        return results

    def evaluate_actions(
        self, x: torch.Tensor,
        src_indices: List[torch.Tensor],
        tgt_indices: List[torch.Tensor],
        ratio_indices: List[torch.Tensor],
        src_mask: torch.Tensor, tgt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-prob, entropy, and value for stored actions."""
        h = self.backbone(x)
        src_logits, tgt_logits, ratio_logits, value = self._forward_heads(h, src_mask, tgt_mask)

        total_log_prob = 0.0
        total_entropy = 0.0

        def _ent(logits):
            p = F.softmax(logits, dim=-1)
            return -(p * F.log_softmax(logits, dim=-1)).sum(-1)

        for i in range(self.n_actions):
            sl, tl, rl = src_logits[i], tgt_logits[i], ratio_logits[i]
            si, ti, ri = src_indices[i], tgt_indices[i], ratio_indices[i]

            slp = F.log_softmax(sl, dim=-1).gather(1, si.unsqueeze(1)).squeeze(1)
            tlp = F.log_softmax(tl, dim=-1).gather(1, ti.unsqueeze(1)).squeeze(1)
            rlp = F.log_softmax(rl, dim=-1).gather(1, ri.unsqueeze(1)).squeeze(1)
            total_log_prob = total_log_prob + slp + tlp + rlp
            total_entropy = total_entropy + _ent(sl) + _ent(tl) + _ent(rl)

        return total_log_prob, total_entropy, value.squeeze(1)


# ---------------------------------------------------------------------------
# Rollout buffer (GAE)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states: List[np.ndarray] = []
        # Store actions as lists per step
        self.actions_src: List[List[int]] = []
        self.actions_tgt: List[List[int]] = []
        self.actions_ratio: List[List[int]] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.src_masks: List[np.ndarray] = []
        self.tgt_masks: List[np.ndarray] = []
        self.returns: Optional[np.ndarray] = None
        self.advantages: Optional[np.ndarray] = None
        return self

    def store(
        self, state,
        src_list: List[int], tgt_list: List[int], ratio_list: List[int],
        reward, done, value, log_prob, src_mask, tgt_mask,
    ):
        self.states.append(state)
        self.actions_src.append(src_list)
        self.actions_tgt.append(tgt_list)
        self.actions_ratio.append(ratio_list)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.src_masks.append(src_mask)
        self.tgt_masks.append(tgt_mask)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        self.returns = returns
        self.advantages = advantages
        return returns, advantages

    def get_batches(self, batch_size: int) -> List[Dict[str, Any]]:
        n = len(self.states)
        idx = np.random.permutation(n)
        adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ix = idx[start:end]
            batches.append({
                "state": torch.FloatTensor(np.array(self.states)[ix]),
                "src_indices": [torch.LongTensor(np.array([s[i] for s in self.actions_src])[ix])
                                for i in range(N_ACTIONS)],
                "tgt_indices": [torch.LongTensor(np.array([s[i] for s in self.actions_tgt])[ix])
                                for i in range(N_ACTIONS)],
                "ratio_indices": [torch.LongTensor(np.array([s[i] for s in self.actions_ratio])[ix])
                                  for i in range(N_ACTIONS)],
                "log_prob_old": torch.FloatTensor(np.array(self.log_probs)[ix]),
                "return": torch.FloatTensor(self.returns[ix]),
                "advantage": torch.FloatTensor(adv[ix]),
                "src_mask": torch.FloatTensor(np.array(self.src_masks)[ix]),
                "tgt_mask": torch.FloatTensor(np.array(self.tgt_masks)[ix]),
            })
        return batches

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_src_mask(raw_planets: List, player: int) -> np.ndarray:
    mask = np.zeros(MAX_PLANETS, dtype=np.float32)
    for i, p in enumerate(raw_planets[:MAX_PLANETS]):
        if p[1] == player and p[5] > 1:  # need at least 2 ships to launch
            mask[i] = 1.0
    if mask.sum() < 0.5:
        mask[0] = 1.0
    return mask


def make_tgt_mask(raw_planets: List, player: int) -> np.ndarray:
    mask = np.ones(MAX_PLANETS, dtype=np.float32)
    for i, p in enumerate(raw_planets[:MAX_PLANETS]):
        if p[1] == player:
            mask[i] = 0.0
    if mask.sum() < 0.5:
        mask[0] = 1.0
    return mask


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """PPO agent with multi-action output."""

    def __init__(
        self,
        player_id: int = 0,
        device: str = "auto",
        load_path: Optional[str] = None,
        train_mode: bool = False,
    ):
        super().__init__(player_id)
        self.name = "PPOAgent"
        self.train_mode = train_mode
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.feature_dim = StateEncoder.get_dim()
        self.policy = ActorCritic(self.feature_dim).to(self.device)
        if load_path is not None:
            self.load(load_path)
        self.buffer = RolloutBuffer()
        self.turn = 0
        self._pending = {}
        self._cached_reward = 0.0
        self._cached_done = False

    def reset(self):
        self.turn = 0
        self._pending = {}
        self._cached_reward = 0.0
        self._cached_done = False

    def act(self, observation: Dict) -> List[List]:
        self.turn += 1

        if isinstance(observation, dict):
            player = observation.get("player", 0)
            raw_planets = observation.get("planets", []) or []
        else:
            player = observation.player
            raw_planets = observation.planets or []

        my_planets = [p for p in raw_planets if p[1] == player]
        if not my_planets:
            return []

        # Encode state
        state = StateEncoder.encode(observation, self.turn)
        src_mask = make_src_mask(raw_planets, player)
        tgt_mask = make_tgt_mask(raw_planets, player)

        # Store previous pending transition
        if self.train_mode and self._pending:
            self.buffer.store(
                self._pending["state"],
                self._pending["src_list"],
                self._pending["tgt_list"],
                self._pending["ratio_list"],
                self._cached_reward,
                self._cached_done,
                self._pending["value"],
                self._pending["log_prob"],
                self._pending["src_mask"],
                self._pending["tgt_mask"],
            )

        # Sample actions
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            sm_t = torch.FloatTensor(src_mask).unsqueeze(0).to(self.device)
            tm_t = torch.FloatTensor(tgt_mask).unsqueeze(0).to(self.device)

            action_results = self.policy.sample_actions(
                s_t, sm_t, tm_t, deterministic=not self.train_mode,
            )
            value = self.policy.get_value(s_t).squeeze().cpu().item()

        # Collect moves, de-duplicating source planets
        moves = []
        src_list, tgt_list, ratio_list = [], [], []
        total_log_prob = 0.0
        used_source_idx = set()

        for src_idx, tgt_idx, ratio_idx, log_prob in action_results:
            si = src_idx.cpu().item()
            ti = tgt_idx.cpu().item()
            ri = ratio_idx.cpu().item()
            total_log_prob += log_prob.cpu().item()

            src_list.append(si)
            tgt_list.append(ti)
            ratio_list.append(ri)

            # Skip duplicate sources
            if si not in used_source_idx and si < len(raw_planets):
                move = self._indices_to_move(si, ti, ri, raw_planets, player)
                if move:
                    moves.extend(move)
                    used_source_idx.add(si)

        # Save pending
        self._pending = dict(
            state=state,
            src_list=src_list, tgt_list=tgt_list, ratio_list=ratio_list,
            value=value, log_prob=total_log_prob,
            src_mask=src_mask.copy(), tgt_mask=tgt_mask.copy(),
        )

        return moves

    def _indices_to_move(self, src_ix: int, tgt_ix: int, ratio_bin: int,
                          raw_planets: List, player: int) -> Optional[List[List]]:
        if src_ix >= len(raw_planets) or tgt_ix >= len(raw_planets):
            return None
        src_p = raw_planets[src_ix]
        tgt_p = raw_planets[tgt_ix]
        if src_p[1] != player:
            return None
        from_id = src_p[0]
        angle = math.atan2(tgt_p[3] - src_p[3], tgt_p[2] - src_p[2])
        ratios = [0.05, 0.12, 0.22, 0.35, 0.50, 0.65, 0.80, 1.0]
        ratio = ratios[min(ratio_bin, N_RATIOS - 1)]
        ships = max(1, int(src_p[5] * ratio))
        if ships >= src_p[5]:
            ships = src_p[5] - 1
        if ships <= 0:
            return None
        return [[from_id, angle, ships]]

    def handle_terminal(self):
        if self.train_mode and self._pending:
            self.buffer.store(
                self._pending["state"],
                self._pending["src_list"],
                self._pending["tgt_list"],
                self._pending["ratio_list"],
                self._cached_reward,
                self._cached_done,
                self._pending["value"],
                self._pending["log_prob"],
                self._pending["src_mask"],
                self._pending["tgt_mask"],
            )
            self._pending = {}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        logger.info(f"Model loaded from {path}")

    def reset_value_head(self):
        """Reinitialize value head — call when switching to harder opponents."""
        self.policy.reset_value_head()


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Orchestrates environment interaction and PPO policy updates."""

    def __init__(
        self,
        env,
        agent: PPOAgent,
        opponent_agents: Optional[List[BaseAgent]] = None,
        opponent_probs: Optional[List[float]] = None,
        lr: float = LR,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
        clip_eps: float = CLIP_EPS,
        vf_coef: float = VF_COEF,
        ent_coef: float = ENT_COEF,
        update_epochs: int = UPDATE_EPOCHS,
        batch_size: int = BATCH_SIZE,
        max_grad_norm: float = MAX_GRAD_NORM,
        target_kl: float = 0.02,
        log_dir: str = "./log",
        device: str = "auto",
    ):
        self.env = env
        self.agent = agent
        self.opponent_agents = opponent_agents or []
        if opponent_probs is not None:
            total = sum(opponent_probs)
            self.opponent_probs = [p / total for p in opponent_probs]
        else:
            self.opponent_probs = None
        self.all_agents = [agent] + self.opponent_agents

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.log_dir = log_dir
        self.total_steps = 0
        self.episode_count = 0
        self._prev_obs: Optional[Dict] = None

    def _select_opponent(self):
        """Sample one opponent from the pool (with optional probability weights)."""
        if len(self.opponent_agents) <= 1:
            return self.opponent_agents[0] if self.opponent_agents else None
        idx = np.random.choice(len(self.opponent_agents), p=self.opponent_probs)
        return self.opponent_agents[idx]

    def _run_episode(self, max_steps: int = 500, render: bool = False) -> Dict:
        # Select opponent for this episode (supports opponent mixing)
        opponent = self._select_opponent()
        if opponent is not None and opponent is not self.all_agents[-1]:
            self.all_agents = [self.agent, opponent]

        observations = self.env.reset()
        if observations is None:
            observations = [None] * len(self.all_agents)

        for a in self.all_agents:
            a.reset()

        for i, a in enumerate(self.all_agents):
            pid = self.env.get_player_id(i)
            if pid is not None:
                a.player_id = pid

        self.agent.train_mode = True
        self.agent._cached_reward = 0.0
        self.agent._cached_done = False
        self._prev_obs = None

        step = 0
        total_rewards = [0.0] * len(self.all_agents)
        done = False

        while step < max_steps and not done:
            actions = []
            for i, agent in enumerate(self.all_agents):
                obs = observations[i] if i < len(observations) else None
                actions.append([] if obs is None else agent.act(obs))

            obs_before = observations[0] if observations else None
            observations, rewards, dones, info = self.env.step(actions)
            obs_after = observations[0] if observations else None

            # Compute shaped reward (delta in state score + terminal outcome)
            player = self.agent.player_id
            terminal = rewards[0] if dones and dones[0] else 0.0
            shaped = compute_shaped_reward(obs_before, obs_after, player, terminal)
            self.agent._cached_reward = shaped
            self.agent._cached_done = dones[0] if dones else False

            for i, r in enumerate(rewards):
                total_rewards[i] += r

            done = all(dones) or any(dones)
            step += 1

            if render and step % 100 == 0:
                self.env.render(mode="ipython", width=800, height=600)

        self.agent.handle_terminal()
        self.total_steps += step
        self.episode_count += 1

        return {
            "steps": step,
            "total_rewards": total_rewards,
            "agent_reward": total_rewards[0] if total_rewards else 0,
            "agent_name": self.agent.name,
            "transitions": len(self.agent.buffer),
        }

    def _update_policy(self):
        buf = self.agent.buffer
        if len(buf) < 2:
            return

        policy = self.agent.policy
        policy.train()

        with torch.no_grad():
            if self.agent._pending:
                last_s = torch.FloatTensor(self.agent._pending["state"]).unsqueeze(0).to(self.device)
                last_val = policy.get_value(last_s).cpu().item()
            else:
                last_val = 0.0
        buf.compute_gae(last_val, self.gamma, self.gae_lambda)

        total_loss = 0.0
        approx_kl = 0.0
        epochs_run = 0
        kl_threshold = self.target_kl * 1.5

        for epoch in range(self.update_epochs):
            batches = buf.get_batches(self.batch_size)
            epoch_loss = 0.0
            epoch_kl = 0.0
            for batch in batches:
                state = batch["state"].to(self.device)
                src_indices = [b.to(self.device) for b in batch["src_indices"]]
                tgt_indices = [b.to(self.device) for b in batch["tgt_indices"]]
                ratio_indices = [b.to(self.device) for b in batch["ratio_indices"]]
                old_lp = batch["log_prob_old"].to(self.device)
                return_ = batch["return"].to(self.device)
                adv = batch["advantage"].to(self.device)
                src_mask = batch["src_mask"].to(self.device)
                tgt_mask = batch["tgt_mask"].to(self.device)

                log_prob, entropy, value = policy.evaluate_actions(
                    state, src_indices, tgt_indices, ratio_indices, src_mask, tgt_mask,
                )

                ratio = torch.exp(log_prob - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, return_)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    epoch_kl += (log_prob - old_lp).mean().item()

            epoch_kl_avg = epoch_kl / max(len(batches), 1)
            approx_kl = epoch_kl_avg
            total_loss += epoch_loss / max(len(batches), 1)
            epochs_run += 1

            # KL early stopping: prevent destructive policy updates
            if epoch_kl_avg > kl_threshold:
                logger.debug(f"KL early stop at epoch {epoch+1}: kl={epoch_kl_avg:.4f} > {kl_threshold:.4f}")
                break

        avg_loss = total_loss / epochs_run
        explained_var = self._explained_variance(buf)
        logger.info(
            f"PPO update | steps={self.total_steps} | loss={avg_loss:.4f} | "
            f"kl={approx_kl:.4f} | ep={epochs_run}/{self.update_epochs} | "
            f"ev={explained_var:.4f} | buf={len(buf)}"
        )

        buf.reset()
        policy.eval()

    def _explained_variance(self, buf: RolloutBuffer) -> float:
        if buf.returns is None or len(buf.returns) < 2:
            return 0.0
        var_y = buf.returns.var()
        if var_y < 1e-8:
            return 1.0
        return 1.0 - ((buf.returns - np.array(buf.values)) ** 2).mean() / var_y

    def train(self, num_episodes: int = 100, render_every: int = 0,
              update_frequency: int = 1, save_every: int = 50,
              save_dir: str = "./checkpoints") -> List[Dict]:
        os.makedirs(save_dir, exist_ok=True)
        results = []
        start = time.time()
        best_reward = -float("inf")

        logger.info(f"PPO training: {num_episodes} episodes, "
                    f"agent={self.agent.name}, "
                    f"opponents={[a.name for a in self.opponent_agents]}")

        # tqdm progress bar (optional)
        try:
            from tqdm import tqdm
            pbar = tqdm(total=num_episodes, desc="PPO", unit="ep", dynamic_ncols=True)
            _use_pbar = True
        except ImportError:
            pbar = None
            _use_pbar = False

        for ep in range(1, num_episodes + 1):
            render = render_every > 0 and ep % render_every == 0
            ep_result = self._run_episode(render=render)
            ep_result["episode"] = ep
            results.append(ep_result)

            if ep % update_frequency == 0 and len(self.agent.buffer) > 0:
                self._update_policy()

            avg_reward = float(np.mean([r["agent_reward"] for r in results[-50:]]))
            logger.info(
                f"Ep {ep}/{num_episodes} | "
                f"rew={ep_result['agent_reward']:.2f} | "
                f"avg={avg_reward:.2f} | "
                f"steps={ep_result['steps']} | "
                f"buf={ep_result['transitions']}"
            )

            if _use_pbar:
                win_rate = float(np.mean([1.0 if r["agent_reward"] > 0 else 0.0
                                          for r in results[-100:]]))
                pbar.set_postfix(
                    rew=f"{ep_result['agent_reward']:.1f}",
                    avg50=f"{avg_reward:.2f}",
                    wr=f"{win_rate:.1%}",
                )
                pbar.update(1)

            if save_every > 0 and ep % save_every == 0:
                self.agent.save(os.path.join(save_dir, f"ppo_ep{ep}.pt"))

            if ep_result["agent_reward"] > best_reward:
                best_reward = ep_result["agent_reward"]
                self.agent.save(os.path.join(save_dir, "ppo_best.pt"))

        if _use_pbar:
            pbar.close()

        total = time.time() - start
        avg_r = float(np.mean([r["agent_reward"] for r in results]))
        logger.info(f"Training done: {num_episodes} episodes in {total:.1f}s, "
                    f"avg_reward={avg_r:.2f}")
        self.agent.save(os.path.join(save_dir, "ppo_final.pt"))
        return results


# ---------------------------------------------------------------------------
# Kaggle-compatible entry point
# ---------------------------------------------------------------------------

_agent_instance: Optional[PPOAgent] = None
_agent_model_path: Optional[str] = None
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_weights_path(path: str) -> str:
    if path is None:
        return path
    if os.path.exists(path):
        return path
    alt = os.path.join(_MODULE_DIR, os.path.basename(path))
    if os.path.exists(alt):
        return alt
    alt = os.path.join(os.getcwd(), path)
    if os.path.exists(alt):
        return alt
    return path


def set_model_path(path: str):
    global _agent_model_path
    _agent_model_path = _resolve_weights_path(path)


def ppo_agent(obs, config=None) -> List[List]:
    global _agent_instance
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if _agent_instance is None or _agent_instance.player_id != player:
        _agent_instance = PPOAgent(
            player_id=player, load_path=_agent_model_path, train_mode=False,
        )
    try:
        return _agent_instance.act(obs)
    except Exception as e:
        import sys, traceback
        print(f"PPO agent error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []
