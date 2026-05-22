"""
Learned value network for Orbit Wars strategic agent.

Replaces hand-crafted target_value() scoring with a neural value function
trained from self-play outcomes. Uses cold-start blending with heuristic
scores to avoid poor decisions early in training.

Architecture:
  - StateEncoder: extracts fixed-size feature vector from WorldModel + planet + mission
  - ValueMLP: 3-layer network that outputs a scalar value score
  - ReplayBuffer: stores (features, heuristic_value, outcome) tuples
  - ValueNetwork: orchestrates prediction, training, and heuristic blending
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
VAL_FEAT_DIM = 28
VAL_H1 = 64
VAL_H2 = 32
VAL_H3 = 16
VAL_LR = 0.003
REPLAY_CAPACITY = 8192
BATCH_SIZE = 128
BLEND_TEMPERATURE_INIT = 2.0
BLEND_TEMPERATURE_MIN = 0.05
BLEND_DECAY = 0.9995


# ── Value MLP ────────────────────────────────────────────────────────────────

class ValueMLP:
    """Three-hidden-layer MLP predicting scalar mission value."""

    def __init__(self, in_dim=VAL_FEAT_DIM, h1=VAL_H1, h2=VAL_H2, h3=VAL_H3,
                 lr=VAL_LR):
        self.lr = lr
        scale = np.sqrt(2.0)
        self.W1 = np.random.randn(in_dim, h1) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, h3) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(h3)
        self.W4 = np.random.randn(h3, 1) * np.sqrt(2.0 / h3)
        self.b4 = np.zeros(1)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = np.maximum(0, self.z3)
        self.z4 = self.a3 @ self.W4 + self.b4
        return self.z4.ravel()

    def predict(self, X):
        return self.forward(X)

    def train_step(self, X, y):
        """Single SGD step with MSE loss. Returns loss."""
        pred = self.forward(X)
        diff = pred - y
        loss = float(np.mean(diff ** 2))

        n = X.shape[0]
        dL_dz4 = (2.0 * diff / n).reshape(-1, 1)

        dW4 = self.a3.T @ dL_dz4
        db4 = dL_dz4.sum(axis=0)
        dL_da3 = dL_dz4 @ self.W4.T
        dL_dz3 = dL_da3 * (self.z3 > 0)
        dW3 = self.a2.T @ dL_dz3
        db3 = dL_dz3.sum(axis=0)
        dL_da2 = dL_dz3 @ self.W3.T
        dL_dz2 = dL_da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dL_dz2
        db2 = dL_dz2.sum(axis=0)
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (self.z1 > 0)
        dW1 = X.T @ dL_dz1
        db1 = dL_dz1.sum(axis=0)

        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss

    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]

    def set_weights(self, weights):
        (self.W1, self.b1, self.W2, self.b2,
         self.W3, self.b3, self.W4, self.b4) = weights


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size experience replay for (features, heuristic_value, outcome)."""

    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.features = deque(maxlen=capacity)
        self.heuristic_values = deque(maxlen=capacity)
        self.outcomes = deque(maxlen=capacity)

    def push(self, features, heuristic_value, outcome):
        self.features.append(features)
        self.heuristic_values.append(heuristic_value)
        self.outcomes.append(outcome)

    def sample(self, batch_size=BATCH_SIZE):
        n = min(batch_size, len(self.features))
        if n == 0:
            return None, None, None
        indices = np.random.choice(len(self.features), size=n, replace=False)
        X = np.stack([self.features[i] for i in indices])
        h = np.array([self.heuristic_values[i] for i in indices], dtype=np.float32)
        o = np.array([self.outcomes[i] for i in indices], dtype=np.float32)
        return X, h, o

    def __len__(self):
        return len(self.features)


# ── State Encoder ────────────────────────────────────────────────────────────

class StateEncoder:
    """
    Encodes (world, planet, mission_type) into a fixed-size feature vector
    suitable as input to the value network.
    """

    MISSION_TYPES = ["capture", "snipe", "swarm", "reinforce", "recapture",
                     "rescue", "crash_exploit"]
    MISSION_TO_IDX = {m: i for i, m in enumerate(MISSION_TYPES)}

    @classmethod
    def encode(cls, world, target, mission_type, arrival_turns,
               policy, modes) -> np.ndarray:
        """
        Build a 28-dim feature vector capturing planet value context.
        """
        mission_idx = cls.MISSION_TO_IDX.get(mission_type, 0)
        mission_onehot = np.zeros(len(cls.MISSION_TYPES), dtype=np.float32)
        mission_onehot[mission_idx] = 1.0

        indirect = world.indirect_feature_map.get(target.id, (0.0, 0.0, 0.0))
        my_t, enemy_t = policy.get("reaction_time_map", {}).get(
            target.id, (10**9, 10**9))

        features = np.array([
            target.production / 5.0,
            target.ships / 200.0,
            1.0 if target.owner == -1 else 0.0,
            1.0 if target.owner not in (-1, world.player) else 0.0,
            1.0 if world.is_static(target.id) else 0.0,
            1.0 if target.id in world.comet_ids else 0.0,
            arrival_turns / 100.0,
            world.remaining_steps / 500.0,
            world.my_total / max(1.0, world.my_total + world.enemy_total),
            world.my_prod / max(1.0, world.my_prod + world.enemy_prod + 1.0),
            min(my_t, 999.0) / 100.0,
            min(enemy_t, 999.0) / 100.0,
            indirect[0] / 10.0,   # friendly indirect
            indirect[1] / 10.0,   # neutral indirect
            indirect[2] / 10.0,   # enemy indirect
            1.0 if modes.get("is_behind") else 0.0,
            1.0 if modes.get("is_ahead") else 0.0,
            1.0 if modes.get("is_finishing") else 0.0,
            1.0 if modes.get("is_dominating") else 0.0,
            world.step / 500.0,
            1.0 if world.is_four_player else 0.0,
        ], dtype=np.float32)

        return np.concatenate([features, mission_onehot])


# ── Value Network ────────────────────────────────────────────────────────────

class ValueNetwork:
    """
    Learned value function for scoring missions.

    Cold-start strategy:
      - blend_temperature controls how much to trust learned vs heuristic values
      - Starts high (trust heuristic more), decays toward 0 (trust learned more)
      - Uses heuristic values as auxiliary training targets when outcome = 0.5 (neutral/unknown)
    """

    def __init__(self):
        self.mlp = ValueMLP()
        self.replay = ReplayBuffer()
        self.blend_temperature = BLEND_TEMPERATURE_INIT
        self.train_step_count = 0

    def reset(self):
        self.blend_temperature = max(
            self.blend_temperature * 0.95, BLEND_TEMPERATURE_MIN
        )

    def predict_value(self, world, target, mission_type, arrival_turns,
                      policy, modes) -> float:
        """Predict the value score for a mission."""
        feats = StateEncoder.encode(
            world, target, mission_type, arrival_turns, policy, modes
        )
        return float(self.mlp.predict(feats.reshape(1, -1))[0])

    def blend_value(self, learned_value, heuristic_value) -> float:
        """
        Blend learned and heuristic values. As training progresses,
        the blend shifts from heuristic-heavy to learned-heavy.
        """
        alpha = np.exp(-self.blend_temperature)
        alpha = max(0.05, min(0.95, alpha))
        return alpha * learned_value + (1.0 - alpha) * heuristic_value

    def record_experience(self, world, target, mission_type, arrival_turns,
                          policy, modes, heuristic_value, outcome=0.5):
        """
        Record a mission evaluation for later training.
        outcome: 1.0 = win, 0.0 = loss, 0.5 = neutral/unknown
        """
        feats = StateEncoder.encode(
            world, target, mission_type, arrival_turns, policy, modes
        )
        target_value_signal = heuristic_value * 0.7 + outcome * 0.3
        self.replay.push(feats, heuristic_value, target_value_signal)

    def train(self):
        """Perform one training step on replay buffer data."""
        if len(self.replay) < BATCH_SIZE:
            return 0.0

        X, h_vals, outcomes = self.replay.sample(BATCH_SIZE)
        if X is None:
            return 0.0

        # Training target: blend of heuristic and outcome-based signal
        target = 0.6 * h_vals + 0.4 * outcomes * np.maximum(h_vals, 1.0)

        loss = self.mlp.train_step(X, target)

        # Decay blend temperature
        self.blend_temperature = max(
            BLEND_TEMPERATURE_MIN,
            self.blend_temperature * BLEND_DECAY,
        )
        self.train_step_count += 1

        return loss

    def record_episode_outcome(self, won: bool):
        """
        Update the most recent experiences with the final outcome.
        This is called after an episode ends.
        """
        outcome_val = 1.0 if won else 0.0
        n_recent = min(len(self.replay), 256)
        if n_recent == 0:
            return
        for i in range(1, n_recent + 1):
            idx = -i
            old_outcome = self.replay.outcomes[idx]
            self.replay.outcomes[idx] = old_outcome * 0.5 + outcome_val * 0.5

    def get_weights(self):
        return {
            "mlp": self.mlp.get_weights(),
            "blend_temperature": self.blend_temperature,
            "train_step_count": self.train_step_count,
        }

    def set_weights(self, weights_dict):
        if "mlp" in weights_dict:
            self.mlp.set_weights(weights_dict["mlp"])
        self.blend_temperature = weights_dict.get(
            "blend_temperature", BLEND_TEMPERATURE_INIT
        )
        self.train_step_count = weights_dict.get("train_step_count", 0)
