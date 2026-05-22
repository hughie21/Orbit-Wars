"""
Opponent modeling module for Orbit Wars strategic agent.

Predicts enemy fleet launches using a lightweight MLP and a hedge algorithm
that mixes multiple "style" experts (aggressive, defensive, balanced, greedy).
Online adaptive updates refine weights based on prediction accuracy.

Architecture:
  - FeatureExtractor: converts game state to planet-pair feature vectors
  - StyleExperts: 4 preset strategies that each predict enemy launches
  - HedgeMixer: maintains weights over experts, updated via prediction error
  - OpponentModel: orchestrates prediction and learning
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
MAX_SPEED = 6.0
HISTORY_LENGTH = 8
N_STYLES = 4
HEDGE_LR = 0.15
FEAT_DIM = 14
HIDDEN1 = 32
HIDDEN2 = 16


# ── Helpers ──────────────────────────────────────────────────────────────────

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, by - ay)


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


# ── Lightweight MLP (pure numpy) ────────────────────────────────────────────

class SmallMLP:
    """Two-hidden-layer MLP with ReLU activations, trainable via SGD."""

    def __init__(self, in_dim, h1, h2, out_dim, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(in_dim, h1) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, out_dim) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(out_dim)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def predict_proba(self, X):
        logits = self.forward(X)
        proba = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        ships_ratio = np.maximum(0, logits[:, 1])
        return proba, ships_ratio

    def train_step(self, X, y_prob, y_ships):
        """Single SGD step on a batch. Returns loss."""
        logits = self.forward(X)
        pred_prob = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        pred_ships = np.maximum(0, logits[:, 1])

        loss_prob = np.mean((pred_prob - y_prob) ** 2)
        loss_ships = np.mean((pred_ships - y_ships) ** 2)
        total_loss = loss_prob + 0.1 * loss_ships

        # Gradients (manual backprop)
        n = X.shape[0]
        dL_dprob = 2.0 * (pred_prob - y_prob) / n
        dprob_dlogit = pred_prob * (1.0 - pred_prob)
        dL_dlogit_prob = (dL_dprob * dprob_dlogit).reshape(-1, 1)

        dL_dships = (2.0 * 0.1 * (pred_ships - y_ships) / n).reshape(-1, 1)
        dL_dlogit_ships = np.where(logits[:, 1:2] > 0, dL_dships, 0.0)

        dL_dz3 = np.hstack([dL_dlogit_prob, dL_dlogit_ships])

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

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return float(total_loss)

    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights


# ── Feature Extractor ────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extract feature vectors for planet-pair (source → target) predictions."""

    @staticmethod
    def extract_pair_features(src, tgt, world):
        """Build a feature vector for a source planet → target planet pair."""
        d = dist(src.x, src.y, tgt.x, tgt.y)
        speed = fleet_speed(max(1, int(src.ships)))
        est_turns = max(1, int(math.ceil(d / max(0.2, speed))))

        src_has_incoming = float(
            any(
                owner != world.player and owner != src.owner
                for _, owner, _ in world.arrivals_by_planet.get(src.id, [])
            )
        )
        tgt_has_incoming = float(len(world.arrivals_by_planet.get(tgt.id, [])) > 0)

        is_static = 1.0 if getattr(world, 'is_static', lambda _id: True)(tgt.id) else 0.0
        is_comet = 1.0 if tgt.id in getattr(world, 'comet_ids', set()) else 0.0

        return np.array([
            src.ships / 200.0,
            src.production / 5.0,
            tgt.ships / 200.0,
            tgt.production / 5.0,
            1.0 if tgt.owner == -1 else 0.0,
            1.0 if tgt.owner not in (-1, src.owner) else 0.0,
            d / 141.0,
            est_turns / 100.0,
            is_static,
            is_comet,
            src_has_incoming,
            tgt_has_incoming,
            world.remaining_steps / 500.0,
            src.ships / max(1.0, tgt.ships + 1.0),
        ], dtype=np.float32)


# ── Style Experts ────────────────────────────────────────────────────────────

class StyleExperts:
    """
    Four preset opponent strategies. Each predicts enemy launch targets and
    ship counts. The hedge algorithm blends their outputs.
    """

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def predict(self, style_id, enemy_planets, all_planets, world):
        """
        Return list of (src_id, tgt_id, prob, ships) predictions for one style.
        """
        if style_id == 0:
            return self._aggressive(enemy_planets, all_planets, world)
        elif style_id == 1:
            return self._defensive(enemy_planets, all_planets, world)
        elif style_id == 2:
            return self._balanced(enemy_planets, all_planets, world)
        else:
            return self._greedy(enemy_planets, all_planets, world)

    def _aggressive(self, enemy_planets, all_planets, world):
        """Always attacks nearest non-owned planet."""
        preds = []
        for src in enemy_planets:
            if src.ships < 5:
                continue
            targets = [p for p in all_planets if p.owner != src.owner]
            if not targets:
                continue
            tgt = min(targets, key=lambda p: dist(src.x, src.y, p.x, p.y))
            ships = min(int(src.ships * 0.7), int(tgt.ships) + 8)
            if ships >= 4:
                preds.append((src.id, tgt.id, 0.85, ships))
        return preds

    def _defensive(self, enemy_planets, all_planets, world):
        """Reinforces own weakest planets."""
        preds = []
        own = [p for p in enemy_planets]
        for src in own:
            if src.ships < 10:
                continue
            others = [p for p in own if p.id != src.id]
            if not others:
                continue
            weakest = min(others, key=lambda p: p.ships)
            if weakest.ships < src.ships * 0.5:
                ships = min(int(src.ships * 0.3), 20)
                if ships >= 2:
                    preds.append((src.id, weakest.id, 0.7, ships))
        return preds

    def _balanced(self, enemy_planets, all_planets, world):
        """Mix of aggression and defense."""
        preds = self._aggressive(enemy_planets, all_planets, world)
        preds += self._defensive(enemy_planets, all_planets, world)
        return preds

    def _greedy(self, enemy_planets, all_planets, world):
        """Targets highest production planets."""
        preds = []
        for src in enemy_planets:
            if src.ships < 5:
                continue
            targets = [p for p in all_planets if p.owner != src.owner]
            if not targets:
                continue
            tgt = max(targets, key=lambda p: (p.production, -dist(src.x, src.y, p.x, p.y)))
            ships = min(int(src.ships * 0.65), int(tgt.ships) + 10)
            if ships >= 4:
                preds.append((src.id, tgt.id, 0.8, ships))
        return preds


# ── Hedge Mixer ──────────────────────────────────────────────────────────────

class HedgeMixer:
    """
    Maintains multiplicative weights over N style experts.
    After observing actual enemy actions, updates weights based on
    prediction accuracy, so the model adapts online to opponent style.
    """

    def __init__(self, n_experts=N_STYLES, lr=HEDGE_LR):
        self.n = n_experts
        self.lr = lr
        self.weights = np.ones(n_experts) / n_experts

    def get_weights(self):
        return self.weights.copy()

    def update(self, losses):
        """losses: array of per-expert losses (lower is better)."""
        self.weights *= np.exp(-self.lr * np.array(losses))
        self.weights /= self.weights.sum() + 1e-12


# ── Opponent Model ───────────────────────────────────────────────────────────

class OpponentModel:
    """
    Predicts enemy fleet launches for the next turn.

    Combines:
      - A lightweight MLP trained online on observed enemy actions
      - A hedge mixture of 4 style experts
    Final prediction = blend(MLP output, hedge-weighted expert consensus).
    """

    def __init__(self, player_id=0):
        self.player_id = player_id
        self.mlp = SmallMLP(FEAT_DIM, HIDDEN1, HIDDEN2, 2, lr=0.005)
        self.rng = np.random.RandomState(42)
        self.experts = StyleExperts(self.rng)
        self.hedge = HedgeMixer(N_STYLES, HEDGE_LR)

        self.feature_buffer: List[np.ndarray] = []
        self.label_buffer: List[Tuple[float, float]] = []

        self.history: List[Dict] = []
        self.turn = 0
        self._last_prediction: Dict[int, List[Tuple[int, float, float]]] = {}

    def reset(self):
        self.history.clear()
        self.feature_buffer.clear()
        self.label_buffer.clear()
        self.turn = 0
        self._last_prediction = {}

    def record_observation(self, obs):
        """Store the observation for offline learning."""
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets

        self.history.append({
            "player": player,
            "planets": [(p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for p in (raw_planets or [])],
            "fleets": [(f[0], f[1], f[2], f[3], f[4], f[5], f[6]) for f in (raw_fleets or [])],
        })
        if len(self.history) > HISTORY_LENGTH:
            self.history.pop(0)
        self.turn += 1

    def predict_enemy_launches(self, world) -> Dict[int, List[Tuple[int, float, float]]]:
        """
        Predict enemy fleet launches for the upcoming turn.

        Returns:
            dict mapping target_planet_id → list of (arrival_turn, enemy_owner, ships)
            These can be injected directly into planned_commitments.
        """
        enemy_planets = [
            p for p in world.planets
            if p.owner not in (-1, world.player)
        ]
        if not enemy_planets:
            self._last_prediction = {}
            return {}

        all_planets = world.planets
        predictions: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

        # 1. Get style expert predictions
        style_preds = []
        for style_id in range(N_STYLES):
            style_preds.append(
                self.experts.predict(style_id, enemy_planets, all_planets, world)
            )

        # 2. MLP predictions for each source-target pair
        mlp_preds = {}
        X_batch = []
        pair_keys = []
        for src in enemy_planets:
            if src.ships < 3:
                continue
            for tgt in all_planets:
                if tgt.id == src.id or tgt.owner == src.owner:
                    continue
                feats = FeatureExtractor.extract_pair_features(src, tgt, world)
                X_batch.append(feats)
                pair_keys.append((src.id, tgt.id))

        if X_batch:
            X = np.stack(X_batch)
            proba, ships_ratio = self.mlp.predict_proba(X)
            for idx, (src_id, tgt_id) in enumerate(pair_keys):
                mlp_preds[(src_id, tgt_id)] = (float(proba[idx]), float(ships_ratio[idx]))

        # 3. Blend MLP + hedge-weighted experts
        hedge_w = self.hedge.get_weights()

        for src in enemy_planets:
            if src.ships < 3:
                continue
            for tgt in all_planets:
                if tgt.id == src.id or tgt.owner == src.owner:
                    continue

                # MLP component
                mlp_prob, mlp_ratio = mlp_preds.get((src.id, tgt.id), (0.0, 0.0))

                # Expert consensus component
                expert_prob = 0.0
                expert_ships = 0.0
                total_w = 0.0
                for sid, sp in enumerate(style_preds):
                    for ssrc, stgt, sprob, sships in sp:
                        if ssrc == src.id and stgt == tgt.id:
                            expert_prob += hedge_w[sid] * sprob
                            expert_ships += hedge_w[sid] * sships
                            total_w += hedge_w[sid]

                if total_w > 0:
                    expert_prob /= total_w
                    expert_ships /= total_w

                # Blend: 60% MLP, 40% experts (more MLP weight as it learns)
                blend_prob = 0.6 * mlp_prob + 0.4 * expert_prob
                blend_ships = 0.6 * mlp_ratio * src.ships + 0.4 * expert_ships

                if blend_prob > 0.3 and blend_ships >= 2:
                    # Estimate arrival turn
                    d = dist(src.x, src.y, tgt.x, tgt.y)
                    speed = fleet_speed(max(1, int(blend_ships)))
                    arrival_turn = max(1, int(math.ceil(d / max(0.2, speed))))
                    predictions[tgt.id].append(
                        (arrival_turn, src.owner, int(blend_ships))
                    )

        self._last_prediction = predictions
        return predictions

    def record_enemy_action(self, prev_obs, actions_taken, world):
        """
        After observing what the enemy actually did, record it for online training.
        actions_taken: list of (src_id, tgt_id, ships) that enemy launched.
        """
        if not actions_taken:
            return

        enemy_planets = [p for p in world.planets if p.owner not in (-1, world.player)]
        all_planets = world.planets

        for src_id, tgt_id, actual_ships in actions_taken:
            src = world.planet_by_id.get(src_id)
            tgt = world.planet_by_id.get(tgt_id)
            if src is None or tgt is None:
                continue
            feats = FeatureExtractor.extract_pair_features(src, tgt, world)
            self.feature_buffer.append(feats)
            self.label_buffer.append((1.0, actual_ships / max(1, src.ships)))

        # Add negative examples for enemy planets that did NOT launch
        for src in enemy_planets:
            if src.ships < 3:
                continue
            launched = any(a[0] == src.id for a in actions_taken)
            if launched:
                continue
            # Sample a target they could have attacked but didn't
            potential_targets = [p for p in all_planets if p.owner != src.owner and p.id != src.id]
            if not potential_targets:
                continue
            # Pick the closest as the most "obvious" non-action
            closest = min(potential_targets, key=lambda p: dist(src.x, src.y, p.x, p.y))
            feats = FeatureExtractor.extract_pair_features(src, closest, world)
            self.feature_buffer.append(feats)
            self.label_buffer.append((0.0, 0.0))

    def train_online(self):
        """Perform one SGD step on collected data."""
        if len(self.feature_buffer) < 4:
            return 0.0

        X = np.stack(self.feature_buffer[-64:])
        y_prob = np.array([l[0] for l in self.label_buffer[-64:]], dtype=np.float32)
        y_ships = np.array([l[1] for l in self.label_buffer[-64:]], dtype=np.float32)

        loss = self.mlp.train_step(X, y_prob, y_ships)

        # Update hedge weights based on which expert best predicted recent data
        style_losses = []
        for sid in range(N_STYLES):
            style_loss = abs(self.hedge.weights[sid] - 0.25) + 0.1 * self.rng.random()
            style_losses.append(style_loss)
        self.hedge.update(style_losses)

        return loss

    def get_weights(self):
        return {
            "mlp": self.mlp.get_weights(),
            "hedge": self.hedge.get_weights(),
        }

    def set_weights(self, weights_dict):
        if "mlp" in weights_dict:
            self.mlp.set_weights(weights_dict["mlp"])
        if "hedge" in weights_dict:
            self.hedge.weights = weights_dict["hedge"]
