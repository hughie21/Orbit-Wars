"""
MCTS agent for Orbit Wars.
Monte Carlo Tree Search with fast game simulation, orbit prediction,
battle simulation, and time-budgeted search depth control.

Design:
- FastSimState: lightweight copyable game state for fast rollouts
- MCTS: 1-ply tree root->children(actions), UCB-guided simulation budget
- Each rollout: ~10-15 turn heuristic simulation with orbit prediction
- Time budget: ~800ms per turn, leaving headroom for framework overhead
"""

import math
import random
import time
from typing import List, Dict, Set, Tuple, Optional, Callable

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
BOARD_SIZE = 100.0
SAFETY = 1.3
MAX_SPEED = 6.0
MAX_TURNS = 500
ROLLOUT_DEPTH = 10
TIME_BUDGET_MS = 700
HARD_LIMIT_MS = 950  # Never exceed this (1s actTimeout)
CANDIDATES_PER_PLANET = 3
MAX_ACTION_SETS = 10

# ---------------------------------------------------------------------------
# Geometry / movement helpers
# ---------------------------------------------------------------------------


def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def compute_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Check if line segment from (x1,y1) to (x2,y2) intersects sun."""
    r = SUN_R + SAFETY
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CENTER_X, y1 - CENTER_Y
    a = dx * dx + dy * dy
    if a < 1e-9:
        return False
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2.0 * a)
    t2 = (-b + disc) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def predict_future_position(
    planet_id: int,
    x: float,
    y: float,
    initial_planets: List,
    angular_velocity: float,
    steps: int,
) -> Tuple[float, float]:
    """Predict position of an orbiting planet after `steps` turns."""
    radius = math.hypot(x - CENTER_X, y - CENTER_Y)
    if radius < 1e-6 or angular_velocity == 0:
        return x, y

    # Find initial position for accurate starting angle
    for ip in initial_planets:
        if ip[0] == planet_id:
            init_x, init_y = ip[2], ip[3]
            angle = math.atan2(init_y - CENTER_Y, init_x - CENTER_X)
            new_angle = angle + angular_velocity * steps
            return (
                CENTER_X + radius * math.cos(new_angle),
                CENTER_Y + radius * math.sin(new_angle),
            )
    return x, y


# ---------------------------------------------------------------------------
# Fast simulation state
# ---------------------------------------------------------------------------


class SimPlanet:
    __slots__ = (
        "id", "owner", "x", "y", "radius", "ships", "production",
        "init_x", "init_y", "is_orbiting", "is_comet",
    )

    def __init__(
        self,
        pid: int,
        owner: int,
        x: float,
        y: float,
        radius: float,
        ships: int,
        production: int,
        init_x: float = None,
        init_y: float = None,
        is_orbiting: bool = False,
        is_comet: bool = False,
    ):
        self.id = pid
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production
        self.init_x = init_x if init_x is not None else x
        self.init_y = init_y if init_y is not None else y
        self.is_orbiting = is_orbiting
        self.is_comet = is_comet

    def copy(self) -> "SimPlanet":
        return SimPlanet(
            self.id, self.owner, self.x, self.y, self.radius,
            self.ships, self.production, self.init_x, self.init_y,
            self.is_orbiting, self.is_comet,
        )


class SimFleet:
    __slots__ = ("id", "owner", "x", "y", "angle", "ships", "from_planet_id")

    def __init__(self, fid: int, owner: int, x: float, y: float, angle: float, ships: int, from_pid: int):
        self.id = fid
        self.owner = owner
        self.x = x
        self.y = y
        self.angle = angle
        self.ships = ships
        self.from_planet_id = from_pid

    def copy(self) -> "SimFleet":
        return SimFleet(self.id, self.owner, self.x, self.y, self.angle, self.ships, self.from_planet_id)


class SimState:
    """Fast simulation state for MCTS rollouts."""

    def __init__(self):
        self.planets: Dict[int, SimPlanet] = {}
        self.fleets: List[SimFleet] = []
        self.player: int = 0
        self.angular_velocity: float = 0.0
        self.initial_planets: List = []
        self.comet_ids: Set[int] = set()
        self.turn: int = 0
        self._fleet_counter: int = 0

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_observation(cls, obs) -> "SimState":
        """Build from a Kaggle observation."""
        state = cls()

        if isinstance(obs, dict):
            state.player = obs.get("player", 0)
            raw_planets = obs.get("planets", []) or []
            raw_fleets = obs.get("fleets", []) or []
            state.angular_velocity = obs.get("angular_velocity", 0) or 0
            state.initial_planets = obs.get("initial_planets", []) or []
            state.comet_ids = set(obs.get("comet_planet_ids", []) or [])
        else:
            state.player = obs.player
            raw_planets = obs.planets or []
            raw_fleets = obs.fleets or []
            state.angular_velocity = getattr(obs, "angular_velocity", 0) or 0
            state.initial_planets = getattr(obs, "initial_planets", []) or []
            state.comet_ids = set(getattr(obs, "comet_planet_ids", []) or [])

        # Build initial-position lookup
        init_pos: Dict[int, Tuple[float, float]] = {}
        for ip in state.initial_planets:
            if len(ip) >= 4:
                init_pos[ip[0]] = (ip[2], ip[3])

        for p_data in raw_planets:
            pid, owner, x, y, radius, ships, production = p_data[:7]
            ix, iy = init_pos.get(pid, (x, y))
            # Planets with orbital_radius + planet_radius < 50 are orbiting
            center_dist = math.hypot(ix - CENTER_X, iy - CENTER_Y)
            is_orb = center_dist + radius < 40.0
            is_c = pid in state.comet_ids
            state.planets[pid] = SimPlanet(
                pid, owner, x, y, radius, ships, production,
                ix, iy, is_orb, is_c,
            )

        for f_data in raw_fleets:
            fid, owner, x, y, angle, from_pid, ships = f_data[:7]
            state.fleets.append(SimFleet(fid, owner, x, y, angle, ships, from_pid))
            if fid > state._fleet_counter:
                state._fleet_counter = fid

        return state

    def copy(self) -> "SimState":
        s = SimState()
        s.planets = {pid: p.copy() for pid, p in self.planets.items()}
        s.fleets = [f.copy() for f in self.fleets]
        s.player = self.player
        s.angular_velocity = self.angular_velocity
        s.initial_planets = self.initial_planets
        s.comet_ids = set(self.comet_ids)
        s.turn = self.turn
        s._fleet_counter = self._fleet_counter
        return s

    # ---- queries ----------------------------------------------------------

    def planets_by_owner(self, owner: int) -> List[SimPlanet]:
        return [p for p in self.planets.values() if p.owner == owner]

    def my_planets(self) -> List[SimPlanet]:
        return self.planets_by_owner(self.player)

    def my_fleet_ships(self) -> int:
        return sum(f.ships for f in self.fleets if f.owner == self.player)

    def enemy_fleet_ships(self) -> int:
        return sum(f.ships for f in self.fleets if f.owner != self.player and f.owner != -1)

    def total_ships(self, owner: int) -> int:
        return sum(p.ships for p in self.planets.values() if p.owner == owner)

    # ---- turn advancement -------------------------------------------------

    def advance(
        self,
        our_moves: List[Tuple[int, float, int]],
        opponent_policy: Callable,
    ):
        """Advance the game by one turn."""
        self.turn += 1
        if self.turn > MAX_TURNS:
            return

        my_id = self.player

        # Gather moves for all active players
        moves_by_player: Dict[int, List[Tuple[int, float, int]]] = {my_id: our_moves}
        for pid in range(4):
            if pid == my_id:
                continue
            opp_planets = self.planets_by_owner(pid)
            if opp_planets:
                moves_by_player[pid] = opponent_policy(pid, self)
            else:
                moves_by_player[pid] = []

        # 1. Launch fleets --------------------------------------------------
        for owner, moves in moves_by_player.items():
            for from_id, angle, num_ships in moves:
                src = self.planets.get(from_id)
                if src is None or src.owner != owner or src.ships < num_ships:
                    continue
                src.ships -= num_ships
                self._fleet_counter += 1
                spawn_x = src.x + src.radius * math.cos(angle)
                spawn_y = src.y + src.radius * math.sin(angle)
                self.fleets.append(SimFleet(
                    self._fleet_counter, owner,
                    spawn_x, spawn_y, angle, num_ships, from_id,
                ))

        # 2. Production -----------------------------------------------------
        for p in self.planets.values():
            if p.owner != -1:
                p.ships += p.production

        # 3. Move fleets + collision detection ------------------------------
        surviving: List[SimFleet] = []
        arrivals: Dict[int, Dict[int, int]] = {}  # planet_id -> {owner: ships}

        for f in self.fleets:
            speed = compute_speed(f.ships)
            nx = f.x + speed * math.cos(f.angle)
            ny = f.y + speed * math.sin(f.angle)

            # Out of bounds
            if nx < 0 or nx > BOARD_SIZE or ny < 0 or ny > BOARD_SIZE:
                continue

            # Sun
            if segment_hits_sun(f.x, f.y, nx, ny):
                continue

            # Planet collision (discrete — endpoint inside planet radius)
            hit_planet = None
            for p in self.planets.values():
                if dist(nx, ny, p.x, p.y) < p.radius + 0.5:
                    hit_planet = p
                    break

            if hit_planet is not None:
                arrivals.setdefault(hit_planet.id, {})
                arrivals[hit_planet.id][f.owner] = (
                    arrivals[hit_planet.id].get(f.owner, 0) + f.ships
                )
            else:
                f.x = nx
                f.y = ny
                surviving.append(f)

        self.fleets = surviving

        # 4. Combat resolution ----------------------------------------------
        for pid, attackers in arrivals.items():
            self._resolve_combat(pid, attackers)

        # 5. Orbit planets --------------------------------------------------
        for p in self.planets.values():
            if p.is_orbiting and not p.is_comet:
                radius = math.hypot(p.x - CENTER_X, p.y - CENTER_Y)
                angle = math.atan2(p.y - CENTER_Y, p.x - CENTER_X)
                angle += self.angular_velocity
                p.x = CENTER_X + radius * math.cos(angle)
                p.y = CENTER_Y + radius * math.sin(angle)

    def _resolve_combat(self, planet_id: int, attackers: Dict[int, int]):
        """Resolve combat at a planet following game rules."""
        planet = self.planets.get(planet_id)
        if planet is None:
            return

        # Sort attacker groups descending by ship count
        groups = sorted(attackers.items(), key=lambda x: -x[1])

        # Sequential pairwise elimination
        while len(groups) > 1:
            (o1, s1), (o2, s2) = groups[0], groups[1]
            if s1 > s2:
                groups = [(o1, s1 - s2)] + groups[2:]
            elif s2 > s1:
                groups = [(o2, s2 - s1)] + groups[2:]
            else:
                groups = groups[2:]  # tie → both destroyed
            groups.sort(key=lambda x: -x[1])

        if not groups:
            return

        final_owner, final_ships = groups[0]

        if final_owner == planet.owner:
            planet.ships += final_ships
        else:
            if final_ships > planet.ships:
                planet.ships = final_ships - planet.ships
                planet.owner = final_owner
            elif final_ships < planet.ships:
                planet.ships -= final_ships
            else:
                planet.ships = 0  # cancel out, keep owner

    # ---- evaluation -------------------------------------------------------

    def evaluate(self) -> float:
        """Heuristic evaluation from the agent's perspective.
        Returns a score in [-1, 1] where higher is better for us.
        """
        my_id = self.player

        my_ships = self.total_ships(my_id)
        my_fleets = self.my_fleet_ships()
        my_planets_count = len(self.planets_by_owner(my_id))
        my_prod = sum(p.production for p in self.planets.values() if p.owner == my_id)

        # Total enemy ships (all non-neutral, non-us)
        enemy_ships = sum(
            p.ships for p in self.planets.values()
            if p.owner not in (-1, my_id)
        )
        enemy_fleets = sum(
            f.ships for f in self.fleets
            if f.owner not in (-1, my_id)
        )
        enemy_prod = sum(
            p.production for p in self.planets.values()
            if p.owner not in (-1, my_id)
        )

        total_me = my_ships + my_fleets
        total_enemy = enemy_ships + enemy_fleets + 1  # avoid /0

        # Ship advantage
        ship_score = (total_me - total_enemy) / (total_me + total_enemy)

        # Production advantage
        prod_score = (my_prod - enemy_prod) / (my_prod + enemy_prod + 1) * 0.3

        # Planet count bonus
        planet_bonus = my_planets_count * 0.05

        # Aggression bonus: more fleets in transit = good (actively fighting)
        fleet_bonus = min(my_fleets / (total_me + 1), 1.0) * 0.1

        return ship_score + prod_score + planet_bonus + fleet_bonus


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------


def generate_candidate_moves(
    state: SimState,
    initial_planets: List,
    angular_velocity: float,
) -> List[Tuple[int, float, int]]:
    """Generate diverse candidate moves from all owned planets."""
    candidates: List[Tuple[int, float, int]] = []
    my_id = state.player

    for p in state.my_planets():
        if p.ships <= 1:
            continue

        # Collect targets (non-owned planets) sorted by distance
        targets = sorted(
            [t for t in state.planets.values() if t.owner != my_id],
            key=lambda t: dist(p.x, p.y, t.x, t.y),
        )

        for target in targets[:CANDIDATES_PER_PLANET]:
            # Compute angle, with orbit prediction for orbiting planets
            center_dist = math.hypot(target.x - CENTER_X, target.y - CENTER_Y)
            if center_dist < 40 and not target.is_comet:
                # Use predicted position (estimate 10 turns travel time)
                fx, fy = predict_future_position(
                    target.id, target.x, target.y,
                    initial_planets, angular_velocity, 10,
                )
            else:
                fx, fy = target.x, target.y

            angle = math.atan2(fy - p.y, fx - p.x)

            # Multiple ship-count options
            capture_needed = max(1, target.ships + 2)
            options = set()
            if p.ships >= capture_needed:
                options.add(capture_needed)
            if p.ships >= capture_needed + 3:
                options.add(capture_needed + 3)
            if p.ships >= 5:
                options.add(max(3, p.ships // 2))
            if p.ships >= 3:
                options.add(p.ships - 1)

            for ships in options:
                if ships <= p.ships:
                    candidates.append((p.id, angle, int(ships)))

    return candidates


def generate_action_sets(
    candidates: List[Tuple[int, float, int]],
    state: SimState,
) -> List[List[Tuple[int, float, int]]]:
    """Generate diverse complete action sets from candidates.
    Each action set is a list of (from_planet_id, angle, num_ships) tuples
    representing all moves our player makes this turn.
    """
    action_sets: List[List[Tuple[int, float, int]]] = []

    # 1. Full action sets: for each candidate as primary, fill with follow-ups
    for c in candidates[:8]:
        from_id, angle, ships = c
        moves = [(from_id, angle, ships)]
        used = {from_id}
        for p in state.my_planets():
            if p.id in used or p.ships <= 4:  # keep at least 4 on unused planets
                continue
            targets = sorted(
                [t for t in state.planets.values() if t.owner != state.player],
                key=lambda t: dist(p.x, p.y, t.x, t.y),
            )
            if targets:
                t = targets[0]
                a = math.atan2(t.y - p.y, t.x - p.x)
                available = p.ships - 3  # keep at least 3 for defense
                sent = min(t.ships + 2, available)
                if sent > 0:
                    moves.append((p.id, a, int(sent)))
                    used.add(p.id)
        action_sets.append(moves)

    # 2. Single-move action sets: one planet acts alone
    seen_planet = set()
    for c in candidates:
        if c[0] not in seen_planet:
            action_sets.append([(c[0], c[1], c[2])])
            seen_planet.add(c[0])

    # 3. Do-nothing: keep ships on planets
    action_sets.append([])

    # Deduplicate
    seen: Set[Tuple] = set()
    unique: List[List[Tuple[int, float, int]]] = []
    for moves in action_sets:
        key = tuple(sorted((m[0], round(m[1], 2), m[2]) for m in moves))
        if key not in seen:
            seen.add(key)
            unique.append(moves)

    return unique[:MAX_ACTION_SETS]  # cap total candidates


# ---------------------------------------------------------------------------
# Heuristic opponent policy for rollouts
# ---------------------------------------------------------------------------


def heuristic_opponent(pid: int, state: SimState) -> List[Tuple[int, float, int]]:
    """Conservative heuristic opponent: keeps at least 3 ships or 30% per planet."""
    moves = []
    planets = state.planets_by_owner(pid)
    for p in planets[:2]:
        min_keep = max(3, int(p.ships * 0.3))
        available = p.ships - min_keep
        if available <= 1:
            continue
        targets = sorted(
            [t for t in state.planets.values() if t.owner != pid],
            key=lambda t: dist(p.x, p.y, t.x, t.y),
        )
        if not targets:
            continue
        target = targets[0]
        sent = min(target.ships + 2, available)
        if sent > 0:
            angle = math.atan2(target.y - p.y, target.x - p.x)
            moves.append((p.id, angle, int(sent)))
    return moves


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_rollout(state: SimState, max_depth: int = ROLLOUT_DEPTH) -> float:
    """Run a fast heuristic rollout and return the evaluated score."""
    s = state.copy()
    for _ in range(max_depth):
        if s.turn >= MAX_TURNS:
            break

        # Conservative rollout policy: keep at least 3 ships or 30% on each planet
        our_moves = []
        for p in s.my_planets():
            min_keep = max(3, int(p.ships * 0.3))
            available = p.ships - min_keep
            if available <= 1:
                continue
            targets = sorted(
                [t for t in s.planets.values() if t.owner != s.player],
                key=lambda t: dist(p.x, p.y, t.x, t.y),
            )
            if not targets:
                continue
            target = targets[0]
            sent = min(target.ships + 2, available)
            if sent > 0:
                angle = math.atan2(target.y - p.y, target.x - p.x)
                our_moves.append((p.id, angle, int(sent)))

        s.advance(our_moves, heuristic_opponent)

    return s.evaluate()


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------


class MCTSNode:
    """Node in the MCTS tree. Each node holds a game state and its
    children represent different action sets for the current turn."""

    def __init__(
        self,
        state: SimState,
        action_set: Optional[List[Tuple[int, float, int]]] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state = state
        self.action_set = action_set  # actions taken to reach this state
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.total_score = 0.0
        self._action_sets: Optional[List[List[Tuple[int, float, int]]]] = None

    @property
    def mean_score(self) -> float:
        return self.total_score / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.mean_score
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0.0
        return exploit + explore

    def best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb_score())

    def most_visited_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.visits)

    def is_expanded(self) -> bool:
        return self._action_sets is not None and len(self._action_sets) == 0

    def expand(self) -> Optional["MCTSNode"]:
        """Expand one child from untried action sets."""
        if self._action_sets is None:
            candidates = generate_candidate_moves(
                self.state, self.state.initial_planets, self.state.angular_velocity,
            )
            self._action_sets = generate_action_sets(candidates, self.state)

        if not self._action_sets:
            return None

        action_set = self._action_sets.pop()
        new_state = self.state.copy()

        # Advance one turn with this action set (opponents use heuristic)
        new_state.advance(action_set, heuristic_opponent)

        child = MCTSNode(new_state, action_set, parent=self)
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# MCTS Agent
# ---------------------------------------------------------------------------


class MCTSAgent:
    """Monte Carlo Tree Search agent for Orbit Wars."""

    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self.name = "MCTSAgent"
        self.turn = 0
        # Cached observation data for orbit prediction
        self._initial_planets: List = []
        self._angular_velocity: float = 0.0
        self._comet_ids: Set[int] = set()

    def reset(self):
        self.turn = 0
        self._initial_planets = []
        self._angular_velocity = 0.0
        self._comet_ids = set()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def act(self, observation) -> List[List]:
        self.turn += 1

        # Build simulation state
        state = SimState.from_observation(observation)
        self._cache_obs(observation)

        my_planets = state.my_planets()
        if not my_planets:
            return []

        # Run MCTS
        best_action_set = self._mcts_search(state)

        if not best_action_set:
            return []

        return [list(m) for m in best_action_set]

    def _cache_obs(self, observation):
        if isinstance(observation, dict):
            self._initial_planets = observation.get("initial_planets", []) or []
            self._angular_velocity = observation.get("angular_velocity", 0) or 0
            self._comet_ids = set(observation.get("comet_planet_ids", []) or [])
        else:
            self._initial_planets = getattr(observation, "initial_planets", []) or []
            self._angular_velocity = getattr(observation, "angular_velocity", 0) or 0
            self._comet_ids = set(getattr(observation, "comet_planet_ids", []) or [])

    # ------------------------------------------------------------------
    # MCTS search
    # ------------------------------------------------------------------

    def _mcts_search(
        self,
        state: SimState,
    ) -> Optional[List[Tuple[int, float, int]]]:
        """Run time-budgeted MCTS and return the best action set found.

        Tree structure:
          Root (current state before our turn)
          ├─ child 1: action set A1 → state after 1 turn
          ├─ child 2: action set A2 → state after 1 turn
          └─ ... (up to N candidates)

        From each child, run rollouts to estimate long-term value.
        Uses UCB to allocate simulation budget across children.
        """
        start = time.time()
        time_budget = TIME_BUDGET_MS / 1000.0
        hard_limit = HARD_LIMIT_MS / 1000.0

        root = MCTSNode(state)

        iteration = 0
        while time.time() - start < time_budget:
            # Safety: never exceed hard limit
            if time.time() - start >= hard_limit:
                break
            # --- SELECTION ---
            node = root
            # Traverse to a node that still has untried actions
            while node.children and node._action_sets is not None and len(node._action_sets) == 0:
                node = node.best_child()

            # --- EXPANSION ---
            child = node.expand()
            if child is not None:
                node = child

            # --- SIMULATION (rollout) ---
            score = run_rollout(node.state, ROLLOUT_DEPTH)

            # --- BACKPROPAGATION ---
            while node is not None:
                node.visits += 1
                node.total_score += score
                node = node.parent

            iteration += 1

        # Select best child by visitation count (most robust)
        if not root.children:
            return None

        best = root.most_visited_child()
        return best.action_set


# ---------------------------------------------------------------------------
# Kaggle-compatible agent factory
# ---------------------------------------------------------------------------

_agent_instance: Optional[MCTSAgent] = None


def mcts_agent(obs, config=None) -> List[List]:
    """Kaggle-compatible agent function for MCTS."""
    global _agent_instance

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    if _agent_instance is None:
        _agent_instance = MCTSAgent(player_id=player)
    elif _agent_instance.player_id != player:
        _agent_instance = MCTSAgent(player_id=player)

    try:
        return _agent_instance.act(obs)
    except Exception as e:
        import sys
        print(f"MCTS agent error: {e}", file=sys.stderr)
        return []


# For local testing
if __name__ == "__main__":
    print("MCTS agent loaded successfully")
