import math
import time
from typing import List, Dict, Set, Tuple, Optional, Callable

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# ---------------------------------------------------------------------------
# 常量（修复：调整关键阈值）
# ---------------------------------------------------------------------------
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
BOARD_SIZE = 100.0
SAFETY = 1.3
MAX_SPEED = 6.0
MAX_TURNS = 500
ROLLOUT_DEPTH = 30
TIME_BUDGET_MS = 900  # 降低时间预算，避免超时
HARD_LIMIT_MS = 950
CANDIDATES_PER_PLANET = 5  # 增加候选目标数
MAX_ACTION_SETS = 20  # 增加动作集数量


# ---------------------------------------------------------------------------
# 几何/移动工具函数（修复：轨道行星判断阈值）
# ---------------------------------------------------------------------------
def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def compute_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
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


def predict_future_position_from_current(
    x: float,
    y: float,
    angular_velocity: float,
    steps: int,
) -> Tuple[float, float]:
    radius = math.hypot(x - CENTER_X, y - CENTER_Y)
    if radius < 1e-6 or angular_velocity == 0:
        return x, y
    angle = math.atan2(y - CENTER_Y, x - CENTER_X)
    new_angle = angle + angular_velocity * steps
    return (
        CENTER_X + radius * math.cos(new_angle),
        CENTER_Y + radius * math.sin(new_angle),
    )


# ---------------------------------------------------------------------------
# 快速模拟状态（修复：轨道行星判断阈值从40→50）
# ---------------------------------------------------------------------------
class SimPlanet:
    __slots__ = (
        "id",
        "owner",
        "x",
        "y",
        "radius",
        "ships",
        "production",
        "init_x",
        "init_y",
        "is_orbiting",
        "is_comet",
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
        init_x: float = 0,
        init_y: float = 0,
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
            self.id,
            self.owner,
            self.x,
            self.y,
            self.radius,
            self.ships,
            self.production,
            self.init_x,
            self.init_y,
            self.is_orbiting,
            self.is_comet,
        )


class SimFleet:
    __slots__ = ("id", "owner", "x", "y", "angle", "ships", "from_planet_id")

    def __init__(
        self,
        fid: int,
        owner: int,
        x: float,
        y: float,
        angle: float,
        ships: int,
        from_pid: int,
    ):
        self.id = fid
        self.owner = owner
        self.x = x
        self.y = y
        self.angle = angle
        self.ships = ships
        self.from_planet_id = from_pid

    def copy(self) -> "SimFleet":
        return SimFleet(
            self.id,
            self.owner,
            self.x,
            self.y,
            self.angle,
            self.ships,
            self.from_planet_id,
        )


class SimState:
    def __init__(self):
        self.planets: Dict[int, SimPlanet] = {}
        self.fleets: List[SimFleet] = []
        self.player: int = 0
        self.angular_velocity: float = 0.0
        self.initial_planets: List = []
        self.comet_ids: Set[int] = set()
        self.turn: int = 0
        self._fleet_counter: int = 0

    @classmethod
    def from_observation(cls, obs) -> "SimState":
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

        init_pos: Dict[int, Tuple[float, float]] = {}
        for ip in state.initial_planets:
            if len(ip) >= 4:
                init_pos[ip[0]] = (ip[2], ip[3])

        for p_data in raw_planets:
            pid, owner, x, y, radius, ships, production = p_data[:7]
            ix, iy = init_pos.get(pid, (x, y))
            # 修复：轨道行星判断阈值从40→50（匹配游戏规则）
            center_dist = math.hypot(ix - CENTER_X, iy - CENTER_Y)
            is_orb = center_dist + radius < 50.0  # 原代码是40，错误！
            is_c = pid in state.comet_ids
            state.planets[pid] = SimPlanet(
                pid,
                owner,
                x,
                y,
                radius,
                ships,
                production,
                ix,
                iy,
                is_orb,
                is_c,
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

    def planets_by_owner(self, owner: int) -> List[SimPlanet]:
        return [p for p in self.planets.values() if p.owner == owner]

    def my_planets(self) -> List[SimPlanet]:
        return self.planets_by_owner(self.player)

    def my_fleet_ships(self) -> int:
        return sum(f.ships for f in self.fleets if f.owner == self.player)

    def enemy_fleet_ships(self) -> int:
        return sum(
            f.ships for f in self.fleets if f.owner != self.player and f.owner != -1
        )

    def total_ships(self, owner: int) -> int:
        return sum(p.ships for p in self.planets.values() if p.owner == owner)

    def advance(
        self,
        our_moves: List[Tuple[int, float, int]],
        opponent_policy: Callable,
    ):
        self.turn += 1
        if self.turn > MAX_TURNS:
            return

        my_id = self.player
        moves_by_player: Dict[int, List[Tuple[int, float, int]]] = {my_id: our_moves}

        for pid in range(4):
            if pid == my_id:
                continue
            opp_planets = self.planets_by_owner(pid)
            if opp_planets:
                moves_by_player[pid] = opponent_policy(pid, self)
            else:
                moves_by_player[pid] = []

        # 1. 发射舰队
        for owner, moves in moves_by_player.items():
            for from_id, angle, num_ships in moves:
                src = self.planets.get(from_id)
                if src is None or src.owner != owner or src.ships < num_ships:
                    continue
                src.ships -= num_ships
                self._fleet_counter += 1
                spawn_x = src.x + src.radius * math.cos(angle)
                spawn_y = src.y + src.radius * math.sin(angle)
                self.fleets.append(
                    SimFleet(
                        self._fleet_counter,
                        owner,
                        spawn_x,
                        spawn_y,
                        angle,
                        num_ships,
                        from_id,
                    )
                )

        # 2. 生产飞船
        for p in self.planets.values():
            if p.owner != -1:
                p.ships += p.production

        # 3. 舰队移动 + 碰撞检测
        surviving: List[SimFleet] = []
        arrivals: Dict[int, Dict[int, int]] = {}

        for f in self.fleets:
            speed = compute_speed(f.ships)
            nx = f.x + speed * math.cos(f.angle)
            ny = f.y + speed * math.sin(f.angle)

            if nx < 0 or nx > BOARD_SIZE or ny < 0 or ny > BOARD_SIZE:
                continue
            if segment_hits_sun(f.x, f.y, nx, ny):
                continue

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

        # 4. 战斗结算
        for pid, attackers in arrivals.items():
            self._resolve_combat(pid, attackers)

        # 5. 行星轨道旋转
        for p in self.planets.values():
            if p.is_orbiting and not p.is_comet:
                radius = math.hypot(p.x - CENTER_X, p.y - CENTER_Y)
                angle = math.atan2(p.y - CENTER_Y, p.x - CENTER_X)
                angle += self.angular_velocity
                p.x = CENTER_X + radius * math.cos(angle)
                p.y = CENTER_Y + radius * math.sin(angle)

    def _resolve_combat(self, planet_id: int, attackers: Dict[int, int]):
        planet = self.planets.get(planet_id)
        if planet is None:
            return

        groups = sorted(attackers.items(), key=lambda x: -x[1])
        while len(groups) > 1:
            (o1, s1), (o2, s2) = groups[0], groups[1]
            if s1 > s2:
                groups = [(o1, s1 - s2)] + groups[2:]
            elif s2 > s1:
                groups = [(o2, s2 - s1)] + groups[2:]
            else:
                groups = groups[2:]
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
                planet.ships = 0

    def evaluate(self) -> float:
        my_id = self.player
        my_ships = self.total_ships(my_id)
        my_fleets = self.my_fleet_ships()
        my_planets_count = len(self.planets_by_owner(my_id))
        my_prod = sum(p.production for p in self.planets.values() if p.owner == my_id)

        enemy_ships = sum(
            p.ships for p in self.planets.values() if p.owner not in (-1, my_id)
        )
        enemy_fleets = sum(f.ships for f in self.fleets if f.owner not in (-1, my_id))
        enemy_prod = sum(
            p.production for p in self.planets.values() if p.owner not in (-1, my_id)
        )

        total_me = my_ships + my_fleets
        total_enemy = enemy_ships + enemy_fleets + 1

        ship_score = (total_me - total_enemy) / (total_me + total_enemy)
        prod_score = (my_prod - enemy_prod) / (my_prod + enemy_prod + 1) * 0.3
        planet_bonus = my_planets_count * 0.05

        fleet_potential = 0
        for f in self.fleets:
            if f.owner == self.player:
                # 找到最近的非己方星球
                min_d = min(
                    [
                        dist(f.x, f.y, p.x, p.y)
                        for p in self.planets.values()
                        if p.owner != self.player
                    ]
                    or [999]
                )
                # 距离越近，飞船价值越高（鼓励进攻）
                fleet_potential += f.ships * (1.0 / (min_d + 1))

        return ship_score + prod_score + planet_bonus + (fleet_potential * 0.1)


# ---------------------------------------------------------------------------
# 动作生成（修复：降低发射阈值、增加候选动作）
# ---------------------------------------------------------------------------
def generate_candidate_moves(
    state: SimState,
    angular_velocity: float,
) -> List[Tuple[int, float, int]]:
    candidates: List[Tuple[int, float, int]] = []
    my_id = state.player

    for p in state.my_planets():
        # 修复：降低飞船保留阈值（原代码是<=1，现在保留至少3艘防御）
        if p.ships <= 3:
            continue

        # 候选目标：非己方行星（包括中立）
        # 排序时对轨道行星（center_dist < 50）加 2.5 倍距离惩罚，
        # 优先选择容易占领的外围静态行星，避免浪费舰队
        targets = sorted(
            [t for t in state.planets.values() if t.owner != my_id],
            key=lambda t: dist(p.x, p.y, t.x, t.y)
            * (
                2.5
                if (math.hypot(t.x - CENTER_X, t.y - CENTER_Y) < 50 and not t.is_comet)
                else 1.0
            ),
        )
        if not targets:
            continue  # 无目标则跳过

        for target in targets[:CANDIDATES_PER_PLANET]:
            center_dist = math.hypot(target.x - CENTER_X, target.y - CENTER_Y)
            is_orbiting = center_dist < 50 and not target.is_comet

            # 估算到达时间，用于计算目标增援和选取发射飞船数量
            est_distance = dist(p.x, p.y, target.x, target.y)
            est_travel_time = est_distance / compute_speed(max(1, p.ships // 2))

            expected_enemy = target.ships + (
                est_travel_time * target.production if target.owner != -1 else 0
            )
            capture_needed = int(expected_enemy + 5)
            options = set()

            if p.ships >= capture_needed:
                options.add(capture_needed)
            if p.ships >= capture_needed + 2:
                options.add(capture_needed + 2)
            options.add(min(p.ships - 3, capture_needed))

            for ships in options:
                if not (1 <= ships <= p.ships - 3):
                    continue

                if is_orbiting:
                    # 根据实际飞船数量和距离计算飞行时间，
                    # 然后预测目标在此时间后的位置，实现精确拦截
                    speed = compute_speed(int(ships))
                    flight_steps = max(1, int(est_distance / speed))
                    fx, fy = predict_future_position_from_current(
                        target.x,
                        target.y,
                        angular_velocity,
                        flight_steps,
                    )
                else:
                    fx, fy = target.x, target.y

                angle = math.atan2(fy - p.y, fx - p.x)
                candidates.append((p.id, angle, int(ships)))

    return candidates


def generate_action_sets(
    candidates: List[Tuple[int, float, int]],
    state: SimState,
) -> List[List[Tuple[int, float, int]]]:
    action_sets: List[List[Tuple[int, float, int]]] = []

    # 1. 生成多行星协同动作集
    for c in candidates[:10]:  # 增加候选动作数量
        from_id, angle, ships = c
        moves = [(from_id, angle, ships)]
        used = {from_id}
        for p in state.my_planets():
            if p.id in used or p.ships <= 4:
                continue
            targets = sorted(
                [t for t in state.planets.values() if t.owner != state.player],
                key=lambda t: dist(p.x, p.y, t.x, t.y)
                * (
                    2.5
                    if (math.hypot(t.x - CENTER_X, t.y - CENTER_Y) < 50
                        and not getattr(t, "is_comet", False))
                    else 1.0
                ),
            )
            if targets:
                t = targets[0]
                a = math.atan2(t.y - p.y, t.x - p.x)
                available = p.ships - 3
                sent = min(t.ships + 1, available)
                if sent > 0:
                    moves.append((p.id, a, int(sent)))
                    used.add(p.id)
        action_sets.append(moves)

    # 2. 单行星动作集
    seen_planet = set()
    for c in candidates:
        if c[0] not in seen_planet:
            action_sets.append([(c[0], c[1], c[2])])
            seen_planet.add(c[0])

    # 修复：空动作放到最后（原代码是insert(0, [])，优先级过高）
    action_sets.append([])

    # 去重
    seen: Set[Tuple] = set()
    unique: List[List[Tuple[int, float, int]]] = []
    for moves in action_sets:
        key = tuple(sorted((m[0], round(m[1], 2), m[2]) for m in moves))
        if key not in seen:
            seen.add(key)
            unique.append(moves)

    return unique[:MAX_ACTION_SETS]


# ---------------------------------------------------------------------------
# 对手策略
# ---------------------------------------------------------------------------
def heuristic_opponent(pid: int, state: SimState) -> List[Tuple[int, float, int]]:
    moves = []
    planets = state.planets_by_owner(pid)
    for p in planets[:3]:  # 增加对手进攻行星数
        min_keep = max(2, int(p.ships * 0.2))  # 降低对手保留阈值
        available = p.ships - min_keep
        if available <= 0:
            continue
        targets = sorted(
            [t for t in state.planets.values() if t.owner != pid],
            key=lambda t: dist(p.x, p.y, t.x, t.y)
            * (
                2.5
                if (math.hypot(t.x - CENTER_X, t.y - CENTER_Y) < 50 and not getattr(t, "is_comet", False))
                else 1.0
            ),
        )
        if not targets:
            continue
        target = targets[0]
        # 对轨道行星做位置预测
        center_dist = math.hypot(target.x - CENTER_X, target.y - CENTER_Y)
        if center_dist < 50 and not getattr(target, "is_comet", False):
            est_speed = compute_speed(max(1, p.ships))
            flight_steps = max(1, int(dist(p.x, p.y, target.x, target.y) / est_speed))
            fx, fy = predict_future_position_from_current(
                target.x,
                target.y,
                state.angular_velocity,
                flight_steps,
            )
        else:
            fx, fy = target.x, target.y
        angle = math.atan2(fy - p.y, fx - p.x)
        sent = min(target.ships + 1, available)
        if sent > 0:
            moves.append((p.id, angle, int(sent)))
    return moves


# ---------------------------------------------------------------------------
# Rollout（修复：我方不再完全被动，加入简易进攻策略）
# ---------------------------------------------------------------------------
def rollout_attack_policy(pid: int, state: SimState) -> List[Tuple[int, float, int]]:
    """Rollout阶段的简易进攻策略，含轨道预测"""
    moves = []
    planets = state.planets_by_owner(pid)
    for p in planets[:2]:
        if p.ships <= 3:
            continue
        targets = sorted(
            [t for t in state.planets.values() if t.owner != pid],
            key=lambda t: dist(p.x, p.y, t.x, t.y)
            * (
                2.5
                if (math.hypot(t.x - CENTER_X, t.y - CENTER_Y) < 50 and not getattr(t, "is_comet", False))
                else 1.0
            ),
        )
        if targets:
            target = targets[0]
            # 对轨道行星做位置预测
            center_dist = math.hypot(target.x - CENTER_X, target.y - CENTER_Y)
            if center_dist < 50 and not getattr(target, "is_comet", False):
                est_speed = compute_speed(max(1, p.ships))
                flight_steps = max(
                    1, int(dist(p.x, p.y, target.x, target.y) / est_speed)
                )
                fx, fy = predict_future_position_from_current(
                    target.x,
                    target.y,
                    state.angular_velocity,
                    flight_steps,
                )
            else:
                fx, fy = target.x, target.y
            angle = math.atan2(fy - p.y, fx - p.x)
            sent = min(target.ships + 1, p.ships - 3)
            if sent > 0:
                moves.append((p.id, angle, sent))
    return moves


def run_rollout(state: SimState, max_depth: int = ROLLOUT_DEPTH) -> float:
    s = state.copy()
    for _ in range(max_depth):
        if s.turn >= MAX_TURNS:
            break
        # 修复：Rollout阶段我方主动进攻（原代码是传[]，完全被动）
        our_moves = rollout_attack_policy(s.player, s)
        s.advance(our_moves, heuristic_opponent)
    return s.evaluate()


# ---------------------------------------------------------------------------
# MCTS节点
# ---------------------------------------------------------------------------
class MCTSNode:
    def __init__(
        self,
        state: SimState,
        action_set: Optional[List[Tuple[int, float, int]]] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state = state
        self.action_set = action_set
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
        explore = (
            c * math.sqrt(math.log(self.parent.visits) / self.visits)
            if self.parent
            else 0.0
        )
        return exploit + explore

    def best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb_score())

    def most_visited_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.visits)

    def is_expanded(self) -> bool:
        return self._action_sets is not None and len(self._action_sets) == 0

    def expand(self) -> Optional["MCTSNode"]:
        if self._action_sets is None:
            candidates = generate_candidate_moves(
                self.state,
                self.state.angular_velocity,
            )
            self._action_sets = generate_action_sets(candidates, self.state)

        if not self._action_sets:
            return None

        action_set = self._action_sets.pop(0)  # 从队首弹出，最近的候选动作集优先被探索
        new_state = self.state.copy()
        new_state.advance(action_set, heuristic_opponent)

        child = MCTSNode(new_state, action_set, parent=self)
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# MCTS Agent（修复：异常日志输出到stdout、增加调试信息）
# ---------------------------------------------------------------------------
class MCTSAgent:
    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self.name = "MCTSAgent"
        self.turn = 0
        self._initial_planets: List = []
        self._angular_velocity: float = 0.0
        self._comet_ids: Set[int] = set()

    def reset(self):
        self.turn = 0
        self._initial_planets = []
        self._angular_velocity = 0.0
        self._comet_ids = set()

    def act(self, observation) -> List[List]:
        self.turn += 1
        print(f"=== Turn {self.turn} ===")  # 调试信息：输出回合数

        try:
            state = SimState.from_observation(observation)
            self._cache_obs(observation)

            my_planets = state.my_planets()
            print(f"My planets count: {len(my_planets)}")  # 调试信息：己方行星数
            if not my_planets:
                print("No owned planets, return empty moves")
                return []

            # 输出己方行星的飞船数
            for p in my_planets:
                print(f"Planet {p.id}: ships={p.ships}, owner={p.owner}")

            best_action_set = self._mcts_search(state)

            if best_action_set is None:
                print("No best action set found")
                return []

            print(f"Selected action set: {best_action_set}")  # 调试信息：输出选中的动作
            return [list(m) for m in best_action_set]

        except Exception as e:
            # 修复：异常信息输出到stdout（Kaggle可见）
            import sys

            print(f"MCTS agent error (turn {self.turn}): {e}", file=sys.stdout)
            import traceback

            traceback.print_exc(file=sys.stdout)
            return []

    def compute_moves(self, observation):
        return self.act(observation)

    def _cache_obs(self, observation):
        if isinstance(observation, dict):
            self._initial_planets = observation.get("initial_planets", []) or []
            self._angular_velocity = observation.get("angular_velocity", 0) or 0
            self._comet_ids = set(observation.get("comet_planet_ids", []) or [])
        else:
            self._initial_planets = getattr(observation, "initial_planets", []) or []
            self._angular_velocity = getattr(observation, "angular_velocity", 0) or 0
            self._comet_ids = set(getattr(observation, "comet_planet_ids", []) or [])

    def _mcts_search(
        self,
        state: SimState,
    ) -> Optional[List[Tuple[int, float, int]]]:
        start = time.time()
        time_budget = TIME_BUDGET_MS / 1000.0
        hard_limit = HARD_LIMIT_MS / 1000.0

        root = MCTSNode(state)
        iteration = 0

        # 强制至少迭代10次（避免超时导致无迭代）
        min_iterations = 10
        while (
            time.time() - start < time_budget or iteration < min_iterations
        ) and time.time() - start < hard_limit:
            # SELECTION
            node = root
            while (
                node.children
                and node._action_sets is not None
                and len(node._action_sets) == 0
            ):
                node = node.best_child()

            # EXPANSION
            child = node.expand()
            if child is not None:
                node = child

            # SIMULATION
            score = run_rollout(node.state, ROLLOUT_DEPTH)

            # BACKPROPAGATION
            while node is not None:
                node.visits += 1
                node.total_score += score
                node = node.parent

            iteration += 1

        print(f"MCTS iterations: {iteration}")  # 调试信息：输出迭代次数
        if not root.children:
            print("No MCTS children nodes")
            return None

        best = root.most_visited_child()
        return best.action_set


# ---------------------------------------------------------------------------
# Kaggle兼容入口
# ---------------------------------------------------------------------------
_agent_instance: Optional[MCTSAgent] = None


def mcts_agent(obs, config=None) -> List[List]:
    global _agent_instance

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    if _agent_instance is None:
        _agent_instance = MCTSAgent(player_id=player)
    elif _agent_instance.player_id != player:
        _agent_instance = MCTSAgent(player_id=player)

    return _agent_instance.act(obs)


# 本地测试
if __name__ == "__main__":
    print("MCTS agent loaded successfully")
