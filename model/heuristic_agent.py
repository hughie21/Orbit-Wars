"""
Heuristic agent for Orbit Wars implementing the five strategies:
1. Prioritize capturing high-production neutral planets (production 3-5) with minimal ships
2. Defense mechanism: detect incoming enemy fleets and recall ships from nearby planets
3. Resource management: keep at least 20% ships in home planet
4. Comet utilization: capture comets immediately, withdraw ships before they leave
5. Orbit prediction: compute future positions of orbiting planets to avoid missed fleets
"""

import math
import logging
from typing import List, Tuple, Dict, Optional
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# Constants from game
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
SAFETY = 1.3
MAX_SPEED = 6.0
ROTATION_LIMIT = 50.0

def compute_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5

def is_orbiting_planet(x: float, y: float, radius: float) -> bool:
    return math.hypot(x - CENTER_X, y - CENTER_Y) + radius < ROTATION_LIMIT

def segment_hits_sun(x1, y1, x2, y2):
    """Check if line segment from (x1,y1) to (x2,y2) hits the sun."""
    r = SUN_R + SAFETY
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CENTER_X, y1 - CENTER_Y
    a = dx*dx + dy*dy
    if a < 1e-9:
        return False
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1, t2 = (-b - disc) / (2*a), (-b + disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

# Set up logger
logger = logging.getLogger(__name__)


class HeuristicAgent:
    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self.home_planet_id = None
        self.comet_ids = set()
        self.turn = 0

    def find_home_planet(self, planets: List[Planet]) -> Optional[Planet]:
        """Find and return the home planet for this player."""
        for p in planets:
            if p.owner == self.player_id:
                # Assume the first planet we own is home (simplification)
                # In reality, home planet is determined at game start
                return p
        return None

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def planet_distance(self, p1: Planet, p2: Planet) -> float:
        """Distance between two planets."""
        return self.distance(p1.x, p1.y, p2.x, p2.y)

    def future_planet_position(self, planet: Planet, initial_planets, angular_velocity, steps: int = 10) -> Tuple[float, float]:
        """Predict planet position after `steps` turns. Static planets return current position."""
        if is_orbiting_planet(planet.x, planet.y, planet.radius):
            # Find initial position for orbital radius
            initial = None
            for ip in initial_planets:
                if ip[0] == planet.id:
                    initial = ip
                    break
            if initial is None:
                return planet.x, planet.y
            init_x, init_y = initial[2], initial[3]
            r = math.hypot(init_x - CENTER_X, init_y - CENTER_Y)
            if r <= 1e-6 or angular_velocity == 0:
                return planet.x, planet.y
            # Use CURRENT angle, not initial angle
            cur_angle = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
            new_angle = cur_angle + angular_velocity * steps
            return (CENTER_X + r * math.cos(new_angle), CENTER_Y + r * math.sin(new_angle))
        return planet.x, planet.y  # static planet: position doesn't change

    def detect_incoming_threats(self, fleets: List[Fleet], my_planets: List[Planet]) -> Dict[int, List[Fleet]]:
        """
        Detect enemy fleets heading towards planets we own.
        Returns dict mapping planet_id -> list of threatening fleets.
        """
        threats = {}
        for fleet in fleets:
            if fleet.owner == self.player_id:
                continue

            # Simplified threat detection: check if fleet is close to any of our planets
            for planet in my_planets:
                dist = self.distance(fleet.x, fleet.y, planet.x, planet.y)
                if dist < 20:  # arbitrary threshold
                    if planet.id not in threats:
                        threats[planet.id] = []
                    threats[planet.id].append(fleet)

        return threats

    def find_nearest_ally_planet(self, target_planet: Planet, my_planets: List[Planet]) -> Optional[Planet]:
        """Find the nearest planet we own to a target planet."""
        if not my_planets:
            return None

        nearest = min(my_planets, key=lambda p: self.planet_distance(p, target_planet))
        return nearest

    def get_comet_leaving_soon(self, planets: List[Planet], comet_planet_ids: List[int], steps: int = 20) -> List[int]:
        """
        Identify comets that will leave the board soon.
        Simplified: assume comets near board edge are leaving soon.
        """
        leaving = []
        for pid in comet_planet_ids:
            planet = next((p for p in planets if p.id == pid), None)
            if planet:
                # Check if near edge
                if planet.x < 10 or planet.x > 90 or planet.y < 10 or planet.y > 90:
                    leaving.append(pid)
        return leaving

    def choose_target_planet(self, planets: List[Planet], my_planets: List[Planet]) -> Optional[Planet]:
        """Choose best target: prefer static neutral high-prod planets using orbit penalty."""
        neutral_planets = [p for p in planets if p.owner == -1]
        high_prod = [p for p in neutral_planets if 3 <= p.production <= 5]
        if not high_prod:
            high_prod = neutral_planets
        if not high_prod:
            high_prod = [p for p in planets if p.owner != self.player_id and p.owner != -1]
        if not high_prod:
            return None

        best_target = None
        best_score = float('inf')

        for target in high_prod:
            for my in my_planets:
                raw_dist = self.planet_distance(my, target)
                score = raw_dist * (2.5 if is_orbiting_planet(target.x, target.y, target.radius) else 1.0)
                if score < best_score:
                    best_score = score
                    best_target = target

        return best_target

    def compute_moves(self, obs) -> List[List]:
        """
        Main method to compute moves based on observation.
        Returns list of [from_planet_id, angle, num_ships]
        """
        self.turn += 1
        logger.info(f"Turn {self.turn}, player {self.player_id}")

        # Parse observation
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
        # Ensure they are not None
        if raw_planets is None:
            raw_planets = []
        if raw_fleets is None:
            raw_fleets = []
        angular_velocity = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity
        if angular_velocity is None:
            angular_velocity = 0
        initial_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
        if initial_planets is None:
            initial_planets = []
        comet_planet_ids = obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids
        if comet_planet_ids is None:
            comet_planet_ids = []

        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]

        my_planets = [p for p in planets if p.owner == player]
        if not my_planets:
            return []

        # Update home planet if not set
        if self.home_planet_id is None and my_planets:
            self.home_planet_id = my_planets[0].id

        home_planet = next((p for p in my_planets if p.id == self.home_planet_id), None)
        if home_planet is None and my_planets:
            home_planet = my_planets[0]
            self.home_planet_id = home_planet.id

        # Update comet IDs
        self.comet_ids = set(comet_planet_ids)

        moves = []

        # Strategy 2: Defense mechanism
        threats = self.detect_incoming_threats(fleets, my_planets)
        for planet_id, threat_fleets in threats.items():
            threatened_planet = next((p for p in my_planets if p.id == planet_id), None)
            if not threatened_planet:
                continue

            # Estimate total threat ships
            threat_ships = sum(f.ships for f in threat_fleets)
            defense_needed = threat_ships - threatened_planet.ships + 1

            if defense_needed > 0:
                # Find nearest ally planet to reinforce
                nearest_ally = self.find_nearest_ally_planet(threatened_planet,
                                                             [p for p in my_planets if p.id != planet_id])
                if nearest_ally and nearest_ally.ships > defense_needed:
                    angle = math.atan2(threatened_planet.y - nearest_ally.y,
                                      threatened_planet.x - nearest_ally.x)
                    ships_to_send = min(defense_needed, nearest_ally.ships - 1)
                    if ships_to_send > 0:
                        moves.append([nearest_ally.id, angle, ships_to_send])
                        logger.info(f"Defense: sending {ships_to_send} ships from {nearest_ally.id} to {planet_id}")

        # Strategy 1: Prioritize capturing high-production neutral planets
        target = self.choose_target_planet(planets, my_planets)
        if target:
            # Find nearest owned planet to target
            nearest_ally = self.find_nearest_ally_planet(target, my_planets)
            if nearest_ally:
                # Calculate ships needed: target.ships + 1-2 extra
                ships_needed = target.ships + 2  # +2 for safety

                # Ensure we don't send more than available (keep some for defense)
                available_ships = nearest_ally.ships
                if nearest_ally.id == self.home_planet_id:
                    # Strategy 3: Keep at least 20% in home planet
                    min_keep = max(1, int(available_ships * 0.2))
                    available_ships -= min_keep
                else:
                    # Keep at least 1 ship
                    available_ships -= 1

                if available_ships >= ships_needed:
                    # Strategy 5: Predict future position for orbiting planets
                    if target.id not in self.comet_ids and is_orbiting_planet(target.x, target.y, target.radius):
                        # Iterative convergence: aim → predict → re-aim
                        speed = compute_speed(max(1, ships_needed))
                        tx, ty = target.x, target.y
                        for _ in range(5):
                            fsteps = max(1, int(math.ceil(self.distance(nearest_ally.x, nearest_ally.y, tx, ty) / speed)))
                            ptx, pty = self.future_planet_position(
                                Planet(target.id, target.owner, tx, ty, target.radius, target.ships, target.production),
                                initial_planets, angular_velocity, fsteps)
                            if abs(ptx - tx) < 0.3 and abs(pty - ty) < 0.3:
                                tx, ty = ptx, pty
                                break
                            tx, ty = ptx, pty
                        angle = math.atan2(ty - nearest_ally.y, tx - nearest_ally.x)
                    else:
                        angle = math.atan2(target.y - nearest_ally.y,
                                          target.x - nearest_ally.x)

                    # Sun-crossing check before launching
                    lx = nearest_ally.x + math.cos(angle) * (nearest_ally.radius + 0.1)
                    ly = nearest_ally.y + math.sin(angle) * (nearest_ally.radius + 0.1)
                    hx = nearest_ally.x + math.cos(angle) * self.distance(nearest_ally.x, nearest_ally.y, target.x, target.y)
                    hy = nearest_ally.y + math.sin(angle) * self.distance(nearest_ally.x, nearest_ally.y, target.x, target.y)
                    if not segment_hits_sun(lx, ly, hx, hy):
                        moves.append([nearest_ally.id, angle, ships_needed])
                        logger.info(f"Attack: sending {ships_needed} ships from {nearest_ally.id} to target {target.id}")

        # Strategy 4: Comet utilization
        comet_planets = [p for p in planets if p.id in self.comet_ids]
        neutral_comets = [p for p in comet_planets if p.owner == -1]

        for comet in neutral_comets:
            nearest_ally = self.find_nearest_ally_planet(comet, my_planets)
            if nearest_ally:
                ships_needed = comet.ships + 2
                available_ships = nearest_ally.ships
                if nearest_ally.id == self.home_planet_id:
                    min_keep = max(1, int(available_ships * 0.2))
                    available_ships -= min_keep
                else:
                    available_ships -= 1

                if available_ships >= ships_needed:
                    angle = math.atan2(comet.y - nearest_ally.y,
                                      comet.x - nearest_ally.x)
                    moves.append([nearest_ally.id, angle, ships_needed])
                    logger.info(f"Comet capture: sending {ships_needed} ships to comet {comet.id}")

        # Strategy 4: Withdraw ships from comets leaving soon
        leaving_comets = self.get_comet_leaving_soon(planets, list(self.comet_ids))
        for comet_id in leaving_comets:
            comet_planet = next((p for p in planets if p.id == comet_id), None)
            if comet_planet and comet_planet.owner == self.player_id:
                # Withdraw all but 1 ship
                if comet_planet.ships > 1:
                    # Find nearest owned planet to send ships to
                    nearest_ally = self.find_nearest_ally_planet(comet_planet,
                                                                 [p for p in my_planets if p.id != comet_id])
                    if nearest_ally:
                        ships_to_send = comet_planet.ships - 1
                        angle = math.atan2(nearest_ally.y - comet_planet.y,
                                          nearest_ally.x - comet_planet.x)
                        moves.append([comet_id, angle, ships_to_send])
                        logger.info(f"Comet withdrawal: sending {ships_to_send} ships from comet {comet_id}")

        # Limit number of moves (game might have limit)
        if len(moves) > 10:
            moves = moves[:10]

        return moves


# Global agent instance for compatibility with Kaggle
_agent_instance = None


def heuristic_agent(obs, config=None):
    """
    Agent function compatible with Kaggle interface.
    """
    global _agent_instance

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    if _agent_instance is None:
        _agent_instance = HeuristicAgent(player_id=player)
    elif _agent_instance.player_id != player:
        # Player ID changed (shouldn't happen but just in case)
        _agent_instance = HeuristicAgent(player_id=player)

    try:
        return _agent_instance.compute_moves(obs)
    except Exception as e:
        # Log error to stderr (visible in Kaggle logs)
        import sys
        print(f"Heuristic agent error: {e}", file=sys.stderr)
        # Return empty moves to avoid crashing validation
        return []