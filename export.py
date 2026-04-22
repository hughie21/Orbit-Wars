#!/usr/bin/env python3
"""
Model export utility for Orbit Wars.
Generates standalone main.py for Kaggle submission based on selected model.
"""

import argparse
import sys
import os

# Template for main.py with placeholders
MAIN_PY_TEMPLATE = '''import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# Constants from game
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
SAFETY = 1.3

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

{agent_code}

# Global agent instance
_agent_instance = None

def agent(obs, config=None):
    global _agent_instance
    try:
        if hasattr(obs, 'player'):
            player = obs.player
        else:
            player = obs.get("player", 0)

        if _agent_instance is None:
            _agent_instance = {agent_class_name}(player_id=player)
        elif _agent_instance.player_id != player:
            _agent_instance = {agent_class_name}(player_id=player)

        return _agent_instance.compute_moves(obs)
    except Exception as e:
        # Return empty moves on any error
        return []

# For local testing
if __name__ == "__main__":
    print("{agent_name} agent loaded successfully")
'''

# Agent code templates
AGENT_TEMPLATES = {
    "heuristic": {
        "class_name": "HeuristicAgent",
        "agent_code": '''
class HeuristicAgent:
    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self.home_planet_id = None
        self.comet_ids = set()
        self.turn = 0

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def planet_distance(self, p1: Planet, p2: Planet) -> float:
        return self.distance(p1.x, p1.y, p2.x, p2.y)

    def future_planet_position(self, planet: Planet, initial_planets, angular_velocity, steps: int = 10):
        # Find initial position from initial_planets
        initial = None
        for ip in initial_planets:
            if ip[0] == planet.id:  # id matches
                initial = ip
                break

        if initial is None:
            return planet.x, planet.y

        initial_x, initial_y = initial[2], initial[3]  # x, y
        center_x, center_y = 50, 50

        # Calculate radius from center
        radius = math.sqrt((initial_x - center_x) ** 2 + (initial_y - center_y) ** 2)
        if radius <= 0:
            return planet.x, planet.y

        # Initial angle
        initial_angle = math.atan2(initial_y - center_y, initial_x - center_x)

        # New angle after rotation
        new_angle = initial_angle + angular_velocity * steps

        # New position
        new_x = center_x + radius * math.cos(new_angle)
        new_y = center_y + radius * math.sin(new_angle)

        return new_x, new_y

    def detect_incoming_threats(self, fleets, my_planets):
        threats = {}
        for fleet in fleets:
            if fleet.owner == self.player_id:
                continue

            # Simplified threat detection
            for planet in my_planets:
                dist = self.distance(fleet.x, fleet.y, planet.x, planet.y)
                if dist < 20:
                    if planet.id not in threats:
                        threats[planet.id] = []
                    threats[planet.id].append(fleet)

        return threats

    def find_nearest_ally_planet(self, target_planet, my_planets):
        if not my_planets:
            return None
        return min(my_planets, key=lambda p: self.planet_distance(p, target_planet))

    def get_comet_leaving_soon(self, planets, comet_planet_ids, steps: int = 20):
        leaving = []
        for pid in comet_planet_ids:
            planet = next((p for p in planets if p.id == pid), None)
            if planet:
                if planet.x < 10 or planet.x > 90 or planet.y < 10 or planet.y > 90:
                    leaving.append(pid)
        return leaving

    def choose_target_planet(self, planets, my_planets):
        neutral_planets = [p for p in planets if p.owner == -1]
        high_prod = [p for p in neutral_planets if 3 <= p.production <= 5]

        if not high_prod:
            high_prod = neutral_planets

        if not high_prod:
            enemy_planets = [p for p in planets if p.owner != self.player_id and p.owner != -1]
            high_prod = enemy_planets

        if not high_prod:
            return None

        # Find closest target
        best_target = None
        best_distance = float('inf')
        for target in high_prod:
            for my in my_planets:
                dist = self.planet_distance(my, target)
                if dist < best_distance:
                    best_distance = dist
                    best_target = target
        return best_target

    def compute_moves(self, obs):
        self.turn += 1

        # Parse observation
        if hasattr(obs, 'player'):
            player = obs.player
            raw_planets = obs.planets if obs.planets else []
            raw_fleets = obs.fleets if obs.fleets else []
            angular_velocity = obs.angular_velocity
            initial_planets = obs.initial_planets if hasattr(obs, 'initial_planets') else []
            comet_planet_ids = obs.comet_planet_ids if hasattr(obs, 'comet_planet_ids') else []
        else:
            player = obs.get("player", 0)
            raw_planets = obs.get("planets", [])
            raw_fleets = obs.get("fleets", [])
            angular_velocity = obs.get("angular_velocity", 0)
            initial_planets = obs.get("initial_planets", [])
            comet_planet_ids = obs.get("comet_planet_ids", [])

        planets = [Planet(*p) for p in raw_planets] if raw_planets else []
        fleets = [Fleet(*f) for f in raw_fleets] if raw_fleets else []

        my_planets = [p for p in planets if p.owner == player]
        if not my_planets:
            return []

        # Update home planet if not set
        if self.home_planet_id is None:
            self.home_planet_id = my_planets[0].id

        home_planet = next((p for p in my_planets if p.id == self.home_planet_id), None)
        if home_planet is None:
            home_planet = my_planets[0]
            self.home_planet_id = home_planet.id

        # Update comet IDs
        self.comet_ids = set(comet_planet_ids) if comet_planet_ids else set()

        moves = []

        # Strategy 2: Defense mechanism
        threats = self.detect_incoming_threats(fleets, my_planets)
        for planet_id, threat_fleets in threats.items():
            threatened_planet = next((p for p in my_planets if p.id == planet_id), None)
            if not threatened_planet:
                continue

            threat_ships = sum(f.ships for f in threat_fleets)
            defense_needed = threat_ships - threatened_planet.ships + 1

            if defense_needed > 0:
                nearest_ally = self.find_nearest_ally_planet(threatened_planet,
                                                             [p for p in my_planets if p.id != planet_id])
                if nearest_ally and nearest_ally.ships > defense_needed:
                    angle = math.atan2(threatened_planet.y - nearest_ally.y,
                                      threatened_planet.x - nearest_ally.x)
                    ships_to_send = min(defense_needed, nearest_ally.ships - 1)
                    if ships_to_send > 0:
                        moves.append([nearest_ally.id, angle, ships_to_send])

        # Strategy 1: Prioritize capturing high-production neutral planets
        target = self.choose_target_planet(planets, my_planets)
        if target:
            nearest_ally = self.find_nearest_ally_planet(target, my_planets)
            if nearest_ally:
                ships_needed = target.ships + 2
                available_ships = nearest_ally.ships
                if nearest_ally.id == self.home_planet_id:
                    min_keep = max(1, int(available_ships * 0.2))
                    available_ships -= min_keep
                else:
                    available_ships -= 1

                if available_ships >= ships_needed:
                    if target.id not in self.comet_ids:
                        center_dist = self.distance(target.x, target.y, 50, 50)
                        if center_dist < 40:
                            future_x, future_y = self.future_planet_position(
                                target, initial_planets, angular_velocity, 10)
                            angle = math.atan2(future_y - nearest_ally.y,
                                              future_x - nearest_ally.x)
                        else:
                            angle = math.atan2(target.y - nearest_ally.y,
                                              target.x - nearest_ally.x)
                    else:
                        angle = math.atan2(target.y - nearest_ally.y,
                                          target.x - nearest_ally.x)

                    moves.append([nearest_ally.id, angle, ships_needed])

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

        # Withdraw ships from comets leaving soon
        leaving_comets = self.get_comet_leaving_soon(planets, list(self.comet_ids))
        for comet_id in leaving_comets:
            comet_planet = next((p for p in planets if p.id == comet_id), None)
            if comet_planet and comet_planet.owner == self.player_id:
                if comet_planet.ships > 1:
                    nearest_ally = self.find_nearest_ally_planet(comet_planet,
                                                                 [p for p in my_planets if p.id != comet_id])
                    if nearest_ally:
                        ships_to_send = comet_planet.ships - 1
                        angle = math.atan2(nearest_ally.y - comet_planet.y,
                                          nearest_ally.x - comet_planet.x)
                        moves.append([comet_id, angle, ships_to_send])

        # Limit number of moves
        if len(moves) > 10:
            moves = moves[:10]

        return moves
'''
    },
    "random": {
        "class_name": "RandomAgent",
        "agent_code": '''
import random

class RandomAgent:
    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self.random = random.Random()

    def compute_moves(self, obs):
        # Parse observation
        if hasattr(obs, 'player'):
            player = obs.player
            raw_planets = obs.planets if obs.planets else []
        else:
            player = obs.get("player", 0)
            raw_planets = obs.get("planets", [])

        planets = [Planet(*p) for p in raw_planets] if raw_planets else []
        my_planets = [p for p in planets if p.owner == player]
        if not my_planets:
            return []

        moves = []
        # Random number of moves (0-2)
        num_moves = self.random.randint(0, 2)
        for _ in range(num_moves):
            source = self.random.choice(my_planets)
            if source.ships <= 1:
                continue

            # Random target (any planet except source)
            target = self.random.choice([p for p in planets if p.id != source.id])
            angle = math.atan2(target.y - source.y, target.x - source.x)
            ships = self.random.randint(1, min(10, source.ships - 1))

            moves.append([source.id, angle, ships])

        return moves
'''
    }
}

def generate_main_py(model_type):
    """Generate main.py content for given model type."""
    if model_type not in AGENT_TEMPLATES:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(AGENT_TEMPLATES.keys())}")

    template = AGENT_TEMPLATES[model_type]
    agent_code = template["agent_code"]
    agent_class_name = template["class_name"]
    agent_name = model_type.capitalize()

    content = MAIN_PY_TEMPLATE.format(
        agent_code=agent_code,
        agent_class_name=agent_class_name,
        agent_name=agent_name
    )
    return content

def main():
    parser = argparse.ArgumentParser(description='Export Orbit Wars model for Kaggle submission')
    parser.add_argument('model', choices=['heuristic', 'random'],
                       help='Model type to export')
    parser.add_argument('--output', '-o', default='main.py',
                       help='Output file path (default: main.py)')
    parser.add_argument('--pack', '-p', action='store_true',
                       help='Automatically run pack.py to create submission.zip')

    args = parser.parse_args()

    print(f"Generating {args.model} agent to {args.output}...")

    try:
        content = generate_main_py(args.model)
        with open(args.output, 'w') as f:
            f.write(content)
        print(f"Successfully generated {args.output}")

        if args.pack:
            print("Running pack.py...")
            import subprocess
            result = subprocess.run([sys.executable, 'pack.py'],
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print("Successfully created submission.zip")
            else:
                print(f"pack.py failed: {result.stderr}")
                sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()