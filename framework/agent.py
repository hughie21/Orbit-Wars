"""
Agent base class and implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from model.heuristic_agent import heuristic_agent as heuristic_agent_fn

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self.name = self.__class__.__name__

    @abstractmethod
    def act(self, observation: Dict) -> List[List]:
        """
        Choose action based on observation.

        Args:
            observation: Observation dict

        Returns:
            List of moves [[from_planet_id, angle, num_ships], ...]
        """
        pass

    def reset(self):
        """Reset agent state for new episode."""
        pass

    def __call__(self, observation: Dict) -> List[List]:
        """Make agent callable like a function."""
        return self.act(observation)


class HeuristicAgent(BaseAgent):
    """
    Wrapper for the heuristic agent implementation.
    """

    def __init__(self, player_id: int = 0):
        super().__init__(player_id)
        self.name = "HeuristicAgent"

    def act(self, observation: Dict) -> List[List]:
        """
        Use the heuristic agent implementation.

        Args:
            observation: Observation dict

        Returns:
            List of moves
        """
        return heuristic_agent_fn(observation, None)


class RandomAgent(BaseAgent):
    """
    Random agent for baseline.
    """

    def __init__(self, player_id: int = 0):
        super().__init__(player_id)
        self.name = "RandomAgent"
        import random
        self.random = random

    def act(self, observation: Dict) -> List[List]:
        """
        Random action.

        Args:
            observation: Observation dict

        Returns:
            Random moves (0-2 moves per turn)
        """
        import math
        player = observation.get("player", 0) if isinstance(observation, dict) else observation.player
        raw_planets = observation.get("planets", []) if isinstance(observation, dict) else observation.planets

        # Parse planets
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
        planets = [Planet(*p) for p in raw_planets]

        # Get planets we own
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


def load_agent(agent_type: str = "heuristic", **kwargs) -> BaseAgent:
    """
    Factory function to load an agent.

    Args:
        agent_type: Type of agent ("heuristic", "random", etc.)
        **kwargs: Agent-specific arguments

    Returns:
        Agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == "heuristic":
        return HeuristicAgent(**kwargs)
    elif agent_type == "random":
        return RandomAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")