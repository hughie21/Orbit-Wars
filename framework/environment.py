"""
Environment wrapper for Orbit Wars.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from kaggle_environments import make

logger = logging.getLogger(__name__)


class OrbitWarsEnvironment:
    """
    Wrapper for Orbit Wars Kaggle environment.
    """

    def __init__(self, debug: bool = False, configuration: Optional[Dict] = None):
        """
        Initialize the environment.

        Args:
            debug: Enable debug mode
            configuration: Environment configuration dict
        """
        self.debug = debug
        self.config = configuration or {
            "episodeSteps": 500,
            "actTimeout": 1,
            "shipSpeed": 6.0,
            "sunRadius": 10.0,
            "boardSize": 100.0,
            "cometSpeed": 4.0,
        }

        self.env = make("orbit_wars", debug=self.debug, configuration=self.config)
        self.state = None

    def reset(self, agents: List[Any] = None) -> List[Dict]:
        """
        Reset the environment.

        Args:
            agents: List of agents (functions or strings). If None, uses placeholder agents.

        Returns:
            Initial observations for all agents
        """
        if agents is None:
            agents = ["random", "random"]

        self.state = self.env.reset(num_agents=len(agents))
        return self.state

    def step(self, actions: List[List]) -> Tuple[List[Dict], List[float], List[bool], Dict]:
        """
        Take a step in the environment.

        Args:
            actions: List of actions for each agent

        Returns:
            observations, rewards, dones, info
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self.state = self.env.step(actions)
        observations = [agent.observation for agent in self.state]
        rewards = [agent.reward for agent in self.state]
        dones = [agent.status != "ACTIVE" for agent in self.state]
        info = {
            "state": self.state,
            "step": self.env.steps
        }

        return observations, rewards, dones, info

    def run(self, agents: List[Any], render: bool = False) -> Dict:
        """
        Run a full episode with given agents.

        Args:
            agents: List of agents
            render: Whether to render the episode

        Returns:
            Episode results
        """
        self.state = self.env.run(agents)

        # Collect results
        results = {
            "steps": self.env.steps,
            "final_state": self.state[-1] if self.state else None,
            "rewards": [agent.reward for agent in self.state[-1]] if self.state else [],
            "statuses": [agent.status for agent in self.state[-1]] if self.state else [],
        }

        if render:
            self.env.render(mode="ipython", width=800, height=600)

        return results

    def get_observation(self, agent_idx: int = 0) -> Optional[Dict]:
        """
        Get observation for a specific agent.

        Args:
            agent_idx: Agent index

        Returns:
            Observation dict or None if not available
        """
        if self.state is None or agent_idx >= len(self.state):
            return None

        return self.state[agent_idx].observation

    def get_player_id(self, agent_idx: int = 0) -> Optional[int]:
        """
        Get player ID for a specific agent from observation.

        Args:
            agent_idx: Agent index

        Returns:
            Player ID or None
        """
        obs = self.get_observation(agent_idx)
        if obs is None:
            return None

        return obs.get("player", 0) if isinstance(obs, dict) else obs.player

    def render(self, mode: str = "ipython", **kwargs):
        """
        Render the current state.

        Args:
            mode: Rendering mode
            **kwargs: Additional arguments for render
        """
        self.env.render(mode=mode, **kwargs)