"""
Trainer for Orbit Wars agents.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from .environment import OrbitWarsEnvironment
from .agent import BaseAgent
from .logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Trainer for running training episodes.
    """

    def __init__(
        self,
        env: OrbitWarsEnvironment,
        agent: BaseAgent,
        opponent_agents: Optional[List[BaseAgent]] = None,
        num_opponents: int = 1,
    ):
        """
        Initialize trainer.

        Args:
            env: Environment instance
            agent: Main agent to train/evaluate
            opponent_agents: List of opponent agents (if None, uses random agents)
            num_opponents: Number of opponents (if opponent_agents not provided)
        """
        self.env = env
        self.agent = agent
        self.opponent_agents = opponent_agents
        self.num_opponents = num_opponents

        if self.opponent_agents is None:
            # Create random opponents
            from .agent import RandomAgent
            self.opponent_agents = [RandomAgent(player_id=i+1) for i in range(num_opponents)]

        # All agents in the game
        self.all_agents = [self.agent] + self.opponent_agents

    def run_episode(
        self,
        render: bool = False,
        max_steps: int = 500,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single episode.

        Args:
            render: Whether to render the episode
            max_steps: Maximum steps per episode
            verbose: Print detailed logs

        Returns:
            Episode results
        """
        # Reset environment
        observations = self.env.reset()
        if observations is None:
            observations = [None] * len(self.all_agents)

        # Reset agents
        for agent in self.all_agents:
            agent.reset()

        # Get initial player IDs
        player_ids = []
        for i in range(len(self.all_agents)):
            pid = self.env.get_player_id(i)
            if pid is not None:
                self.all_agents[i].player_id = pid
            player_ids.append(self.all_agents[i].player_id)

        logger.info(f"Starting episode with agents: {[agent.name for agent in self.all_agents]}")
        logger.info(f"Player IDs: {player_ids}")

        step = 0
        total_rewards = [0.0] * len(self.all_agents)
        done = False

        while step < max_steps and not done:
            if verbose:
                logger.debug(f"Step {step}")

            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.all_agents):
                obs = observations[i] if i < len(observations) else None
                if obs is None:
                    actions.append([])  # Empty action
                else:
                    try:
                        action = agent.act(obs)
                        actions.append(action)
                    except Exception as e:
                        logger.error(f"Agent {agent.name} failed to act: {e}")
                        actions.append([])

            # Step environment
            observations, rewards, dones, info = self.env.step(actions)

            # Update total rewards
            for i, reward in enumerate(rewards):
                total_rewards[i] += reward

            # Check if episode is done
            done = all(dones) or any(dones)  # All done or any agent done

            step += 1

            if render and step % 50 == 0:
                self.env.render(mode="ipython", width=800, height=600)

        # Final results
        results = {
            "steps": step,
            "total_rewards": total_rewards,
            "agent_reward": total_rewards[0] if total_rewards else 0,
            "agent_name": self.agent.name,
            "opponent_names": [agent.name for agent in self.opponent_agents],
            "done_reason": "max_steps" if step >= max_steps else "early_termination",
        }

        logger.info(f"Episode finished after {step} steps. Agent reward: {total_rewards[0]:.2f}")

        return results

    def train(
        self,
        num_episodes: int = 100,
        render_every: int = 0,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple training episodes.

        Args:
            num_episodes: Number of episodes to run
            render_every: Render every N episodes (0 = never)
            verbose: Print detailed logs

        Returns:
            List of episode results
        """
        logger.info(f"Starting training for {num_episodes} episodes")

        results = []
        start_time = time.time()

        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")

            render = (render_every > 0 and (episode + 1) % render_every == 0)
            episode_result = self.run_episode(
                render=render,
                verbose=verbose,
            )
            episode_result["episode"] = episode + 1
            results.append(episode_result)

            # Log progress
            avg_reward = sum(r["agent_reward"] for r in results) / len(results)
            logger.info(f"Episode {episode + 1} reward: {episode_result['agent_reward']:.2f}, "
                       f"Average: {avg_reward:.2f}")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Average reward: {sum(r['agent_reward'] for r in results) / len(results):.2f}")

        return results