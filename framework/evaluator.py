"""
Evaluator for Orbit Wars agents.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import statistics
from .environment import OrbitWarsEnvironment
from .agent import BaseAgent
from .logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluator for benchmarking agents.
    """

    def __init__(
        self,
        env: OrbitWarsEnvironment,
        agents: List[BaseAgent],
        agent_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            env: Environment instance
            agents: List of agents to evaluate
            agent_names: Optional names for agents (defaults to agent class names)
        """
        self.env = env
        self.agents = agents
        self.agent_names = agent_names or [agent.name for agent in agents]

        if len(self.agents) != len(self.agent_names):
            raise ValueError("Number of agents must match number of agent names")

    def evaluate_matchup(
        self,
        agent_indices: List[int],
        num_episodes: int = 10,
        max_steps: int = 500,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a specific matchup of agents.

        Args:
            agent_indices: Indices of agents in self.agents to use in the match
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            verbose: Print detailed logs

        Returns:
            Evaluation results
        """
        selected_agents = [self.agents[i] for i in agent_indices]
        selected_names = [self.agent_names[i] for i in agent_indices]

        logger.info(f"Evaluating matchup: {selected_names}")

        results = {
            "agents": selected_names,
            "episodes": [],
            "total_wins": [0] * len(selected_agents),
            "total_rewards": [0.0] * len(selected_agents),
            "total_steps": [],
        }

        for episode in range(num_episodes):
            logger.debug(f"Matchup episode {episode + 1}/{num_episodes}")

            # Reset environment with selected agents
            observations = self.env.reset()
            if observations is None:
                observations = [None] * len(selected_agents)

            # Reset agents
            for agent in selected_agents:
                agent.reset()

            # Update player IDs
            for i, agent in enumerate(selected_agents):
                pid = self.env.get_player_id(i)
                if pid is not None:
                    agent.player_id = pid

            step = 0
            rewards = [0.0] * len(selected_agents)
            done = False

            while step < max_steps and not done:
                # Get actions
                actions = []
                for i, agent in enumerate(selected_agents):
                    obs = observations[i] if i < len(observations) else None
                    if obs is None:
                        actions.append([])
                    else:
                        try:
                            action = agent.act(obs)
                            actions.append(action)
                        except Exception as e:
                            logger.error(f"Agent {agent.name} failed: {e}")
                            actions.append([])

                # Step
                observations, step_rewards, dones, info = self.env.step(actions)

                # Accumulate rewards
                for i, reward in enumerate(step_rewards):
                    rewards[i] += reward

                done = all(dones) or any(dones)
                step += 1

            # Determine winner (highest reward)
            winner_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            results["total_wins"][winner_idx] += 1
            for i, reward in enumerate(rewards):
                results["total_rewards"][i] += reward
            results["total_steps"].append(step)

            episode_result = {
                "episode": episode + 1,
                "steps": step,
                "rewards": rewards.copy(),
                "winner": winner_idx,
                "winner_name": selected_names[winner_idx],
            }
            results["episodes"].append(episode_result)

            if verbose:
                logger.info(f"Episode {episode + 1}: {selected_names[winner_idx]} wins with reward {rewards[winner_idx]:.2f}")

        # Calculate statistics
        results["win_rates"] = [wins / num_episodes for wins in results["total_wins"]]
        results["avg_rewards"] = [total / num_episodes for total in results["total_rewards"]]
        results["avg_steps"] = statistics.mean(results["total_steps"]) if results["total_steps"] else 0

        logger.info(f"Matchup completed. Win rates: {dict(zip(selected_names, results['win_rates']))}")
        logger.info(f"Average rewards: {dict(zip(selected_names, results['avg_rewards']))}")

        return results

    def evaluate_agent(
        self,
        agent_index: int,
        opponents: List[int],
        num_episodes: int = 20,
        max_steps: int = 500,
    ) -> Dict[str, Any]:
        """
        Evaluate a single agent against multiple opponents.

        Args:
            agent_index: Index of agent to evaluate
            opponents: List of opponent indices
            num_episodes: Number of episodes per opponent
            max_steps: Maximum steps per episode

        Returns:
            Evaluation results
        """
        agent_name = self.agent_names[agent_index]
        logger.info(f"Evaluating agent {agent_name} against {len(opponents)} opponents")

        all_results = []

        for opp_idx in opponents:
            matchup_indices = [agent_index, opp_idx]
            results = self.evaluate_matchup(
                matchup_indices,
                num_episodes=num_episodes,
                max_steps=max_steps,
                verbose=False,
            )
            results["opponent"] = self.agent_names[opp_idx]
            all_results.append(results)

        # Aggregate results
        total_wins = 0
        total_episodes = 0
        total_reward = 0.0

        for result in all_results:
            # Agent is always at index 0 in matchup results
            total_wins += result["total_wins"][0]
            total_reward += result["total_rewards"][0]
            total_episodes += len(result["episodes"])

        win_rate = total_wins / total_episodes if total_episodes > 0 else 0
        avg_reward = total_reward / total_episodes if total_episodes > 0 else 0

        summary = {
            "agent": agent_name,
            "total_episodes": total_episodes,
            "total_wins": total_wins,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "matchups": all_results,
        }

        logger.info(f"Evaluation completed for {agent_name}: "
                   f"win rate={win_rate:.3f}, avg_reward={avg_reward:.2f}")

        return summary