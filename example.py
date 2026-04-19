#!/usr/bin/env python3
"""
Example usage of the Orbit Wars training framework.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.logger import setup_logger
from framework.environment import OrbitWarsEnvironment
from framework.agent import HeuristicAgent, RandomAgent
from framework.trainer import Trainer
from framework.evaluator import Evaluator


def example_basic():
    """Basic example: run a single episode."""
    print("=== Example 1: Basic Episode ===")

    # Setup logging
    logger = setup_logger(console=True, file=False)

    # Create environment
    env = OrbitWarsEnvironment(debug=False)

    # Create agents
    agent1 = HeuristicAgent(player_id=0)
    agent2 = RandomAgent(player_id=1)

    # Run a single episode
    results = env.run([agent1, agent2])

    print(f"Episode completed in {len(env.env.steps)} steps")
    print(f"Final rewards: {[agent.reward for agent in results[-1]]}")
    print()


def example_training():
    """Example: training against random opponent."""
    print("=== Example 2: Training ===")

    # Setup logging
    logger = setup_logger(console=True, file=False)

    # Create environment
    env = OrbitWarsEnvironment(debug=False)

    # Create agents
    agent = HeuristicAgent(player_id=0)
    opponent = RandomAgent(player_id=1)

    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        opponent_agents=[opponent],
    )

    # Run short training
    results = trainer.train(num_episodes=3, render_every=0, verbose=False)

    print(f"Training completed: {len(results)} episodes")
    for i, r in enumerate(results):
        print(f"  Episode {i+1}: reward = {r['agent_reward']:.2f}")
    print()


def example_evaluation():
    """Example: evaluating multiple agents."""
    print("=== Example 3: Evaluation ===")

    # Setup logging
    logger = setup_logger(console=True, file=False)

    # Create environment
    env = OrbitWarsEnvironment(debug=False)

    # Create multiple agents
    agents = [
        HeuristicAgent(player_id=0),
        RandomAgent(player_id=1),
        RandomAgent(player_id=2),
    ]
    agent_names = ["Heuristic", "Random1", "Random2"]

    # Create evaluator
    evaluator = Evaluator(
        env=env,
        agents=agents,
        agent_names=agent_names,
    )

    # Evaluate heuristic agent against random agents
    results = evaluator.evaluate_agent(
        agent_index=0,
        opponents=[1, 2],
        num_episodes=2,  # Small number for example
    )

    print(f"Agent: {results['agent']}")
    print(f"Win rate: {results['win_rate']:.3f}")
    print(f"Average reward: {results['avg_reward']:.2f}")
    print()


def main():
    """Run all examples."""
    print("Orbit Wars Framework Examples")
    print("=" * 40)
    print()

    # Note: These examples require kaggle_environments to be installed
    try:
        from kaggle_environments import make
        example_basic()
        example_training()
        example_evaluation()

        print("All examples completed successfully!")
        print()
        print("Next steps:")
        print("1. Run full training: python train.py --agent heuristic --num_episodes 100")
        print("2. Evaluate agents: python test.py --agents heuristic random --num_episodes 50")
        print("3. Submit to Kaggle: kaggle competitions submit orbit-wars -f main.py")

    except ImportError as e:
        print("Error: kaggle_environments not installed.")
        print("Please install it first:")
        print("  pip install -r requirements.txt")
        print("  or")
        print("  pip install kaggle_environments")
        sys.exit(1)


if __name__ == "__main__":
    main()