#!/usr/bin/env python3
"""
Testing/evaluation script for Orbit Wars agents.
"""

import argparse
import logging
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.logger import setup_logger
from framework.environment import OrbitWarsEnvironment
from framework.agent import load_agent
from framework.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Orbit Wars agents")
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["heuristic", "random"],
                        help="Agent types to evaluate")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes per matchup")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="Log directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--output", type=str, default="./eval_results.json",
                        help="Output file for results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logger(
        name="orbit_wars_eval",
        log_dir=args.log_dir,
        level=logging.DEBUG if args.verbose else logging.INFO,
        console=True,
        file=True,
        log_prefix="eval"
    )

    logger.info("Starting Orbit Wars evaluation")
    logger.info(f"Arguments: {vars(args)}")

    # Create environment
    env = OrbitWarsEnvironment(debug=args.debug)
    logger.info("Environment created")

    # Create agents
    agents = []
    agent_names = []
    for i, agent_type in enumerate(args.agents):
        agent = load_agent(agent_type)
        agents.append(agent)
        agent_names.append(f"{agent_type}_{i}")

    logger.info(f"Agents: {agent_names}")

    # Create evaluator
    evaluator = Evaluator(
        env=env,
        agents=agents,
        agent_names=agent_names,
    )

    # Evaluate all matchups
    all_results = []

    # Evaluate each agent against others
    for i, agent_name in enumerate(agent_names):
        # Opponents are all other agents
        opponents = [j for j in range(len(agents)) if j != i]

        logger.info(f"Evaluating {agent_name} against {len(opponents)} opponents")

        results = evaluator.evaluate_agent(
            agent_index=i,
            opponents=opponents,
            num_episodes=args.num_episodes,
        )
        all_results.append(results)

    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary:")
    for result in all_results:
        logger.info(f"  Agent: {result['agent']}")
        logger.info(f"    Win rate: {result['win_rate']:.3f}")
        logger.info(f"    Avg reward: {result['avg_reward']:.2f}")
        logger.info(f"    Episodes: {result['total_episodes']}")
        logger.info("")

    logger.info("=" * 60)

    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()