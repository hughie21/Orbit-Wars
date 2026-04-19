#!/usr/bin/env python3
"""
Training script for Orbit Wars agents.
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
from framework.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Orbit Wars agent")
    parser.add_argument("--agent", type=str, default="heuristic",
                        help="Agent type (heuristic, random)")
    parser.add_argument("--opponent", type=str, default="random",
                        help="Opponent agent type")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of training episodes")
    parser.add_argument("--render_every", type=int, default=0,
                        help="Render every N episodes (0 = never)")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="Log directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logger(
        name="orbit_wars_train",
        log_dir=args.log_dir,
        level=logging.DEBUG if args.verbose else logging.INFO,
        console=True,
        file=True,
        log_prefix="train"
    )

    logger.info("Starting Orbit Wars training")
    logger.info(f"Arguments: {vars(args)}")

    # Create environment
    env = OrbitWarsEnvironment(debug=args.debug)
    logger.info("Environment created")

    # Create agents
    agent = load_agent(args.agent)
    opponent = load_agent(args.opponent)

    logger.info(f"Main agent: {agent.name}")
    logger.info(f"Opponent agent: {opponent.name}")

    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        opponent_agents=[opponent],
        num_opponents=1,
    )

    # Run training episodes
    results = trainer.train(
        num_episodes=args.num_episodes,
        render_every=args.render_every,
        verbose=args.verbose,
    )

    # Print summary
    avg_reward = sum(r["agent_reward"] for r in results) / len(results)
    logger.info("=" * 50)
    logger.info("Training Summary:")
    logger.info(f"  Episodes: {len(results)}")
    logger.info(f"  Average reward: {avg_reward:.2f}")
    logger.info(f"  Agent: {agent.name}")
    logger.info(f"  Opponent: {opponent.name}")
    logger.info("=" * 50)

    # Save results to file
    import json
    results_file = os.path.join(args.log_dir, "training_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    logger.info("Training completed")


if __name__ == "__main__":
    main()