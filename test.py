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
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["heuristic", "random"],
        help="Agent types to evaluate (heuristic, random, mcts, ppo)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20, help="Number of episodes per matchup"
    )
    parser.add_argument("--log_dir", type=str, default="./log", help="Log directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--output",
        type=str,
        default="./eval_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--strategic_load", type=str, default=None,
        help="Path to load pre-trained strategic model weights (.pkl)"
    )
    parser.add_argument(
        "--strategic_learn", action="store_true",
        help="Enable online learning for strategic agent during evaluation"
    )
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
        log_prefix="eval",
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
        kwargs = {}
        if agent_type == "strategic":
            kwargs["enable_learning"] = args.strategic_learn
        agent = load_agent(agent_type, **kwargs)
        # Load strategic weights if provided (only for first strategic agent)
        if agent_type == "strategic" and args.strategic_load:
            if agent.load_weights(args.strategic_load):
                logger.info(f"Loaded strategic weights from: {args.strategic_load}")
            else:
                logger.warning(f"Could not load strategic weights from: {args.strategic_load}")
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

    # ── Strategic agent: post-evaluation training ─────────────────────
    if args.strategic_learn:
        for i, agent in enumerate(agents):
            if agent.name == "StrategicAgent" and agent.enable_learning:
                # Find results for this agent
                for result in all_results:
                    if result.get("agent") == agent_names[i]:
                        for ep in result.get("episodes", []):
                            # Agent is index 0 in matchup, check if it won
                            won = ep.get("winner", -1) == 0
                            agent.record_episode_outcome(won)
                        agent.train_models()
                        logger.info(f"Trained strategic agent {agent_names[i]} on evaluation outcomes")

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

