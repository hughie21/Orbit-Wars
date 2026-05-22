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
    parser.add_argument(
        "--agent",
        type=str,
        default="heuristic",
        help="Agent type (heuristic, random, mcts, ppo)",
    )
    parser.add_argument(
        "--opponent", type=str, default="random", help="Opponent agent type"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of training episodes"
    )
    parser.add_argument(
        "--render_every",
        type=int,
        default=0,
        help="Render every N episodes (0 = never)",
    )
    parser.add_argument("--log_dir", type=str, default="./log", help="Log directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    # PPO-specific arguments
    parser.add_argument(
        "--ppo_train", action="store_true", help="Use PPOTrainer for PPO weight updates"
    )
    parser.add_argument(
        "--ppo_lr", type=float, default=3e-4, help="PPO learning rate"
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=10, help="PPO update epochs"
    )
    parser.add_argument(
        "--save_every", type=int, default=50,
        help="Save model checkpoint every N episodes (PPO)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to a checkpoint .pt file to resume training from"
    )
    parser.add_argument(
        "--opponent_mix", type=str, default=None,
        help="Secondary opponent type for mixing (e.g., 'random' to mix with main opponent)"
    )
    parser.add_argument(
        "--opponent_mix_ratio", type=float, default=0.3,
        help="Probability of using the mix opponent (0.0-1.0, default 0.3)"
    )
    parser.add_argument(
        "--ppo_target_kl", type=float, default=0.02,
        help="Target KL divergence for early stopping (lower = more conservative updates)"
    )
    parser.add_argument(
        "--reset_value_head", action="store_true",
        help="Reset value head when loading checkpoint (use when switching opponent type)"
    )
    # Strategic agent learning arguments
    parser.add_argument(
        "--strategic_learn", action="store_true",
        help="Enable online learning for strategic agent (opponent model + value network)"
    )
    parser.add_argument(
        "--strategic_load", type=str, default=None,
        help="Path to load pre-trained strategic model weights (.pkl)"
    )
    parser.add_argument(
        "--strategic_save", type=str, default=None,
        help="Path to save strategic model weights after training (.pkl)"
    )
    parser.add_argument(
        "--strategic_train_every", type=int, default=5,
        help="Train strategic models every N episodes (default 5)"
    )
    parser.add_argument(
        "--strategic_opponent_load", type=str, default=None,
        help="Path to load opponent strategic agent weights (.pkl) for self-play"
    )
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
        log_prefix="train",
    )

    logger.info("Starting Orbit Wars training")
    logger.info(f"Arguments: {vars(args)}")

    # Create environment
    env = OrbitWarsEnvironment(debug=args.debug)
    logger.info("Environment created")

    # Create agents
    agent_kwargs = {}
    if args.load and args.agent == "ppo":
        agent_kwargs["load_path"] = args.load
        logger.info(f"Loading model from: {args.load}")

    # Strategic agent: enable learning and load pre-trained weights
    if args.agent == "strategic" and args.strategic_learn:
        agent_kwargs["enable_learning"] = True
        logger.info("Strategic agent learning enabled")
    elif args.agent == "strategic":
        agent_kwargs["enable_learning"] = False
        logger.info("Strategic agent learning disabled (use --strategic_learn to enable)")

    agent = load_agent(args.agent, **agent_kwargs)

    # Load strategic model weights if provided
    if args.agent == "strategic" and args.strategic_load:
        if agent.load_weights(args.strategic_load):
            logger.info(f"Loaded strategic weights from: {args.strategic_load}")
        else:
            logger.warning(f"Could not load strategic weights from: {args.strategic_load}")

    opponent = load_agent(args.opponent)

    # Load opponent strategic weights for self-play training
    if args.opponent == "strategic" and args.strategic_opponent_load:
        if opponent.load_weights(args.strategic_opponent_load):
            logger.info(f"Loaded opponent strategic weights from: {args.strategic_opponent_load}")
        else:
            logger.warning(f"Could not load opponent strategic weights from: {args.strategic_opponent_load}")

    # Opponent mixing: secondary opponent for curriculum training
    opponent_mix = None
    if args.opponent_mix:
        opponent_mix = load_agent(args.opponent_mix)
        logger.info(f"Mix opponent: {opponent_mix.name} (ratio={args.opponent_mix_ratio})")

    # Reset value head when switching opponent types (prevents death spiral)
    if args.reset_value_head and args.load and args.agent == "ppo":
        agent.reset_value_head()
        logger.info("Value head reset (--reset_value_head)")
        # Also reset optimizer state for the value head
        if args.ppo_lr:
            logger.info(f"Using fine-tune LR: {args.ppo_lr}")

    logger.info(f"Main agent: {agent.name}")
    logger.info(f"Opponent agent: {opponent.name}")

    # PPO training path
    if args.agent == "ppo" and args.ppo_train:
        from model.ppo_agent import PPOTrainer

        # Build opponent list and probability weights for mixing
        if opponent_mix:
            opponents = [opponent, opponent_mix]
            probs = [1.0 - args.opponent_mix_ratio, args.opponent_mix_ratio]
            logger.info(f"Opponent mix: {opponent.name}({probs[0]:.0%}) + {opponent_mix.name}({probs[1]:.0%})")
        else:
            opponents = [opponent]
            probs = None

        trainer = PPOTrainer(
            env=env,
            agent=agent,
            opponent_agents=opponents,
            opponent_probs=probs,
            lr=args.ppo_lr,
            update_epochs=args.ppo_epochs,
            target_kl=args.ppo_target_kl,
            log_dir=args.log_dir,
        )
        results = trainer.train(
            num_episodes=args.num_episodes,
            render_every=args.render_every,
            save_every=args.save_every,
            save_dir=args.save_dir,
        )
    else:
        # Standard training (heuristic, random, mcts, or ppo eval-only)
        trainer = Trainer(
            env=env,
            agent=agent,
            opponent_agents=[opponent],
            num_opponents=1,
        )
        results = trainer.train(
            num_episodes=args.num_episodes,
            render_every=args.render_every,
            verbose=args.verbose,
        )

    # ── Strategic agent: post-training model updates ──────────────────
    if args.agent == "strategic" and args.strategic_learn and results:
        logger.info("Training strategic models on episode outcomes...")
        wins = 0
        for ep_idx, ep_result in enumerate(results):
            # Determine win/loss: agent is player 0, reward > 0 means won
            agent_reward = ep_result.get("agent_reward", 0)
            won = agent_reward > 0
            if won:
                wins += 1
            agent.record_episode_outcome(won)

            # Train models periodically
            if (ep_idx + 1) % args.strategic_train_every == 0:
                agent.train_models()
                logger.debug(f"  Trained models after episode {ep_idx + 1}")

        # Final training pass
        agent.train_models()
        logger.info(f"Strategic training complete: {wins}/{len(results)} wins ({wins/len(results)*100:.1f}%)")

        # Save weights
        if args.strategic_save:
            agent.save_weights(args.strategic_save)
            logger.info(f"Saved strategic weights to: {args.strategic_save}")

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

