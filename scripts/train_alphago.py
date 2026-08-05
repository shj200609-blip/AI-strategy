#!/usr/bin/env python3
"""CLI entry point for AlphaGo Zero battle AI training.

Usage:
    # Quick test with small settings
    python scripts/train_alphago.py --sims 100 --games 10 --iterations 5 \
        --train-steps 100 --device cpu

    # Full training run
    python scripts/train_alphago.py --sims 800 --games 100 --iterations 100 \
        --device cuda --checkpoint-dir checkpoints/run1

    # Resume from checkpoint
    python scripts/train_alphago.py --resume checkpoints/run1/best.pt \
        --iterations 50
"""

import argparse
import json
import os
import sys

# Ensure the alphago-battle-ai package is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alphago.config import AlphaGoConfig
from alphago.pipeline import AlphaGoPipeline


def load_json_config(path: str) -> dict:
    """Load a battle configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AlphaGo Zero battle AI training for fheroes2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── MCTS ────────────────────────────────────────────────────────
    mcts = p.add_argument_group("MCTS")
    mcts.add_argument("--sims", type=int, default=800,
                      help="MCTS simulations per move")
    mcts.add_argument("--cpuct", type=float, default=2.5,
                      help="PUCT exploration constant")
    mcts.add_argument("--dirichlet-alpha", type=float, default=0.03,
                      help="Dirichlet noise concentration")
    mcts.add_argument("--dirichlet-eps", type=float, default=0.25,
                      help="Dirichlet noise mixing ratio")
    mcts.add_argument("--temp-threshold", type=int, default=15,
                      help="Temperature for first N moves")
    mcts.add_argument("--eval-sims", type=int, default=400,
                      help="MCTS simulations during evaluation")

    # ── Self-Play ───────────────────────────────────────────────────
    sp = p.add_argument_group("Self-Play")
    sp.add_argument("--games", type=int, default=100,
                    help="Self-play games per iteration")

    # ── Replay Buffer ───────────────────────────────────────────────
    buf = p.add_argument_group("Replay Buffer")
    buf.add_argument("--buffer-capacity", type=int, default=500_000,
                     help="Max stored (s, pi, z) examples")
    buf.add_argument("--min-buffer", type=int, default=10_000,
                     help="Minimum buffer size before training")

    # ── Training ────────────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument("--batch-size", type=int, default=512,
                       help="Mini-batch size")
    train.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate (SGD)")
    train.add_argument("--momentum", type=float, default=0.9,
                       help="SGD momentum")
    train.add_argument("--weight-decay", type=float, default=1e-4,
                       help="L2 weight decay")
    train.add_argument("--train-steps", type=int, default=1000,
                       help="Training steps per iteration")

    # ── Evaluation ──────────────────────────────────────────────────
    ev = p.add_argument_group("Evaluation")
    ev.add_argument("--eval-games", type=int, default=50,
                    help="Pit evaluation games")
    ev.add_argument("--win-threshold", type=float, default=0.55,
                    help="Win rate to promote new model")

    # ── Pipeline ────────────────────────────────────────────────────
    pl = p.add_argument_group("Pipeline")
    pl.add_argument("--iterations", type=int, default=100,
                    help="Number of pipeline iterations")
    pl.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                    help="Directory for model checkpoints")
    pl.add_argument("--log-interval", type=int, default=10,
                    help="Log training metrics every N steps")
    pl.add_argument("--tensorboard", action="store_true",
                    help="Enable TensorBoard logging")
    pl.add_argument("--final-eval", action="store_true", default=True,
                    help="After training, pit best_model vs ClassicAI "
                         "and report final win rate (default: enabled)")
    pl.add_argument("--no-final-eval", dest="final_eval", action="store_false",
                    help="Disable the final best-vs-ClassicAI evaluation")
    pl.add_argument("--final-eval-games", type=int, default=100,
                    help="Number of games for the final best-vs-ClassicAI "
                         "evaluation (0 -> use --eval-games)")
    pl.add_argument("--final-eval-sims", type=int, default=800,
                    help="MCTS sims per move during final eval "
                         "(0 -> use --eval-sims)")

    # ── Battle Config ───────────────────────────────────────────────
    cfg = p.add_argument_group("Battle Config")
    cfg.add_argument("--config", type=str, nargs="+", default=None,
                     help="JSON battle config file(s) for self-play")
    cfg.add_argument("--eval-config", type=str, nargs="+", default=None,
                     help="JSON config(s) for evaluation (hold-out set)")
    pl.add_argument("--early-stop", type=int, default=0,
                    help="Stop if eval win rate unchanged for N iters")

    # ── Hardware ────────────────────────────────────────────────────
    hw = p.add_argument_group("Hardware")
    hw.add_argument("--device", type=str, default="cpu",
                    help="Torch device: cpu, cuda, cuda:0")
    hw.add_argument("--seed", type=int, default=42,
                    help="Base random seed")

    # ── Resume ──────────────────────────────────────────────────────
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint for resuming")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load battle configs
    if args.config:
        battle_configs = sum([load_json_config(p) for p in args.config], [])
    else:
        # Default: 1v1 Swordsman mirror match
        battle_configs = [{
            "units": [
                {"team": 0, "type": "Swordsman", "col": 5, "row": 3},
                {"team": 1, "type": "Swordsman", "col": 5, "row": 6},
            ]
        }]

    config = AlphaGoConfig(
        num_simulations=args.sims,
        c_puct=args.cpuct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_eps,
        temperature_threshold=args.temp_threshold,
        eval_mcts_simulations=args.eval_sims,
        games_per_iteration=args.games,
        buffer_capacity=args.buffer_capacity,
        min_buffer_size=args.min_buffer,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        train_steps_per_iter=args.train_steps,
        eval_games=args.eval_games,
        win_rate_threshold=args.win_threshold,
        num_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        battle_configs=battle_configs,
        device=args.device,
        tensorboard=args.tensorboard,
        early_stop_patience=args.early_stop,
        final_eval_enabled=args.final_eval,
        final_eval_games=args.final_eval_games,
        final_eval_mcts_simulations=args.final_eval_sims,
    )

    # Eval configs (hold-out set for detecting overfitting)
    if args.eval_config:
        eval_battle_configs = sum([load_json_config(p) for p in args.eval_config], [])
        config = AlphaGoConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            temperature_threshold=config.temperature_threshold,
            eval_mcts_simulations=config.eval_mcts_simulations,
            games_per_iteration=config.games_per_iteration,
            buffer_capacity=config.buffer_capacity,
            min_buffer_size=config.min_buffer_size,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            train_steps_per_iter=config.train_steps_per_iter,
            eval_games=config.eval_games,
            win_rate_threshold=config.win_rate_threshold,
            num_iterations=config.num_iterations,
            checkpoint_dir=config.checkpoint_dir,
            log_interval=config.log_interval,
            battle_configs=config.battle_configs,
            eval_battle_configs=eval_battle_configs,
            device=config.device,
            tensorboard=config.tensorboard,
            early_stop_patience=config.early_stop_patience,
            final_eval_enabled=config.final_eval_enabled,
            final_eval_games=config.final_eval_games,
            final_eval_mcts_simulations=config.final_eval_mcts_simulations,
        )

    pipeline = AlphaGoPipeline(config)

    if args.resume:
        pipeline.load_checkpoint(args.resume)

    pipeline.run()


if __name__ == "__main__":
    main()
