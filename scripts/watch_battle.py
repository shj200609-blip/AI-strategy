#!/usr/bin/env python3
"""观察 AI 对战 — 终端 ASCII 或 Matplotlib PNG 两种模式。

Usage:
    python scripts/watch_battle.py --mode ascii
    python scripts/watch_battle.py --mode png --output screenshots/
    python scripts/watch_battle.py --model checkpoints/best.pt --sims 200
"""

import argparse, os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from alphago.self_play import _build_battle
from alphago.mcts import advance_to_next_unit
from alphago.visualization import print_battle, render_battle
from ai_core.action_space import index_to_action
from alphago.player import MCTSAIPlayer
from alphago.config import AlphaGoConfig


def parse_args():
    p = argparse.ArgumentParser(description="Watch AI play a battle")
    p.add_argument("--mode", choices=["ascii", "png"], default="ascii")
    p.add_argument("--output", type=str, default="screenshots")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-steps", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    config = AlphaGoConfig(num_simulations=args.sims, device=args.device)
    ai = MCTSAIPlayer(model_path=args.model, config=config, device=args.device)
    print(f"AI: MCTSAI(sims={args.sims}) | Model: {args.model or 'random'}")

    battle = _build_battle({
        "units": [
            {"team": 0, "type": "Swordsman", "col": 2, "row": 3, "count": 5},
            {"team": 0, "type": "Archer",    "col": 3, "row": 4, "count": 5},
            {"team": 0, "type": "Cavalry",   "col": 4, "row": 3, "count": 3},
            {"team": 1, "type": "Pikeman",   "col": 7, "row": 5, "count": 6},
            {"team": 1, "type": "Ranger",    "col": 6, "row": 6, "count": 5},
            {"team": 1, "type": "Griffin",   "col": 8, "row": 4, "count": 3},
        ],
    })
    ai.battle_begins()
    unit = advance_to_next_unit(battle)
    step = 0

    if args.mode == "png":
        os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*50}\nBattle Start: {len(battle.alive())} units\n{'='*50}")

    while not battle.is_over() and unit and step < args.max_steps:
        team = "BLUE" if unit.team == 0 else "RED"
        if args.mode == "ascii":
            print(f"\n--- Step {step} | Round {battle.round_num} | {team} ---")
            print_battle(battle, unit)
        else:
            path = os.path.join(args.output, f"step_{step:04d}.png")
            render_battle(battle, unit,
                          title=f"Step {step} | Round {battle.round_num} | {team}",
                          save_path=path)
            print(f"[{step}] {team} {unit.name} → {path}")

        action, desc = ai.decide(battle, unit)
        unit._acted = True  # 关键：标记单位已行动，回合才能推进
        result = battle.execute(action)
        print(f"  → {desc} | {result.get('desc','?')}")
        unit = advance_to_next_unit(battle)
        step += 1

    w = battle.winner() if battle.is_over() else "Draw"
    print(f"\n{'='*50}\nGame Over! Winner: Team {w} | Steps: {step}\n{'='*50}")
    if args.mode == "ascii":
        print_battle(battle, None)
    else:
        path = os.path.join(args.output, f"step_{step:04d}_final.png")
        render_battle(battle, None, title=f"Game Over | Winner: Team {w}",
                      save_path=path)
        print(f"Final → {path}  ({step+1} frames in {args.output}/)")


if __name__ == "__main__":
    main()
