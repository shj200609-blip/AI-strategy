"""Self-Play — generate training data by playing games with MCTS.

Uses the current best network to play complete battles.  Each game produces
a trajectory of (state, MCTS policy π, outcome z) tuples for training.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ai_core.action_space import ACTION_DIM, index_to_action, legal_mask
from ai_core.base import AIPlayer
from ai_core.model import BattleNet
from ai_core.observation import encode_observation
from engine.actions import Action
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.hero import Hero
from alphago.config import AlphaGoConfig
from alphago.mcts import MCTS, sample_action_from_policy, advance_to_next_unit
from alphago.replay_buffer import TrainingExample


class SelfPlayRunner:
    """Generates training data by playing games with MCTS-guided self-play.

    Parameters
    ----------
    network : BattleNet
        The neural network used for MCTS evaluations.
    config : AlphaGoConfig
        Pipeline hyperparameters.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        network: BattleNet,
        config: AlphaGoConfig,
        device: str = "cpu",
    ):
        self.network = network
        self.config = config
        self.device = device
        self.mcts = MCTS(config)

    def play_game(self, battle_config: dict, seed: int,
                  opponent_network: Optional[BattleNet] = None,
                  opponent_ai: Optional["AIPlayer"] = None
                  ) -> List[TrainingExample]:
        """Play one complete self-play game.

        Opponent selection (priority order):
          1. ``opponent_ai`` (a rule-based ``AIPlayer`` like ClassicAI) — if
             provided, team 1 uses its greedy decisions; team 0 uses MCTS.
          2. ``opponent_network`` (a historical ``BattleNet``) — if provided,
             team 1 uses its MCTS while team 0 still uses self.network.
          3. None — pure self-play (both teams use self.network).

        Trajectory recording is **only** done for the *learning* team (team 0).
        When team 1 is the rule-based opponent, we still capture team 0's
        decisions because they're the only ones using MCTS.
        """
        if opponent_network is not None:
            opponent_network.eval()
        if opponent_ai is not None:
            # ClassicAI has no eval() — but it has no learnable params.
            pass
        random.seed(seed)
        np.random.seed(seed)

        # Build initial battle state
        battle = _build_battle(battle_config)
        if opponent_ai is not None:
            opponent_ai.battle_begins()

        trajectory: List[_TrajectoryEntry] = []
        step = 0

        # Get initial unit
        current_unit: Optional[Unit] = advance_to_next_unit(battle)
        if current_unit is None:
            return []

        while (
            not battle.is_over()
            and current_unit is not None
            and step < self.config.max_moves_per_game
        ):
            # ── Decide whose policy we follow for this unit ────────────
            if opponent_ai is not None and current_unit.team == 1:
                # Rule-based opponent: NO MCTS, NO trajectory record.
                # The opponent's decisions are not the model's policy —
                # we just need to advance the game state.
                action, _desc = opponent_ai.decide(battle, current_unit)
                battle.execute(action)
                current_unit._acted = True
                step += 1
                current_unit = advance_to_next_unit(battle)
                continue

            # ── Team 0 (the learner) runs MCTS ─────────────────────────
            net = (
                self.network
                if current_unit.team == 0 or opponent_network is None
                else opponent_network
            )
            policy = self.mcts.search(battle, current_unit, net, self.device)

            if not policy:
                break  # No legal actions — shouldn't happen, but safety

            # Encode observation for this decision
            grid, gvec = encode_observation(battle, current_unit)
            mask = legal_mask(battle, current_unit)

            # Record the MCTS policy as a dense vector
            policy_vector = np.zeros(ACTION_DIM, dtype=np.float32)
            for act_idx, prob in policy.items():
                policy_vector[act_idx] = prob

            # Sample action (temperature for exploration in early moves)
            temperature = 1.0 if step < self.config.temperature_threshold else 0.0
            action_idx = sample_action_from_policy(policy, temperature)

            # Record trajectory entry
            acting_team = current_unit.team
            trajectory.append(_TrajectoryEntry(
                grid=grid,
                global_vec=gvec,
                mask=mask,
                policy_vector=policy_vector,
                team=acting_team,
            ))

            # Execute the chosen action
            action = index_to_action(action_idx, battle, current_unit)
            current_unit._acted = True
            battle.execute(action)

            step += 1

            # Advance to next unit
            current_unit = advance_to_next_unit(battle)

        # Determine outcome
        if battle.is_over():
            winner = battle.winner()
        else:
            winner = _determine_winner_by_strength(battle)

        # Fill outcomes from the acting team's perspective.
        # When team 1 is the rule-based AI, only team 0 entries exist —
        # which is exactly what we want for training.
        examples = []
        for entry in trajectory:
            outcome = 1.0 if entry.team == winner else (-1.0 if winner is not None else 0.0)
            examples.append(TrainingExample(
                grid=entry.grid,
                global_vec=entry.global_vec,
                mask=entry.mask,
                policy=entry.policy_vector,
                outcome=outcome,
            ))

        return examples

    def run_batch(self, seeds: List[int],
                  opponent_network=None, opponent_ratio: float = 0.5,
                  opponent_ai=None, classic_ai_ratio: float = 0.0,
                  ) -> List[TrainingExample]:
        """Play games; opponent choice follows priority order per game.

        Per-game opponent selection:
          1. If ``classic_ai_ratio > 0`` and ``opponent_ai`` is not None:
             ``classic_ai_ratio`` chance of using the rule-based opponent.
          2. Else if ``opponent_network`` is not None and ``random()<opponent_ratio``:
             use the historical neural network opponent.
          3. Else: pure self-play.

        The mixture lets the trainer blend "stable rule-based feedback"
        with "diverse historical strategy" — this is critical early in
        training when historical snapshots are weak.
        """
        all_examples: List[TrainingExample] = []
        total = len(seeds)
        for i, seed in enumerate(seeds):
            cfg = random.choice(self.config.battle_configs)
            # Decide opponent
            use_classic = (opponent_ai is not None
                           and classic_ai_ratio > 0
                           and random.random() < classic_ai_ratio)
            opp_net = None
            opp_ai = opponent_ai if use_classic else None
            if not use_classic and opponent_network is not None:
                if random.random() < opponent_ratio:
                    opp_net = opponent_network
            examples = self.play_game(cfg, seed, opp_net, opp_ai)
            all_examples.extend(examples)
            if (i + 1) % 5 == 0 or i == 0:
                tag = "classic" if opp_ai is not None else (
                    "pool" if opp_net is not None else "self")
                print(f"  [Self-Play] game {i+1}/{total} done [{tag}], "
                      f"{len(examples)} examples", flush=True)
        return all_examples


# ── Internal Helpers ─────────────────────────────────────────────────────────


@dataclass
class _TrajectoryEntry:
    """Intermediate entry before outcome is known."""
    grid: np.ndarray
    global_vec: np.ndarray
    mask: np.ndarray
    policy_vector: np.ndarray
    team: int


def _build_battle(battle_config: dict) -> BattleState:
    """Build a BattleState from a config dict.

    Supports both list-style (bare units) and dict-style (with heroes, morale, etc.).
    """
    from config.units import UNIT_TYPES
    from engine.hex_grid import HexGrid
    from engine.unit import Unit
    from engine.hero import Hero
    from engine.castle import Castle

    # Extract units — support both list and dict config formats
    if isinstance(battle_config, list):
        units_data = battle_config
        hero_data = {}
        extra = {}
    else:
        units_data = battle_config.get("units", [])
        hero_data = battle_config.get("heroes", {})
        extra = battle_config

    grid = HexGrid()
    all_units = []

    for item in units_data:
        if isinstance(item, dict):
            name = item["type"]
            team = item["team"]
            col = item.get("col", 5)
            row = item.get("row", 3)
            count = item.get("count", None)
        elif isinstance(item, (list, tuple)):
            # Tuple format: (name, team, col, row[, count])
            name = item[0]
            team = item[1]
            col = item[2] if len(item) > 2 else 5
            row = item[3] if len(item) > 3 else 3
            count = item[4] if len(item) > 4 else None
        else:
            raise ValueError(f"Unsupported unit config format: {type(item)}")

        utype = UNIT_TYPES[name]
        unit = Unit(
            name=name,
            team=team,
            col=col,
            row=row,
            count=count if count else utype["count"],
            attack=utype["attack"],
            defense=utype["defense"],
            hp=utype["hp"],
            damage_min=utype["damage_min"],
            damage_max=utype["damage_max"],
            speed=utype["speed"],
            is_archer=utype["is_archer"],
            is_flying=utype["is_flying"],
            is_wide=utype["is_wide"],
            abilities=set(utype["abilities"]),
            ability_params={k: v for k, v in utype["ability_params"].items()},
            tags=set(),
        )
        all_units.append(unit)

    # Heroes
    heroes: Dict[int, Optional[Hero]] = {0: None, 1: None}
    for team_str, hdata in hero_data.items():
        team = int(team_str)
        if hdata is not None:
            # fheroes2: GetMaxSpellPoints() = 10 * knowledge (heroes.cpp:967).
            # ``spell_points`` is accepted for backwards compat but is now
            # always derived from knowledge; we read it to validate the
            # caller's expectation and discard.
            _legacy_sp = hdata.get("spell_points", None)
            hero = Hero(
                name=hdata.get("name", "Hero"),
                power=hdata.get("power", 1),
                attack=hdata.get("attack", 0),
                defense=hdata.get("defense", 0),
                knowledge=hdata.get("knowledge", 1),
                spells=hdata.get("spells", []),
                skills=hdata.get("skills", {}),
                has_spell_book=hdata.get("has_spell_book", True),
            )
            # Allow an explicit spell_points override in test configs only
            # when knowledge is 0 (i.e. testing the "no SP" path).
            if _legacy_sp is not None and hero.knowledge == 0:
                hero.spell_points = int(_legacy_sp)
                hero.max_spell_points = int(_legacy_sp)
            heroes[team] = hero

    # Castle
    castle = None
    if extra.get("siege", False):
        castle = Castle()

    # Morale & luck
    morale = extra.get("morale", {0: 0, 1: 0})
    luck = extra.get("luck", {0: 0, 1: 0})
    difficulty = extra.get("difficulty", "Normal")

    # Determine attacker (first team that has units)
    attacker_team = 0
    for u in all_units:
        if u.is_alive:
            attacker_team = u.team
            break

    battle = BattleState(
        units=all_units,
        grid=grid,
        heroes=heroes,
        castle=castle,
        attacker_team=attacker_team,
        first_team=random.choice([0, 1]),
        difficulty=difficulty,
        morale={int(k): int(v) for k, v in morale.items()},
        luck={int(k): int(v) for k, v in luck.items()},
    )
    return battle


def _determine_winner_by_strength(battle: BattleState) -> Optional[int]:
    """If battle hits max moves, determine winner by remaining army strength."""
    team_strength = {0: 0.0, 1: 0.0}
    for u in battle.units:
        if u.is_alive:
            team_strength[u.team] += u.strength
    if team_strength[0] > team_strength[1]:
        return 0
    elif team_strength[1] > team_strength[0]:
        return 1
    return battle.attacker_team
