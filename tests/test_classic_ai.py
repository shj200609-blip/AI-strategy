"""Smoke tests for ``ai_core.classic_ai.ClassicAI``.

These tests exercise the public interface of ClassicAI against a minimal
``BattleState`` to make sure the refactor preserved behaviour:

* ``check_retreat`` returns the three-state ``(decision, payload)`` tuple.
* ``decide`` returns an ``Action`` and reason string.
* SkipAction is single-arg (no `defending` field) — fheroes2 Command::SKIP
  carries no per-stack "defend" bit; "defend" is a UI hot-key, not a
  command variant.
* Hypnotized units are classified as effective enemies (their ``_effective_team``
  flips to the opposing side) for the duration of the spell.
* Spell helpers report no candidate when the hero lacks the spell or SP.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when pytest is invoked from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.battle_state import BattleState
from engine.hero import Hero
from engine.hex_grid import HexGrid
from engine.spells import SPELLS, make_effect
from engine.unit import Unit
from engine.actions import SkipAction, AttackAction, MoveAction, CastAction, RetreatAction

from ai_core.classic_ai import (
    ClassicAI,
    RETREAT_NONE,
    RETREAT_RETREAT,
    RETREAT_SURRENDER,
    _is_hypnotized,
    _effective_team,
    _effective_friends,
    _effective_enemies,
)
from ai_core.action_space import (
    _can_attack_from_pos as action_space_can_attack,
    _tail_dir as action_space_tail_dir,
)
from ai_core.battle_geometry import _can_attack_from_pos, _tail_dir


# ── fixtures ──────────────────────────────────────────────────────────

def _make_unit(name: str, team: int, col: int, row: int,
               *, attack: int = 5, defense: int = 5, hp: int = 10,
               speed: int = 4, count: int = 5, damage: int = 5,
               is_archer: bool = False, is_flying: bool = False,
               abilities=()) -> Unit:
    return Unit(name, team, col, row,
                attack=attack, defense=defense, hp=hp,
                speed=speed, damage=damage, count=count,
                is_archer=is_archer, is_flying=is_flying,
                abilities=abilities)


def _make_battle(units, heroes=None, cols=15, rows=11):
    grid = HexGrid(cols=cols, rows=rows)
    return BattleState(grid, units, heroes=heroes)


# ── Hypnotize helpers ─────────────────────────────────────────────────

def test_hypnotized_flip_makes_friend_into_enemy():
    """A hypnotized unit is no longer our ally for attack/move planning."""
    friend = _make_unit("Ally", 0, col=2, row=5)
    enemy = _make_unit("Foe", 1, col=8, row=5)
    hypnotized = _make_unit("Traitor", 0, col=5, row=5)
    # Engine models Hypnotize as a Hypnotize Effect with ``is_hypnotize``
    # set; ``Unit.is_hypnotized`` reflects that. Apply via the same path
    # combat uses (``Unit.add_effect(make_effect(spell, power))``).
    hypnotized.add_effect(make_effect(SPELLS["Hypnotize"], power=1))

    battle = _make_battle([friend, enemy, hypnotized])
    ai = ClassicAI()

    # Engine queries still report original teams.
    assert {u.name for u in battle.friends_of(friend)} == {"Ally", "Traitor"}
    # ClassicAI helpers expose the hypnotized unit as an enemy to friend.
    eff_friends = {u.name for u in _effective_friends(battle, friend)}
    eff_enemies = {u.name for u in _effective_enemies(battle, friend)}
    assert eff_friends == {"Ally"}
    assert eff_enemies == {"Foe", "Traitor"}
    assert _is_hypnotized(hypnotized) is True
    assert _effective_team(hypnotized) == 1  # flipped to opposing side


def test_non_hypnotized_passthrough():
    friend = _make_unit("Ally", 0, col=2, row=5)
    enemy = _make_unit("Foe", 1, col=8, row=5)
    battle = _make_battle([friend, enemy])
    assert _is_hypnotized(friend) is False
    assert _effective_team(friend) == 0
    assert _effective_team(enemy) == 1


# ── check_retreat three-state output ──────────────────────────────────

def test_check_retreat_none_when_safe():
    friend = _make_unit("Ally", 0, col=2, row=5, count=50)
    enemy = _make_unit("Foe", 1, col=8, row=5, count=2)
    battle = _make_battle([friend, enemy])
    ai = ClassicAI()
    decision, payload = ai.check_retreat(battle, friend)
    assert decision == RETREAT_NONE
    assert payload is None


def test_check_retreat_advises_retreat():
    """Overwhelming enemy should trigger a RETREAT_RETREAT decision."""
    # Hero must have total primary skill level >= 10 AND no valuable
    # artifacts AND be possible-to-rehire to mirror the C++ Outcome
    # lambda's final branch (`if ( primary >= 10 ) return Retreat`).
    hero = Hero(knowledge=0, spells=[],
                attack=20, defense=20, power=20)
    weak = _make_unit("Ally", 0, col=2, row=5, count=1, hp=1)
    strong = _make_unit("Foe", 1, col=8, row=5, count=200, hp=100,
                        attack=20, defense=20)
    battle = _make_battle([weak, strong], heroes={0: hero, 1: None})
    ai = ClassicAI()
    decision, payload = ai.check_retreat(battle, weak)
    assert decision in (RETREAT_RETREAT, RETREAT_SURRENDER)
    if decision == RETREAT_RETREAT:
        # payload is the (cast-or-none, RetreatAction) tuple from fheroes2.
        _, retreat_action = payload
        assert isinstance(retreat_action, RetreatAction)


# ── decide action structure ───────────────────────────────────────────

def test_decide_returns_action_and_reason():
    friend = _make_unit("Ally", 0, col=2, row=5, count=5)
    enemy = _make_unit("Foe", 1, col=5, row=5, count=5)
    battle = _make_battle([friend, enemy])
    ai = ClassicAI()
    action, reason = ai.decide(battle, friend)
    assert isinstance(action, (AttackAction, MoveAction, SkipAction, CastAction))
    assert isinstance(reason, str) and reason


def test_skipaction_cpp_parity():
    """fheroes2 Battle::Command SKIP carries only ``UID``.

    No defending flag — defending is a UI hot-key, not a command variant
    (battle_command.h:82).  Engine applies ``TR_SKIP | TR_MOVED`` so the
    same unit can't be issued more actions in the same round.  This test
    locks in both shape (single arg, no flag) and that the engine
    actually flips ``unit._acted`` after dispatch.
    """
    unit = _make_unit("Ally", 0, 2, 5)
    # 1. Constructor shape: only ``unit``, no ``defending``.
    a = SkipAction(unit)
    assert a.unit is unit
    assert not hasattr(a, "defending")

    # 2. Engine path: dispatch a SkipAction through a BattleState and
    #    verify TR_MOVED (modelled as ``unit._acted``) flips.
    friend = _make_unit("Friend", 0, col=1, row=1)
    enemy = _make_unit("Foe", 1, col=8, row=5)
    battle = _make_battle([friend, enemy])
    assert friend._acted is False
    battle.execute(SkipAction(friend))
    assert friend._acted is True


# ── Spell helpers ─────────────────────────────────────────────────────

def test_maybe_cast_spell_no_hero_returns_none():
    friend = _make_unit("Ally", 0, col=2, row=5)
    enemy = _make_unit("Foe", 1, col=8, row=5)
    battle = _make_battle([friend, enemy], heroes={0: None, 1: None})
    ai = ClassicAI()
    assert ai.maybe_cast_spell(battle, friend) is None


def test_maybe_cast_spell_hero_without_sufficient_sp_returns_none():
    """Hero with 0 spell points can't cast anything."""
    hero = Hero(knowledge=0, spells=[])
    friend = _make_unit("Ally", 0, col=2, row=5)
    enemy = _make_unit("Foe", 1, col=8, row=5)
    battle = _make_battle([friend, enemy], heroes={0: hero, 1: None})
    ai = ClassicAI()
    assert ai.maybe_cast_spell(battle, friend) is None


def test_spell_damage_score_forwarder_preserves_static_method_contract():
    assert ClassicAI._spell_damage_score(30, SPELLS["Magic Arrow"]) > 0


# ── analyze battle state ──────────────────────────────────────────────

def test_analyze_battle_state_basic_keys():
    """After _analyze_battle_state, the AI must populate its per-turn
    instance attributes that mirror BattlePlanner's member variables."""
    friend = _make_unit("Ally", 0, col=2, row=5, count=10, hp=10)
    enemy = _make_unit("Foe", 1, col=8, row=5, count=10, hp=10)
    battle = _make_battle([friend, enemy])
    ai = ClassicAI()
    assert ai._analyze_battle_state(battle, friend) is None  # sets attrs
    # All key analysis fields must be populated.
    for attr in ("_my_army_strength", "_enemy_army_strength",
                 "_my_shooters_strength", "_enemy_shooters_strength",
                 "_my_army_average_speed", "_enemy_average_speed",
                 "_attacking_castle", "_defending_castle",
                 "_consider_retreat", "_defensive_tactics",
                 "_cautious_offensive", "_avoid_stacking_units"):
        assert hasattr(ai, attr), attr
    assert ai._my_army_strength > 0
    assert ai._enemy_army_strength > 0
    # No castle in this test → both flags are False.
    assert ai._attacking_castle is False
    assert ai._defending_castle is False


# ── Lifecycle and shared geometry ──────────────────────────────────────

def test_battle_begins_resets_configured_turn_limit():
    ai = ClassicAI(turn_limit=7)
    ai._current_turn_number = 9
    ai._remaining_turns_without_deaths = 1
    ai._attacker_dead_total = 12
    ai._defender_dead_total = 8

    ai.battle_begins()

    assert ai._current_turn_number == 0
    assert ai._remaining_turns_without_deaths == 7
    assert ai._attacker_dead_total == 0
    assert ai._defender_dead_total == 0


def test_action_space_and_classic_ai_share_wide_unit_geometry():
    wide = _make_unit("Wide", 0, col=3, row=4)
    wide.is_wide = True
    target = _make_unit("Target", 1, col=1, row=4)
    grid = HexGrid(cols=15, rows=11)

    assert action_space_tail_dir is _tail_dir
    assert action_space_can_attack is _can_attack_from_pos
    assert _tail_dir(wide) == -1
    assert _can_attack_from_pos(grid, wide, target, wide.pos) is True
    assert _can_attack_from_pos(grid, wide, target, (8, 4)) is False
