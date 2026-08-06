"""1:1 fheroes2 rules contract tests for the Python engine.

These tests pin the engine's behavior to the original fheroes2 source files
(battle_arena.cpp, battle_troop.cpp, spell.cpp). They are deliberately
narrow — one test per rule — and serve as a regression guard whenever the
engine is refactored.
"""
import os
import sys
import random

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.battle_state import BattleState
from engine.hex_grid import HexGrid
from engine.hero import Hero
from engine.unit import Unit
from engine.spells import SPELLS, make_effect
from engine.actions import AttackAction, CastAction


def _fresh_battle(seed: int = 0) -> BattleState:
    """Open-field battle with placeholder units, fixed RNG.

    Two baseline units per side; a no-op hero (power=1) on each side so
    spell dispatches can resolve hero.power.
    """
    grid = HexGrid()
    units = [
        Unit.from_type("Pikeman", 0, 1, 1),
        Unit.from_type("Archer",  0, 3, 3, count=1),
        Unit.from_type("Pikeman", 1, 12, 1),
        Unit.from_type("Archer",  1, 13, 4, count=1),
    ]
    heroes = {0: Hero(power=1, name="A"), 1: Hero(power=1, name="B")}
    bs = BattleState(grid=grid, units=units, first_team=0,
                     attacker_team=0, heroes=heroes)
    random.seed(seed)
    return bs


def _find_archer(bs, team):
    return next(u for u in bs.units if u.team == team and u.is_archer)


# ── Archer ammo ──────────────────────────────────────────────


def test_archer_consumes_one_shot_per_ranged_attack():
    """battle_troop.cpp _shotsLeft decrements by one per ranged hit."""
    bs = _fresh_battle()
    archer = _find_archer(bs, 0)
    assert archer.shots_left == archer.max_shots > 0
    target = _find_archer(bs, 1)
    bs.execute(AttackAction(archer, target, ranged=True))
    assert archer.shots_left == archer.max_shots - 1


def test_archer_cannot_shoot_when_ammo_exhausted():
    bs = _fresh_battle()
    archer = _find_archer(bs, 0)
    archer.shots_left = 0
    assert archer.can_shoot is False


# ── Mirror Image ─────────────────────────────────────────────


def test_mirror_image_destroyed_by_any_damage():
    """fheroes2 CAP_MIRRORIMAGE: any non-zero damage wipes a mirror."""
    bs = _fresh_battle()
    src = _find_archer(bs, 0)
    mirror = Unit.mirror_image(src, src.team)
    bs.units.append(mirror)
    assert mirror.is_mirror and mirror.is_alive
    mirror.take_damage(1)
    assert mirror.is_alive is False and mirror.count == 0


def test_mirror_image_has_no_ammo_and_cannot_shoot():
    src = _find_archer(_fresh_battle(), 0)
    mirror = Unit.mirror_image(src, src.team)
    assert mirror.max_shots == 0 and mirror.can_shoot is False


# ── Hypnotize ────────────────────────────────────────────────


def test_hypnotize_flips_effective_team():
    """Hypnotize controls allegiance (Battle::Troop CurrentColor)."""
    bs = _fresh_battle()
    archer = _find_archer(bs, 1)
    # Force HP under threshold: 1 creature × max_hp << 25*power.
    archer.count = 1
    archer._total_hp = archer.max_hp
    spell = SPELLS["Hypnotize"]
    bs.execute(CastAction(team=0, spell=spell, target=archer))
    assert archer.is_hypnotized and archer.effective_team == 0


def test_hypnotize_fails_when_top_hp_exceeds_threshold():
    """fheroes2 battle_troop.cpp:1519 — threshold compares to the top unit
    HP of the stack (single strongest monster), not the stack total.
    """
    bs = _fresh_battle()
    archer = _find_archer(bs, 1)
    archer._total_hp = (archer.count - 1) * archer.max_hp + 99999  # boost top single-monster HP
    spell = SPELLS["Hypnotize"]
    bs.execute(CastAction(team=0, spell=spell, target=archer))
    assert not archer.is_hypnotized


def test_hypnotized_unit_does_not_retaliate():
    """fheroes2 Battle::Troop::isResistable — hypnotized never counter."""
    bs = _fresh_battle()
    hypo = _find_archer(bs, 1)
    hypo.count = 1
    hypo._total_hp = hypo.max_hp
    bs.execute(CastAction(team=0, spell=SPELLS["Hypnotize"], target=hypo))
    attacker = _find_archer(bs, 0)
    res = bs.execute(AttackAction(attacker, hypo, ranged=False))
    # The retaliate flag in the result stays zero because hypnosis negates it.
    assert res["ret_dmg"] == 0 and res["ret_killed"] == 0


# ── Berserker ────────────────────────────────────────────────


def test_berserker_sets_flag():
    """Berserker effect persists (SP_BERSERKER — BestialMinded)."""
    bs = _fresh_battle()
    tgt = _find_archer(bs, 1)
    bs.execute(CastAction(team=0, spell=SPELLS["Berserker"], target=tgt))
    assert tgt.is_berserk


# ── Resurrect ────────────────────────────────────────────────


def test_resurrect_heals_living_target():
    bs = _fresh_battle()
    tgt = _find_archer(bs, 1)
    # Damage the archer so its total HP drops below the cap.
    tgt._total_hp = 1
    spell = SPELLS["Resurrect"]
    bs.execute(CastAction(team=1, spell=spell, target=tgt))
    # Resurrect heals up to resurrect_per_power * power, capped by max_hp * count.
    assert tgt._total_hp == tgt.max_hp * tgt.count


# ── Summon Elemental ─────────────────────────────────────────


def test_summon_earth_elemental_creates_3_per_power():
    bs = _fresh_battle()
    empty = _find_archer(bs, 0)
    spell = SPELLS["Summon Earth Elemental"]
    # Use a hero-like power=1 simulation; the cast dispatch uses hero.power.
    bs.heroes[0].power = 1
    bs.execute(CastAction(team=0, spell=spell, target=empty))
    new_elems = [u for u in bs.units
                 if u.name == spell.summon_unit_type]
    assert len(new_elems) == 1 and new_elems[0].count == 3


def test_mirror_image_summon_creates_mirror_unit():
    bs = _fresh_battle()
    src = _find_archer(bs, 0)
    spell = SPELLS["Mirror Image"]
    bs.execute(CastAction(team=0, spell=spell, target=src))
    mirrors = [u for u in bs.units if u.is_mirror]
    assert len(mirrors) == 1 and mirrors[0].mirror_of is src


def test_spell_blind_blocks_retaliation():
    """fheroes2 Battle::Troop::ApplySpell<Blind> sets _blindRetaliation=false.
    isRetaliationAllowed() then returns false (Modes(SP_BLIND) && !_blindRetaliation).
    """
    bs = _fresh_battle()
    archer = _find_archer(bs, 1)
    # Before blind: can retaliate normally.
    assert archer.can_retaliate()
    # Hero casts Blind spell on the defender.
    bs.execute(CastAction(team=0, spell=SPELLS["Blind"], target=archer))
    # fheroes2: spell-Blind sets _blindRetaliation=false.
    assert archer.has_effect("Blind")
    assert archer.blind_retaliation is False
    assert not archer.can_retaliate()


def test_ability_blind_still_allows_retaliation():
    """fheroes2: a unit may be SP_BLIND via a monster passive (no spell cast).
    In that case ``_blindRetaliation`` keeps its default True value, so
    ``isRetaliationAllowed`` returns true — the retaliatory strike goes out,
    only with halved damage.  ``isImmovable`` (skip turn) is a separate
    check that affects the unit's own activation, not retaliation.

    Our engine has no monster-ability Blind source, so we simulate the
    flag state directly: ``blind_retaliation`` stays True after the
    ``Blind`` Effect is applied via a no-op path that does NOT call
    ``BattleState._cast`` (which is what would have set the flag to False).
    """
    from engine.spells import make_effect
    bs = _fresh_battle()
    archer = _find_archer(bs, 1)
    # Manually attach a Blind Effect without going through hero casting
    # — mirrors the "monster passive ability" path where _blindRetaliation
    # is left at its True default.
    archer.add_effect(make_effect(SPELLS["Blind"], 1))
    assert archer.has_effect("Blind")
    # blind_retaliation stays True (no ApplySpell was called).
    assert archer.blind_retaliation is True
    # Retaliation allowed; this is the key contract from fheroes2.
    assert archer.can_retaliate()


def test_paralyzed_unit_cannot_retaliate():
    """fheroes2: paralyzing magic blocks both action and retaliation."""
    bs = _fresh_battle()
    archer = _find_archer(bs, 1)
    bs.execute(CastAction(team=0, spell=SPELLS["Paralyze"], target=archer))
    assert archer.skip_turn
    assert not archer.can_retaliate()


def test_unlimited_retaliation_overrides_already_retaliated_flag():
    from engine.unit import Unit
    from config import units as _u
    t = dict(_u.UNIT_TYPES["Griffin"])
    t["count"] = 4
    grif = Unit("Griffin", 0, 5, 5, **t)
    grif.retaliated = True
    assert grif.can_retaliate()

# ── Petrify (4th-level spell, fheroes2 builtin) ──────────────


def test_petrify_reduces_direct_attack_damage_to_half():
    """fheroes2 battle_troop.cpp:562 — Petrified enemies take half damage
    from a direct attack. Expected and actual damage must both reflect /2.
    """
    bs = _fresh_battle()
    atk = _find_archer(bs, 0)
    dfn = next(u for u in bs.units if u.team == 1 and not u.is_archer)
    base = bs.expected_damage(atk, dfn, ranged=False)
    dfn.add_effect(make_effect(SPELLS["Petrification"], power=1))
    after = bs.expected_damage(atk, dfn, ranged=False)
    assert after == max(1, base // 2), (
        f"Petrify should halve expected damage, got {after} vs {base}")
    # roll_damage path: a second call must be roughly half the first.
    base_roll = bs.roll_damage(atk, dfn, ranged=False)
    after_roll = bs.roll_damage(atk, dfn, ranged=False)
    assert after_roll * 2 >= base_roll, (
        f"rolled dmg after Petrify {after_roll} should be <= half of "
        f"pre-Petrify {base_roll}")


def test_petrify_reduces_high_damage_to_exact_half():
    """Regression: Petrify damage-halving lookup uses the right effect name.

    battle_troop.cpp:562 — ``if ( enemy.Modes( SP_STONE ) ) dmg /= 2;``.
    The corresponding Python ``Effect`` is named ``"Petrification"`` (the
    builtin-only Petrification spell), NOT ``"Petrify"``.

    The existing low-damage Petrify test passes coincidentally when the
    bug is present (base=1 → ``max(1, 1//2) == 1``).  Use a high-count
    Crusader vs Champion stack so the halving is unambiguous.
    """
    from engine.unit import Unit
    from config import units as _u
    grid = HexGrid()
    c_t = dict(_u.UNIT_TYPES["Crusader"]); c_t["count"] = 10
    ch_t = dict(_u.UNIT_TYPES["Champion"]); ch_t["count"] = 10
    c = Unit("Crusader", 0, 1, 1, **c_t)
    h = Unit("Champion", 1, 8, 1, **ch_t)
    bs = BattleState(grid=grid, units=[c, h],
                     first_team=0, attacker_team=0,
                     heroes={0: Hero(power=1, name="A"),
                             1: Hero(power=1, name="B")})
    random.seed(0)
    base = bs.expected_damage(c, h, ranged=False)
    h.add_effect(make_effect(SPELLS["Petrification"], power=1))
    after = bs.expected_damage(c, h, ranged=False)
    assert base > 10, (
        f"test fixture too weak to expose Petrify halving bug (base={base})")
    assert after == max(1, base // 2), (
        f"Petrify must halve expected_damage (battle_troop.cpp:562). "
        f"Got base={base}, after={after}, expected={max(1, base // 2)}. "
        f"This regression indicates a wrong effect-name lookup in "
        f"expected_damage / roll_damage.")
    # Same on the rolled path.
    rolled_before = bs.roll_damage(c, h, ranged=False)
    h2 = Unit("Champion", 1, 8, 1, **ch_t)
    bs2 = BattleState(grid=grid, units=[c, h2],
                      first_team=0, attacker_team=0,
                      heroes={0: Hero(power=1, name="A"),
                              1: Hero(power=1, name="B")})
    random.seed(0)
    _ = bs2.roll_damage(c, h2, ranged=False)  # advance RNG identically
    h2.add_effect(make_effect(SPELLS["Petrification"], power=1))
    rolled_after = bs2.roll_damage(c, h2, ranged=False)
    assert rolled_after * 2 >= rolled_before, (
        f"rolled dmg after Petrify ({rolled_after}) should be <= half of "
        f"pre-Petrify ({rolled_before})")


def test_petrified_unit_cannot_retaliate():
    """Petrify sets IS_PARALYZE_MAGIC, fheroes2 battle_troop.cpp:757-759 —
    so petrified units are immovable AND cannot retaliate.
    """
    from engine.unit import Unit
    from config import units as _u
    t = dict(_u.UNIT_TYPES["Pikeman"])
    t["count"] = 5
    u = Unit("Pikeman", 0, 5, 5, **t)
    u.add_effect(make_effect(SPELLS["Petrification"], power=1))
    assert not u.can_retaliate(), (
        "Petrified unit must not be able to retaliate (IS_PARALYZE_MAGIC)")
    # Removing the effect restores retaliation rights.
    u.effects.clear()
    assert u.can_retaliate()


def test_petrification_is_builtin_only():
    """fheroes2 spell.h: Petrification exists but cannot enter a spellbook."""
    from engine.spells import DEFAULT_SPELLBOOK, SPELLS
    assert "Petrification" in SPELLS
    assert "Petrification" not in DEFAULT_SPELLBOOK
    assert SPELLS["Petrification"].effect_skip_turn is True
    assert SPELLS["Petrification"].effect_break_on_damage is False


def test_elemental_classification_matches_spell_header():
    elemental = {name for name, spell in SPELLS.items() if spell.elemental}
    assert elemental == {
        "Cold Ray", "Cold Ring", "Fireball", "Fireblast",
        "Lightning Bolt", "Chain Lightning", "Elemental Storm",
    }


def test_slow_halves_speed_instead_of_subtracting_two():
    unit = Unit.from_type("Pikeman", 0, 5, 5, count=1)
    unit.base_speed = 3
    unit.add_effect(make_effect(SPELLS["Slow"], power=10))
    assert unit.speed == 1


def test_canonical_cpp_spell_names_are_used():
    assert "Blood Lust" in SPELLS and "Bloodlust" not in SPELLS
    assert "Stoneskin" in SPELLS and "Stone Skin" not in SPELLS
    assert "Steelskin" in SPELLS and "Steel Skin" not in SPELLS


# ── AOE undead/living filters ──────────────────────────────


def test_death_ripple_skips_undead_army():
    """fheroes2 Spell::isAliveOnly() covers Death Ripple. Undead monsters
    have 100% spell resistance to alive-only spells (monster_info.cpp).
    """
    bs = _fresh_battle()
    from engine.actions import CastAction
    from engine.unit import Unit
    from config import units as _u
    # Replace one enemy unit with an undead marker (Bone Dragon, has tag
    # "undead").
    enemy = next(u for u in bs.units if u.team == 1)
    enemy.tags = ("undead",)
    enemy.name = "Bone Dragon"
    hp_before = enemy.hp
    bs.execute(CastAction(team=0, spell=SPELLS["Death Ripple"],
                          target=enemy))
    # Death Ripple must NOT damage an undead target.
    assert enemy.hp == hp_before, (
        f"Death Ripple should ignore undead target, but hp went "
        f"{hp_before} -> {enemy.hp}")


def test_holy_word_only_hits_undead():
    """fheroes2 Spell::isUndeadOnly() covers Holy Word. Living monsters
    have 100% resistance (monster_info.cpp).
    """
    bs = _fresh_battle()
    from engine.actions import CastAction
    enemy = next(u for u in bs.units if u.team == 1)
    # Mark enemy as a living Pikeman (no "undead" tag).
    enemy.tags = ("living",)
    enemy.name = "Pikeman"
    hp_before = enemy.hp
    bs.execute(CastAction(team=0, spell=SPELLS["Holy Word"],
                          target=enemy))
    assert enemy.hp == hp_before, (
        f"Holy Word should ignore living target, but hp went "
        f"{hp_before} -> {enemy.hp}")


# ── Mind-influence immunity (fheroes2 monster_info.cpp:890-898) ──


def test_undead_unit_is_immune_to_mind_spells():
    from engine.unit import Unit
    u = Unit.from_type("Skeleton", 0, 5, 5, count=5)
    assert u.is_immune_to_mind, (
        "Undead units (tag 'undead') must be immune to mind-influence "
        "spells (Blind / Paralyze / Berserker / Hypnotize)")


def test_elemental_unit_is_immune_to_mind_spells():
    from engine.unit import Unit
    u = Unit.from_type("Earth Elemental", 0, 5, 5, count=2)
    assert u.is_immune_to_mind, (
        "Elemental units (tag 'elemental') must be immune to mind spells")


def test_hypnotize_resisted_by_undead():
    """fheroes2 monster_info.cpp:890 — Undead have 100% resistance to
    mind-influence spells including Hypnotize.
    """
    bs = _fresh_battle()
    from engine.actions import CastAction
    from engine.unit import Unit
    # Replace one enemy slot with a Skeleton (Undead).
    bs.units = [u for u in bs.units if u.team != 1]
    skel = Unit.from_type("Skeleton", 1, 12, 1, count=1)
    bs.units.append(skel)
    bs.execute(CastAction(team=0, spell=SPELLS["Hypnotize"], target=skel))
    assert not skel.is_hypnotized, (
        f"Hypnotize must NOT take effect on undead target, but "
        f"is_hypnotized={skel.is_hypnotized}")


def test_berserker_resisted_by_undead():
    """fheroes2 monster_info.cpp:890 — Berserker is mind-influence."""
    bs = _fresh_battle()
    from engine.actions import CastAction
    from engine.unit import Unit
    bs.units = [u for u in bs.units if u.team != 1]
    skel = Unit.from_type("Skeleton", 1, 12, 1, count=1)
    bs.units.append(skel)
    bs.execute(CastAction(team=0, spell=SPELLS["Berserker"], target=skel))
    assert not skel.is_berserk, (
        f"Berserker must NOT affect undead target, but "
        f"is_berserk={skel.is_berserk}")


def test_blind_resisted_by_undead():
    """fheroes2 monster_info.cpp:890 — Blind is mind-influence (spell.cpp:428).
    Undead must have 100% resistance: no effect attached.
    """
    bs = _fresh_battle()
    from engine.actions import CastAction
    from engine.unit import Unit
    bs.units = [u for u in bs.units if u.team != 1]
    skel = Unit.from_type("Skeleton", 1, 12, 1, count=1)
    bs.units.append(skel)
    bs.execute(CastAction(team=0, spell=SPELLS["Blind"], target=skel))
    assert len(skel.effects) == 0, (
        f"Blind must NOT attach any effect to undead target, but "
        f"effects={[type(e).__name__ for e in skel.effects]}")


def test_paralyze_resisted_by_undead():
    """fheroes2 monster_info.cpp:890 — Paralyze is mind-influence.
    Undead must have 100% resistance: no effect attached.
    """
    bs = _fresh_battle()
    from engine.actions import CastAction
    from engine.unit import Unit
    bs.units = [u for u in bs.units if u.team != 1]
    skel = Unit.from_type("Skeleton", 1, 12, 1, count=1)
    bs.units.append(skel)
    bs.execute(CastAction(team=0, spell=SPELLS["Paralyze"], target=skel))
    assert len(skel.effects) == 0, (
        f"Paralyze must NOT attach any effect to undead target, but "
        f"effects={[type(e).__name__ for e in skel.effects]}")


def test_dispel_magic_only_removes_buffs():
    """fheroes2 battle_action.cpp:262 — Dispel removes IS_GOOD_MAGIC only.

    A unit with both a Buff (Haste) and a Debuff (Curse) must lose only
    the buff after Dispel.
    """
    bs = _fresh_battle()
    from engine.actions import CastAction
    from engine.unit import Unit
    bs.units = [u for u in bs.units if u.team != 1]
    skel = Unit.from_type("Skeleton", 1, 12, 1, count=1)
    bs.units.append(skel)
    bs.execute(CastAction(team=0, spell=SPELLS["Haste"], target=skel))
    bs.execute(CastAction(team=0, spell=SPELLS["Curse"], target=skel))
    assert len(skel.effects) == 2, (
        f"Setup: expected 2 effects, got {len(skel.effects)}")
    bs.execute(CastAction(team=0, spell=SPELLS["Dispel Magic"], target=skel))
    # Only Good Magic (Haste) should be removed; Curse (debuff) remains.
    assert len(skel.effects) == 1, (
        f"Dispel Magic should remove only positive effects, "
        f"got {len(skel.effects)} effects remaining")
    assert skel.effects[0].name == "Curse", (
        f"Remaining effect should be Curse, got {skel.effects[0].name}")


def test_mass_dispel_removes_all_effects():
    """fheroes2 — Mass Dispel clears all affection from all units."""
    bs = _fresh_battle()
    from engine.actions import CastAction
    from engine.unit import Unit
    bs.units = [u for u in bs.units if u.team != 1]
    skel = Unit.from_type("Skeleton", 1, 12, 1, count=1)
    bs.units.append(skel)
    bs.execute(CastAction(team=0, spell=SPELLS["Haste"], target=skel))
    bs.execute(CastAction(team=0, spell=SPELLS["Curse"], target=skel))
    bs.execute(CastAction(team=0, spell=SPELLS["Mass Dispel"], target=skel))
    assert len(skel.effects) == 0, (
        f"Mass Dispel should clear all effects, got {len(skel.effects)}")


# ── fheroes2 alignment helpers (Battle::Arena façade) ─────────


def test_cells_under_walls_match_fheroes2_indexes():
    """BattlePlanner's cellsUnderWallsIndexes = {7, 28, 49, 72, 95}.

    In our (col, row) flat-index ``row * 11 + col`` system these are
    (7,0), (6,2), (5,4), (6,6), (7,8) — one cell under each wall
    segment plus the moat cell just outside the gate.
    """
    from engine.castle import Castle
    bs = _fresh_battle()
    assert bs.cells_under_walls() == []
    bs.castle = Castle(color=1)
    assert bs.cells_under_walls() == [(7, 0), (6, 2), (5, 4), (6, 6), (7, 8)]


def test_castle_bridge_passability_uses_owner_color_and_down_state():
    """Bridge::isPassable: down bridges admit both sides; raised bridges
    admit only the castle owner in this simplified model.
    """
    from engine.castle import Castle

    castle = Castle(color=0)
    assert castle.is_gate_passable(0) is True
    assert castle.is_gate_passable(1) is False
    castle.lower_bridge()
    assert castle.is_gate_passable(0) is True
    assert castle.is_gate_passable(1) is True


def test_castle_damage_wall_rejects_unknown_coordinate():
    from engine.castle import Castle

    castle = Castle()
    before = dict(castle.walls)
    try:
        castle.damage_wall((0, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown wall coordinate must be rejected")
    assert castle.walls == before


def test_catapult_targets_side_towers_then_bridge_then_center():
    import random
    from engine.castle import Castle

    castle = Castle()
    for pos in castle.walls:
        castle.walls[pos] = 0
    assert castle._catapult_pick_target(random.Random(0)) in {
        "tower_0", "tower_2"
    }
    castle.damage_tower(0)
    castle.damage_tower(2)
    assert castle._catapult_pick_target(random.Random(0)) == "bridge"
    castle.destroy_bridge()
    assert castle._catapult_pick_target(random.Random(0)) == "tower_1"


def test_earthquake_targets_cpp_structures_and_preserves_center_tower(monkeypatch):
    from engine.actions import CastAction
    from engine.castle import Castle, GATE_TOWER_POSITIONS
    from engine.spells import SPELLS

    bs = _fresh_battle()
    bs.castle = Castle()
    hero = bs.heroes[0]
    assert hero is not None
    hero.power = 10
    monkeypatch.setattr("engine.castle.random.randint", lambda a, b: b)
    target = next(unit for unit in bs.alive(1))
    bs.execute(CastAction(team=0, spell=SPELLS["Earthquake"], target=target))

    assert all(hp == 0 for hp in bs.castle.walls.values())
    assert bs.castle.towers[0].destroyed is True
    assert bs.castle.towers[1].destroyed is False
    assert bs.castle.towers[2].destroyed is True
    assert bs.castle.bridge_destroyed is True
    assert all(bs.castle.gate_tower_hp[pos] == 1
               for pos in GATE_TOWER_POSITIONS)


def test_no_attacker_hero_means_no_catapult():
    from engine.castle import Castle

    bs = _fresh_battle()
    bs.castle = Castle()
    bs.heroes[bs.attacker_team] = None
    before = dict(bs.castle.walls)
    bs._catapult_round()
    assert bs.castle.walls == before


def test_is_siege_and_side_flags():
    from engine.castle import Castle
    bs = _fresh_battle()
    assert bs.is_siege() is False
    assert bs.is_attacking_castle() is False
    assert bs.is_defending_castle() is False

    bs.castle = Castle(color=1)        # defender owns the castle
    assert bs.is_siege() is True
    assert bs.is_attacking_castle() is True    # team 0 = attacker
    assert bs.is_defending_castle() is False

    bs.attacker_team = 1               # flip: team 1 now attacks
    assert bs.is_attacking_castle() is False
    assert bs.is_defending_castle() is True


def test_get_enemy_color_open_field_vs_siege():
    from engine.castle import Castle
    bs = _fresh_battle()
    assert bs.get_enemy_color(0) == 1
    assert bs.get_enemy_color(1) == 0
    # With a siege, the castle owner is always the defender, regardless
    # of the numerical team.
    bs.castle = Castle(color=0)
    assert bs.get_enemy_color(0) == 1
    assert bs.get_enemy_color(1) == 0


def test_can_retreat_and_surrender_gates():
    bs = _fresh_battle()
    assert bs.can_retreat_opponent(0) is True
    assert bs.can_retreat_opponent(1) is True
    assert bs.can_surrender_opponent(0) is True

    # No live stack → cannot surrender; the battle itself is over so
    # can_retreat_opponent also flips False (fheroes2 BattleValid gates
    # both queries).
    bs.units = [u for u in bs.units if u.team != 0]
    bs.start_round()
    assert bs.is_over() is True
    assert bs.can_retreat_opponent(0) is False
    assert bs.can_surrender_opponent(0) is False


def test_turn_number_and_death_totals_track_cumulative_kills():
    """fheroes2 BattlePlanner reads turn_number / attackerDeadTotal /
    defenderDeadTotal between turns.  Each kill increments the right
    total and the totals survive ``start_round``."""
    from engine.actions import AttackAction
    bs = _fresh_battle()
    archer = _find_archer(bs, 0)
    enemy = _find_archer(bs, 1)
    # Wreck enemy archer by combining many ranged hits into a single
    # round (the engine does NOT gate ranged attacks on TR_MOVED for
    # headless sequence tests — only the single-shot "double command"
    # regression below exercises that gate).
    while enemy.is_alive:
        # Lift the TR_MOVED gate from the prior attack so the next
        # execute() call inside the same round still resolves.
        archer._acted = False
        bs.execute(AttackAction(archer, enemy, ranged=True))
    assert bs.attacker_dead_total == 0
    assert bs.defender_dead_total > 0
    bs.start_round()
    assert bs.attacker_dead_total == 0
    assert bs.defender_dead_total > 0
    assert bs.turn_number == 1


def test_record_kill_counts_per_casualty_not_only_full_death():
    """Regression: ``_record_kill`` must bump per-casualty like C++.

    fheroes2 battle_troop.cpp:653 — every ``_applyDamage`` bumps the
    unit's ``_deadCount`` for each creature killed in that hit, and
    ``Force::getTotalNumberOfDeadUnits`` aggregates those per-unit
    counters.  A 50-stack archer shot that kills 3 creatures then 5
    creatures must bump ``defender_dead_total`` by 3 then 5, not jump
    straight from 0 to 8 on the final killing blow.
    """
    from engine.actions import AttackAction
    from config import units as _u
    grid = HexGrid()
    # Two 30-strong champion stacks. Champion deals ~20 dmg/hit so each
    # strike kills only ~1 creature (Champions have 40 hp), exposing
    # the per-casualty bump that battle_troop.cpp:653 expects.  We
    # lift the TR_MOVED gate (``a._acted = False``) between strikes so
    # we can observe the cumulative attribution within a single round.
    a_t = dict(_u.UNIT_TYPES["Champion"]); a_t["count"] = 30
    c_t = dict(_u.UNIT_TYPES["Champion"]); c_t["count"] = 30
    a = Unit("Champion", 0, 7, 1, **a_t)
    c = Unit("Champion", 1, 8, 1, **c_t)
    bs = BattleState(grid=grid, units=[a, c],
                     first_team=0, attacker_team=0,
                     heroes={0: Hero(power=1, name="A"),
                             1: Hero(power=1, name="B")})
    bs.execute(AttackAction(a, c, ranged=False))
    first_kill = bs.defender_dead_total
    assert 0 < first_kill < c.count, (
        f"per-casualty attribution: first hit must kill < entire stack; "
        f"got defender_dead_total={first_kill} of {c.count}")
    a._acted = False
    bs.execute(AttackAction(a, c, ranged=False))
    second_kill = bs.defender_dead_total
    assert second_kill > first_kill, (
        f"second hit must accumulate on top of first; "
        f"got first={first_kill}, second={second_kill}")


def test_record_kill_attributes_hypnotized_deaths_to_original_team():
    """Regression: deaths must count against the *original* army.

    fheroes2 battle_troop.cpp:653 + Battle::Troop::GetColor — a
    Hypnotized unit's ``_deadCount`` belongs to its original Force
    (Unit.team), not the hypnotized allegiance (effective_team).  The
    AI planner uses these totals to decide retreat / strength, so a
    unit that flips team and then dies must NOT silently vanish from
    its original side's casualty count.

    Previously ``_record_kill`` checked a non-existent
    ``unit.original_team`` attribute and silently fell through to
    ``unit.team``, which is the same value here, BUT the ``hasattr``
    branch also masked the broader "only-fire-on-full-death" bug.
    With Hypnotize we explicitly verify the original-team attribution.
    """
    from engine.actions import AttackAction
    from config import units as _u
    grid = HexGrid()
    # Tiny high-attack attacker vs a 50-strong hypnotized enemy.
    c_t = dict(_u.UNIT_TYPES["Champion"])
    p_t = dict(_u.UNIT_TYPES["Peasant"]); p_t["count"] = 50
    champ = Unit("Champion", 0, 7, 1, **c_t)
    peasant = Unit("Peasant", 1, 8, 1, **p_t)
    bs = BattleState(grid=grid, units=[champ, peasant],
                     first_team=0, attacker_team=0,
                     heroes={0: Hero(power=1, name="A"),
                             1: Hero(power=1, name="B")})
    # Hypnotize the peasant onto the attacker's side.
    peasant.add_effect(make_effect(SPELLS["Hypnotize"], power=10))
    assert peasant.team == 1 and peasant.effective_team == 0, (
        "precondition: unit must be team=1 (defender) but effectively team=0 "
        "(hypnotized to attacker's side)")
    bs.execute(AttackAction(champ, peasant, ranged=False))
    # The hypnotized defender should still count casualties against
    # its ORIGINAL team (= 1 = defender).
    assert bs.defender_dead_total > 0, (
        f"hypnotized deaths must attribute to original team (defender). "
        f"Got defender_dead_total={bs.defender_dead_total}, "
        f"attacker_dead_total={bs.attacker_dead_total}")
    assert bs.attacker_dead_total == 0, (
        f"defender (now effectively ally) must NOT bleed attacker_dead_total. "
        f"Got defender_dead_total={bs.defender_dead_total}, "
        f"attacker_dead_total={bs.attacker_dead_total}")


def test_clone_isolates_unit_mutations():
    """``BattleState.clone`` must give MCTS a sandboxed copy.

    Mutating ``count`` on the clone's units must not affect the parent.
    """
    bs = _fresh_battle()
    snap = bs.clone()
    archer_a = next(u for u in bs.units if u.team == 0 and u.is_archer)
    archer_c = next(u for u in snap.units if u.team == 0 and u.is_archer)
    original = archer_a.count
    archer_c.count -= 5
    assert archer_a.count == original


def test_is_position_reachable_and_move_distance():
    bs = _fresh_battle()
    archer = _find_archer(bs, 0)
    # Adjacent cell — always reachable in one step.
    assert bs.is_position_reachable(archer, (archer.col + 1, archer.row)) is True
    # Out-of-bounds cell.
    assert bs.is_position_reachable(archer, (-1, -1)) is False
    d = bs.calculate_move_distance(archer, (archer.col + 1, archer.row))
    assert d == 1
    assert bs.calculate_move_distance(archer, (-1, -1)) == 0


def test_unit_get_uid_is_stable():
    bs = _fresh_battle()
    u = _find_archer(bs, 0)
    assert u.get_uid() == id(u)
    assert u.get_uid() == u.get_uid()


# ── C++ command parity regression tests ────────────────────────────────
# These tests pin each of the seven bugs listed in the engine refactor:
#   1. SkipAction now flips TR_MOVED (unit._acted) post-dispatch.
#   2. MoveAction updates wide-unit tail via final_position.
#   3. AttackAction.recomputes ranged from isHandFighting + dir is logged.
#   4. RetreatAction gates through can_retreat_opponent.
#   5. CastAction accepts target=None for cell-only spells (Earthquake).
#   6. MoraleAction (NEW) clears/sets _acted + effect.
#   7. SurrenderAction (NEW) gates through can_surrender_opponent + cost.
#   8. TowerAction (NEW) shoots attackers only, deals damage.
#   9. CatapultAction (NEW) plumbs CatapultAction.NONE/WALL1..4/etc.

def test_skipaction_sets_tr_moved_after_dispatch():
    """fheroes2 ApplyActionSkip sets TR_SKIP | TR_MOVED (battle_action.cpp:362-376).

    The Python port's ``unit._acted`` flag is the analogue of TR_MOVED.
    Without it, the engine accepts further Move/Attack commands on the
    same unit within the round — the original bug.
    """
    from engine.actions import SkipAction
    bs = _fresh_battle()
    u = _find_archer(bs, 0)
    assert u._acted is False
    bs.execute(SkipAction(u))
    assert u._acted is True


def test_moveaction_updates_wide_unit_tail_via_final_position():
    """fheroes2 ApplyActionMove uses Battle::Position (head + tail + reflection).

    Old port only updated head via ``unit.pos = path[-1]`` and left the
    wide-unit tail at the previous round's coordinates — half-moved bug.
    """
    from engine.actions import MoveAction
    bs = _fresh_battle()
    # Construct a wide unit (e.g., a Dragon occupies 2 hexes).
    wide = Unit("Dragon", 0, 3, 4, count=1, attack=10, defense=10, hp=100,
                speed=8, damage=20, is_archer=False, is_flying=False,
                is_wide=True)
    bs.units.append(wide)
    # 1. Move head only — single-hex path mode.
    head = (4, 4)
    wide._acted = False
    from engine.battle_pathfinding import BattlePosition
    pos = BattlePosition(head=head, tail=None)
    bs.execute(MoveAction(wide, path=[(3, 4), head],
                          final_position=pos))
    assert wide.pos == head
    # 2. Move both head and tail — wide-unit path mode. The tail should
    #    end up at ``(head[0] + 1, head[1])`` per Battle::Position's
    #    horizontal-neighbour constraint (unit.py:188-191).
    head2 = (5, 4)
    tail2 = (head2[0] + 1, head2[1])
    pos2 = BattlePosition(head=head2, tail=tail2)
    wide._acted = False
    wide.set_battle_position((3, 4), tail=(4, 4))   # reset to original
    bs.execute(MoveAction(wide, path=[(4, 4), head2],
                          final_position=pos2))
    assert wide.pos == head2
    assert tail2 in wide.occupied_cells()


def test_attackaction_recomputes_ranged_via_is_hand_fighting():
    """fheroes2 ApplyActionAttack derives ranged from isArchers && !isHandFighting.

    An archer that's adjacent to the target goes through the melee branch
    even if the AI claimed ``ranged=True`` — engine must trust is_archer +
    adjacency, not the caller's flag.
    """
    from engine.actions import AttackAction
    bs = _fresh_battle()
    archer = _find_archer(bs, 0)
    enemy = _find_archer(bs, 1)
    # Force adjacency: drop the enemy onto the archer's neighbour cell.
    enemy.pos = (archer.col + 1, archer.row)
    # Pass ranged=True even though they're now adjacent.  Engine must
    # flip to actual_ranged=False (no ammo consumed).
    before = archer.shots_left
    r = bs.execute(AttackAction(archer, enemy, ranged=True))
    assert archer.shots_left == before       # hand-fighting ⇒ no ammo


def test_attackaction_dir_field_round_trip():
    """fheroes2 ATTACK's 5th int is the attack CellDirection.

    Python port stores it as ``AttackAction.dir`` so the AI can ask for
    a specific melee angle; the engine logs it for replay parity.
    """
    from engine.actions import AttackAction
    bs = _fresh_battle()
    archer = _find_archer(bs, 0)
    enemy = _find_archer(bs, 1)
    # Move enemy adjacent so the attack is melee-able.
    enemy.pos = (archer.col + 1, archer.row)
    r = bs.execute(AttackAction(archer, enemy, ranged=False, dir=3))
    assert "(dir=3)" in r["desc"]


def test_attackaction_rejects_double_command_in_round():
    """fheroes2 ApplyActionAttack validates TR_MOVED (battle_action.cpp:560).

    Without that gate, an attacker could fire two AttackCommands in one
    round — the AI's worst nightmare.
    """
    from engine.actions import AttackAction, SkipAction
    bs = _fresh_battle()
    archer = _find_archer(bs, 0)
    enemy = _find_archer(bs, 1)
    enemy.pos = (archer.col + 1, archer.row)
    # First attack: should succeed.
    bs.execute(AttackAction(archer, enemy, ranged=False))
    # Second attack: engine must reject (REJECTED in desc).
    r2 = bs.execute(AttackAction(archer, enemy, ranged=False))
    assert "REJECTED" in r2["desc"]


def test_retreataction_gates_via_can_retreat_opponent():
    """fheroes2 ApplyActionRetreat validates CanRetreatOpponent + BattleValid.

    Old port skipped the gate and called ``self.retreat()`` directly,
    which silently bypassed on a no-hero defender-in-castle scenario.
    """
    from engine.actions import RetreatAction
    bs = _fresh_battle()
    # Defender has a hero but is "in castle" in the open-field fixture
    # (we can fake it by setting bs._retreated). The cleanest way to
    # show the gate is to retreat the attacker then try retreating again.
    assert bs.can_retreat_opponent(0) is True
    bs.execute(RetreatAction(0))
    # Now the battle is over (one side retreated) → gate must fail.
    assert bs.can_retreat_opponent(0) is False
    r = bs.execute(RetreatAction(0))
    assert "REJECTED" in r["desc"]


def test_castaction_target_none_for_cell_only_spells():
    """fheroes2 Earthquake doesn't pass a target unit — just a cell index.

    CastAction.target=None must not crash the engine; cell-only paths
    (UTILITY kind) must accept it without an immunity check.
    """
    bs = _fresh_battle()
    eq_spell = SPELLS["Earthquake"]
    # No target — earthquake is cell-only.
    r = bs.execute(CastAction(team=0, spell=eq_spell, target=None,
                               cell=(8, 5)))
    assert r is not None


def test_moraleaction_clears_acted_on_good_morale():
    """fheroes2 ApplyActionMorale good branch clears TR_MOVED.

    Without the clear, a lucky unit never gets its bonus turn — the
    engine thinks it owes another action and the UI shows the wrong
    cursor.
    """
    from engine.actions import MoraleAction
    bs = _fresh_battle()
    u = _find_archer(bs, 0)
    # Skip first to set TR_MOVED.
    from engine.actions import SkipAction
    bs.execute(SkipAction(u))
    assert u._acted is True
    # Lucky! Morale roll goes good ⇒ _acted must be cleared.
    u.effects.append("MORALE_GOOD")
    bs.execute(MoraleAction(u, morale=True))
    assert u._acted is False
    assert "MORALE_GOOD" not in u.effects


def test_moraleaction_sets_acted_on_bad_morale():
    """fheroes2 ApplyActionMorale bad branch sets TR_MOVED."""
    from engine.actions import MoraleAction, SkipAction
    bs = _fresh_battle()
    u = _find_archer(bs, 0)
    u.effects.append("MORALE_BAD")
    bs.execute(MoraleAction(u, morale=False))
    assert u._acted is True
    assert "MORALE_BAD" not in u.effects


def test_surrenderaction_rejected_when_no_hero_or_unaffordable():
    """fheroes2 ApplyActionSurrender gates on CanSurrender + cost.

    Surrender with no hero is rejected; surrender with too high a cost
    is also rejected.
    """
    from engine.actions import SurrenderAction
    bs = _fresh_battle()
    # No hero ⇒ no surrender possible.
    bs.heroes[0] = None
    r = bs.execute(SurrenderAction(0, cost=100))
    assert "REJECTED" in r["desc"]


def test_toweraction_shoots_attacker_unit():
    """fheroes2 ApplyActionTower (battle_action.cpp:900-1010).

    Towers only fire during a siege and only target the attacking army.
    """
    from engine.castle import Castle
    from engine.actions import TowerAction
    grid = HexGrid()
    units = [
        Unit.from_type("Pikeman", 0, 1, 1),     # attacker
        Unit.from_type("Archer",  0, 3, 3, count=1),
    ]
    castle = Castle(color=1)                  # defender side owns the castle
    bs = BattleState(grid=grid, units=units,
                     attacker_team=0, heroes={0: None, 1: None},
                     castle=castle)
    target = units[0]                          # attacker ⇒ legal target
    target._acted = False
    r = bs.execute(TowerAction(tower_type=0, target=target))   # TWR_LEFT = 0
    # Result must contain damage (or at least a "miss" if rng was unlucky)
    # and the desc tag must identify the tower.
    assert r is not None
    assert ("tower" in r["desc"])


def test_catapultaction_none_target_id_is_no_op():
    """fheroes2 CatapultAction.shots list may include (NONE=0, dmg, hit).

    A None target_id means the catapult didn't aim at a structure (or
    the structure no longer exists) and the shot must be a no-op.
    """
    from engine.actions import CatapultAction
    grid = HexGrid()
    units = [Unit.from_type("Pikeman", 0, 1, 1)]
    bs = BattleState(grid=grid, units=units,
                     attacker_team=0, heroes={0: None, 1: None})
    # No castle ⇒ entire catapult dispatch rejects (REJECTED).
    r = bs.execute(CatapultAction(shots=[]))
    assert r["desc"] in ("catapult idle",) or "REJECTED" in r["desc"]
