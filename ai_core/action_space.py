"""R3 action space — flat discrete encoding + legality mask.

Encodes every possible battle action into a single integer index and provides
a binary mask indicating which actions are legal in the current state.

Action layout (14 655 total):
  Index 0              → Wait
  Index 1              → Defend
  Indices 2–100        → Move(hex[0..98])
  Indices 101–9901     → Attack(pos, target)  = pos × 99 + target
  Indices 9902–14653   → Cast(spell[0..47], hex[0..98])
  Index 14654          → Retreat

Hex indexing: row-major  = row × 11 + col  (0–98).

Teleport is excluded (needs two hexes; niche spell).
Wait and Defend both map to engine SkipAction (no distinct semantics yet).
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from engine.actions import (Action, MoveAction, AttackAction, SkipAction,
                            CastAction, RetreatAction)
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.spells import (Spell, SPELLS,
                            DAMAGE, AOE, BUFF, DEBUFF, CONTROL,
                            DISPEL, CURE, UTILITY)

# ── Grid constants ──────────────────────────────────────────────

GRID_ROWS = 9
GRID_COLS = 11
GRID_CELLS = GRID_ROWS * GRID_COLS  # 99


def cell_to_index(col: int, row: int) -> int:
    """(col, row) → flat index 0–98."""
    return row * GRID_COLS + col


def index_to_cell(idx: int) -> Tuple[int, int]:
    """Flat index 0–98 → (col, row)."""
    return idx % GRID_COLS, idx // GRID_COLS


# ── Spell ordering (alphabetical, excluding Teleport) ──────────

_SPELL_ORDER: List[str] = sorted(n for n in SPELLS if n != "Teleport")
_SPELL_INDEX: dict = {n: i for i, n in enumerate(_SPELL_ORDER)}  # 0–36
NUM_SPELLS = len(_SPELL_ORDER)  # 37

# ── Action-range boundaries ────────────────────────────────────

WAIT_IDX    = 0
DEFEND_IDX  = 1
MOVE_START  = 2
MOVE_END    = MOVE_START + GRID_CELLS - 1              # 100
ATTACK_START = MOVE_END + 1                             # 101
ATTACK_END   = ATTACK_START + GRID_CELLS ** 2 - 1      # 9901
CAST_START   = ATTACK_END + 1                           # 9902
CAST_END     = CAST_START + NUM_SPELLS * GRID_CELLS - 1 # 13564
RETREAT_IDX  = CAST_END + 1                             # 13565
ACTION_DIM   = RETREAT_IDX + 1                          # 14655


from ai_core.battle_geometry import _can_attack_from_pos, _tail_dir


# ── Spell legality helpers ─────────────────────────────────────

def _spell_target_team(spell: Spell) -> Optional[int]:
    """Which team can this spell target?  None = both (Dispel)."""
    if spell.kind in (BUFF, CURE):
        return 0  # side_friendly — resolved per-caster at call site
    if spell.kind in (DAMAGE, DEBUFF, CONTROL):
        return 1  # enemy — resolved per-caster at call site
    return None  # DISPEL: either team


def _is_mass_or_armywide(spell: Spell) -> bool:
    """True if the spell doesn't need an individual target hex."""
    return (spell.is_mass
            or spell.aoe_pattern in ("all_tagged", "all_units")
            or (spell.kind == UTILITY and spell.name == "Earthquake"))


def _is_ring_aoe(spell: Spell) -> bool:
    """True for AOE spells that take a center cell."""
    return spell.aoe_pattern in ("ring1", "ring2", "ring_outer")


# ── Core API ───────────────────────────────────────────────────

def action_to_index(action: Action, battle: BattleState,
                    current_unit: Unit) -> int:
    """Convert an Action object to its flat index.

    SkipAction maps to WAIT_IDX by default (Defend distinction lost).
    """
    if isinstance(action, SkipAction):
        return WAIT_IDX
    if isinstance(action, RetreatAction):
        return RETREAT_IDX
    if isinstance(action, MoveAction):
        return MOVE_START + cell_to_index(*action.path[-1])
    if isinstance(action, AttackAction):
        tgt_idx = cell_to_index(*action.target.pos)
        if action.ranged:
            pos_idx = cell_to_index(*action.attacker.pos)
        else:
            from_cell = action.from_pos if action.from_pos else action.attacker.pos
            pos_idx = cell_to_index(*from_cell)
        return ATTACK_START + pos_idx * GRID_CELLS + tgt_idx
    if isinstance(action, CastAction):
        slot = _SPELL_INDEX.get(action.spell.name)
        if slot is None:
            raise ValueError(f"Spell '{action.spell.name}' not in action space "
                             "(Teleport is excluded)")
        if action.cell is not None:
            hex_idx = cell_to_index(*action.cell)
        elif action.target is not None:
            hex_idx = cell_to_index(*action.target.pos)
        else:
            hex_idx = 0
        return CAST_START + slot * GRID_CELLS + hex_idx
    raise ValueError(f"Unknown action type: {type(action).__name__}")


def index_to_action(index: int, battle: BattleState,
                    current_unit: Unit) -> Action:
    """Convert a flat index back to an Action object.

    Returns SkipAction as fallback for invalid/out-of-range indices.
    """
    # ── Wait / Defend ──
    if index == WAIT_IDX or index == DEFEND_IDX:
        return SkipAction(current_unit)

    # ── Retreat ──
    if index == RETREAT_IDX:
        return RetreatAction(current_unit.team)

    # ── Move ──
    if MOVE_START <= index <= MOVE_END:
        hex_idx = index - MOVE_START
        col, row = index_to_cell(hex_idx)
        dest = (col, row)
        built = battle.build_path(current_unit, dest)
        if built is None:
            return SkipAction(current_unit)
        path, final_position = built
        return MoveAction(current_unit, path, final_position=final_position)

    # ── Attack ──
    if ATTACK_START <= index <= ATTACK_END:
        offset = index - ATTACK_START
        pos_idx = offset // GRID_CELLS
        tgt_idx = offset % GRID_CELLS
        tgt_col, tgt_row = index_to_cell(tgt_idx)
        target = battle.unit_at((tgt_col, tgt_row))
        if target is None:
            return SkipAction(current_unit)
        attacker_idx = cell_to_index(*current_unit.pos)
        pos_col, pos_row = index_to_cell(pos_idx)
        if pos_idx == attacker_idx and current_unit.is_archer:
            return AttackAction(current_unit, target, ranged=True)
        else:
            from_cell = (pos_col, pos_row)
            from_position = battle._reachable_position(
                current_unit, from_cell, True)
            if from_position is None:
                return SkipAction(current_unit)
            return AttackAction(current_unit, target,
                                from_pos=from_position.head,
                                from_position=from_position,
                                ranged=False)

    # ── Cast ──
    if CAST_START <= index <= CAST_END:
        offset = index - CAST_START
        spell_slot = offset // GRID_CELLS
        hex_idx = offset % GRID_CELLS
        hex_col, hex_row = index_to_cell(hex_idx)
        spell_name = _SPELL_ORDER[spell_slot]
        spell = SPELLS[spell_name]
        team = current_unit.team
        hero = battle.heroes.get(team)

        if hero is None:
            return SkipAction(current_unit)

        # Determine target unit and optional cell/destination
        target = battle.unit_at((hex_col, hex_row))
        cell = None

        if _is_ring_aoe(spell):
            cell = (hex_col, hex_row)
            if target is None:
                # Ring AOE can center on any cell; pick a placeholder target
                alive = battle.alive(1 - team) or battle.alive(team)
                target = alive[0] if alive else None
        elif spell.aoe_pattern == "chain":
            if target is None:
                enemies = battle.alive(1 - team)
                target = enemies[0] if enemies else None
        elif _is_mass_or_armywide(spell):
            if target is None:
                side = team if spell.side_friendly else (1 - team)
                if spell.name == "Earthquake":
                    side = 0  # placeholder
                candidates = battle.alive(side)
                target = candidates[0] if candidates else None

        if target is None:
            return SkipAction(current_unit)

        return CastAction(team, spell, target, cell=cell)

    # Out of range → fallback
    return SkipAction(current_unit)


# ── Legal mask ─────────────────────────────────────────────────

def legal_mask(battle: BattleState, current_unit: Unit) -> np.ndarray:
    """Return a float32 binary mask of shape (ACTION_DIM,).

    1.0 = legal, 0.0 = illegal.  Always non-empty (Wait is always legal).
    """
    mask = np.zeros(ACTION_DIM, dtype=np.float32)

    # Wait / Defend — always legal
    mask[WAIT_IDX] = 1.0
    mask[DEFEND_IDX] = 1.0

    # Retreat — legal if that side has a hero
    hero = battle.heroes.get(current_unit.team)
    if hero is not None:
        mask[RETREAT_IDX] = 1.0

    # ── Move ──
    reachable = battle.get_all_available_moves(current_unit)
    for cell in reachable:
        if cell == current_unit.pos:
            continue  # staying put is not a "move"
        mask[MOVE_START + cell_to_index(*cell)] = 1.0

    # ── Attack ──
    _mark_attack_legal(mask, battle, current_unit, reachable, set(), None)

    # ── Cast ──
    if hero is not None and not hero._cast_this_round:
        _mark_cast_legal(mask, battle, current_unit, hero)

    return mask


def _mark_attack_legal(mask: np.ndarray, battle: BattleState,
                        unit: Unit, reachable: Set[Tuple[int, int]],
                        occ: Set[Tuple[int, int]], moat) -> None:
    """Mark legal melee and ranged attack positions."""
    grid = battle.grid
    enemies = battle.enemies_of(unit)
    attacker_idx = cell_to_index(*unit.pos)

    for enemy in enemies:
        tgt_idx = cell_to_index(*enemy.pos)

        # ── Ranged (archer only) ──
        if unit.is_archer:
            mask[ATTACK_START + attacker_idx * GRID_CELLS + tgt_idx] = 1.0

        # ── Melee ──
        for ac in set(reachable) | {unit.pos}:
            position = battle._reachable_position(unit, ac, True)
            if position is None:
                continue
            body = {position.head}
            if position.tail is not None:
                body.add(position.tail)
            if min(grid.distance(a, b)
                   for a in body
                   for b in enemy.occupied_cells()) != 1:
                continue
            pos_idx = cell_to_index(*position.head)
            mask[ATTACK_START + pos_idx * GRID_CELLS + tgt_idx] = 1.0


def _mark_cast_legal(mask: np.ndarray, battle: BattleState,
                      unit: Unit, hero) -> None:
    """Mark legal spell-casting actions for the hero of *unit*'s team."""
    team = unit.team
    friendly = battle.alive(team)
    enemies = battle.alive(1 - team)

    for spell_slot, spell_name in enumerate(_SPELL_ORDER):
        spell = SPELLS[spell_name]

        # Check hero can afford and hasn't cast this round
        if spell_name not in [s.name for s in hero.spellbook]:
            continue
        if not hero.can_cast(spell):
            continue

        base = CAST_START + spell_slot * GRID_CELLS

        if _is_mass_or_armywide(spell):
            # All hexes legal (target ignored at execution)
            mask[base:base + GRID_CELLS] = 1.0
            continue

        if _is_ring_aoe(spell):
            # Any cell can be the center of the AOE
            mask[base:base + GRID_CELLS] = 1.0
            continue

        if spell.aoe_pattern == "chain":
            # Chain Lightning: hex must have an alive enemy (first bounce)
            for e in enemies:
                if e.is_immune_to_spells:
                    continue
                idx = cell_to_index(*e.pos)
                mask[base + idx] = 1.0
            continue

        # ── Single-target spells ──
        _mark_single_target_spell(mask, base, battle, spell, team,
                                   friendly, enemies)


def _mark_single_target_spell(mask: np.ndarray, base: int,
                                battle: BattleState, spell: Spell,
                                team: int,
                                friendly: list, enemies: list) -> None:
    """Mark legal hexes for a single-target spell."""
    # Determine which team the spell can target
    if spell.side_friendly or spell.kind in (BUFF, CURE):
        candidates = friendly
    elif spell.kind == DISPEL:
        # Dispel can target either side
        candidates = friendly + enemies
    else:
        # DAMAGE, DEBUFF, CONTROL
        candidates = enemies

    for unit in candidates:
        if not unit.is_alive:
            continue
        if unit.is_immune_to_spells:
            continue
        # Already has this buff/debuff → skip
        if spell.kind in (BUFF, DEBUFF, CONTROL) and unit.has_effect(spell.name):
            continue
        # Tag exclusions (e.g. undead can't be Blessed/Cursed)
        if spell.exclude_tags:
            if any(unit.has_tag(t) for t in spell.exclude_tags):
                continue
        idx = cell_to_index(*unit.pos)
        mask[base + idx] = 1.0


# ── Convenience ────────────────────────────────────────────────

def enumerate_legal(battle: BattleState, current_unit: Unit) -> List[int]:
    """Return sorted list of all legal action indices."""
    m = legal_mask(battle, current_unit)
    return sorted(int(i) for i in np.nonzero(m)[0])


def action_type_label(index: int) -> str:
    """Human-readable label for an action index (for debugging)."""
    if index == WAIT_IDX:
        return "Wait"
    if index == DEFEND_IDX:
        return "Defend"
    if MOVE_START <= index <= MOVE_END:
        col, row = index_to_cell(index - MOVE_START)
        return f"Move({col},{row})"
    if ATTACK_START <= index <= ATTACK_END:
        offset = index - ATTACK_START
        pos_idx = offset // GRID_CELLS
        tgt_idx = offset % GRID_CELLS
        pc, pr = index_to_cell(pos_idx)
        tc, tr = index_to_cell(tgt_idx)
        return f"Attack(({pc},{pr})->({tc},{tr}))"
    if CAST_START <= index <= CAST_END:
        offset = index - CAST_START
        slot = offset // GRID_CELLS
        hex_idx = offset % GRID_CELLS
        hc, hr = index_to_cell(hex_idx)
        return f"Cast({_SPELL_ORDER[slot]}@({hc},{hr}))"
    if index == RETREAT_IDX:
        return "Retreat"
    return f"Unknown({index})"
