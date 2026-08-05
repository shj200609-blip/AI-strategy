"""Observation encoder: BattleState → neural network input tensors.

Encodes the battle state as player-relative feature maps (35×9×11 grid tensor)
and a global scalar vector (20-dim) for input to a CNN-based policy network.

Player-relative encoding
------------------------
Always encoded from the current acting unit's perspective.  When team 0 acts,
"my" = team 0; when team 1 acts, "my" = team 1.  This enables parameter sharing
across both sides (AlphaStar-style), doubling data efficiency.

Grid channel layout (35 channels × 9 rows × 11 cols)
-----------------------------------------------------
 0–9:  My units   (existence, hp, count, atk, def, spd, archer, flyer, wide_tail, acted)
10–19: Enemy units (same layout, offset by 10)
20–29: Status effects (property-based detection — see ``_encode_effects``)
30–32: Siege structures (wall HP, moat, towers)
33:    My unit type index (normalised 0–1, 0 = no unit)
34:    Enemy unit type index (normalised 0–1, 0 = no unit)

Global vector (20 dims): round, attacker team, unit counts, HP totals,
hero stats, siege state, morale/luck, current-unit index.
"""

import numpy as np
from typing import Tuple

from config.units import UNIT_TYPE_INDEX, NUM_UNIT_TYPES
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.castle import MOAT_CELLS

# ── Public constants ───────────────────────────────────────────

GRID_ROWS = 9
GRID_COLS = 11
NUM_GRID_CHANNELS = 35
GLOBAL_DIM = 20

# ── Internal normalisation constants ───────────────────────────

_MAX_ROUNDS = 200        # BattleState.MAX_ROUNDS
_MAX_STAT = 30           # attack / defence ceiling
_MAX_SPEED = 10          # speed ceiling
_MAX_SPELL_POINTS = 100  # hero spell-points ceiling
_MAX_POWER = 15          # hero power ceiling
_MAX_HERO_STAT = 15      # hero primary-attribute ceiling
_MAX_DR_STACKS = 5       # Disrupting Ray practical ceiling
_MAX_TYPE_INDEX = NUM_UNIT_TYPES - 1  # max real unit index (66)


# ── Channel index helpers ──────────────────────────────────────

# Unit attribute channels (relative to base: 0 for my, 10 for enemy)
_CH_EXISTENCE = 0
_CH_HP_RATIO = 1
_CH_COUNT_RATIO = 2
_CH_ATTACK = 3
_CH_DEFENSE = 4
_CH_SPEED = 5
_CH_ARCHER = 6
_CH_FLYER = 7
_CH_WIDE_TAIL = 8
_CH_ACTED = 9

# Effect channels (absolute)
_CH_HASTE = 20
_CH_SLOW = 21
_CH_BLESS = 22
_CH_CURSE = 23
_CH_BLIND = 24
_CH_ATK_BUFF = 25
_CH_DEF_BUFF = 26
_CH_SHIELD = 27
_CH_ANTI_MAGIC = 28
_CH_DISRUPTING = 29

# Siege channels (absolute)
_CH_WALL = 30
_CH_MOAT = 31
_CH_TOWER = 32

# Unit type channels (absolute — T8)
_CH_MY_TYPE = 33
_CH_ENEMY_TYPE = 34


# ── Public API ─────────────────────────────────────────────────


def encode_observation(
    battle: BattleState,
    current_unit: Unit,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the battle state from *current_unit*'s perspective.

    Args:
        battle:       Current battle state (must have ``_initial_counts`` set).
        current_unit: The unit about to act (determines player-relative view).

    Returns:
        grid_tensor:   ``float32`` array of shape ``(35, 9, 11)``
        global_vector: ``float32`` array of shape ``(20,)``
    """
    my_team = current_unit.team

    grid = np.zeros((NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS), dtype=np.float32)

    _encode_units(grid, battle, my_team)
    _encode_effects(grid, battle)
    _encode_siege(grid, battle)

    gvec = _encode_global(battle, current_unit)

    return grid, gvec


# ── Internal helpers ───────────────────────────────────────────


def _set_on_cells(grid: np.ndarray, ch: int, unit: Unit, value: float = 1.0):
    """Set *value* on channel *ch* at every cell occupied by *unit*."""
    c, r = unit.pos
    grid[ch, r, c] = value
    if unit.is_wide and unit.tail_cell:
        tc, tr = unit.tail_cell
        grid[ch, tr, tc] = value


def _encode_units(grid: np.ndarray, battle: BattleState, my_team: int):
    """Fill channels 0–9 (my), 10–19 (enemy), and 33–34 (type index)."""
    initial_counts = getattr(battle, "_initial_counts", {})

    for unit in battle.alive():
        base = 0 if unit.team == my_team else 10
        type_ch = _CH_MY_TYPE if unit.team == my_team else _CH_ENEMY_TYPE

        # Normalised attributes
        hp_ratio = (
            unit._total_hp / unit._max_total_hp if unit._max_total_hp > 0 else 0.0
        )
        init_count = initial_counts.get(id(unit), unit.count)
        count_ratio = unit.count / init_count if init_count > 0 else 0.0
        atk = unit.effective_attack / _MAX_STAT
        dfn = unit.effective_defense / _MAX_STAT
        spd = unit.speed / _MAX_SPEED

        # Unit type index (T8): 0 = no unit, 1–66 = real units
        type_idx = UNIT_TYPE_INDEX.get(unit.name, 0)
        type_norm = type_idx / _MAX_TYPE_INDEX

        # Head cell — full attribute suite
        c, r = unit.pos
        grid[base + _CH_EXISTENCE, r, c] = 1.0
        grid[base + _CH_HP_RATIO, r, c] = hp_ratio
        grid[base + _CH_COUNT_RATIO, r, c] = count_ratio
        grid[base + _CH_ATTACK, r, c] = atk
        grid[base + _CH_DEFENSE, r, c] = dfn
        grid[base + _CH_SPEED, r, c] = spd
        grid[base + _CH_ARCHER, r, c] = float(unit.is_archer)
        grid[base + _CH_FLYER, r, c] = float(unit.is_flying)
        # _CH_WIDE_TAIL: only on the tail cell, not the head
        grid[base + _CH_ACTED, r, c] = float(unit._acted)

        # Type index on head cell
        grid[type_ch, r, c] = type_norm

        # Tail cell — existence + wide-tail marker + type index
        # Bounds check: wide unit tail may extend beyond grid edge
        if unit.is_wide and unit.tail_cell:
            tc, tr = unit.tail_cell
            if 0 <= tc < GRID_COLS and 0 <= tr < GRID_ROWS:
                grid[base + _CH_EXISTENCE, tr, tc] = 1.0
                grid[base + _CH_WIDE_TAIL, tr, tc] = 1.0
                grid[type_ch, tr, tc] = type_norm

def _encode_effects(grid: np.ndarray, battle: BattleState):
    """Fill channels 20–29 with status-effect markers.

    Effects are detected by *property* (not by spell name) so that the agent
    perceives the consequence rather than the label.  The one exception is
    Disrupting Ray (channel 29), which encodes the *stack count* as a
    continuous value.
    """
    for unit in battle.alive():
        for eff in unit.effects:
            if eff.speed_delta > 0:
                _set_on_cells(grid, _CH_HASTE, unit)
            if eff.speed_delta < 0:
                _set_on_cells(grid, _CH_SLOW, unit)
            if eff.name == "Bless":
                _set_on_cells(grid, _CH_BLESS, unit)
            if eff.name == "Curse":
                _set_on_cells(grid, _CH_CURSE, unit)
            if eff.skip_turn:
                _set_on_cells(grid, _CH_BLIND, unit)
            if eff.attack_delta > 0:
                _set_on_cells(grid, _CH_ATK_BUFF, unit)
            if eff.defense_delta > 0:
                _set_on_cells(grid, _CH_DEF_BUFF, unit)
            if eff.ranged_shield < 1.0:
                _set_on_cells(grid, _CH_SHIELD, unit)
            if eff.anti_magic:
                _set_on_cells(grid, _CH_ANTI_MAGIC, unit)

        # Disrupting Ray stacks (continuous 0–1)
        dr_stacks = sum(1 for e in unit.effects if e.name == "Disrupting Ray")
        if dr_stacks > 0:
            _set_on_cells(grid, _CH_DISRUPTING, unit,
                          min(1.0, dr_stacks / _MAX_DR_STACKS))


def _encode_siege(grid: np.ndarray, battle: BattleState):
    """Fill channels 30–32 with siege structure data."""
    castle = battle.castle
    if castle is None:
        return

    # Channel 30: wall HP (0 / 0.5 / 1.0)
    for (c, r), hp in castle.walls.items():
        grid[_CH_WALL, r, c] = hp / 2.0

    # Channel 31: moat cells
    for c, r in MOAT_CELLS:
        grid[_CH_MOAT, r, c] = 1.0

    # Channel 32: active tower positions.
    # towers: [left(0) at (8,1), center(1) — no grid cell, right(2) at (8,7)]
    tower_grid_pos = [(8, 1), (8, 7)]  # left, right archer towers
    for i, (c, r) in enumerate(tower_grid_pos):
        idx = 0 if i == 0 else 2  # map to towers[0] (left) or towers[2] (right)
        if castle.towers[idx].is_valid:
            grid[_CH_TOWER, r, c] = 1.0


def _encode_global(battle: BattleState, current_unit: Unit) -> np.ndarray:
    """Build the 20-dimensional global scalar vector."""
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)

    my_team = current_unit.team
    enemy_team = 1 - my_team

    # 0: round / MAX_ROUNDS
    g[0] = battle.round_num / _MAX_ROUNDS

    # 1: attacker team (0/1)
    g[1] = float(battle.attacker_team)

    # 2–3: alive unit counts / 7
    g[2] = len(battle.alive(my_team)) / 7.0
    g[3] = len(battle.alive(enemy_team)) / 7.0

    # 4–5: total HP ratio (current / initial, includes dead units in denominator)
    for idx, team in ((4, my_team), (5, enemy_team)):
        hp = sum(u._total_hp for u in battle.units if u.team == team)
        max_hp = sum(u._max_total_hp for u in battle.units if u.team == team)
        g[idx] = hp / max_hp if max_hp > 0 else 0.0

    # 6–7: hero spell points
    my_hero = battle.heroes.get(my_team)
    en_hero = battle.heroes.get(enemy_team)
    g[6] = my_hero.spell_points / _MAX_SPELL_POINTS if my_hero else 0.0
    g[7] = en_hero.spell_points / _MAX_SPELL_POINTS if en_hero else 0.0

    # 8–9: hero power
    g[8] = my_hero.power / _MAX_POWER if my_hero else 0.0
    g[9] = en_hero.power / _MAX_POWER if en_hero else 0.0

    # 10–11: hero attack
    g[10] = my_hero.attack / _MAX_HERO_STAT if my_hero else 0.0
    g[11] = en_hero.attack / _MAX_HERO_STAT if en_hero else 0.0

    # 12–13: hero defense
    g[12] = my_hero.defense / _MAX_HERO_STAT if my_hero else 0.0
    g[13] = en_hero.defense / _MAX_HERO_STAT if en_hero else 0.0

    # 14: is siege
    g[14] = float(battle.castle is not None)

    # 15: active towers / 3
    if battle.castle:
        g[15] = sum(1 for t in battle.castle.towers if t.is_valid) / 3.0

    # 16: intact walls / 4
    if battle.castle:
        g[16] = sum(1 for hp in battle.castle.walls.values() if hp > 0) / 4.0

    # 17: my morale / 3 → [-1, 1]
    g[17] = battle.morale.get(my_team, 0) / 3.0

    # 18: my luck / 3 → [-1, 1]
    g[18] = battle.luck.get(my_team, 0) / 3.0

    # 19: current unit index in turn order / 14
    order = battle.turn_order()
    try:
        ui = order.index(current_unit)
    except ValueError:
        ui = 0
    g[19] = ui / 14.0

    return g
