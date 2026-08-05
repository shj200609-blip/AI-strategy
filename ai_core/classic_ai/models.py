"""Small data holders used by the classic battle planner."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

import config
from engine.unit import Unit


# ── C++ ``std::numeric_limits<double>::lowest()`` analogue ──────────────
# ``-sys.float_info.max`` is the most-negative finite Python float. We do
# NOT use ``float("-inf")`` here because
# ``abs(float("-inf") - float("-inf")) == nan`` which would break the
# 0.001 epsilon tertiary comparison in C++ ``ValueHasImproved``.
_DOUBLE_LOWEST: float = -sys.float_info.max


# ── cell identity: C++ linear index ↔ Python (col, row) ─────────────────
# Every ``cell`` / ``index`` field in ai_battle.h is a single ``int32_t``
# board index; fheroes2 lays the 11x9 arena out row-major, so
#
#     index = row * ARENAW + col        (ARENAW == 11 == config.GRID_COLS)
#
# This port addresses cells by their ``(col, row)`` offset pair instead —
# the convention used everywhere in ``engine.hex_grid`` and by
# ``Unit.pos`` — because the Python engine never materialises a flat
# board array. The two representations are interchangeable via the
# helpers below; they exist so parity tests and log comparisons against
# the C++ AI can speak either dialect.
#
# The other half of the mapping is the "no cell" sentinel: C++ spells its
# absent index as ``-1`` (``BattleTargetPair::cell``,
# ``SpellSelection::cell`` / ``destinationCell``, ``SpellcastOutcome``),
# while Python uses ``None``. Any code that ports a ``cell == -1`` guard
# must therefore become an ``is None`` check, and vice versa —
# ``_index_to_cell``/``_cell_to_index`` round-trip that sentinel too.

_NO_CELL: int = -1


def _cell_to_index(cell: Optional[Tuple[int, int]]) -> int:
    """``(col, row)`` → fheroes2 board index (``None`` → ``-1``)."""
    if cell is None:
        return _NO_CELL
    col, row = cell
    return row * config.GRID_COLS + col


def _index_to_cell(index: int) -> Optional[Tuple[int, int]]:
    """fheroes2 board index → ``(col, row)`` (``-1`` → ``None``)."""
    if index is None or index < 0:
        return None
    return (index % config.GRID_COLS, index // config.GRID_COLS)


@dataclass
class _TargetPair:
    """Mirrors fheroes2 ``BattleTargetPair``.

    ``cell`` is the destination / source cell (depending on the call
    site). ``unit`` is the enemy to engage. ``from_index`` is the
    C++-chosen attack-origin cell, populated by the melee port so
    callers can preserve the chosen source position through to
    ``AttackAction.from_pos``.

    Representation note: C++ stores ``int cell{ -1 }`` — a single board
    index — whereas both cell fields here hold a ``(col, row)`` pair and
    use ``None`` for "unset". See the module-level index↔coordinate
    section above; ``_cell_to_index`` / ``_index_to_cell`` convert.
    ``from_index`` has no C++ counterpart in ``BattleTargetPair``: the
    C++ melee code returns the attack origin through the separate
    ``Position`` it already holds, which this port does not thread
    through, so it rides along on the pair instead.
    """
    cell: Optional[Tuple[int, int]] = None
    unit: Optional[Unit] = None
    from_index: Optional[Tuple[int, int]] = None


@dataclass
class _PositionCharacteristics:
    """Mutable holder for archer retreat cell evaluation."""
    threatening: Set[int] = field(default_factory=set)
    distance: int = 10**9


@dataclass
class _MeleePosition:
    """fheroes2 ``Battle::Position``.

    A logical position has a head cell and (for wide units) an optional
    tail cell; ``reflected`` mirrors ``Position::isReflect()`` (the head
    sits *behind* the tail from the unit's natural orientation). The C++
    struct also stores the owning unit, but Python already threads the
    unit through the helpers that build positions, so it is omitted here.
    """
    head: Tuple[int, int]
    tail: Optional[Tuple[int, int]] = None
    reflected: bool = False

    def get_head(self) -> Tuple[int, int]:
        return self.head


@dataclass
class _MeleeAttackOutcome:
    """fheroes2 ``AI::BattlePlanner::MeleeAttackOutcome``.

    Mirrors the C++ aggregate verbatim:

      * ``fromIndex``      — the head cell from which the attack is made
                              (``int32_t`` board index in C++, ``(col,
                              row)`` here — see the module-level
                              index↔coordinate note; ``None`` == ``-1``).
      * ``attackValue``    — best attack score at the candidate position.
      * ``positionValue``  — position-only score (used for the second-
                              level comparison when ``attackValue`` is
                              close to the prior best).
      * ``canAttackImmediately`` — at least one enemy can be engaged from
                              the candidate position this turn.

    ``getMeleeBestOutcome`` ranks candidates by the three-level
    ``IsOutcomeImproved`` comparator:
      1. ``canAttackImmediately`` beats a non-immediate position;
      2. within the same immediacy, the larger ``positionValue`` wins;
      3. within an epsilon (0.001) on ``positionValue``, the larger
         ``attackValue`` wins.

    All numeric fields use ``float`` and the most-negative finite sentinel
    (``-sys.float_info.max``) to mirror C++
    ``std::numeric_limits<double>::lowest()``.
    """
    from_index: Optional[Tuple[int, int]] = None
    attack_value: float = _DOUBLE_LOWEST
    position_value: float = _DOUBLE_LOWEST
    can_attack_immediately: bool = False


# ── spell-planning aggregates (ai_battle.h:47-71) ───────────────────────

@dataclass
class _SpellcastOutcome:
    """fheroes2 ``AI::SpellcastOutcome``.

    The accumulator every ``spell*Value`` scorer fills in while walking
    its candidate targets. ``cell`` is the chosen target cell (C++
    ``int32_t`` index, ``(col, row)`` here, ``None`` == ``-1``);
    ``destination_cell`` is only used by Teleport, which needs both a
    source and a landing cell.
    """
    cell: Optional[Tuple[int, int]] = None
    value: float = 0.0
    destination_cell: Optional[Tuple[int, int]] = None

    def update_outcome(self, potential_value: float,
                       target_cell: Optional[Tuple[int, int]],
                       is_mass_effect: bool = False) -> None:
        """Port of ``SpellcastOutcome::updateOutcome`` — verbatim.

        Two distinct accumulation modes:

          * *mass effect* (``spell.isMassActions()``) — the spell hits
            every valid target at once, so its worth is the **sum** over
            all of them. ``potential_value`` is added unconditionally,
            negatives included, and ``cell`` is deliberately left alone
            (a mass spell has no target cell; C++ keeps it at ``-1``).
          * *single target* — keep the **maximum**, recording the cell
            that produced it. Because ``value`` starts at ``0.0`` and the
            comparison is strict, a candidate worth <= 0 never wins and
            ``cell`` stays unset.
        """
        if is_mass_effect:
            self.value += potential_value
        elif potential_value > self.value:
            self.value = potential_value
            self.cell = target_cell


@dataclass
class _SpellSelection:
    """fheroes2 ``AI::SpellSelection`` — ``selectBestSpell``'s return.

    ``spell_id`` identifies the winning spell (C++ ``int`` spell ID,
    ``-1`` for "cast nothing"); this port keys spells by name, so
    ``spell_id`` carries the ``Spell`` object / name used by
    ``engine.spells.SPELLS`` and ``None`` is the empty selection. The
    remaining three fields are the winning ``_SpellcastOutcome`` copied
    out verbatim.
    """
    spell_id: Optional[object] = None
    cell: Optional[Tuple[int, int]] = None
    value: float = 0.0
    destination_cell: Optional[Tuple[int, int]] = None

    @classmethod
    def from_outcome(cls, spell_id, outcome: "_SpellcastOutcome"
                     ) -> "_SpellSelection":
        """C++ ``bestSpell = { spell.GetID(), outcome.cell, ... }``."""
        return cls(spell_id=spell_id, cell=outcome.cell,
                   value=outcome.value,
                   destination_cell=outcome.destination_cell)


# ── C++ ``ValueHasImproved`` / ``IsOutcomeImproved`` ────────────────────

def _value_has_improved(primary: float, primary_max: float,
                        secondary: float, secondary_max: float,
                        epsilon: float = 0.001) -> bool:
    """fheroes2 ``ValueHasImproved``.

    Returns True iff ``primary`` strictly improves over ``primary_max``,
    OR ``primary`` ties ``primary_max`` within *epsilon* AND ``secondary``
    strictly improves over ``secondary_max``.
    """
    if primary_max < primary:
        return True
    return (abs(primary_max - primary) < epsilon
            and secondary_max < secondary)


def _is_outcome_improved(new_outcome: _MeleeAttackOutcome,
                         previous: _MeleeAttackOutcome) -> bool:
    """fheroes2 ``IsOutcomeImproved`` (ai_battle.cpp:84-96).

    Three-level comparison: any immediate-attack candidate strictly
    outranks a non-immediate one. Within the same immediacy, prefer
    larger ``positionValue``; ties on ``positionValue`` (within 0.001)
    are broken by larger ``attackValue``.
    """
    if new_outcome.can_attack_immediately and not previous.can_attack_immediately:
        return True
    if new_outcome.can_attack_immediately != previous.can_attack_immediately:
        return False
    return _value_has_improved(
        new_outcome.position_value, previous.position_value,
        new_outcome.attack_value, previous.attack_value,
    )