"""ClassicAI movement helpers backed by the engine BattlePathfinder."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from engine.battle_pathfinding import BattlePosition
from engine.battle_state import BattleState
from engine.unit import Unit


def _reachable_cells(self, battle: BattleState, unit: Unit
                      ) -> Dict[Tuple[int, int], int]:
    return battle.get_all_available_moves(unit)


def _path_to(self, battle: BattleState, unit: Unit,
             goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Return the C++ ``GetPath`` current-turn prefix toward *goal*."""
    result = battle.build_path(unit, goal)
    return result[0] if result is not None else None


def _unit_movement_position(self, battle: BattleState, unit: Unit,
                            dest: Tuple[int, int]
                            ) -> Optional[BattlePosition]:
    """C++ ``getUnitMovementTarget`` retaining the resolved orientation."""
    reachable = battle._reachable_position(unit, dest, True)
    if reachable is not None:
        return reachable

    destination = battle._position_for_cell(unit, dest)
    if destination is None:
        return None
    return battle._pathfinder(unit).closest_reachable_position(destination)


def _unit_movement_target(self, battle: BattleState, unit: Unit,
                          dest: Tuple[int, int]) -> Tuple[int, int]:
    position = _unit_movement_position(self, battle, unit, dest)
    if position is None:
        raise ValueError(f"unreachable movement target: {dest}")
    return position.head
