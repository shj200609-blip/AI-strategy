"""Shared wide-unit battle geometry.

Both the rule-based planner and the neural-network action encoder use these
helpers so their movement and melee-legality rules cannot drift apart.
"""

from typing import List, Optional, Tuple

from engine.unit import Unit


def _tail_dir(unit: Unit) -> Optional[int]:
    """Column offset of a wide unit's tail; None for single-hex units."""
    return unit.tail_offset


def _attack_cells(grid, target: Unit) -> List[Tuple[int, int]]:
    """Cells from which a melee attacker can strike *target*."""
    cells = list(grid.neighbors(*target.pos))
    if target.is_wide:
        body = target.occupied_cells()
        for tc in grid.neighbors(*target.tail_cell):
            if tc not in cells and tc not in body:
                cells.append(tc)
    return cells


def _can_attack_from_pos(grid, unit: Unit, target: Unit,
                         pos: Tuple[int, int], moat=None) -> bool:
    """Validate that a melee attacker at *pos* can strike *target*."""
    if not unit.is_wide:
        return True

    td = _tail_dir(unit)
    tail = (pos[0] + td, pos[1]) if td is not None else pos

    head_adj = _pos_dist(grid, pos, target) <= 1
    tail_adj = _pos_dist(grid, tail, target) <= 1
    if not head_adj and not tail_adj:
        return False

    if moat and pos in moat and not unit.is_flying:
        if pos != unit.pos and (not unit.is_wide or pos != unit.tail_cell):
            return False

    return True


def _pos_dist(grid, pos: Tuple[int, int], unit: Unit) -> int:
    """Minimum distance from a bare cell to a unit's body."""
    if not unit.is_wide:
        return grid.distance(pos, unit.pos)
    return min(grid.distance(pos, cell) for cell in unit.occupied_cells())
