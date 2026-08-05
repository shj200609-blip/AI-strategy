"""Hex grid engine: pure geometry and pathfinding (no rendering).

Pointy-top hex grid with even-r offset coordinates.
Matches the original HoMM2 battle board (11x9) and the C++
``Battle::Board`` layout in fheroes2 (odd rows are shifted LEFT by
half a hex width — ``Battle::Cell::SetArea`` subtracts ``widthPx/2``
for ``(index / widthInCells) % 2 == 1``).

This module is deliberately free of pygame / pixel concerns so that
``engine`` and ``ai`` can be imported and unit-tested headlessly.
Pixel coordinates and drawing live in ``ui/hex_renderer.py``.
"""

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import config

# Neighbor offsets: (dcol, drow) for even / odd rows.
# even-r: odd rows are shifted LEFT, so the upper/lower diagonal neighbors
# in an odd row are at (col-1, row±1), while in an even row they are at
# (col, row±1). Matches ``Battle::Board::GetIndexDirection`` exactly.
#
# Order: E, W, NW, NE, SW, SE (keeps the list flat and easy to read).
EVEN_NEIGHBORS = [(1, 0), (-1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]
ODD_NEIGHBORS = [(1, 0), (-1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]


class HexGrid:
    def __init__(self, cols: Optional[int] = None, rows: Optional[int] = None):
        self.cols = config.GRID_COLS if cols is None else cols
        self.rows = config.GRID_ROWS if rows is None else rows

    # ── coordinate math ────────────────────────────────────

    def _to_cube(self, col: int, row: int) -> Tuple[int, int, int]:
        # Matches ``Battle::Board::GetDistanceIndexes`` axial conversion:
        #   q = col - (row + (row % 2)) // 2
        # The +1 on odd rows makes ``row=1`` map to q=col-1 (because odd
        # rows are visually shifted left by half a hex).
        q = col - (row + (row & 1)) // 2
        r = row
        return (q, r, -q - r)

    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        ca, cb = self._to_cube(*a), self._to_cube(*b)
        return max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1]), abs(ca[2] - cb[2]))

    def neighbors(self, col: int, row: int) -> List[Tuple[int, int]]:
        offsets = ODD_NEIGHBORS if row & 1 else EVEN_NEIGHBORS
        return [(col + dc, row + dr) for dc, dr in offsets
                if 0 <= col + dc < self.cols and 0 <= row + dr < self.rows]

    def is_valid(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def half_of(self, col: int) -> int:
        return 0 if col < self.cols // 2 else 1

    def cell_behind(self, from_pos: Tuple[int, int],
                    target_pos: Tuple[int, int],
                    target_occupied: Optional[Set[Tuple[int, int]]] = None
                    ) -> Optional[Tuple[int, int]]:
        """Return the cell one step behind *target* from *from*'s perspective.

        Used by TWO_CELL_MELEE_ATTACK: the attacker hits both the target and
        whatever unit sits one hex further in the same direction.  Computed via
        cube-coordinate direction normalisation so it is correct for all hex
        offsets, not just the east-west axis.

        ``target_occupied`` — when supplied, the cell is rejected if it is one
        of the target's own cells (head or tail). This prevents a wide
        target's tail from being mistakenly chosen as a splash target.
        Returns ``None`` for out-of-board cells.
        """
        fc = self._to_cube(*from_pos)
        tc = self._to_cube(*target_pos)
        # Cube-space direction, normalised to a unit step.
        def _sign(x):
            return (1 if x > 0 else -1) if x != 0 else 0
        nd = tuple(map(_sign, (tc[0] - fc[0], tc[1] - fc[1], tc[2] - fc[2])))
        bq, br = tc[0] + nd[0], tc[1] + nd[1]
        # Inverse of ``_to_cube`` in even-r (matches
        # ``Battle::Board::GetDistanceIndexes`` line ``x = q + (r + r%2)/2``).
        col = bq + (br + (br & 1)) // 2
        row = br
        if not self.is_valid(col, row):
            return None
        if target_occupied is not None and (col, row) in target_occupied:
            return None
        return (col, row)

    # ── fheroes2 board geometry ─────────────────────────────
    #
    # Static helpers that mirror ``Battle::Board::isCastleIndex`` and
    # ``Battle::Board::isOutOfWallsIndex``. Wall cells count as BOTH
    # "in the castle" (defender side) and "out of walls" (they are the
    # wall itself) — the C++ does the same because the wall is the
    # boundary between the two halves.  These let the AI reason about
    # "are we inside the castle yet" without depending on the Castle
    # object (which is only constructed in siege mode).

    _CASTLE_MIN_COL = [8, 9, 7, 8, 7, 8, 7, 9, 8]
    _OUT_OF_WALLS_MAX_COL = [8, 8, 7, 7, 6, 7, 7, 8, 8]

    @staticmethod
    def isCastleIndex(col: int, row: int) -> bool:
        """True if *col,row* lies inside the castle (or on a wall).

        Mirrors ``Battle::Board::isCastleIndex``: wall cells count as
        inside so the AI can tell that the attacker has reached the
        defender's side even when the path goes through a wall segment.
        """
        if not (0 <= row < len(HexGrid._CASTLE_MIN_COL)):
            return False
        return col >= HexGrid._CASTLE_MIN_COL[row]

    @staticmethod
    def isOutOfWallsIndex(col: int, row: int) -> bool:
        """True if *col,row* lies on the attacker side of the walls (or on a wall).

        Mirrors ``Battle::Board::isOutOfWallsIndex``.
        """
        if not (0 <= row < len(HexGrid._OUT_OF_WALLS_MAX_COL)):
            return False
        return col <= HexGrid._OUT_OF_WALLS_MAX_COL[row]

    # ── pathfinding ─────────────────────────────────────────
    #
    # All three functions track the *head* cell. ``tail_dir`` is the column
    # offset of a wide unit's tail (-1 for team 0, +1 for team 1); None means a
    # single-hex unit, in which case the single-hex code path is byte-for-byte
    # unchanged. For wide units a head cell is only usable when its tail cell is
    # in-grid and (for non-flyers) unoccupied — see ``_tail_ok``.

    def _tail_ok(self, head: Tuple[int, int], tail_dir: int,
                 occupied: Set[Tuple[int, int]], flying: bool,
                 goal: Optional[Tuple[int, int]] = None) -> bool:
        # C++ ``Battle::Cell::isPassableFromAdjacent`` always derives the
        # tail as a horizontal neighbour of the head (``LEFT`` or ``RIGHT``
        # in ``CellDirection``). The reflection state is determined by
        # the move direction, not by the unit's stored orientation; here
        # the caller passes the correct ``tail_dir`` for the new head, so
        # the simple (head[0] + tail_dir, head[1]) formula is correct
        # for all cases including U-turns (where ``isOutOfWallsIndex``
        # / C++'s special case lets the unit land on its own tail cell).
        tail = (head[0] + tail_dir, head[1])
        if not self.is_valid(*tail):
            return False
        if not flying and tail in occupied and tail != goal:
            return False
        return True

    def reachable(self, start: Tuple[int, int], speed: int,
                  occupied: Set[Tuple[int, int]], flying: bool = False,
                  tail_dir: Optional[int] = None,
                  moat_cells: Optional[FrozenSet[Tuple[int, int]]] = None
                  ) -> Dict[Tuple[int, int], int]:
        """BFS reachable cells within *speed* steps.

        ``moat_cells`` (siege): non-flying units can enter a moat cell but
        their movement terminates there — the cell is added to results but
        the BFS does not expand from it.  ``None`` (default) = no moat.
        """
        result: Dict[Tuple[int, int], int] = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            pos, d = queue.popleft()
            if d >= speed:
                continue
            for nb in self.neighbors(*pos):
                if nb in result:
                    continue
                if nb in occupied and not flying:
                    continue
                if tail_dir is not None and not self._tail_ok(nb, tail_dir, occupied, flying):
                    continue
                result[nb] = d + 1
                # Moat: non-flying units stop here (can enter but not continue).
                if moat_cells and not flying and nb in moat_cells:
                    continue  # don't expand from moat
                queue.append((nb, d + 1))
        return result

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  occupied: Set[Tuple[int, int]], flying: bool = False,
                  max_len: int = 99, tail_dir: Optional[int] = None,
                  moat_cells: Optional[FrozenSet[Tuple[int, int]]] = None
                  ) -> Optional[List[Tuple[int, int]]]:
        """BFS shortest path from *start* to *goal*.

        ``moat_cells`` (siege): non-flying units may traverse at most one moat
        cell (the one they stop on).  The BFS does not expand further from a
        moat cell unless it is the goal itself.
        """
        if start == goal:
            return [start]
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
        queue = deque([(start, 0)])
        while queue:
            pos, d = queue.popleft()
            if d >= max_len:
                continue
            for nb in self.neighbors(*pos):
                if nb in prev:
                    continue
                if nb in occupied and not flying and nb != goal:
                    continue
                if tail_dir is not None and not self._tail_ok(nb, tail_dir, occupied, flying, goal):
                    continue
                prev[nb] = pos
                if nb == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                # Moat: non-flying units stop here — don't expand further.
                if moat_cells and not flying and nb in moat_cells:
                    continue
                queue.append((nb, d + 1))
        return None

    def nearest_cell_next_to(self, start: Tuple[int, int], target: Tuple[int, int],
                             occupied: Set[Tuple[int, int]], flying: bool = False,
                             max_dist: int = 99, tail_dir: Optional[int] = None,
                             moat_cells: Optional[FrozenSet[Tuple[int, int]]] = None
                             ) -> Optional[Tuple[int, int]]:
        best_pos, best_d = None, float('inf')
        for nb in self.neighbors(*target):
            if nb in occupied:
                continue
            if tail_dir is not None and not self._tail_ok(nb, tail_dir, occupied, flying):
                continue
            path = self.find_path(start, nb, occupied, flying, max_dist,
                                  tail_dir, moat_cells)
            if path and len(path) - 1 < best_d:
                best_d = len(path) - 1
                best_pos = nb
        return best_pos
