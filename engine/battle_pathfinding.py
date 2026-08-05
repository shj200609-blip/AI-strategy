"""fheroes2-compatible battle pathfinding.

The C++ pathfinder stores a graph of complete unit positions (head and tail),
not merely head cells.  This module keeps that distinction in the engine so
ClassicAI, the action encoder, and battle execution all use the same rules.

Differences from the C++ ``Battle::BattlePathfinder``:

* Cells are ``(col, row)`` tuples rather than row-major ``int32_t`` indices.
  The fheroes2 encoding ``index = row * widthInCells + col`` is used wherever
  C++ compares cell order (the reflection flag, the std::set ordering of
  available moves, etc.).
* ``is_moat_built`` defaults to ``True`` so callers that don't track the
  Castle object (yet) keep the previous behaviour; pass ``False`` for the
  fheroes2 ``!isBuild(BUILD_MOAT)`` case.
* Flying wide positions enumerate both orientations (``Position::GetPosition``
  primary + fallback) so the pathfinder reaches cells whose natural head-side
  is off the board.
* The ordinary search uses ``heapq`` (Dijkstra) instead of the C++
  label-correcting FIFO vector — equivalent here because every edge weight is
  non-negative and ties are not load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .hex_grid import HexGrid
from .unit import Unit

Cell = Tuple[int, int]
# fheroes2 ``Battle::Board::widthInCells`` — cell index = row*WIDTH + col.
WIDTH_IN_CELLS = 11
MOAT_PENALTY = 0xFFFF


def _cell_index(cell: Cell) -> int:
    """fheroes2 ``Battle::Cell::GetIndex()`` — row-major encoding."""
    return cell[1] * WIDTH_IN_CELLS + cell[0]


@dataclass(frozen=True)
class BattlePosition:
    """A C++ ``Battle::Position`` represented by Python cells."""

    head: Cell
    tail: Optional[Cell] = None

    @property
    def is_wide(self) -> bool:
        return self.tail is not None

    @property
    def is_reflected(self) -> bool:
        # fheroes2 ``Battle::Position::isReflect()`` — compares
        # ``first->GetIndex() < second->GetIndex()`` (battle_cell.cpp:211-214).
        # Same-row cells fall out identical to the simpler tuple compare; the
        # explicit row-major encoding stays correct in the cross-row edge case
        # and matches the C++ cache key semantics (battle_pathfinding.cpp:62).
        return self.tail is not None and _cell_index(self.head) < _cell_index(self.tail)

    def contains(self, cell: Cell) -> bool:
        return self.head == cell or self.tail == cell

    def swapped(self) -> "BattlePosition":
        if self.tail is None:
            return self
        return BattlePosition(self.tail, self.head)


@dataclass
class PathNode:
    previous: Optional[BattlePosition]
    cost: int
    distance: int


class BattlePathfinder:
    """Build the in-principle graph, then apply the current-turn cost cap."""

    def __init__(self, grid: HexGrid, unit: Unit,
                 occupied: Set[Cell],
                 is_moat: Callable[[Unit, Cell], bool],
                 is_moat_built: bool = True):
        self.grid = grid
        self.unit = unit
        self.occupied = occupied
        self.is_moat = is_moat
        # fheroes2 ``isMoatBuilt = castle && castle->isBuild(BUILD_MOAT)``
        # (battle_pathfinding.cpp:73-74). Without this gate the ``is_moat``
        # callback alone would still report "moat" cells outside a siege (or
        # before the moat was rebuilt) and the penalty would be applied.
        self.is_moat_built = is_moat_built
        self.start = self.position_from_unit(unit)
        self.nodes: Dict[BattlePosition, PathNode] = {}
        self._build()

    @staticmethod
    def position_from_unit(unit: Unit) -> BattlePosition:
        return BattlePosition(unit.pos, unit.tail_cell)

    def _cell_free(self, cell: Cell) -> bool:
        # fheroes2 ``Cell::isPassable(true)`` for the single-hex case.
        # The unit's own current body is treated as free — the unit vacates
        # those cells as it moves. This matches the C++ carve-out in
        # ``Cell::isPassableFromAdjacent`` (battle_cell.cpp:334-346), which
        # permits the new head to land on the unit's current tail during a
        # horizontal U-turn without separately validating the new tail that
        # falls on the unit's current head.
        if not self.grid.is_valid(*cell):
            return False
        if cell not in self.occupied:
            return True
        return self._unit_owns(cell)

    def _unit_owns(self, cell: Cell) -> bool:
        """True if *cell* is part of the unit's current body."""
        return cell == self.unit.pos or (
            self.unit.is_wide and cell == self.unit.tail_cell)

    def _valid_position(self, pos: BattlePosition) -> bool:
        if self.unit.is_wide != pos.is_wide:
            return False
        if not self.grid.is_valid(*pos.head):
            return False
        if pos.tail is None:
            return self._cell_free(pos.head)
        return (self._cell_free(pos.head)
                and self._cell_free(pos.tail)
                and pos.head != pos.tail)

    def _initial_moat_cell(self) -> Optional[Cell]:
        # fheroes2 ``pathStartMoatCellIdx`` (battle_pathfinding.cpp:107-127).
        # A non-siege battle (or one without moat) returns -1 there.
        if self.unit.is_flying or not self.is_moat_built:
            return None
        for cell in (self.start.head, self.start.tail):
            if cell is not None and self.is_moat(self.unit, cell):
                return cell
        return None

    def _position_in_moat(self, pos: BattlePosition) -> bool:
        # fheroes2 gates this on ``isMoatBuilt`` (battle_pathfinding.cpp:74, 150, 190).
        if not self.is_moat_built or self.unit.is_flying:
            return False
        return (self.is_moat(self.unit, pos.head)
                or (pos.tail is not None
                    and self.is_moat(self.unit, pos.tail)))

    def _moat_ignored(self, pos: BattlePosition,
                      initial_moat: Optional[Cell]) -> bool:
        return initial_moat is not None and pos.contains(initial_moat)

    # ── direction helpers ──────────────────────────────────────
    #
    # fheroes2 ``Battle::Board::GetMoveWideIndexes`` enumerates four candidate
    # next heads (battle_board.cpp:732-766) and the tail for each is derived
    # via ``Cell::isPassableFromAdjacent`` / ``isLeftSide(GetDirection(...))``
    # (battle_cell.cpp:324-347, battle_pathfinding.cpp:161). Translating the
    # bit-flag direction into even-r offset coordinates requires per-row
    # parity because "visual left" maps to different ``(dc, dr)`` patterns in
    # odd and even rows.

    @staticmethod
    def _is_left_side(dc: int, dr: int, new_head_row: int) -> bool:
        """C++ ``isLeftSide(GetDirection(currentHead, newHead))`` in even-r.

        ``True`` when the unit's head is moving towards the visual left side,
        so its newly-derived tail must sit on the RIGHT (``col + 1``).
        """
        if dr == 0:
            # Purely horizontal: dc = -1 is LEFT, dc = +1 is RIGHT.
            return dc < 0
        if dc < 0:
            return True   # dc = -1 with any dr is a left-diagonal
        if dc > 0:
            return False  # dc = +1 with any dr is a right-diagonal
        # dc == 0 with dr != 0: even-r maps (dc=0, dr=±1) to TOP/BOTTOM_LEFT
        # in even rows and TOP/BOTTOM_RIGHT in odd rows (battle_board.cpp
        # ``GetIndexDirection`` offsets).
        return (new_head_row & 1) == 0

    @classmethod
    def _wide_tail_offset(cls, dc: int, dr: int, new_head_row: int) -> int:
        """Column offset of the new tail from the new head."""
        return 1 if cls._is_left_side(dc, dr, new_head_row) else -1

    @staticmethod
    def _wide_neighbor_offsets(pos: BattlePosition
                               ) -> List[Tuple[int, int]]:
        """The 4 next-head offsets (``Battle::Board::GetMoveWideIndexes``)."""
        # LEFT and RIGHT are always allowed.
        offsets: List[Tuple[int, int]] = [(-1, 0), (1, 0)]
        if pos.is_reflected:
            # Left-diagonals — same column on even rows, one less on odd rows.
            if pos.head[1] & 1:
                offsets.extend([(-1, -1), (-1, 1)])
            else:
                offsets.extend([(0, -1), (0, 1)])
        else:
            # Right-diagonals — one more on even rows, same column on odd rows.
            if pos.head[1] & 1:
                offsets.extend([(0, -1), (0, 1)])
            else:
                offsets.extend([(1, -1), (1, 1)])
        return offsets

    def _wide_neighbors(self, pos: BattlePosition
                        ) -> Iterable[BattlePosition]:
        assert pos.tail is not None
        for dc, dr in self._wide_neighbor_offsets(pos):
            head = (pos.head[0] + dc, pos.head[1] + dr)
            if not self.grid.is_valid(*head):
                continue
            tail_dc = self._wide_tail_offset(dc, dr, head[1])
            tail = (head[0] + tail_dc, head[1])
            if not self.grid.is_valid(*tail):
                continue
            # fheroes2 ``Cell::isPassableFromAdjacent`` (battle_cell.cpp:324-347):
            #   - LEFT/RIGHT direction: passable OR the new head is the unit's
            #     own current tail cell (U-turn through own body).
            #   - diagonal: both this cell and its newly-derived tail must
            #     be passable.
            if dr == 0:
                head_ok = self._cell_free(head) or self._unit_owns(head)
            else:
                head_ok = self._cell_free(head) and self._cell_free(tail)
            if not head_ok:
                continue
            candidate = BattlePosition(head, tail)
            if self._valid_position(candidate):
                yield candidate

    def _neighbors(self, pos: BattlePosition
                   ) -> Iterable[BattlePosition]:
        if pos.tail is None:
            for cell in self.grid.neighbors(*pos.head):
                candidate = BattlePosition(cell)
                if self._valid_position(candidate):
                    yield candidate
        else:
            yield from self._wide_neighbors(pos)

    # ── flying distance (Battle::Board::GetDistance, Position × Position) ──
    #
    # fheroes2 returns the minimum over all (start head/tail × target head/tail)
    # pairings (battle_board.cpp:146-172). A wide unit which is itself the
    # "target" gets a minimum distance of 1 even when its head and tail
    # already coincide with the start body's cells (battle_pathfinding.cpp:96).

    def _flying_distance(self, target: BattlePosition) -> int:
        starts = [self.start.head]
        if self.start.tail is not None:
            starts.append(self.start.tail)
        ends = [target.head]
        if target.tail is not None:
            ends.append(target.tail)
        best: Optional[int] = None
        for a in starts:
            for b in ends:
                d = self.grid.distance(a, b)
                if best is None or d < best:
                    best = d
        # fheroes2 floors the distance at 1 — a wide unit that overlaps its
        # current position is still considered to have "moved" (battle_pathfinding.cpp:96).
        return max(best if best is not None else 1, 1)

    # ── flying wide positions (Battle::Position::GetPosition) ──
    #
    # fheroes2 prefers the unit's current reflection (the "natural" landing)
    # and falls back to the mirrored orientation when the natural tail cell
    # is off-grid or otherwise blocked (battle_cell.cpp:78-129). We emit
    # both candidates per destination cell — when neither is valid the
    # position is dropped by ``_valid_position`` below.

    def _flying_candidates_for(self, head: Cell
                               ) -> Iterable[BattlePosition]:
        if not self.unit.is_wide:
            yield BattlePosition(head)
            return
        # Mirrors ``tailDirection = isReflect() ? RIGHT : LEFT`` (battle_cell.cpp:100).
        primary_tail = (head[0] + (1 if self.unit.is_reflected else -1),
                        head[1])
        secondary_head = (head[0] - (1 if self.unit.is_reflected else -1),
                          head[1])
        yield BattlePosition(head, primary_tail)
        # Fallback: this cell becomes the TAIL, the head sits one step in
        # the opposite direction (battle_cell.cpp:109-118).
        yield BattlePosition(secondary_head, head)

    def _build(self) -> None:
        self.nodes = {self.start: PathNode(None, 0, 0)}

        # C++ treats flying movement as direct landing rather than a normal
        # BFS.  Enumerate the same complete positions and retain their direct
        # distance estimates.
        if self.unit.is_flying:
            for row in range(self.grid.rows):
                for col in range(self.grid.cols):
                    head = (col, row)
                    for pos in self._flying_candidates_for(head):
                        if pos == self.start or not self._valid_position(pos):
                            continue
                        self.nodes[pos] = PathNode(
                            self.start, 1, self._flying_distance(pos))
            return

        initial_moat = self._initial_moat_cell()
        queue: List[Tuple[int, int, int, BattlePosition]] = []
        sequence = 0
        heapq.heappush(queue, (0, 0, sequence, self.start))
        while queue:
            cost, _, _, current = heapq.heappop(queue)
            node = self.nodes[current]
            if cost != node.cost:
                continue
            if self._moat_ignored(current, initial_moat) or not self._position_in_moat(current):
                penalty = 1
            else:
                penalty = MOAT_PENALTY
            for candidate in self._neighbors(current):
                if candidate == self.start:
                    continue
                is_reversal = (current.tail is not None
                                and candidate == current.swapped())
                edge_cost = 0 if is_reversal else penalty
                edge_distance = 0 if is_reversal else 1
                new_cost = node.cost + edge_cost
                new_distance = node.distance + edge_distance
                old = self.nodes.get(candidate)
                if old is not None and old.cost <= new_cost:
                    continue
                self.nodes[candidate] = PathNode(
                    current, new_cost, new_distance)
                sequence += 1
                heapq.heappush(queue, (new_cost, new_distance,
                                       sequence, candidate))

    def node_for(self, pos: BattlePosition) -> Optional[PathNode]:
        return self.nodes.get(pos)

    def is_position_reachable(self, pos: BattlePosition,
                              on_current_turn: bool) -> bool:
        node = self.node_for(pos)
        if node is None:
            return False
        if pos == self.start:
            return True
        return node.cost <= self.unit.speed if on_current_turn else True

    def get_cost(self, pos: BattlePosition) -> int:
        node = self.nodes[pos]
        return node.cost

    def get_distance(self, pos: BattlePosition) -> int:
        node = self.nodes[pos]
        return node.distance

    def all_available_moves(self) -> Dict[Cell, int]:
        result: Dict[Cell, int] = {}
        for pos, node in self.nodes.items():
            if pos == self.start or node.cost > self.unit.speed:
                continue
            result[pos.head] = min(result.get(pos.head, node.distance),
                                   node.distance)
        return result

    def _trace(self, destination: BattlePosition
               ) -> Tuple[List[BattlePosition], Optional[BattlePosition]]:
        if destination not in self.nodes:
            return [], None
        result: List[BattlePosition] = []
        last_reachable: Optional[BattlePosition] = None
        current = destination
        while current != self.start:
            node = self.nodes.get(current)
            if node is None or node.previous is None:
                return [], None
            previous = node.previous
            if node.cost <= self.unit.speed:
                if last_reachable is None:
                    last_reachable = current
                result.append(current)
            current = previous
        result.reverse()
        return result, last_reachable

    def build_path(self, destination: BattlePosition
                   ) -> Optional[Tuple[List[Cell], BattlePosition]]:
        nodes, last_reachable = self._trace(destination)
        if not nodes or last_reachable is None:
            return None
        path = [self.start.head] + [pos.head for pos in nodes]
        # fheroes2 ``BattlePathfinder::buildPath`` (battle_pathfinding.cpp:344-355):
        # when the path's last reachable position's reflection differs from
        # the destination's, the wide unit must take an extra U-turn step —
        # append the last reachable's tail cell so the engine's animation
        # walks through both orientations.
        if (destination.is_wide
                and last_reachable.is_reflected != destination.is_reflected):
            path.append(last_reachable.tail)  # type: ignore[arg-type]
        final = BattlePosition(path[-1], None)
        if destination.is_wide:
            if path[-1] == last_reachable.tail:
                final = last_reachable.swapped()
            else:
                final = nodes[-1]
        return path, final

    def closest_reachable_position(self, destination: BattlePosition
                                   ) -> Optional[BattlePosition]:
        nodes, last_reachable = self._trace(destination)
        if not nodes or last_reachable is None:
            return None
        result = last_reachable
        if (destination.is_wide
                and result.is_reflected != destination.is_reflected):
            result = result.swapped()
        return result
