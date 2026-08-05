"""Battle logger — records AI decisions and combat results to file.

Each battle produces one file in log/ named by timestamp, e.g.:
  log/2026-05-30_04-15-23.log
"""

import os
from datetime import datetime
from typing import List, Optional

_LOG_DIR = os.path.join(os.getcwd(), "log")


class BattleLogger:
    """Collects battle events and writes them to a log file."""

    def __init__(self):
        self._lines: List[str] = []
        self._path: Optional[str] = None

    def start(self, units: list):
        """Begin a new battle log — write roster."""
        os.makedirs(_LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = os.path.join(_LOG_DIR, f"{ts}.log")
        self._lines = []
        self._lines.append(f"=== Battle Start ===")
        self._lines.append("")
        for team in (0, 1):
            team_units = [u for u in units if u.team == team]
            names = ", ".join(
                f"{u.name} x{u.count} ({u.col},{u.row})" for u in team_units)
            self._lines.append(f"Team {team}: {names}")
        self._lines.append("")

    def round_start(self, round_num: int):
        self._lines.append(f"--- Round {round_num} ---")

    def action(self, ai_desc: str, result_desc: str):
        """Record one unit's turn: AI decision + execution result."""
        self._lines.append(f"  {ai_desc}")
        if result_desc:
            self._lines.append(f"  -> {result_desc}")

    def end(self, winner: Optional[int], round_num: int, timeout: bool = False):
        """Finish the log — write result and flush to disk."""
        self._lines.append("")
        if winner is not None:
            how = "by strength (round limit)" if timeout else ""
            self._lines.append(
                f"=== Team {winner} wins {how}(Round {round_num}) ===")
        else:
            self._lines.append(f"=== Battle reset (Round {round_num}) ===")
        if self._path:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._lines) + "\n")
            print(f"Battle log saved to {self._path}")

    def reset(self):
        self._lines = []
        self._path = None
