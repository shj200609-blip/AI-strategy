"""训练可视化：TensorBoard 日志 + Matplotlib 战斗渲染 + 复盘 GIF。

三个独立工具：
  - TensorBoardLogger: 记录训练指标
  - render_battle: Matplotlib 六角格战斗渲染
  - print_battle: 终端 ASCII 快速查看
  - generate_replay_gif: 生成战斗复盘 GIF
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.battle_state import BattleState
from engine.unit import Unit


# ═══════════════════════════════════════════════════════════════════════════════
# TensorBoard 日志
# ═══════════════════════════════════════════════════════════════════════════════

class TensorBoardLogger:
    """轻量 TensorBoard 封装，导入失败时优雅降级。"""

    def __init__(self, log_dir: str = "runs/alphago"):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self._active = True
        except ImportError:
            print("[TensorBoard] torch.utils.tensorboard not available, skipping")
            self._active = False

    @property
    def active(self) -> bool:
        return self._active and self.writer is not None

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """记录标量指标 (loss, accuracy 等)。"""
        if not self.active:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def log_histogram(self, tag: str, values, step: int):
        """记录分布直方图。"""
        if not self.active:
            return
        self.writer.add_histogram(tag, values, step)

    def close(self):
        if self.writer:
            self.writer.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Matplotlib 战斗渲染
# ═══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "team0": "#4A90D9", "team1": "#D94A4A",
    "team0_light": "#A8C8E8", "team1_light": "#E8A8A8",
    "grid": "#E8E0D0", "grid_line": "#C0B8A8",
    "wall": "#8B7355", "moat": "#6495ED",
    "highlight": "#FFD700", "text": "#333333",
}

_UNIT_SYMBOL = {
    "archer": "▲", "ranger": "▲", "pikeman": "■",
    "swordsman": "■", "cavalry": "◆", "champion": "◆",
    "paladin": "◆", "crusader": "◆", "griffin": "◇",
    "dragon": "◇", "phoenix": "◇", "gargoyle": "◇",
    "mage": "●", "archmage": "●",
}


def _sym(unit: Unit) -> str:
    name = unit.name.lower()
    for k, v in _UNIT_SYMBOL.items():
        if k in name:
            return v
    if unit.is_archer:
        return "▲"
    if unit.is_flying:
        return "◇"
    return "■"


def render_battle(
    battle: BattleState,
    current_unit: Optional[Unit] = None,
    title: str = "fheroes2 Battle",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """Matplotlib 六角格战斗局面渲染。

    Parameters
    ----------
    battle : BattleState
    current_unit : Unit or None
        高亮显示的单位。
    title : str
    save_path : str or None
        保存 PNG 路径。None 则弹出窗口。
    figsize : (w, h)

    Returns
    -------
    fig, ax
    """
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    radius = 1.0
    dx_val = np.sqrt(3) * radius
    dy_val = 1.5 * radius

    # 格子
    for row in range(9):
        for col in range(11):
            x = col * dx_val + (row % 2) * dx_val / 2
            y = row * dy_val
            ax.add_patch(RegularPolygon(
                (x, y), numVertices=6, radius=radius,
                orientation=np.pi / 6,
                facecolor=_COLORS["grid"],
                edgecolor=_COLORS["grid_line"], linewidth=0.5,
            ))

    # 城堡
    if battle.castle is not None:
        _draw_castle(ax, dx_val, dy_val)

    # 单位
    for unit in battle.units:
        if not unit.is_alive:
            continue
        x = unit.col * dx_val + (unit.row % 2) * dx_val / 2
        y = unit.row * dy_val
        c0 = _COLORS["team0"] if unit.team == 0 else _COLORS["team1"]
        cl = _COLORS["team0_light"] if unit.team == 0 else _COLORS["team1_light"]

        ax.add_patch(plt.Circle((x, y), radius * 0.6, facecolor=cl,
                                edgecolor=c0, linewidth=2, zorder=3))
        ax.text(x, y + 0.15, _sym(unit), ha="center", va="center",
                fontsize=14, fontweight="bold", color=c0, zorder=4)
        hp_total = unit._total_hp if hasattr(unit, '_total_hp') else unit.count * unit.max_hp
        hp_r = hp_total / max(unit.count * unit.max_hp, 1)
        hp_c = "green" if hp_r > 0.6 else ("orange" if hp_r > 0.3 else "red")
        ax.text(x, y - 0.35, f"{unit.count}", ha="center", va="center",
                fontsize=7, color=hp_c, zorder=4)
        ax.text(x, y - 0.55, unit.name[:3], ha="center", va="center",
                fontsize=5, color=_COLORS["text"], alpha=0.7, zorder=4)

        if unit.is_wide:
            tc = unit.col - 1 if unit.team == 0 else unit.col + 1
            if 0 <= tc < 11:
                tx = tc * dx_val + (unit.row % 2) * dx_val / 2
                ax.plot([x, tx], [y, y], color=c0, linewidth=2, alpha=0.4, zorder=2)

    # 当前单位高亮
    if current_unit and current_unit.is_alive:
        cx = current_unit.col * dx_val + (current_unit.row % 2) * dx_val / 2
        cy = current_unit.row * dy_val
        ax.add_patch(plt.Circle((cx, cy), radius * 0.7, fill=False,
                                edgecolor=_COLORS["highlight"], linewidth=3,
                                linestyle="--", zorder=5))

    ax.legend(handles=[
        mpatches.Patch(facecolor=_COLORS["team0_light"], edgecolor=_COLORS["team0"],
                       label="Team 0"),
        mpatches.Patch(facecolor=_COLORS["team1_light"], edgecolor=_COLORS["team1"],
                       label="Team 1"),
        mpatches.Patch(facecolor="none", edgecolor=_COLORS["highlight"],
                       label="Current Unit", linestyle="--"),
    ], loc="upper right", fontsize=8)

    ax.set_xlim(-dx_val, 11 * dx_val)
    ax.set_ylim(-dy_val, 9.5 * dy_val)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def _draw_castle(ax, dx_val, dy_val):
    """绘制城堡结构。"""
    from matplotlib.patches import Rectangle
    for row in range(9):
        rect = Rectangle((-0.5, row * dy_val - 0.5), 1.0, 1.0,
                         facecolor=_COLORS["wall"], alpha=0.5, zorder=1)
        ax.add_patch(rect)


# ═══════════════════════════════════════════════════════════════════════════════
# 终端 ASCII 渲染
# ═══════════════════════════════════════════════════════════════════════════════

def print_battle(battle: BattleState, current_unit: Optional[Unit] = None):
    """终端 ASCII 渲染，不需要 matplotlib，适合快速 debug。"""
    grid = [[" . " for _ in range(11)] for _ in range(9)]
    for unit in battle.units:
        if not unit.is_alive:
            continue
        s = _sym(unit)
        t = "B" if unit.team == 0 else "R"
        grid[unit.row][unit.col] = f"{t}{s}"[:2].ljust(3)
    if current_unit and current_unit.is_alive:
        cu = current_unit
        s = _sym(cu)
        t = "B" if cu.team == 0 else "R"
        grid[cu.row][cu.col] = f"[{t}{s}]"[:3].ljust(3)
    print("   " + "".join(f"{c:^3}" for c in range(11)))
    for row in range(9):
        prefix = "  " if row % 2 == 0 else ""
        print(f"{row} {prefix}" + "".join(grid[row]))
    alive = {0: [], 1: []}
    for u in battle.units:
        if u.is_alive:
            alive[u.team].append(f"{u.name}({u.count})")
    print(f"  Blue: {', '.join(alive[0]) or 'none'}")
    print(f"  Red:  {', '.join(alive[1]) or 'none'}")
    print(f"  Round: {battle.round_num}")
    if current_unit:
        print(f"  Acting: {current_unit.name} (team {current_unit.team})")


# ═══════════════════════════════════════════════════════════════════════════════
# 复盘 GIF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_replay_gif(
    battle_config: dict,
    ai_player,
    output_path: str = "replay.gif",
    max_frames: int = 100,
    fps: int = 2,
):
    """生成战斗复盘 GIF。需要: pip install imageio

    Parameters
    ----------
    battle_config : 战斗配置 dict
    ai_player : AIPlayer
    output_path : GIF 输出路径
    max_frames : 最大帧数
    fps : 每秒帧数
    """
    try:
        import imageio
    except ImportError:
        print("需要 imageio: pip install imageio")
        return

    from alphago.self_play import _build_battle
    from alphago.mcts import advance_to_next_unit
    from ai_core.action_space import index_to_action
    import tempfile

    battle = _build_battle(battle_config)
    ai_player.battle_begins()
    unit = advance_to_next_unit(battle)
    frames, step = [], 0

    with tempfile.TemporaryDirectory() as td:
        while not battle.is_over() and unit and step < max_frames:
            fp = os.path.join(td, f"f_{step:04d}.png")
            render_battle(battle, unit,
                          title=f"Step {step} | Round {battle.round_num}",
                          save_path=fp)
            frames.append(imageio.imread(fp))
            action, _ = ai_player.decide(battle, unit)
            battle.execute(action)
            unit = advance_to_next_unit(battle)
            step += 1
        fp = os.path.join(td, f"f_{step:04d}.png")
        w = battle.winner() if battle.is_over() else "?"
        render_battle(battle, None, title=f"Game Over | Winner: Team {w}",
                      save_path=fp)
        frames.append(imageio.imread(fp))
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"GIF saved: {output_path} ({len(frames)} frames)")
