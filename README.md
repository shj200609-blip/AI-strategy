# AlphaGo Battle AI

用 **AlphaGo Zero** 风格的深度强化学习，为 *fheroes2* 战斗系统训练神经网络 AI。

- 神经网络 + MCTS + 自对弈，参考 [AlphaGo Zero](https://arxiv.org/abs/1712.01815)
- 自实现的 fheroes2 战斗引擎（六角格寻路、法术、攻城、英雄技能）
- 零人工特征，端到端学习

> 本仓库**不包含** fheroes2 游戏本体，仅包含自实现的战斗引擎与 AI 训练代码。

## 安装

需要 Python ≥ 3.11、PyTorch ≥ 2.0、NumPy ≥ 2.0。

```bash
pip install -e .
pip install pytest tensorboard   # tensorboard 可选
```

## 训练

```bash
# 小规模测试
python scripts/train_alphago.py --sims 100 --games 10 --iterations 5 --train-steps 100

# 完整训练（GPU 推荐）
python scripts/train_alphago.py --sims 800 --games 100 --iterations 100 \
    --device cuda --checkpoint-dir checkpoints/run1 --tensorboard

# 从 checkpoint 恢复
python scripts/train_alphago.py --resume checkpoints/run1/best.pt
```

## 观看 AI 对战

```bash
# 终端 ASCII
python scripts/watch_battle.py --mode ascii --sims 200

# 输出 PNG 截图序列
python scripts/watch_battle.py --mode png --output screenshots/ --model checkpoints/best.pt
```

## 测试

```bash
pytest -q
```

## 目录

```
ai_core/      AI 接口层（AIPlayer、共享战场几何、动作空间、观测编码、BattleNet）
  classic_ai/ fheroes2 BattlePlanner 基线 AI（分析、射手、近战、移动、撤退、法术）
alphago/      AlphaGo Zero 框架（MCTS、自对弈、回放缓冲、训练、对手池、Pipeline）
engine/       fheroes2 战斗引擎（六角格、单位、法术、攻城、英雄）
config/       共享常量（单位表、颜色等）
configs/      训练 / 评估阵型 JSON
scripts/      命令行入口（train_alphago / watch_battle）
tests/        单元测试
```

## 致谢

- AlphaGo Zero 论文：Silver et al., 2017
- [fheroes2](https://github.com/ihhub/fheroes2) — 战斗数据来源（GPL-2.0）