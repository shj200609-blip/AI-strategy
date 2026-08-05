# AlphaGo Battle AI — 完整项目框架文档

> **目的**:本文件是 *alphago-battle-ai* 项目的完整工程框架说明,
> 便于后续维护、二次开发、写作更新日志。
> 项目基线于 **2026-07-31**。

---

## 0. 项目定位

- 用 **AlphaGo Zero** 风格的深度强化学习,为 *fheroes2* 战斗系统训练神经网络 AI。
- 自实现 fheroes2 战斗引擎(六角格、单位、法术、攻城、英雄技能)。
- **零人工特征、端到端学习**:网络直接从局面特征图 + 全局向量学习策略与价值。
- 仓库不包含 fheroes2 游戏本体,仅含战斗引擎与 AI 训练代码。

---

## 1. 目录结构

```
alphago-battle-ai/
├── ai_core/             # 神经网络接口层(AIPlayer 契约、动作空间、观测编码、BattleNet)
├── alphago/             # AlphaGo Zero 框架(MCTS、自对弈、回放缓冲、训练、对手池、Pipeline)
├── engine/              # fheroes2 战斗引擎(六角格、单位、法术、攻城、英雄、回合状态)
├── config/              # 共享常量(单位表、颜色、布局预设、动画时序)
├── configs/             # 训练 / 评估阵型 JSON(被 alphago 读取)
├── scripts/             # 命令行入口(train_alphago.py / watch_battle.py)
├── tests/               # 单元测试(test_classic_ai / test_engine_rules)
├── fheroes2/            # ⚠️ 原版游戏参考(不参与编译,作为常量/规则的"事实来源")
└── pyproject.toml       # 包元数据 + 命令行入口注册
```

---

## 2. 顶层接口契约

整个系统的扩展点只有 **三个抽象接口**:

| 接口 | 文件 | 作用 | 实现 |
|---|---|---|---|
| `AIPlayer` | `ai_core/base.py` | 战斗中单位决策的统一契约 | `ClassicAI`(rule-base)、`MCTSAIPlayer`(AlphaGo) |
| `Action` | `engine/actions.py` | 引擎可执行的最小动作类型 | `Move/Attack/Skip/Cast/Retreat` |
| `BattleState` | `engine/battle_state.py` | 引擎的唯一状态机,所有 AI/可视化都从它读 | — |

**铁律**:任何上层模块(alphago/*、scripts/*、tests/*)只能依赖这三个抽象。
所有可视化、日志、对战都通过 `engine.battle_state` + `engine.unit` + `engine.actions` 读取数据。

---

## 3. 模块详解

### 3.1 `engine/` — 自实现 fheroes2 战斗引擎

> 所有 AI、可视化、对战脚本都依赖此模块。
> 它是 fheroes2 行为的事实来源。

**fheroes2 对齐面(Battle::Arena façade)**:为了让把 C++ AI 移植到 Python 时的查询名保持一致,
`BattleState` 同时暴露了与 fheroes2 ``Battle::Arena`` 同名/同义的查询:

| fheroes2 方法 | Python 等价物 | 备注 |
|---|---|---|
| `Battle::Arena::GetTurnNumber` | `BattleState.turn_number` | `round_num` 的别名,带 setter |
| `Battle::Arena::getAttackingArmyColor` / `getDefendingArmyColor` | `BattleState.attacker_team` / `defender_team()` | 已存在 |
| `Battle::Arena::GetOppositeColor(c)` | `BattleState.get_enemy_color(team)` | 新增 |
| `Battle::Arena::isPositionReachable` | `BattleState.is_position_reachable(unit, pos, on_current_turn)` | 新增 |
| `Battle::Arena::CalculateMoveDistance` | `BattleState.calculate_move_distance(unit, pos)` | 新增 |
| `Battle::Arena::CalculateMoveCost` | `BattleState.calculate_move_cost(unit, pos)` | 新增 |
| `Battle::Arena::getAllAvailableMoves` | `BattleState.get_all_available_moves(unit)` | 新增 |
| `Battle::Arena::getClosestReachablePosition` | `BattleState.get_closest_reachable_position(unit, pos)` | 新增 |
| `Battle::Arena::CanRetreatOpponent(c)` | `BattleState.can_retreat_opponent(team)` | 新增 |
| `Battle::Arena::CanSurrenderOpponent(c)` | `BattleState.can_surrender_opponent(team)` | 新增 |
| `Battle::Arena::GetCastle()` | `BattleState.castle` (属性) | 已存在 |
| `Battle::Arena::GetFreePositionNearHero(c)` | `BattleState.get_free_position_near_hero(team)` | 新增 |
| `Battle::Unit::GetUID` | `Unit.get_uid()` | 新增 (`id(self)`) |
| `Battle::Arena::cellsUnderWallsIndexes` | `BattleState.cells_under_walls()` | 新增 — 用于 siege AI |

**siege 标记(BattlePlanner 字段)**:`BattleState.is_attacking_castle()` /
`is_defending_castle()` / `is_siege()` 对应 BattlePlanner 的
`_attackingCastle` / `_defendingCastle`。

**死亡累计(Plan-unit turn 守卫)**:`attacker_dead_total` /
`defender_dead_total` 在每次 `execute()` 中累计,跨 `start_round()` 保持。
这对应 fheroes2 BattlePlanner 的 ``_attackerForceTotalNumberOfDeadUnits`` /
``_defenderForceTotalNumberOfDeadUnits``,用于 `isLimitOfTurnsExceeded`
判断。

**克隆(MCTS 沙箱)**:`BattleState.clone()` 返回浅拷贝,unit / hero / tower
各自 copy.copy,`castle` 整体替换为浅拷贝(tower / walls 在每轮 catapult
后才会变,MCTS 子树内不需深拷贝)。`alphago/mcts.fast_clone_battle()` 作为
兼容垫片继续存在。

**新增 siege 常量** `engine/castle.CELLS_UNDER_WALLS`:五格列表
`(7,0), (6,2), (5,4), (6,6), (7,8)` — 即 fheroes2 的
``cellsUnderWallsIndexes = {7, 28, 49, 72, 95}``,四个墙根 + 一个门前
护城河格。

| 文件 | 责任 | 关键 API |
|---|---|---|
| `hex_grid.py` | 六角格几何 + BFS 寻路 + wide-unit tail 校验 | `HexGrid.{neighbors, distance, reachable, find_path, cell_behind}` |
| `unit.py` | 单个单位(可能含 wide tail)的状态与计算 | `Unit.{pos, occupied_cells, effective_attack/defense, take_damage, heal, is_alive, effects, get_uid}` |
| `hero.py` | 英雄(法力、技能、spellbook) | `Hero.{spellbook, can_cast, cast, get_skill_value}` |
| `spells.py` | 38 个法术 + Effect 数据 + `spell_damage` / `make_effect` | `SPELLS`, `DAMAGE/AOE/BUFF/DEBUFF/CONTROL/DISPEL/CURE/UTILITY/RESURRECT/SUMMON/HYPNOTIZE/BERSERKER` |
| `castle.py` | 城墙、护城河、箭塔、Gate、Catapult | `Castle.{walls, towers, has_moat, catapult_round, lower_bridge, destroy_bridge}` |
| `actions.py` | 5 种动作类型 + 工厂 | `MoveAction / AttackAction / SkipAction / CastAction / RetreatAction` |
| `battle_state.py` | **核心状态机**:回合顺序、伤害、施法、胜负 + fheroes2 Battle::Arena 对齐面 | `BattleState.{execute, is_over, winner, start_round, turn_order, unit_at, alive, enemies_of, expected_damage, roll_damage, is_siege, is_attacking_castle, is_defending_castle, cells_under_walls, can_retreat_opponent, can_surrender_opponent, get_enemy_color, is_position_reachable, calculate_move_distance, get_all_available_moves, get_closest_reachable_position, clone, turn_number, attacker_dead_total, defender_dead_total}` |
| `battle_logger.py` | 可选:日志落盘(log/*.log) | `BattleLogger.{start, action, end}` |

**胜负条件**(影响训练终止与价值标记):
- 一方全部死亡 / 一方 retreat / 连续 50 回合无死亡(stalemate → 攻击方判负)/ 达到 `MAX_ROUNDS = 200`。
- `winner()` 返回 0 或 1;若 MAX_ROUNDS 到期,按双方剩余 `strength` 取较大者。

**坐标系**:(col, row),col 在 0..10,row 在 0..8。寻路在 cube 坐标中算距离,但 BFS 仍在原始 (col, row) 网格上做。

---

### 3.2 `ai_core/` — 神经网络接口层

> 这一层把"游戏局面"翻译成 PyTorch 张量,并把整数动作翻译回 `Action`。

| 文件 | 责任 | 关键 API |
|---|---|---|
| `base.py` | `AIPlayer` 抽象与每场战斗生命周期 | `battle_begins`, `check_retreat`, `maybe_cast_spell`, `decide` |
| `battle_geometry.py` | 规则 AI 与动作编码共享的 wide-unit 几何 | `_tail_dir`, `_attack_cells`, `_can_attack_from_pos` |
| `action_space.py` | 离散动作空间 13,566 维 + 合法掩码 | `action_to_index`, `index_to_action`, `legal_mask`, `enumerate_legal`, `ACTION_DIM` |
| `observation.py` | 玩家相对编码:35 通道网格 + 20 维全局向量 | `encode_observation(battle, unit) → (grid, gvec)` |
| `model.py` | CNN+ResNet+Embedding 双头网络(`BattleNet`) | `BattleNet.forward(grid, gvec, mask) → (logits, value)` |
| `classic_ai/` | 按分析/射手/近战/移动/撤退/法术拆分的 fheroes2 `BattlePlanner` 基线 AI | `ClassicAI` |

**`AIPlayer` 接口**(每场先初始化,随后每单位回合依次调用):

```python
battle_begins() -> None
check_retreat(battle, unit) -> (int, Optional[(farewell_cast, RetreatAction)])
maybe_cast_spell(battle, unit) -> Optional[(CastAction, str)]
decide(battle, unit) -> (Action, str)
```

`battle_begins()` 默认是 no-op；有每场状态的 AI（例如 `ClassicAI`）覆盖它。所有新建 `BattleState` 并驱动 AI 的 runner 必须在第一次决策前调用一次。

**`BattleNet` 架构**(约 4.15M 参数):

```
grid (35,9,11)                         global (20,)
     |                                       |
Conv2d(33→128,3×3) + GN + ReLU              |
     |                                       |
ResBlock × 6  (128→128)                     |
     |                                       |
+ Embedding(67,16) × 2 (from type idx)      |
     |                                       |
concat → flatten → +global → Linear(15860,384)
     |
   ┌─┴──┐
policy(13566)                         value(1) + tanh
```

**动作空间布局**(共 13,566):

| 区间 | 含义 | 大小 |
|---|---|---|
| 0 | Wait | 1 |
| 1 | Defend | 1 |
| 2..100 | Move(col,row) | 99 |
| 101..9901 | Attack(pos, target) = pos × 99 + target | 99² |
| 9902..13564 | Cast(spell, hex) = spell × 99 + hex | 37 × 99 |
| 13565 | Retreat | 1 |

**观测网格通道**(35, 9, 11):

| 通道 | 含义 |
|---|---|
| 0..9 | 我方单位(exist / hp / count / atk / def / spd / archer / flyer / wide_tail / acted) |
| 10..19 | 敌方单位(同上) |
| 20..29 | 状态效果(按属性而非名称:Haste / Slow / Bless / Curse / Blind / AtkBuff / DefBuff / Shield / AntiMagic / Disrupting) |
| 30 | 城墙 HP (0/0.5/1) |
| 31 | 护城河 |
| 32 | 活跃箭塔 |
| 33 | 我方单位类型索引(归一化) |
| 34 | 敌方单位类型索引(归一化) |

**全局向量**(20 维):round、attacker、双方存活数、双方 HP 比、双方法力/攻击/防御、是否 siege、活跃塔数、完整墙数、士气、幸运、当前回合顺序索引。

---

### 3.3 `alphago/` — AlphaGo Zero 框架

| 文件 | 责任 | 关键 API |
|---|---|---|
| `config.py` | 全部超参数 dataclass | `AlphaGoConfig(...)` |
| `mcts.py` | PUCT MCTS,根节点加 Dirichlet 噪声 | `MCTS.search(battle, unit, network) → policy` |
| `self_play.py` | 自对弈一局/多局 | `SelfPlayRunner.play_game / run_batch` |
| `replay_buffer.py` | FIFO 经验回放 | `ReplayBuffer.{add, sample, is_ready}` |
| `trainer.py` | 监督学习:π cross-entropy + v MSE | `AlphaGoTrainer.train_step / train / save_checkpoint / load_checkpoint` |
| `opponent_pool.py` | 历史模型池(防止策略坍塌) | `OpponentPool.{add, sample, load_from_disk}` |
| `player.py` | `MCTSAIPlayer(AIPlayer)` —— 让 AlphaGo 也能接入 engine | `MCTSAIPlayer.decide` |
| `pipeline.py` | **总编排**:self-play → train → pit → promote | `AlphaGoPipeline.run` |
| `visualization.py` | TensorBoard + Matplotlib 六角格 + ASCII + GIF | `TensorBoardLogger, render_battle, print_battle, generate_replay_gif` |

**`AlphaGoPipeline.run()` 的循环**(每轮):

1. **Self-Play**:`SelfPlayRunner.run_batch(...)` 用 MCTS 跑 `games_per_iteration` 局,产出 (s,π,z) 三元组并写入 replay buffer。
2. **Training**:`AlphaGoTrainer.train(buffer, train_steps_per_iter)` 用 SGD(momentum=0.9)更新 `new_model`,梯度裁剪 1.0。
3. **Pit Evaluation**:`new_model` vs `best_model`(或在前几轮 vs `ClassicAI`),跑 `eval_games` 局,统计胜率。
4. **Promote**:若胜率 ≥ `win_rate_threshold`,把 `new_model` 拷到 `best_model`;否则 `new_model` 重置回 `best_model`,下一轮重新挑战。
5. **Opponent Pool**:每 5 轮把当前 `new_model` 入池(保存在 `checkpoints/opponent_pool/pool_*.pt`)。
6. **Checkpoint**:每 10 轮保存 `iter_XXXX.pt`;最终保存 `final.pt` + `final_eval.json`。

**课程式对手**(pipeline 的关键设计):

```
前 classic_opponent_iters 轮:
  team 0 = MCTS(self.network), team 1 = ClassicAI   ← 稳定基线
之后 classic_opponent_iters + k 轮:
  team 1 以 classic_opponent_decay^k 概率仍是 ClassicAI
  否则从 OpponentPool 随机抽一个历史模型
再往后:
  纯自对弈(双方都是 self.network)
```

**MCTS 节点**(`MCTSNode`):
- `visit_counts[a]`、`total_values[a]`、`prior_probs[a]`、`children[a]`。
- 每次模拟:SELECT(PUCT) → `fast_clone_battle` → `index_to_action` → `battle.execute` → EXPAND(network 评估) → BACKUP(取反传播)。

---

### 3.4 `config/` — 共享常量

| 文件 | 内容 |
|---|---|
| `__init__.py` | 重新导出,游戏状态、布局 |
| `colors.py` | UI 调色板(深色 Pixel Retro) |
| `units.py` | 67 个单位完整数据 + UNIT_TAGS(用于法术定向) + UNIT_TYPE_INDEX(供 embedding) |
| `presets.py` | 10 个预设阵型(`PRESETS["Balanced"]` 等) |
| `timing.py` | 动画时序(FPS、暂停秒数) |

`config.UNIT_TYPES["Swordsman"]` 是单位数据的唯一来源,所有单位都通过 `Unit.from_type(...)` 创建。

---

### 3.5 `configs/` — 训练 / 评估 JSON

`train_configs.json`:6 个阵型用于自对弈采样;
`eval_configs.json`:3 个独立阵型用作"过拟合监测"集(自对弈时混入)。

每个 config 是 `{"units": [{"team": 0|1, "type": str, "col": int, "row": int, "count": int}]}` 格式;可加 `"heroes"`、`"siege": true`、`"morale"`、`"luck"`、`"difficulty"` 字段。

---

### 3.6 `scripts/` — 命令行入口

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `train_alphago.py` | 启动 AlphaGo Zero 训练 | `--sims --games --iterations --train-steps --device --resume` |
| `watch_battle.py` | 用训练好的 `best.pt` 走一局并可视化 | `--mode ascii/png --model --sims --output` |

`pyproject.toml` 注册 `train-alphago = "scripts.train_alphago:main"`(别名)。

---

### 3.7 `tests/`

| 测试 | 覆盖 |
|---|---|
| `test_classic_ai.py` | `ClassicAI` 接口契约、退却判定、Hypnotize 切换、退化/非法输入 |
| `test_engine_rules.py` | 引擎 fheroes2 规则合规(伤害公式、宽单位、法术、攻城、护城河) |

---

## 4. 训练是怎么做的(数据流)

```
                       ┌────────────────────────────────────────────┐
                       │ AlphaGoPipeline.run() 主循环                │
                       └────────────────────────────────────────────┘
                                          │
   ┌──────────────────────┐               │              ┌──────────────────────┐
   │ 1) Self-Play         │               │              │ 2) Replay Buffer     │
   │   SelfPlayRunner     │  TrainingExample ──────────►  │ ReplayBuffer.add()   │
   │     .play_game()     │  (grid, gvec, mask,           │ FIFO 500K            │
   │     .run_batch()     │   policy, outcome)            │  sample(batch)       │
   │   team 0 = MCTS(net) │                               └──────────┬───────────┘
   │   team 1 = ClassicAI │                                          │
   │   or pool, or self   │                                          ▼
   └──────────────────────┘                               ┌──────────────────────┐
                                                          │ 3) Trainer           │
                                                          │ AlphaGoTrainer.train │
                                                          │   policy_loss = CE   │
                                                          │   value_loss  = MSE  │
                                                          │   + clip_grad 1.0    │
                                                          └──────────┬───────────┘
                                                                     │
                                                                     ▼
   ┌──────────────────────┐                               ┌──────────────────────┐
   │ 4) Pit Eval          │ ◄─── win_rate ─── new_model ──►│ promote if ≥0.55    │
   │   vs best OR Classic │                               │ best ← new_state_dict│
   └──────────────────────┘                               └──────────────────────┘
                                          │
                                          ▼
                                  ┌──────────────────────┐
                                  │ 5) Checkpoint        │
                                  │   iter_XXXX.pt       │
                                  │   final.pt           │
                                  │   final_eval.json    │
                                  └──────────────────────┘
```

**训练目标**(AlphaGo Zero 标准):
```
L = (z - v)² + CE(π || p)         # 监督学习:价值 + 策略
   + c·||θ||²                      # L2 weight decay(由 optimizer 处理)
```

**采样流程**:
1. `MCTS.search(state, unit, net)` → 访问计数分布 `π`。
2. `temperature_threshold` 步之前按 `π^(1/τ)` 采样动作;之后 greedy。
3. 写入 `_TrajectoryEntry`(包含当前观察 + π + 行动队伍)。
4. 整局结束 → `outcome = +1/-1` 按行动队伍视角回填 → 转 `TrainingExample`。

---

## 5. 可调范围(参数全景)

### 5.1 MCTS(`AlphaGoConfig`)
| 参数 | 默认 | 推荐范围 | 含义 |
|---|---|---|---|
| `num_simulations` | 800 | 100–1600 | 每步 MCTS 模拟次数 |
| `c_puct` | 2.5 | 1.0–5.0 | 探索常数 |
| `dirichlet_alpha` | 0.03 | 0.03–0.3 | 根节点 Dirichlet 浓度 |
| `dirichlet_epsilon` | 0.25 | 0.10–0.30 | 噪声混合比例 |
| `temperature_threshold` | 8 | 4–30 | 多少步之后用 greedy |
| `eval_mcts_simulations` | 400 | 100–800 | 评估时降低模拟数(加速) |

### 5.2 自对弈
| 参数 | 默认 | 含义 |
|---|---|---|
| `games_per_iteration` | 100 | 每轮自对弈局数 |
| `max_moves_per_game` | 200 | 单局最大步数(`MAX_ROUNDS` 上限) |

### 5.3 Replay Buffer
| 参数 | 默认 | 含义 |
|---|---|---|
| `buffer_capacity` | 500_000 | FIFO 上限 |
| `min_buffer_size` | 10_000 | 训练启动门槛 |

### 5.4 训练
| 参数 | 默认 | 含义 |
|---|---|---|
| `batch_size` | 512 | SGD mini-batch |
| `learning_rate` | 0.01 | SGD 学习率 |
| `momentum` | 0.9 | SGD 动量 |
| `weight_decay` | 1e-4 | L2 正则 |
| `train_steps_per_iter` | 1000 | 每轮 SGD 更新次数 |
| `log_interval` | 10 | 控制台打印周期 |

### 5.5 评估
| 参数 | 默认 | 含义 |
|---|---|---|
| `eval_games` | 50 | pit 局数 |
| `win_rate_threshold` | 0.55 | 升级阈值(>0.5 即可) |
| `early_stop_patience` | 0 | N 轮无提升则停(0=关) |

### 5.6 Pipeline / 课程
| 参数 | 默认 | 含义 |
|---|---|---|
| `num_iterations` | 100 | 总迭代数 |
| `opponent_pool_size` | 5 | 历史模型池容量(0=纯自对弈) |
| `use_classic_opponent` | True | 前几轮用 ClassicAI 作对手 |
| `classic_opponent_iters` | 5 | ClassicAI 强制阶段轮数 |
| `classic_opponent_difficulty` | "Normal" | ClassicAI 难度 |
| `classic_opponent_randomize` | 0.05 | ClassicAI 决策抖动 |
| `classic_opponent_decay` | 0.5 | ClassicAI 概率衰减底 |

### 5.7 终评
| 参数 | 默认 | 含义 |
|---|---|---|
| `final_eval_enabled` | True | 训练完 best vs ClassicAI |
| `final_eval_games` | 100 | 终评局数 |
| `final_eval_mcts_simulations` | 800 | 终评 MCTS 模拟数 |

### 5.8 硬件
| 参数 | 默认 | 含义 |
|---|---|---|
| `device` | "cpu" | "cpu" / "cuda" / "cuda:0" |
| `num_workers` | 1 | 并行自对弈进程(⚠️ 尚未实现) |

### 5.9 引擎侧可调(`engine/spells.py`, `config/units.py`)
- 新增单位:在 `UNIT_TYPES` 加条目 + 在 `UNIT_TAGS` 加标签(若需特殊定向)。
- 新增法术:在 `SPELLS` 加 `Spell(...)`,设置 `kind` / `cost` / `aoe_pattern` 等。
- `BattleState.MAX_ROUNDS` 默认 200;`MAX_TURNS_WITHOUT_DEATHS = 50` 控制 stalemate。

### 5.10 网络结构(`ai_core/model.py`)
模块顶部常量:`_CONV_CHANNELS=128`, `_NUM_RES_BLOCKS=6`, `_BOTTLENECK_DIM=384`, `_EMBED_DIM=16`。
Dropout = 0.15。

---

## 6. 接口怎么"接"

> 所有改动必须保证下面的"接缝"仍然成立。

```
┌────────────────────────────────────────────────────────────────────────────┐
│  scripts/*  (CLI 入口)                                                     │
│      ↓                                                                     │
│  alphago.pipeline.AlphaGoPipeline.run()                                    │
│      ├── alphago.self_play.SelfPlayRunner                                  │
│      │     └── ai_core.model.BattleNet     ← 唯一的策略 / 价值网络          │
│      │     └── ai_core.action_space.legal_mask / index_to_action            │
│      │     └── engine.battle_state.BattleState.execute / turn_order        │
│      ├── alphago.trainer.AlphaGoTrainer                                    │
│      │     └── ai_core.model.BattleNet                                     │
│      │     └── alphago.replay_buffer.ReplayBuffer                          │
│      ├── alphago.opponent_pool.OpponentPool                                │
│      ├── alphago.player.MCTSAIPlayer  (implements ai_core.base.AIPlayer)   │
│      ├── ai_core.classic_ai.ClassicAI    (implements ai_core.base.AIPlayer)│
│      │     └── ai_core.battle_geometry   (与 action_space 共享 wide-unit 规则)│
│      └── engine.battle_state.BattleState                                   │
│                                                                            │
│  任何新 AI 想接入:  implement ai_core.base.AIPlayer                        │
│  任何新动作:        engine.actions.Action 子类 + ai_core.action_space 同步  │
│  任何新特征:        ai_core.observation._encode_* + NUM_GRID_CHANNELS++     │
└────────────────────────────────────────────────────────────────────────────┘
```

**接缝稳定原则**(便于升级):

1. **新 AI**:`class FooAI(AIPlayer)` → 实现三个决策方法；若持有每场状态则覆盖 `battle_begins()` → 替换 `pipeline._classic_opponent_ai()` 或加入 `OpponentPool`。
2. **新动作**:在 `engine/actions.py` 加子类,同时在 `ai_core/action_space.py`:
   - 划分区间 → 更新 `WAIT_IDX/MOVE_START/ATTACK_START/CAST_START/RETREAT_IDX/ACTION_DIM`
   - 在 `action_to_index` / `index_to_action` 加分支
   - 在 `legal_mask` 加对应合法掩码生成
3. **新观测通道**:`ai_core/observation.py` 增加通道 → `NUM_GRID_CHANNELS` +1,`_encode_*` 函数补逻辑 → `BattleNet` 输入通道数自动跟随。
4. **新超参**:`AlphaGoConfig` 加字段 → CLI 加 `--xxx` → 文档本节登记。
5. **新引擎规则**:`engine/battle_state.py.execute` 加分支;`tests/test_engine_rules.py` 立即补单测。

---

## 7. 关键文件修改 checklist

| 想做的事 | 改哪些文件 |
|---|---|
| 加新 AI | `ai_core/xxx_ai.py` 或职责包(实现 `AIPlayer`)+ 接入 `pipeline.py` + 新战斗入口调用 `battle_begins()` |
| 加新动作 | `engine/actions.py` + `ai_core/action_space.py` |
| 加新观测通道 | `ai_core/observation.py`(`NUM_GRID_CHANNELS`) |
| 加新法术 | `engine/spells.py`(`SPELLS`)+ `action_space._SPELL_ORDER` |
| 加新单位 | `config/units.py`(`UNIT_TYPES` + `UNIT_TAGS`) |
| 加新阵型 | `configs/train_configs.json` / `configs/eval_configs.json` |
| 改训练超参 | `alphago/config.py`(默认值)+ `scripts/train_alphago.py`(CLI) |
| 改网络结构 | `ai_core/model.py`(`_CONV_CHANNELS` 等常量) |
| 改胜负规则 | `engine/battle_state.py`(`MAX_ROUNDS` / `winner`) |

---

## 8. 训练产物路径

```
checkpoints/
├── best.pt                    # 当前最优模型(评估后晋升)
├── final.pt                   # 训练结束时的最终模型
├── iter_0010.pt ... iter_NNNN.pt  # 周期备份
├── final_eval.json            # 终评结果({win_rate_vs_classic, n_games, mcts_sims, ...})
├── opponent_pool/pool_*.pt    # 历史对手模型
└── tensorboard/               # (可选)TensorBoard 日志
```

---

## 9. 常见故障排查

| 现象 | 排查 |
|---|---|
| `Action space mismatch` | 同步 `action_space.ACTION_DIM` 与 `BattleNet` 输出维度(均默认 13,566) |
| `legal_mask` 全 0 | `BattleState._initial_counts` 未设置;或 `current_unit` 已死亡 |
| 训练 loss 不下降 | 调低 `learning_rate`、`num_simulations` 是否过小、`batch_size` 是否过大 |
| `Promote` 永远失败 | `win_rate_threshold=0.55` 太苛刻;或 `eval_games` 太少导致噪声大 |
| 视频/PNG 渲染空白 | `pip install imageio` / `matplotlib` |
| 复盘 GIF 失败 | `pip install imageio`;`generate_replay_gif()` 在 `alphago/visualization.py` |

---

## 10. 更新日志模板

> 每次有重大改动,建议在仓库根建一个 `CHANGELOG.md`,按下面格式记录。
> 把改动对应到第 6 节"接缝",便于追溯。

```markdown
# 更新日志

## [YYYY-MM-DD] — <一句话标题>

### 改动
- engine/...: <改动点,引用文件:行号>
- ai_core/...: <改动点>
- alphago/...: <改动点>
- scripts/...: <改动点>

### 接缝影响
- 动作空间:`ACTION_DIM` 从 13566 → 13568 (新增 XxxAction)
- 观测通道:`NUM_GRID_CHANNELS` 从 35 → 36 (新增 Xxx 通道)
- AIPlayer 接口:无 / 新增方法 X / 签名变更
- 新超参:`AlphaGoConfig.xxx`,CLI `--xxx`

### 测试
- tests/test_engine_rules.py 新增 Xxx
- 训练结果对照:`checkpoints/before/` vs `checkpoints/after/`

### 回滚方法
- 退回 commit `xxxx`;或对 `engine.battle_state` 中 Xxx 函数 revert
```

---

## 11. 一句话项目心法

> **改哪个文件最对?**
> - 改游戏规则 → `engine/`
> - 改 AI 决策接口 → `ai_core/base.py` + 实现类
> - 改网络输入/输出 → `ai_core/observation.py` 或 `ai_core/action_space.py`
> - 改训练流程 → `alphago/pipeline.py` / `alphago/mcts.py` / `alphago/trainer.py`
> - 改命令行 → `scripts/train_alphago.py`
> - 改超参默认值 → `alphago/config.py`

---

*文档基线:2026-08-01 — AlphaGo Battle AI v0.2.0 (engine ↔ fheroes2 对齐)*