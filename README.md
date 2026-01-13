# 🔍 POMDP Gridworld Learning Framework

本项目旨在通过构建一系列逐步复杂化的 Gridworld 环境，系统学习与实现从 **MCTS → POMDP → POMCP** 的完整技术栈。

项目最终目标是使用 POMCP 智能体，在一个含有钥匙、怪物、门的部分可观测 Gridworld 中进行规划与决策。

---

## 📁 项目结构

```

pomdp_mcts_learning/
├── core/               # 核心模块（环境 + Agent + 算法）
│   ├── env.py
│   ├── mcts.py
│   ├── pomcp.py
│   ├── belief.py
│   ├── agent.py
│   └── **init**.py
│
├── utils/              # 工具模块（评估、绘图、实验基类等）
│   ├── base_experiment.py
│   ├── metrics.py
│   ├── utils.py
│   ├── trajectory_replay.py
│   └── metrics_plotter.py
│
├── notebooks/          # Jupyter Notebook 学习笔记
│   ├── 01_mcts_workflow.ipynb
│   ├── 02_belief_update_demo.ipynb
│   ├── 03_pomcp_visualization.ipynb
│   └── 04_comparative_analysis.ipynb
│
├── results/            # 实验输出 JSON（自动生成）
├── scripts/            # 脚本（可选，用于批量运行）
└── README.md

````

---

## 🚀 快速开始

### 1. 安装依赖（可选）

本项目无重依赖，核心依赖仅：

```bash
pip install numpy matplotlib
````

---

### 2. 运行 MCTS Agent 示例（完全可观测）

```python
from core import DeterministicGridworld, MCTSAgent

env = DeterministicGridworld()
agent = MCTSAgent(env)

obs = env.reset()
done = False

while not done:
    env.render()
    action = agent.act(obs)
    obs, reward, done = env.step(action)
```

---

## 📊 实验方式

### ✅ 使用 Notebook 进行实验与分析：

| Notebook                        | 内容                                |
| ------------------------------- | --------------------------------- |
| `01_mcts_workflow.ipynb`        | MCTS 参数调优与性能分析                    |
| `02_belief_update_demo.ipynb`   | POMDP 中观测噪声影响分析                   |
| `03_pomcp_visualization.ipynb`  | 粒子变化与信念分析                         |
| `04_comparative_analysis.ipynb` | 多智能体对比实验（POMCP vs MCTS vs Random） |

---

## 📐 支持指标

每轮实验可生成如下指标：

* ✅ 成功率 (`success_rate`)
* ✅ 平均奖励 (`avg_reward`)
* ✅ 平均步数 (`avg_steps`)
* ✅ 规划时间 (`avg_time`)
* ⏳ 信念误差（可扩展）
* ✅ JSON 保存 + 可视化

---

## 🧠 模块支持概览

| 模块                   | 功能                                            |
| -------------------- | --------------------------------------------- |
| `env.py`             | 多种 Gridworld 环境（Deterministic / POMDP / 怪物世界） |
| `mcts.py`            | 通用 MCTS / UCT 搜索器                             |
| `pomcp.py`           | 基于历史与粒子的 POMCP 算法                             |
| `belief.py`          | 粒子滤波器 + 观测模型                                  |
| `agent.py`           | Agent 封装（MCTS / POMCP / Greedy / Random）      |
| `base_experiment.py` | 批量实验运行框架                                      |
| `metrics.py`         | 指标记录与对比图绘制                                    |

---

## 🧩 扩展建议

* 加入 Q-MDP、DESPOT、BAMCP 等近似或强化策略
* 结合 Gym API 接入 RLlib 或 PyTorch 训练框架
* 增加高维 Gridworld（如带颜色、多个物体等）
* 多智能体对抗 / 协作 Gridworld 场景

---

## 📚 参考资源

* Silver et al., [POMCP: Partially Observable Monte Carlo Planning](https://www.cs.ubc.ca/~poole/cs532/2011/readings/silver-uctpomdp.pdf)
* AI Planning Resources: [http://ai-planning.org/](http://ai-planning.org/)
* [Partially Observable Markov Decision Processes](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)

---

## © License

For educational use. Feel free to fork and build upon it!


---