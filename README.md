# 🔍 POMDP Gridworld Learning Framework

本项目旨在通过构建一系列逐步复杂化的 Gridworld 环境，系统学习与实现从 **贝叶斯滤波 → MCTS → POMCP** 的完整技术栈。

项目最终目标是使用 POMCP 智能体，在一个含有钥匙、怪物、门的部分可观测 Gridworld 中进行规划与决策。

---

## 📁 项目结构

```

searching_exercise/
├── core/                      # 核心模块
│   ├── __init__.py
│   ├── env.py                 # Gridworld 环境
│   ├── agent.py               # 智能体类
│   ├── bayesfilter.py         # 贝叶斯滤波
│   ├── belief.py              # 粒子滤波与信念模型
│   ├── mcts.py                # MCTS/UCT 搜索
│   └── pomcp.py               # POMCP 规划算法
│
├── utils/                     # 工具模块
│   ├── base_experiment.py     # 实验框架
│   ├── metrics.py             # 指标收集
│   └── utils.py               # 辅助函数
│
├── notbooks/                  # Jupyter Notebook
│   ├── 00_bayes_filter_hw.ipynb
│   ├── 01_mcts_workflow.ipynb
│   ├── 02_pomcp_update_demo.ipynb
│   ├── 03_comparative_analysis.ipynb
│   └── results/               # 实验结果输出
│
├── demo.py                    # 演示脚本
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

### 2. 学习路径

按以下顺序学习最佳：

1. **贝叶斯滤波** (`00_bayes_filter_hw.ipynb`)
   - 学习粒子滤波原理
   - 理解观测与状态更新

2. **MCTS 搜索** (`01_mcts_workflow.ipynb`)
   - 完全可观测环境下的规划
   - UCB 与树搜索

3. **POMCP 算法** (`02_pomcp_update_demo.ipynb`)
   - 部分可观测问题求解
   - 历史树与粒子信念

4. **对比分析** (`03_comparative_analysis.ipynb`)
   - 多算法性能评测

---

## 📊 实验方式

### ✅ 使用 Notebook 进行实验与分析：

| Notebook                        | 内容                                |
| ------------------------------- | --------------------------------- |
| `00_bayes_filter_hw.ipynb`      | 贝叶斯滤波基础与应用实验                    |
| `01_mcts_workflow.ipynb`        | MCTS 参数调优与性能分析                    |
| `02_pomcp_update_demo.ipynb`    | POMCP 更新演示与可视化分析                   |
| `03_comparative_analysis.ipynb` | 多智能体对比实验（POMCP vs MCTS vs Random） |

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
| `bayesfilter.py`     | 贝叶斯滤波 / 粒子滤波基础实现                           |
| `belief.py`          | 粒子滤波器 + 观测模型                                  |
| `mcts.py`            | 通用 MCTS / UCT 搜索器                             |
| `pomcp.py`           | 基于历史与粒子的 POMCP 算法                             |
| `agent.py`           | Agent 封装（MCTS / POMCP / Greedy / Random）      |
| `base_experiment.py` | 批量实验运行框架                                      |
| `metrics.py`         | 指标记录与对比图绘制                                    |

---

## 📚 参考资源

* Silver et al., [POMCP: Partially Observable Monte Carlo Planning](https://www.cs.ubc.ca/~poole/cs532/2011/readings/silver-uctpomdp.pdf)
* AI Planning Resources: [http://ai-planning.org/](http://ai-planning.org/)
* [Partially Observable Markov Decision Processes](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)

---

## © License

For educational use. Feel free to fork and build upon it!


---