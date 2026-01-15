# 蒙特卡洛树搜索（MCTS）实现

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable


# =========================================================
# MCTS Node
# =========================================================
class MCTSNode:
    """
    MCTS 树节点
    """

    def __init__(
        self,
        state: Any,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        depth: int = 0,
    ):
        self.state = state
        self.parent = parent
        self.action = action  # 从父节点到该节点的动作
        self.children: Dict[int, MCTSNode] = {}

        self.visit_count = 0
        self.total_reward = 0.0
        self.depth = depth

    # -----------------------------
    # UCB / UCT 计算
    # -----------------------------
    def ucb_score(self, c: float) -> float:
        """
        UCB1 / UCT 公式：

            Q(s,a)/N(s,a) + c * sqrt( ln(N(s)) / N(s,a) )

        其中：
            - Q(s,a): 累计回报
            - N(s,a): 子节点访问次数
            - N(s): 父节点访问次数
            - c: 探索参数
        """
        if self.visit_count == 0:
            return float("inf")

        exploitation = self.total_reward / self.visit_count
        exploration = c * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration

    # -----------------------------
    # Tree utilities
    # -----------------------------
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def expand(self, action: int, next_state: Any) -> "MCTSNode":
        """扩展一个新子节点"""
        child = MCTSNode(
            state=next_state,
            parent=self,
            action=action,
            depth=self.depth + 1,
        )
        self.children[action] = child
        return child

    def best_child(self, c: float) -> "MCTSNode":
        """按 UCB 选择最优子节点"""
        return max(self.children.values(), key=lambda n: n.ucb_score(c))

    def backpropagate(self, reward: float):
        """回传Backpropagation"""
        self.visit_count += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


# =========================================================
# MCTS Algorithm
# =========================================================
class MCTS:
    """
    通用 MCTS / UCT 实现
    """

    def __init__(
        self,
        env,
        n_simulations: int = 1000,
        max_depth: int = 50,
        c: float = 1.4,
        rollout_policy: Optional[Callable] = None,
        heuristic_fn: Optional[Callable] = None,
        progressive_widening: bool = False,
        max_children: int = 5,
        parallel: bool = False,
        rng: Optional[random.Random] = None,
        np_rng=None,
    ):
        """
        参数说明：
        env: 环境实例（必须支持 step / is_terminal 方法）
        n_simulations: 模拟次数
        max_depth: 最大搜索深度
        c: UCT 探索系数
        rollout_policy: rollout 阶段策略, None 表示随机采样
        heuristic_fn: 启发式评估函数（用于非终止 rollout 状态）
        progressive_widening: 是否启用渐进式扩展
        max_children: 渐进式扩展最大子节点数
        parallel: 是否启用并行模拟（此处预留接口）
        """
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.c = c

        self.rollout_policy = rollout_policy
        self.heuristic_fn = heuristic_fn

        self.progressive_widening = progressive_widening
        self.max_children = max_children

        self.parallel = parallel  # 预留

        # 独立随机源，避免污染全局随机序列
        self.rng = rng or random.Random()
        self.np_rng = np_rng

    # =====================================================
    # Public API
    # =====================================================
    def search(self, root_state: Any) -> int:
        """
        执行 MCTS 搜索，返回最优动作
        """
        root = MCTSNode(state=root_state)

        for _ in range(self.n_simulations):
            node = self._selection(root)
            reward = self._simulation(node)
            node.backpropagate(reward)

        # 返回访问次数最多的动作
        best_child = max(
            root.children.values(),
            key=lambda n: n.visit_count,
        )
        return best_child.action

    # =====================================================
    # MCTS Four Steps
    # =====================================================
    def _selection(self, node: MCTSNode) -> MCTSNode:
        """
        Selection + Expansion
        """
        while not self.env.is_terminal(node.state) and node.depth < self.max_depth:
            if node.is_leaf():
                return self._expand(node)
            else:
                node = node.best_child(self.c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion
        """
        actions = list(self.env.ACTIONS.keys())

        if self.progressive_widening:
            if len(node.children) >= self.max_children:
                return node.best_child(self.c)

        action = self.rng.choice(actions)
        next_state, _, _ = self._simulate_step(node.state, action)
        return node.expand(action, next_state)

    def _simulation(self, node: MCTSNode) -> float:
        """
        Simulation / Rollout
        """
        state = node.state
        depth = node.depth
        total_reward = 0.0
        discount = 1.0

        while not self.env.is_terminal(state) and depth < self.max_depth:
            if self.rollout_policy:
                action = self.rollout_policy(state)
            else:
                action = self.rng.choice(list(self.env.ACTIONS.keys()))

            state, reward, _ = self._simulate_step(state, action)
            total_reward += discount * reward  # 累计折扣奖励
            discount *= 0.99
            depth += 1

        # 启发式评估
        if self.heuristic_fn and not self.env.is_terminal(state):
            total_reward += self.heuristic_fn(state)

        return total_reward

    # =====================================================
    # Utilities
    # =====================================================
    def _simulate_step(self, state, action):
        """
        用于 MCTS 的环境 step（不污染真实环境）
        """
        next_state, reward, done, _ = self.env.simulate_step(
            state, action, rng=self.rng, np_rng=self.np_rng
        )
        return next_state, reward, done
