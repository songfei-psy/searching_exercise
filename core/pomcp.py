# POMCP 算法实现

import math
import random
from collections import defaultdict
from typing import Any, Dict, Tuple, List, Optional


# ==========================
# POMCP 节点类（历史为主）
# ==========================
class POMCPNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children: Dict[Any, "POMCPNode"] = {}  # 观测 -> 子节点
        self.action_children: Dict[int, "POMCPActionNode"] = {}  # 动作 -> 动作节点
        self.particles: List[Any] = []  # 状态粒子
        self.visit_count = 0


class POMCPActionNode:
    def __init__(self, action: int, parent: POMCPNode):
        self.action = action
        self.parent = parent
        self.children: Dict[Any, POMCPNode] = {}  # 观测 -> 下一个 POMCPNode
        self.visit_count = 0
        self.total_value = 0.0

    def ucb_score(self, c: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        avg_value = self.total_value / self.visit_count
        exploration = c * math.sqrt(
            math.log(self.parent.visit_count + 1) / self.visit_count
        )
        return avg_value + exploration


# ==========================
# POMCP 主算法类
# ==========================
class POMCP:
    def __init__(
        self,
        env,
        belief,
        n_simulations: int = 500,
        max_depth: int = 20,
        c: float = 1.0,
        rollout_policy: Optional[callable] = None,
        discount: float = 0.95,
        rng: Optional[random.Random] = None,
        np_rng=None,
    ):
        """
        参数说明：
        - env: 部分可观测环境
        - belief: 当前根节点的粒子滤波器
        - n_simulations: 模拟次数
        - max_depth: 最大模拟深度
        - c: UCB 参数
        - rollout_policy: 默认策略
        - discount: 折扣因子
        """
        self.env = env
        self.root = POMCPNode()
        self.belief = belief
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.c = c
        self.rollout_policy = rollout_policy
        self.discount = discount
        self.rng = rng or random.Random()
        self.np_rng = np_rng

    def search(self) -> int:
        """
        从根粒子中启动多次模拟，选取最佳动作
        """
        # 同步树根的粒子集，保持与当前 belief 一致
        self.root.particles = list(self.belief.particles)

        for _ in range(self.n_simulations):
            state = self.rng.choice(self.belief.particles)
            self._simulate(state, self.root, depth=0)

        # 选取访问次数最多的动作
        best_action = max(
            self.root.action_children.items(),
            key=lambda kv: kv[1].visit_count
        )[0]
        return best_action

    def _simulate(self, state, node: POMCPNode, depth: int) -> float:
        if depth >= self.max_depth or self.env.is_terminal(state):
            return 0.0

        node.visit_count += 1

        # 动作集合
        legal_actions = list(self.env.ACTIONS.keys())

        # 动作扩展（按需）
        for action in legal_actions:
            if action not in node.action_children:
                node.action_children[action] = POMCPActionNode(action, parent=node)

        # UCB 选择动作
        best_action_node = max(
            node.action_children.values(), key=lambda a: a.ucb_score(self.c)
        )
        action = best_action_node.action

        # 模拟一步环境（不污染真实环境）
        next_state, reward, done, obs = self.env.simulate_step(
            state, action, rng=self.rng, np_rng=self.np_rng
        )

        obs_key = self._obs_to_key(obs)

        # 添加粒子（用于后续 belief 更新）
        if len(node.particles) < 500:
            node.particles.append(state)

        # 观测分支扩展
        if obs_key not in best_action_node.children:
            best_action_node.children[obs_key] = POMCPNode(parent=best_action_node)

        obs_node = best_action_node.children[obs_key]

        # 递归模拟
        value = reward + self.discount * self._simulate(next_state, obs_node, depth + 1)

        # 回传
        best_action_node.visit_count += 1
        best_action_node.total_value += value

        return value

    def update_belief(self, action, observation):
        """
        1. 将根节点更新为当前观测后的子节点
        2. 对新根构建新的粒子集合
        """
        obs_key = self._obs_to_key(observation)

        if action in self.root.action_children:
            if obs_key in self.root.action_children[action].children:
                self.root = self.root.action_children[action].children[obs_key]
                self.root.particles = list(self.belief.particles)
                return
        # 如果观测不在树中，重建根节点
        self.root = POMCPNode()
        self.root.particles = list(self.belief.particles)

    @staticmethod
    def _obs_to_key(obs):
        """将观测转换为可哈希的键，用于树索引。"""
        try:
            hash(obs)
            return obs
        except TypeError:
            if isinstance(obs, dict):
                return tuple(sorted((k, POMCP._obs_to_key(v)) for k, v in obs.items()))
            if isinstance(obs, (list, tuple)):
                return tuple(POMCP._obs_to_key(v) for v in obs)
            return str(obs)

    def clear(self):
        """
        清空搜索树（用于重置）
        """
        self.root = POMCPNode()
