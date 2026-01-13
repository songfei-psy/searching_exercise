# 信念状态表示与更新

import random
import numpy as np
from abc import ABC, abstractmethod
from collections import Counter


class BeliefState(ABC):
    """
    抽象信念状态类，表示对真实状态的概率分布（或粒子近似）
    """

    @abstractmethod
    def sample(self, n: int = 1):
        """从信念中采样状态"""
        pass

    @abstractmethod
    def update(self, action, observation):
        """给定动作和观测，更新信念"""
        pass

    @abstractmethod
    def get_distribution(self):
        """返回当前粒子分布（用于可视化）"""
        pass


class ParticleFilter(BeliefState):
    def __init__(self, initial_particles, transition_fn, observation_fn, resample_threshold=0.5, max_particles=100):
        """
        参数：
        - initial_particles: 初始状态粒子集合
        - transition_fn(s, a): 状态转移函数
        - observation_fn(s, a, o): 返回观测概率 O(o|s,a)
        - resample_threshold: 粒子有效数量阈值
        """
        self.particles = initial_particles
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.max_particles = max_particles
        self.resample_threshold = resample_threshold

    def sample(self, n=1):
        return random.choices(self.particles, k=n)

    def update(self, action, observation):
        """
        核心：重要性采样 + 重采样
        """
        weights = []
        new_particles = []

        for s in self.particles:
            s_prime = self.transition_fn(s, action)
            weight = self.observation_fn(s_prime, action, observation)
            weights.append(weight)
            new_particles.append(s_prime)

        weights = np.array(weights)
        weights_sum = weights.sum()

        if weights_sum == 0:
            # 粒子耗尽
            print("[Warning] 粒子权重全部为 0，使用均匀重置")
            self.particles = random.choices(self.particles, k=self.max_particles)
            return

        weights /= weights_sum
        N_eff = 1.0 / np.sum(weights ** 2)

        # 重采样条件
        if N_eff < self.resample_threshold * len(new_particles):
            self.particles = random.choices(new_particles, weights=weights, k=self.max_particles)
        else:
            self.particles = new_particles

    def get_distribution(self):
        """
        返回粒子计数（Counter）用于可视化
        """
        return Counter(self.particles)


class BeliefUpdater:
    def __init__(self, env, observation_model):
        """
        参数：
        - env: 环境对象，提供 transition() 函数
        - observation_model(s', a, o): 返回观测概率
        """
        self.env = env
        self.observation_model = observation_model

    def bayes_update(self, belief: BeliefState, action, observation):
        """
        使用贝叶斯公式更新给定信念
        """
        belief.update(action, observation)
        
        
class GridworldBelief(ParticleFilter):
    def __init__(self, env, num_particles=100):
        """
        env 必须有：
        - agent_pos (用于初始化)
        - _transition (从状态和动作得到新状态)
        - get_observation (观测函数)
        """
        grid = env.grid_size
        valid_positions = [
            (x, y)
            for x in range(grid[0])
            for y in range(grid[1])
            if (x, y) not in env.obstacles
        ]

        initial_particles = random.choices(valid_positions, k=num_particles)

        def transition_fn(s, a):
            # 模拟一次 MDP 转移（不修改 env）
            saved = env.agent_pos
            env.agent_pos = s
            next_state, _, _ = env.step(a)
            env.agent_pos = saved
            return next_state

        def observation_fn(s, a, o):
            # 使用高斯距离作为观测概率
            sx, sy = s
            ox, oy = o
            dist_sq = (sx - ox) ** 2 + (sy - oy) ** 2
            sigma = 1.0
            prob = np.exp(-dist_sq / (2 * sigma ** 2))
            return prob

        super().__init__(
            initial_particles=initial_particles,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            max_particles=num_particles,
        )


class MonsterBelief(ParticleFilter):
    def __init__(self, env, num_particles=100):
        """
        env: MonsterGridworld
        粒子格式: ((agent_x, agent_y), (monster_x, monster_y), has_key)
        """
        agent_positions = [
            (x, y)
            for x in range(env.grid_size[0])
            for y in range(env.grid_size[1])
            if (x, y) not in env.obstacles
        ]
        monster_positions = env.monster_path

        initial_particles = [
            (random.choice(agent_positions),
             random.choice(monster_positions),
             False)
            for _ in range(num_particles)
        ]

        def transition_fn(s, a):
            agent_pos, monster_pos, has_key = s

            # 模拟怪物走一步
            idx = env.monster_path.index(monster_pos)
            new_monster = env.monster_path[(idx + 1) % len(env.monster_path)]

            # 模拟智能体移动（同 Gridworld）
            saved = env.agent_pos
            env.agent_pos = agent_pos
            next_state, _, _ = env.step(a)
            env.agent_pos = saved

            new_has_key = has_key or (next_state == env.key_pos)

            return (next_state, new_monster, new_has_key)

        def observation_fn(s, a, o):
            agent_pos, monster_pos, has_key = s
            obs_pos = o["agent"]
            obs_monster = o["monster"]
            obs_key = o["has_key"]

            ax, ay = agent_pos
            ox, oy = obs_pos
            dist_sq = (ax - ox) ** 2 + (ay - oy) ** 2
            agent_prob = np.exp(-dist_sq / (2 * 1.0))

            monster_prob = 1.0 if monster_pos == obs_monster else 0.1
            key_prob = 1.0 if has_key == obs_key else 0.1

            return agent_prob * monster_prob * key_prob

        super().__init__(
            initial_particles=initial_particles,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            max_particles=num_particles,
        )



