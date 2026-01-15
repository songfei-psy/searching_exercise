import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'，确保图形后端支持
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from abc import ABC, abstractmethod
import time


# === 基类 ===
class BaseEnv(ABC):
    ACTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }

    def __init__(self, grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4), obstacles=None, reward_fn=None):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.agent_pos = start_pos
        self.obstacles = obstacles if obstacles else set()
        self.reward_fn = reward_fn if reward_fn else self.default_reward_fn

        # 图形相关
        self._fig = None
        self._ax = None

    def reset(self):
        self.agent_pos = self.start_pos
        obs = self.get_observation(self.agent_pos)
        # 记录最近观测，便于渲染标注
        self._last_obs = obs
        return obs

    def step(self, action):
        next_pos = self._transition(self.agent_pos, action)
        if self._valid_position(next_pos):
            self.agent_pos = next_pos
        reward = self.reward_fn(self.agent_pos)
        done = self.is_terminal(self.agent_pos)
        obs = self.get_observation(self.agent_pos)
        self._last_obs = obs
        return obs, reward, done

    def simulate_step(self, state, action, rng=None, np_rng=None):
        """
        在给定状态上进行一步仿真，不修改环境自身的内部状态，用于规划算法。
        rng: random.Random 实例（用于离散随机）
        np_rng: numpy 随机生成器（用于连续噪声）
        """
        rng = rng or random.Random()
        np_rng = np_rng or np.random.default_rng()

        next_state = self._transition_from_state(state, action, rng)
        if not self._valid_position(next_state):
            next_state = state

        reward = self.get_reward_from_state(next_state)
        done = self.is_terminal(next_state)
        obs = self.get_observation_from_state(next_state, np_rng)
        return next_state, reward, done, obs

    def _valid_position(self, pos):
        """检查位置是否合法（在网格内且不在障碍物上）"""
        x, y = pos
        rows, cols = self.grid_size
        return 0 <= x < rows and 0 <= y < cols and pos not in self.obstacles

    @abstractmethod
    def _transition(self, current_pos, action):
        pass

    def _transition_from_state(self, state, action, rng):
        """纯函数版本的转移，用于规划/仿真。默认与 _transition 等价。"""
        return self._transition(state, action)

    @abstractmethod
    def get_observation(self, state):
        pass

    def get_observation_from_state(self, state, np_rng=None):
        """纯函数版本观测，可在仿真中使用，不依赖环境当前状态。"""
        return self.get_observation(state)

    def get_reward_from_state(self, state):
        return self.reward_fn(state)

    def is_terminal(self, state):
        return state == self.goal_pos

    def default_reward_fn(self, pos):
        return 1.0 if pos == self.goal_pos else -0.1  # 到达目标奖励1，否则每步惩罚0.1

    def render(self, mode='human'):
        rows, cols = self.grid_size
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots()
            self._ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
            self._ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
            self._ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            self._ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        self._ax.clear()
        grid_data = np.zeros(self.grid_size)
        for (x, y) in self.obstacles:
            grid_data[x, y] = 1

        cmap = colors.ListedColormap(['white', 'black'])
        self._ax.imshow(grid_data, cmap=cmap)

        # Agent
        ax, ay = self.agent_pos
        self._ax.add_patch(patches.Circle((ay, ax), 0.3, color='blue'))

        # Goal
        gx, gy = self.goal_pos
        self._ax.add_patch(patches.Rectangle((gy - 0.4, gx - 0.4), 0.8, 0.8,
                                             edgecolor='green', facecolor='none', lw=2))

        self._ax.set_title(self.__class__.__name__)
        # 图例与旁注
        self._draw_legend()
        self._draw_annotation()
        self._fig.canvas.draw()
        plt.pause(0.1)

    def _legend_handles(self):
        return [
            patches.Patch(color='black', label='Obstacle'),
            patches.Circle((0, 0), 0.3, color='blue', label='Agent'),
            patches.Rectangle((0, 0), 0.8, 0.8, edgecolor='green', facecolor='none', lw=2, label='Goal'),
        ]

    def _draw_legend(self):
        try:
            handles = self._legend_handles()
            labels = [h.get_label() for h in handles]
            self._ax.legend(handles, labels, loc='upper right', framealpha=0.8)
        except Exception:
            pass

    def _get_annotation_text(self):
        # 默认展示最近观测或当前位置
        if hasattr(self, '_last_obs') and self._last_obs is not None:
            return f"obs={self._last_obs}"
        return f"state={self.agent_pos}"

    def _draw_annotation(self):
        ax, ay = self.agent_pos
        info = self._get_annotation_text()
        self._ax.text(ay + 0.3, ax - 0.3, info, fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7))


# === 确定性环境 ===
class DeterministicGridworld(BaseEnv):
    def _transition(self, current_pos, action):
        """确定性转移函数，根据动作返回下一个位置"""
        dx, dy = self.ACTIONS[action]
        x, y = current_pos
        return (x + dx, y + dy)

    def get_observation(self, state):
        """获取观测，默认观测即为状态"""
        return state


# === 随机环境 ===
class StochasticGridworld(BaseEnv):
    def _transition(self, current_pos, action):
        """随机转移函数，有一定概率执行随机动作"""
        prob = random.random()
        if prob < 0.8:
            chosen_action = action
        else:
            chosen_action = random.choice(list(self.ACTIONS.keys()))
        dx, dy = self.ACTIONS[chosen_action]
        x, y = current_pos
        return (x + dx, y + dy)

    def _transition_from_state(self, state, action, rng):
        prob = rng.random()
        if prob < 0.8:
            chosen_action = action
        else:
            chosen_action = rng.choice(list(self.ACTIONS.keys()))
        dx, dy = self.ACTIONS[chosen_action]
        x, y = state
        return (x + dx, y + dy)

    def get_observation(self, state): 
        return state


# === 部分可观测环境 ===
class PartialObservableGrid(StochasticGridworld):
    def __init__(self, obs_noise_std=0.5, **kwargs):
        super().__init__(**kwargs)
        self.obs_noise_std = obs_noise_std

    def get_observation(self, state):
        """获取带噪声的观测"""
        x, y = state
        noisy_x = np.random.normal(x, self.obs_noise_std)
        noisy_y = np.random.normal(y, self.obs_noise_std)
        return (noisy_x, noisy_y)

    def get_observation_from_state(self, state, np_rng=None):
        np_rng = np_rng or np.random.default_rng()
        x, y = state
        noisy_x = np_rng.normal(x, self.obs_noise_std)
        noisy_y = np_rng.normal(y, self.obs_noise_std)
        return (noisy_x, noisy_y)


# === 怪物格子世界 ===
class MonsterGridworld(PartialObservableGrid):
    def __init__(self, monster_path=None, key_pos=(2, 2), door_pos=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.monster_path = monster_path or [(1, 1), (1, 2), (2, 2), (2, 1)]
        self.monster_index = 0
        self.key_pos = key_pos
        self.door_pos = door_pos
        self.has_key = False

    def reset(self):
        self.has_key = False
        self.monster_index = 0
        # 不使用父类的 get_observation_from_state，以避免复合状态解包错误
        self.agent_pos = self.start_pos
        obs = self.get_observation(self.agent_pos)
        self._last_obs = obs
        return obs

    def _transition(self, current_pos, action):
        self.monster_index = (self.monster_index + 1) % len(self.monster_path)
        monster_pos = self.monster_path[self.monster_index]
        next_pos = super()._transition(current_pos, action)

        if next_pos == monster_pos:
            return self.start_pos  # 被怪物抓，回起点

        if next_pos == self.key_pos:
            self.has_key = True

        if next_pos == self.door_pos and not self.has_key:
            return current_pos  # 没钥匙不能通过门

        return next_pos

    def get_observation(self, state):
        noisy_pos = super().get_observation(state)
        monster_pos = self.monster_path[self.monster_index]
        return {
            "agent": noisy_pos,
            "monster": monster_pos,
            "has_key": self.has_key
        }

    def _transition_from_state(self, state, action, rng):
        agent_pos, monster_idx, has_key = state

        monster_idx = (monster_idx + 1) % len(self.monster_path)
        monster_pos = self.monster_path[monster_idx]

        dx, dy = self.ACTIONS[action]
        x, y = agent_pos
        candidate = (x + dx, y + dy)
        next_pos = candidate if self._valid_position(candidate) else agent_pos

        if next_pos == monster_pos:
            next_pos = self.start_pos
            has_key = False

        if next_pos == self.key_pos:
            has_key = True

        if next_pos == self.door_pos and not has_key:
            next_pos = agent_pos

        return (next_pos, monster_idx, has_key)

    def get_observation_from_state(self, state, np_rng=None):
        np_rng = np_rng or np.random.default_rng()
        agent_pos, monster_idx, has_key = state
        noisy_pos = super().get_observation_from_state(agent_pos, np_rng)
        monster_pos = self.monster_path[monster_idx]
        return {
            "agent": noisy_pos,
            "monster": monster_pos,
            "has_key": has_key,
        }

    def get_reward_from_state(self, state):
        agent_pos, _, _ = state
        return self.reward_fn(agent_pos)

    def simulate_step(self, state, action, rng=None, np_rng=None):
        rng = rng or random.Random()
        np_rng = np_rng or np.random.default_rng()

        next_state = self._transition_from_state(state, action, rng)
        reward = self.get_reward_from_state(next_state)
        done = self.is_terminal(next_state)
        obs = self.get_observation_from_state(next_state, np_rng)
        return next_state, reward, done, obs

    def is_terminal(self, state):
        if isinstance(state, tuple) and len(state) == 3:
            agent_pos, _, has_key = state
            return agent_pos == self.goal_pos and has_key
        return state == self.goal_pos and self.has_key

    def render(self, mode='human'):
        super().render(mode)  # 基础可视化

        # 绘制怪物、钥匙、门
        mx, my = self.monster_path[self.monster_index]
        self._ax.add_patch(patches.Circle((my, mx), 0.3, color='red'))

        if not self.has_key:
            kx, ky = self.key_pos
            self._ax.add_patch(patches.RegularPolygon((ky, kx), numVertices=5, radius=0.3, color='orange'))

        dx, dy = self.door_pos
        self._ax.add_patch(patches.Rectangle((dy - 0.4, dx - 0.4), 0.8, 0.8,
                                             edgecolor='brown', facecolor='none', lw=2))

        self._fig.canvas.draw()
        plt.pause(0.1)

    def _legend_handles(self):
        base = super()._legend_handles()
        # 追加怪物 / 钥匙 / 门的图例
        base += [
            patches.Circle((0, 0), 0.3, color='red', label='Monster'),
            patches.RegularPolygon((0, 0), numVertices=5, radius=0.3, color='orange', label='Key'),
            patches.Rectangle((0, 0), 0.8, 0.8, edgecolor='brown', facecolor='none', lw=2, label='Door'),
        ]
        return base

    def _get_annotation_text(self):
        # 展示最近观测的关键字段
        if hasattr(self, '_last_obs') and isinstance(self._last_obs, dict):
            agent_obs = self._last_obs.get('agent')
            monster_obs = self._last_obs.get('monster')
            has_key = self._last_obs.get('has_key')
            return f"obs.agent={agent_obs}, obs.monster={monster_obs}, has_key={has_key}"
        return f"state={self.agent_pos}, has_key={self.has_key}"


# === main函数：环境测试 ===
def main():
    envs = [
        ("Deterministic", DeterministicGridworld(grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4), obstacles={(2, 2)})),
        ("Stochastic", StochasticGridworld(grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4), obstacles={(2, 2)})),
        ("Partial Observable", PartialObservableGrid(grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4), obstacles={(2, 2)})),
        ("Monster", MonsterGridworld(grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4), obstacles={(1, 3), (3, 1)})),
    ]

    for name, env in envs:
        print(f"\n=== Testing {name} Environment ===\n")
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 20:
            env.render()
            action = random.choice(list(BaseEnv.ACTIONS.keys()))
            obs, reward, done = env.step(action)
            print(f"Step {steps}: Action={action}, Obs={obs}, Reward={reward:.2f}, Done={done}")
            steps += 1
            time.sleep(0.1)

        plt.ioff()
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
