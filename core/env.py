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
        return self.get_observation(self.agent_pos)

    def step(self, action):
        next_pos = self._transition(self.agent_pos, action)
        if self._valid_position(next_pos):
            self.agent_pos = next_pos
        reward = self.reward_fn(self.agent_pos)
        done = self.is_terminal(self.agent_pos)
        obs = self.get_observation(self.agent_pos)
        return obs, reward, done

    def _valid_position(self, pos):
        x, y = pos
        rows, cols = self.grid_size
        return 0 <= x < rows and 0 <= y < cols and pos not in self.obstacles

    @abstractmethod
    def _transition(self, current_pos, action):
        pass

    @abstractmethod
    def get_observation(self, state):
        pass

    def is_terminal(self, state):
        return state == self.goal_pos

    def default_reward_fn(self, pos):
        return 1.0 if pos == self.goal_pos else -0.1

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
        self._fig.canvas.draw()
        plt.pause(0.1)


# === 确定性环境 ===
class DeterministicGridworld(BaseEnv):
    def _transition(self, current_pos, action):
        dx, dy = self.ACTIONS[action]
        x, y = current_pos
        return (x + dx, y + dy)

    def get_observation(self, state):
        return state


# === 随机环境 ===
class StochasticGridworld(BaseEnv):
    def _transition(self, current_pos, action):
        prob = random.random()
        if prob < 0.8:
            chosen_action = action
        else:
            chosen_action = random.choice(list(self.ACTIONS.keys()))
        dx, dy = self.ACTIONS[chosen_action]
        x, y = current_pos
        return (x + dx, y + dy)

    def get_observation(self, state):
        return state


# === 部分可观测环境 ===
class PartialObservableGrid(StochasticGridworld):
    def __init__(self, obs_noise_std=0.5, **kwargs):
        super().__init__(**kwargs)
        self.obs_noise_std = obs_noise_std

    def get_observation(self, state):
        x, y = state
        noisy_x = np.random.normal(x, self.obs_noise_std)
        noisy_y = np.random.normal(y, self.obs_noise_std)
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
        return super().reset()

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

    def is_terminal(self, state):
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
