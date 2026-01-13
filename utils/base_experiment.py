# experiments/base_experiment.py

import time
import numpy as np
import random
from collections import defaultdict
from utils.metrics import compute_metrics
from utils.utils import save_results, set_seed


class BaseExperiment:
    def __init__(self, env_class, agent_class, config):
        """
        config: dict
            {
                "grid_size": (5,5),
                "num_episodes": 100,
                "max_steps": 100,
                "seed": 42,
                "agent_args": {...}
            }
        """
        self.config = config
        set_seed(config.get("seed", 0))
        self.env_class = env_class
        self.agent_class = agent_class

        # 独立随机源，便于重现实验且不污染全局随机流
        seed = config.get("seed", 0)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.num_episodes = config["num_episodes"]
        self.max_steps = config["max_steps"]

    def run(self):
        results = []
        for ep in range(self.num_episodes):
            env = self.env_class(**self.config.get("env_args", {}))

            agent_args = dict(self.config.get("agent_args", {}))
            agent_args.setdefault("rng", self.rng)
            agent_args.setdefault("np_rng", self.np_rng)

            agent = self.agent_class(env, **agent_args)
            episode_log = self.run_episode(env, agent)
            results.append(episode_log)

        metrics = compute_metrics(results)
        save_results(metrics, self.config.get("output_file", "results.json"))
        return metrics

    def run_episode(self, env, agent):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()

        trajectory = []

        while not done and steps < self.max_steps:
            action = agent.act(obs)
            new_obs, reward, done = env.step(action)
            if hasattr(agent, "observe"):
                agent.observe(action, new_obs)

            total_reward += reward
            trajectory.append((obs, action, reward))
            obs = new_obs
            steps += 1

        end_time = time.time()

        return {
            "success": env.is_terminal(env.agent_pos),
            "reward": total_reward,
            "steps": steps,
            "time": end_time - start_time,
            "trajectory": trajectory,
        }
