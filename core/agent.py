# Agents for MDP and POMDP environments

import random
from core.mcts import MCTS
from core.pomcp import POMCP
from core.belief import GridworldBelief, MonsterBelief


class BaseAgent:
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        raise NotImplementedError


# ===================================================
# MCTS Agent：用于完全可观测 MDP Gridworld
# ===================================================
class MCTSAgent(BaseAgent):
    def __init__(self, env, n_simulations=100, max_depth=20, c=1.4, rng=None, np_rng=None):
        super().__init__(env)
        self.planner = MCTS(env, n_simulations=n_simulations, max_depth=max_depth, c=c, rng=rng, np_rng=np_rng)

    def act(self, observation):
        return self.planner.search(observation)


# ===================================================
# POMCP Agent：用于 POMDP Gridworld
# ===================================================
class POMCPAgent(BaseAgent):
    def __init__(
        self,
        env,
        belief_type="gridworld",  # or "monster"
        n_simulations=500,
        max_depth=25,
        c=1.0,
        num_particles=200,
        rollout_policy=None,
        rng=None,
        np_rng=None,
    ):
        super().__init__(env)

        if belief_type == "gridworld":
            self.belief = GridworldBelief(env, num_particles=num_particles)
            self._belief_factory = lambda: GridworldBelief(env, num_particles=num_particles)
        elif belief_type == "monster":
            self.belief = MonsterBelief(env, num_particles=num_particles)
            self._belief_factory = lambda: MonsterBelief(env, num_particles=num_particles)
        else:
            raise ValueError("Invalid belief type")

        self.planner = POMCP(
            env,
            belief=self.belief,
            n_simulations=n_simulations,
            max_depth=max_depth,
            c=c,
            rollout_policy=rollout_policy,
            rng=rng,
            np_rng=np_rng,
        )

    def act(self, observation):
        action = self.planner.search()
        return action

    def observe(self, action, observation):
        """
        在环境中执行完 action 并获得 observation 后，更新内部 belief 和 planner 树
        """
        self.belief.update(action, observation)
        self.planner.update_belief(action, observation)

    def reset(self):
        self.belief = self._belief_factory()
        self.planner.clear()
