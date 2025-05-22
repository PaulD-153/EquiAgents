# multi_agent_env_minimal.py

import numpy as np
import gym
from gym import spaces

class MinimalResourceEnv(gym.Env):
    """
    A minimal environment for resource claiming.
    Each timestep: agents claim, resource limited by capacity.
    """
    def __init__(self, n_agents, resource_capacity=1, max_steps=50):
        super().__init__()
        self.n_agents = n_agents
        self.resource_capacity = resource_capacity
        self.max_steps = max_steps

        self.current_step = 0

        # Action space: claim amount per agent (scalar [0, 1])
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n_agents,), dtype=np.float32)

        # Observation space: dummy (no complex state needed)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_agents,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return np.zeros(self.n_agents)

    def step(self, claim_vector):
        """
        claim_vector: np.array of shape (n_agents,) with values in [0, 1]
        """
        rewards = np.zeros(self.n_agents)

        total_claims = np.sum(claim_vector)

        if total_claims <= self.resource_capacity:
            rewards = claim_vector.copy()  # Agents get what they claim
        else:
            # Normalize claims proportionally to fit capacity
            scaling = self.resource_capacity / total_claims
            rewards = claim_vector * scaling

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return np.zeros(self.n_agents), np.sum(rewards), done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")

    def seed(self, seed=None):
        np.random.seed(seed)
