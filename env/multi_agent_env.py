import numpy as np
import gym
from gym import spaces

class MultiAgentEnv(gym.Env):
    """
    A simplified environment for multi-agent flow planning.
    Tracks occupancy of resources and rewards for moving between them.
    """
    def __init__(self, n_resources, n_agents, max_steps=50, resource_capacity=20, reward_matrix=None):
        super().__init__()
        self.n_resources = n_resources
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.resource_capacity = resource_capacity

        self.current_step = 0
        self.occupancy = np.zeros(n_resources)

        self.reward_matrix = reward_matrix if reward_matrix is not None else np.random.rand(n_resources, n_resources)

        # Action space: not used in LP-driven mode, but keeping gym API
        self.action_space = spaces.Box(low=0, high=self.n_agents, shape=(n_resources, n_resources), dtype=np.float32)

        # Observation: occupancy at resources
        self.observation_space = spaces.Box(low=0, high=resource_capacity, shape=(n_resources,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.occupancy = np.zeros(self.n_resources)
        self.occupancy[0] = self.n_agents  # Start all agents at resource 0 (or customize)
        return self.occupancy.copy()

    def step(self, flow_matrix):
        """
        flow_matrix: np.array of shape (n_resources, n_resources)
        flow_matrix[i, j] = number of agents moving from i to j.
        """
        rewards = np.sum(flow_matrix * self.reward_matrix)

        # Update occupancy
        new_occupancy = np.zeros(self.n_resources)
        for i in range(self.n_resources):
            for j in range(self.n_resources):
                new_occupancy[j] += flow_matrix[i, j]

        # Clip to resource capacity
        new_occupancy = np.clip(new_occupancy, 0, self.resource_capacity)

        self.occupancy = new_occupancy

        # Advance step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.occupancy.copy(), rewards, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Occupancy: {self.occupancy}")

    def seed(self, seed=None):
        np.random.seed(seed)
