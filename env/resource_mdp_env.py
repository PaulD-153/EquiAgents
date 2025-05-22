import numpy as np
import gym
from gym import spaces

class ResourceMDPEnv(gym.Env):
    """
    A stateful multi-agent environment for resource-constrained planning.
    Agents act by submitting claim probabilities.
    """

    def __init__(self, n_agents, resource_capacity=1, max_steps=5, reward_profile=None):
        self.n_agents = n_agents
        self.resource_capacity = resource_capacity
        self.max_steps = max_steps
        self.reward_profile = reward_profile or {i: (1.0, 1.0) for i in range(n_agents)}  # fallback

        self.timestep = 0
        self.usage_vector = np.zeros(n_agents)  # Cumulative claims
        self.last_claims = np.zeros(n_agents)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1 + n_agents,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_agents,),
            dtype=np.float32
        )

    def reset(self):
        self.timestep = 0
        self.usage_vector = np.zeros(self.n_agents)
        self.last_claims = np.zeros(self.n_agents)
        return self._get_obs()

    def _get_obs(self):
        normalized_time = self.timestep / self.max_steps
        return np.concatenate([[normalized_time], self.usage_vector]).astype(np.float32)

    def step(self, actions):
        """
        actions: np.array of shape (n_agents,), values ∈ [0, 1]
        Each value is the probability that agent claims the resource.
        """
        assert actions.shape == (self.n_agents,)

        # Sample which agents actually make a claim (Bernoulli trial)
        sampled_claims = (np.random.rand(self.n_agents) < actions).astype(int)

        # Assign rewards based on sampled claim and reward profile
        reward = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if sampled_claims[i] == 1:
                low, high = self.reward_profile.get(i, (1.0, 1.0))
                reward[i] = np.random.uniform(low, high)

        self.usage_vector += sampled_claims
        self.last_claims = sampled_claims
        self.timestep += 1
        done = self.timestep >= self.max_steps

        return self._get_obs(), reward, done, {"claims": self.last_claims.copy()}

    def render(self, mode='human'):
        print(f"Step {self.timestep}, Usage: {self.usage_vector}, Last Claims: {self.last_claims}")