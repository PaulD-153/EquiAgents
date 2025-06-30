import numpy as np
import gym
from gym import spaces

class ResourceMDPEnv(gym.Env):
    """
    A stateful multi-agent environment for resource-constrained planning.
    Agents act by submitting claim probabilities.
    """

    def __init__(self, n_agents, resource_capacity=1, max_steps=5, reward_profile=None, SL_states=None, TL=None, limit_fn=None, initial_sL=None):
        self.n_agents = n_agents
        # static “base” capacity (used only to build default limit_fn)
        self._base_capacity = resource_capacity
        # exogenous limit‐state chain
        # Explicit None checks so we don’t do "array or ..."
        self.SL_states = SL_states if SL_states is not None else [0]
        self.TL = TL if TL is not None else np.array([[1.0]])
        # limit_fn(t, sL) → capacity at time t when in SL‐state sL
        self.limit_fn = limit_fn if limit_fn is not None else (lambda t, sL: self._base_capacity)

        # initialize SL
        self.sL = initial_sL if initial_sL is not None else self.SL_states[0]
        self.sL_history = []
        self.max_steps = max_steps
        self.reward_profile = reward_profile or {i: (1.0, 1.0) for i in range(n_agents)}  # fallback

        assert np.allclose(self.TL.sum(axis=1), 1.0), "Each row of TL must sum to 1"

        self.resource_capacity = self.limit_fn(0, self.sL)

        self.timestep = 0
        self.usage_vector = np.zeros(n_agents)  # Cumulative claims
        self.last_claims = np.zeros(n_agents)

        obs_dim = 1 + n_agents + len(self.SL_states)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
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
        self.resource_capacity = self.limit_fn(0, self.sL)
        # reset SL trajectory
        self.sL = self.SL_states[0]
        self.sL_history = [self.sL]
        return self._get_obs()

    def _get_obs(self):
        t_norm = self.timestep / self.max_steps
        # one-hot encode current sL
        one_hot = np.zeros(len(self.SL_states), dtype=np.float32)
        idx = self.SL_states.index(self.sL)
        one_hot[idx] = 1.0
        return np.concatenate([[t_norm], self.usage_vector, one_hot]).astype(np.float32)

    def step(self, actions):
        # --- New: advance SL and recompute capacity ---
        # sample next SL state
        p = self.TL[self.SL_states.index(self.sL)]
        self.sL = np.random.choice(self.SL_states, p=p)
        self.sL_history.append(self.sL)
        # update instantaneous capacity via limit_fn
        self.resource_capacity = self.limit_fn(self.timestep + 1, self.sL)

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
        return self._get_obs(), reward, done, {
            "claims": self.last_claims.copy(),
            "sL":       self.sL
        }

    def render(self, mode='human'):
        print(f"Step {self.timestep}, Usage: {self.usage_vector}, Last Claims: {self.last_claims}")