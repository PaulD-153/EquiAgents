import numpy as np
import gym
from gym import spaces
from gym_factored.envs.difficult_cmdp import DifficultCMDPEnv

class MultiAgentDifficultCMDPWrapper(gym.Env):
    """
    Multi-Agent wrapper for the DifficultCMDP environment.
    Introduces multiple agents and enforces a capacity constraint.
    """
    def __init__(self, num_agents=3, capacity_limit=1, prob_y_zero=0.1):
        super().__init__()
        
        self.num_agents = num_agents
        self.capacity_limit = capacity_limit  # Max agents allowed to act at once
        self.envs = [DifficultCMDPEnv(prob_y_zero=prob_y_zero) for _ in range(num_agents)]
        self.current_states = [env.reset() for env in self.envs]
        
        self.action_space = spaces.MultiDiscrete([self.envs[0].action_space.n] * num_agents)
        self.observation_space = spaces.Tuple([self.envs[0].observation_space] * num_agents)

        # 🔹 Forward state/action space attributes from the first agent
        self.nS = self.envs[0].ns  # Number of states
        self.nA = self.envs[0].action_space.n  # Number of actions
        self.P = self.envs[0].P  # Transition probability dictionary
        self.isd = self.envs[0].isd  # Initial state distribution

    def step(self, actions):
        """ Steps multiple agents with a shared capacity constraint """
        rewards, next_states, done_flags, infos = [], [], [], []
        
        # Enforce capacity constraint: Only allow `capacity_limit` agents to act
        acting_agents = np.random.choice(self.num_agents, self.capacity_limit, replace=False)
        
        for i in range(self.num_agents):
            if i in acting_agents:
                state, reward, done, info = self.envs[i].step(actions[i])
            else:
                # Agents not allowed to act get zero reward and stay in the same state
                state, reward, done, info = self.current_states[i], 0, False, {}
                info['skipped'] = True  # Custom info for fairness analysis

            next_states.append(state)
            rewards.append(reward)
            done_flags.append(done)
            infos.append(info)

        self.current_states = next_states  # Update state tracking
        return next_states, rewards, done_flags, infos

    def reset(self):
        """ Reset all agents """
        self.current_states = [env.reset() for env in self.envs]
        return self.current_states

    def render(self, mode='human'):
        for i, env in enumerate(self.envs):
            print(f"Agent {i} State:")
            env.render(mode)

class SharedResourceCMDPWrapper(gym.Env):
    """
    A wrapper that creates a shared environment in which multiple agents
    compete for a limited shared resource.
    """
    def __init__(self, prob_y_zero=0.1, initial_resource=10, capacity_limit=1, num_agents=3, replenishment_rate=1):
        # Create one instance of the base environment.
        self.base_env = DifficultCMDPEnv(prob_y_zero=prob_y_zero)
        self.nS = self.base_env.ns
        self.nA = self.base_env.action_space.n
        self.P = self.base_env.P
        self.isd = self.base_env.isd
        self.domains = self.base_env.domains

        # Shared resource and capacity.
        self.initial_resource = initial_resource
        self.resource_level = initial_resource
        self.capacity_limit = capacity_limit  # How many agents are allowed to act in a time step.
        self.num_agents = num_agents
        self.replenishment_rate = replenishment_rate

        # Observation: each agent sees the base state and the current shared resource.
        # Here we return a single observation (common to all agents).
        self.observation_space = spaces.Dict({
            'state': self.base_env.observation_space,
            'resource': spaces.Box(low=0, high=initial_resource, shape=(1,), dtype=np.float32)
        })
        # Action: each agent chooses an action.
        # We use MultiDiscrete so that a list of actions can be passed.
        self.action_space = spaces.MultiDiscrete([self.nA] * self.num_agents)

    def decode(self, state_id):
        # Forward the decode call to the base environment.
        return self.base_env.decode(state_id)

    def reset(self):
        state = self.base_env.reset()
        self.resource_level = self.initial_resource
        observation = {
            'state': state,
            'resource': np.array([self.resource_level], dtype=np.float32)
        }
        return observation  # Shared observation for all agents

    def step(self, actions):
        """
        actions: an array (or list) of actions, one per agent.
        The training loop should already decide which agent gets to act.
        For agents with a dummy action (e.g., -1), we do nothing.
        """
        rewards = []
        infos = []
        for i in range(self.num_agents):
            action = actions[i]
            if action == -1:  # assume -1 means do nothing
                # For agents that do nothing, return current state and zero reward.
                state = self.base_env.s  # current state
                reward = 0
                done = False
                info = {'acted': False}
            else:
                state, reward, done, info = self.base_env.step(action)
                info['acted'] = True
                # Deplete the shared resource for every agent that acts.
                self.resource_level = max(0, self.resource_level - 1)
            rewards.append(reward)
            infos.append(info)
        
        # Compute total consumption (number of agents that acted)
        consumption = sum(1 for info in infos if info.get('acted', False))
        # Replenish resource after consumption.
        self.resource_level = min(self.initial_resource, self.resource_level - consumption + self.replenishment_rate)
        
        observation = {
            'state': state,  # using the latest state
            'resource': np.array([self.resource_level], dtype=np.float32)
        }
        return observation, rewards, done, infos