import numpy as np
import gym
import random
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
    def __init__(self, num_agents=3, initial_resource=10, capacity_limit=1, max_steps=50, replenishment_rate=1):
        # Create one instance of the base environment.
        self.num_agents = num_agents
        self.capacity_limit = capacity_limit  # How many agents are allowed to act in a time step.
        self.max_steps = max_steps
        self.env = SharedResourceMultiStepEnv(max_steps=max_steps, nS=6, nA=2,
                                                   initial_resource=initial_resource, replenishment_rate=replenishment_rate)
        
        self.current_step = 0
        self.resource_level = initial_resource
        self.observation_space = self.env.observation_space
        self.action_space = spaces.MultiDiscrete([self.env.nA] * num_agents)

    def reset(self):
        self.current_step = 0
        return self.env.reset()

    def step(self, actions):
        """
        actions: an array (or list) of actions, one per agent.
        The training loop should already decide which agent gets to act.
        For agents with a dummy action (e.g., -1), we do nothing.
        """
        next_state, rewards, done, info = self.env.step(actions)

        # If we need to enforce capacity limits, ensure only `capacity_limit` agents act at a time
        acting_agents = np.random.choice(self.num_agents, self.capacity_limit, replace=False)

        actions = [actions[agent_idx] if agent_idx in acting_agents else -1 for agent_idx in range(self.num_agents)]
        return next_state, rewards, done, info
    
    def render(self, mode='human'):
        return self.env.render(mode)

class SharedResourceMultiStepEnv(gym.Env):
    """
    A custom finite-horizon environment with a shared resource.
    The environment state is a tuple: (base_state, resource_level).
    Base state evolves according to simple deterministic dynamics.
    Each valid action (not a dummy action) consumes 1 unit of resource.
    At each step, the resource is replenished by a fixed amount.
    """
    def __init__(self, max_steps=50, nS=6, nA=2,
                 initial_resource=20, replenishment_rate=2):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0

        self.nS = nS
        self.nA = nA
        self.initial_resource = initial_resource
        self.resource_level = initial_resource
        self.replenishment_rate = replenishment_rate
    
        # Add the initial state distribution: start always at state 0.
        self.isd = np.zeros(nS)
        self.isd[0] = 1.0

        # Define a simple base state space (states 0 to nS-1)
        self.base_state_space = spaces.Discrete(nS)
        # Define the action space: valid actions 0 to nA-1.
        # We'll use a dummy action (-1) to indicate "do nothing" (i.e. not consuming resource).
        self.action_space = spaces.Discrete(nA)  # For the base state
        # For observation, we combine the base state and the resource level.
        self.observation_space = spaces.Dict({
            'state': self.base_state_space,
            'resource': spaces.Box(low=0, high=initial_resource, shape=(1,), dtype=np.float32)
        })

        # Define a simple deterministic transition function for the base state.
        # For example, action 0 moves forward by 1 (mod nS); action 1 moves forward by 2.
        self.P = {}
        for s in range(nS):
            self.P[s] = {}
            for a in range(nA):
                if a == 0:
                    next_state = (s + 1) % nS
                else:
                    next_state = (s + 2) % nS
                # Reward could be defined arbitrarily; here, say reward=1 for action 1, 0 for action 0.
                reward = 1 if a == 1 else 0
                # We don't impose terminal conditions here beyond max_steps.
                self.P[s][a] = [(1.0, next_state, reward)]
        # Start state always 0.
        self.s = 0

    def reset(self):
        self.current_step = 0
        self.resource_level = self.initial_resource
        self.s = 0
        return {'state': self.s, 'resource': np.array([self.resource_level], dtype=np.float32)}

    def step(self, actions):
        """
        In this setting, assume that the training loop (or centralized decision maker)
        provides an action for each agent. For each agent:
          - If the action is a valid action (0 or 1), it is executed and costs 1 unit of resource.
          - If the action is the dummy action (-1), it means "do nothing" and costs 0.
        For simplicity, we'll assume that only one agent's action is actually used to change the base state,
        or you can design your logic such that when multiple agents act, the base state transition is aggregated.
        Here we simply apply the first valid action encountered (if any), and ignore the others.
        The resource level is decreased by the number of valid actions and then replenished.
        """
        self.current_step += 1
        
        valid_actions = [a for a in actions if a != -1]
        num_valid = len(valid_actions)
        
        # Determine the new base state.
        # For example, if any valid action is found, we use the first one to update the state.
        if num_valid > 0:
            chosen_action = valid_actions[0]
            p, next_state, base_reward = self.P[self.s][chosen_action][0]
        else:
            # If no agent acts, no state change and zero reward.
            next_state = self.s
            base_reward = 0
        
        # Update the base state.
        self.s = next_state
        
        # Deplete the resource for every valid action.
        print(f"Resource level: {self.resource_level}, Valid actions: {num_valid}")
        self.resource_level = max(0, self.resource_level - num_valid)
        # Replenish resource.
        self.replenishment_rate = random.randint(1, 3)
        self.resource_level = min(self.initial_resource, self.resource_level + self.replenishment_rate)
        
        # Build the observation (shared by all agents).
        obs = {'state': self.s, 'resource': np.array([self.resource_level], dtype=np.float32)}
        
        # For simplicity, let's say done is True only when max_steps is reached.
        done = self.current_step >= self.max_steps
        
        # Build a reward and info per agent.
        # For the agent whose valid action was used, we return the reward;
        # others get 0. Also, mark in info whether an agent acted.
        rewards = []
        infos = []
        for a in actions:
            if a != -1:
                # Only the first valid action counted
                rewards.append(base_reward)
                infos.append({'acted': True, 'cost': 1})
                # To avoid counting multiple valid actions, change subsequent ones to dummy:
            else:
                rewards.append(0)
                infos.append({'acted': False, 'cost': 0})
        
        return obs, rewards, done, infos

    def render(self, mode='human'):
        print(f"State: {self.s}, Step: {self.current_step}, Resource: {self.resource_level}")

    def decode(self, state_id):
        return [state_id]
