import numpy as np
import gym
import random
from gym import spaces


class MultiAgentEnv(gym.Env):
    """
    A custom finite-horizon environment with a shared resource.
    The environment state is a tuple: (base_state).
    Base state evolves according to simple deterministic dynamics.
    Each valid action (not a dummy action) consumes 1 unit of resource.
    At each step, the resource is replenished by a fixed amount.
    """
    def __init__(self, num_agents, max_steps=50, nS=6, resource_pool=None):
        super().__init__()
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0

        self.nS = nS
        self.resource_pool = resource_pool
    
        # Add the initial state distribution: start always at state 0.
        self.isd = np.zeros(nS)
        self.isd[0] = 1.0

        # Define a simple base state space (states 0 to nS-1)
        self.base_state_space = spaces.Discrete(nS)
        
        # Define the action space: valid actions 0 to nA-1.
        total_resources = sum(len(v) for v in self.resource_pool.values())
        self.action_space = spaces.Discrete(total_resources)

        self.nA = total_resources
        self.P = {}
        for s in range(nS):
            self.P[s] = {}
            for a in range(self.nA):
                # For a dummy transition, you could simply move to the next state or keep the state constant.
                next_state = (s + 1) % nS  # or even: next_state = s
                # You might want reward = 0 here if state transitions do not contribute to reward.
                reward = 0
                self.P[s][a] = [(1.0, next_state, reward)]

        # Start state always 0.
        self.s = 0

        # These will be set later
        self.agent_types = None
        self.visited = None

    def reset(self):
        self.current_step = 0
        self.s = 0
        # Initialize visited resources per agent as an empty set.
        self.visited = {i: set() for i in range(self.num_agents)}
        return {'state': self.s}

    def step(self, actions):
        # For each agent, interpret the action as selecting a resource from a flat list.
        rewards = []
        infos = []
        
        for agent_idx, action in enumerate(actions):
            # Map action index to a (resource_type, resource_id)
            resource_type, resource_id = self.map_action_to_resource(action)
            
            # Assign reward based on agent-resource type matching
            agent_type = self.agent_types[agent_idx] if self.agent_types is not None else None
            if agent_type is not None and agent_type == resource_type:
                reward = 10  # higher reward for a match
            else:
                reward = 5   # lower reward otherwise


            # Check if this resource was already visited.
            already_visited = self.has_visited(agent_idx, resource_type, resource_id)
            info = {'visited_before': self.visited[agent_idx], 'cost': 1}

            # Optionally, you might want to penalize if already visited.
            if already_visited:
                reward = 0  # or apply another penalty
            
            # Mark the resource as visited.
            self.mark_visited(agent_idx, resource_type, resource_id)
            
            rewards.append(reward)
            infos.append(info)
        
        # Update the base state – here, we simply increment the state (as a placeholder).
        new_state = self.compute_new_state()
        done = self.current_step >= self.max_steps
        self.current_step += 1
        return new_state, rewards, done, infos
    
    def compute_new_state(self):
        """
        A simple example: move to the next state modulo nS.
        """
        self.s = (self.s + 1) % self.nS
        return {'state': self.s}


    def render(self, mode='human'):
        print(f"State: {self.s}, Step: {self.current_step}")

    def decode(self, state_id):
        return [state_id]
    
    def map_action_to_resource(self, action):
        """
        Map the action index to a resource type and resource id.
        """
        total_resources = sum(len(v) for v in self.resource_pool.values())
        resource_type = None
        resource_id = None
        if 0 <= action < total_resources:
            for rtype, resources in self.resource_pool.items():
                if action < len(resources):
                    resource_type = rtype
                    resource_id = resources[action]
                    break
                action -= len(resources)
        return resource_type, resource_id
    
    def mark_visited(self, agent_idx, resource_type, resource_id):
        """
        Record that the given agent has visited the specified resource.
        """
        if self.visited is None:
            self.visited = {i: set() for i in range(self.num_agents)}
        self.visited[agent_idx].add((resource_type, resource_id))

    def has_visited(self, agent_idx, resource_type, resource_id):
        """
        Check if the given agent has already visited the specified resource.
        """
        if self.visited is None:
            return False
        return (resource_type, resource_id) in self.visited[agent_idx]