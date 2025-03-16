import numpy as np

from planners.lp import LinearProgrammingPlanner
from util.mdp import get_mdp_functions
from util.mdp import get_mdp_functions_partial
from env.multi_agent_env import MultiAgentEnv

np.seterr(invalid='ignore', divide='ignore')


class AbsOptCMDPAgent:
    def __init__(self,
                 ns: int,
                 na: int,
                 terminal: np.array,
                 isd: np.array,
                 env,
                 max_reward, min_reward,
                 horizon=3,
                 cost_bound=None,
                 policy_type='ground',
                 cost_bound_coefficient=1,
                 solver='grb',  # grb, cvxpy
                 verbose=False,
                 reward_scale=1.0,
                 cost_scale=1.0,
                 agent_id=0,
                 agent_type=None,
                 lambda_penalty=0.0):

        self.ns, self.na = ns, na
        self.terminal = terminal
        self.isd = isd
        self.env = env
        self.horizon = horizon
        self.cost_bound = cost_bound
        self.policy_type = policy_type
        self.cost_bound_coefficient = cost_bound_coefficient
        self.verbose = verbose

        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.agent_id = agent_id
        self.agent_type = agent_type

        self.episode = 0
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.0
        self.epsilon_decay = 0.004

        self.visited_resources = []

        self.max_reward = max(max_reward, 0)
        self.min_reward = min_reward
        if terminal.any():
            self.max_reward = max(self.max_reward, 0)

        self.estimated_transition = np.full((ns, na, ns), fill_value=1/ns)
        self.estimated_reward = np.full((ns, na), fill_value=self.max_reward)
        self.ensure_terminal_states_are_absorbing()

        self.counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.new_counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.acc_reward = np.zeros((ns, na))

        self.lambda_penalty = lambda_penalty

        self.solver = solver
        self.planner = self.instantiate_planner()

        # computing initial policy
        self.planner.solve()
        

    def instantiate_planner(self):
        if (self.counter_sas > 0).any():
            inverse_counter = 1 / np.maximum(self.counter_sas.sum(axis=2), 1)
            var_transition = self.estimated_transition * (1 - self.estimated_transition)

            transition_ci = np.sqrt(var_transition * inverse_counter[:, :, np.newaxis]) + inverse_counter[:, :, np.newaxis]
            transition_ci[self.terminal] = 0

            if False:
                # this is the theoretical upper-bound on the confidence interval
                reward_ci = np.sqrt(inverse_counter) * (self.max_reward - self.min_reward)
            else:
                reward_ci = inverse_counter * (self.max_reward - self.min_reward)
            reward_ci[self.terminal] = 0
        else:
            transition_ci=np.full((self.ns, self.na, self.ns), fill_value=1.0)
            reward_ci=np.full((self.ns, self.na), fill_value=self.max_reward - self.min_reward)

        return LinearProgrammingPlanner(
            self.estimated_transition, self.estimated_reward, None, self.terminal, self.isd,
            self.env, self.max_reward, self.min_reward, 0, 0,
            cost_bound=self.cost_bound,
            horizon=self.horizon,
            verbose=self.verbose,
            solver=self.solver
        )

    @classmethod
    def from_discrete_env(cls, env: MultiAgentEnv, agent_id=0, **kwargs):
        if hasattr(env, 'envs'):
            agent_env = env.envs[agent_id]
        elif hasattr(env, 'env'):
            agent_env = env.env
        else:
            agent_env = env
        print("Printing agent environment inside agent:", agent_env)
        transition, reward, _, terminal = get_mdp_functions(agent_env)
        
        # Save original terminal mask and override it.
        original_terminal = terminal.copy()
        terminal = np.zeros_like(terminal, dtype=bool)
        for s in range(agent_env.nS):
            if original_terminal[s]:
                transition[s, :, :] = 0
                transition[s, :, s] = 1
                reward[s, :] = 0  # originally set to zero
        # Instead of leaving rewards at zero, update them based on the agent's type.
        agent_type = kwargs.get('agent_type', None)
        for s in range(agent_env.nS):
            for a in range(agent_env.nA):
                res_type, res_id = agent_env.map_action_to_resource(a)
                if agent_type is not None:
                    # Set a higher baseline reward for a match.
                    if agent_type == res_type:
                        reward[s, a] = 10
                    else:
                        reward[s, a] = 5
                else:
                    reward[s, a] = 5  # default reward if agent_type is not set.

        max_reward, min_reward = reward.max(), reward.min()
        
        kwargs.pop('features', None)
        
        return cls(
            agent_env.nS, agent_env.nA, terminal, agent_env.isd, agent_env,
            max_reward, min_reward, **kwargs, agent_id=agent_id
        )

    def ensure_terminal_states_are_absorbing(self):
        for s in np.arange(self.ns)[self.terminal]:
            self.estimated_transition[s, :, :] = 0
            self.estimated_transition[s, :, s] = 1
            self.estimated_reward[s, :] = 0

    def act(self, state, evaluation=False):
        """ This act method does not use the self.planner.act() function!"""
    # Compute decaying epsilon: it decays as the number of episodes increases.
        epsilon = max(self.min_epsilon, self.initial_epsilon / (1 + self.episode * self.epsilon_decay))

        # If in evaluation mode, set epsilon to zero.
        if evaluation:
            epsilon = 0.0

        # With probability epsilon, choose a random action from available ones.
        if self.planner.rng.random() < epsilon:
            available = [a for a in range(self.planner.na) if a not in self.visited_resources]
            if available:
                action = self.planner.rng.choice(available)
            else:
                action = self.planner.rng.choice(list(range(self.planner.na)))
        else:
            # Otherwise, use the LP-derived policy:
            probabilities = self.planner.policy[self.planner.time_step][state].copy()
            # Filter out visited actions:
            for a in self.visited_resources:
                probabilities[a] = 0
            total = probabilities.sum()
            if total > 0:
                probabilities /= total
            else:
                probabilities = np.full(self.planner.na, 1.0 / self.planner.na)
            action = self.planner.rng.choice(list(range(self.planner.na)), p=probabilities)
        
        # Record the chosen action as visited:
        self.visited_resources.append(action)
        # Increment the time step if not at the last one.
        if self.planner.time_step < self.horizon - 1:
            self.planner.time_step += 1
        return action

    def add_transition(self, state, reward, action, next_state, done, info=None):
        self.acc_reward[state, action] += reward
        self.new_counter_sas[state, action, next_state] += 1
        self.planner.add_transition(state, reward, action, next_state, done, info)

    def end_episode(self, evaluation=False):
        if not evaluation:
            self.aggregate_new_samples()
            self.update_estimate()
            self.update_planner()
            self.planner.solve()
        self.visited_resources = []
        self.planner.end_episode()
        self.episode += 1

    def enough_new_samples_collected(self):
        return (self.new_counter_sas > self.counter_sas).any()

    def aggregate_new_samples(self):
        self.counter_sas += self.new_counter_sas
        self.new_counter_sas.fill(0)

    def update_estimate(self):
        counter_sa = np.maximum(self.counter_sas.sum(axis=2), 1)
        alpha = 0.1
        new_estimated_reward = (self.acc_reward / counter_sa) * self.reward_scale
        self.estimated_reward = (1 - alpha) * self.estimated_reward + alpha * new_estimated_reward
        self.estimated_transition = self.counter_sas / counter_sa[:, :, np.newaxis]
        self.ensure_terminal_states_are_absorbing()

    def update_planner(self):
        self.planner = self.instantiate_planner()

    def expected_value(self, _) -> float:
        return self.planner.expected_value(self.isd)

    def get_expected_cost(self) -> float:
        return self.planner.get_expected_cost()

    def seed(self, seed):
        self.planner.seed(seed)

    def get_expected_reward(self, state):
        """
        Compute the expected reward at a given state.
        `state` should be a single state index (e.g. an integer).
        """
        # Ensure that the planner's policy is available.
        # We assume that self.planner.policy is an array of shape (horizon, ns, na)
        try:
            policy_at_state = self.planner.policy[self.planner.time_step][state]
        except Exception as e:
            print("Error accessing policy at time step", self.planner.time_step, "state", state, ":", e)
            return 0.0
        # Compute the expected reward as the dot-product between the policy and the estimated rewards.
        expected_reward = np.dot(policy_at_state, self.estimated_reward[state])
        return expected_reward


