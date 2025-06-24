import cvxpy as cv
import numpy as np
from util.fairness_penalties import (
    variance_penalty, jain_index_penalty, nsw_penalty, minshare_penalty, gini_penalty, envy_scaled_penalty
)


class MasterProblem:
    def __init__(self, agents, resource_capacity=1, fairness=False, fairness_constraint=False, langrangian=False, langrangian_weight=1.0, fairness_type="variance", reward_scaling=None):
        self.agents = agents
        self.horizon = agents[0].horizon
        self.num_agents = len(agents)
        self.resource_capacity = resource_capacity
        self.fairness = fairness
        self.fairness_constraint = fairness_constraint
        self.langrangian = langrangian
        self.langrangian_weight = langrangian_weight
        self.fairness_type = fairness_type
        self.reward_scaling = reward_scaling if reward_scaling is not None else [1.0] * self.num_agents
        self.decision_vars = []
        self.lp = None
        self.resource_constraints = []  # Save constraints to access duals
        self.fairness_constraints = []  # Save fairness constraints to access duals

    def solve(self):
        constraints = []

        self.decision_vars = []
        
        # Create decision variables (nonnegative for stochastic plan selection)
        for agent in self.agents:
            agent_vars = []
            for _ in agent.get_columns():
                var = cv.Variable(nonneg=True)  # Stochastic: non-negative fractional
                agent_vars.append(var)
            self.decision_vars.append(agent_vars)

        # Each agent must select or probabilistic combination of columns
        for a in range(self.num_agents):
            constraints.append(cv.sum(self.decision_vars[a]) == 1)

        self.resource_constraints = []
        for t in range(self.horizon):
            expected_total_claims = 0
            for a, agent in enumerate(self.agents):
                columns = agent.get_columns()
                for c, column in enumerate(columns):
                    expected_total_claims += self.decision_vars[a][c] * column["claims"][t]
            constraint = (expected_total_claims <= self.resource_capacity)
            constraints.append(constraint)
            self.resource_constraints.append(constraint)

        # Objective: maximize expected total reward
        total_expected_reward = 0

        if self.langrangian:
            for t in range(self.horizon):
                expected_claims_t = []
                for a in range(self.num_agents):
                    expr = 0
                    for c, column in enumerate(self.agents[a].get_columns()):
                        expr += self.decision_vars[a][c] * column["claims"][t]
                    expected_claims_t.append(expr)

                # Choose penalty based on selected fairness_type
                if self.fairness_type == "variance":
                    fairness_penalty = variance_penalty(expected_claims_t)
                elif self.fairness_type == "jain":
                    fairness_penalty = jain_index_penalty(expected_claims_t)
                elif self.fairness_type == "nsw":
                    fairness_penalty = nsw_penalty(expected_claims_t)
                elif self.fairness_type == "minshare":
                    fairness_penalty = minshare_penalty(expected_claims_t)
                elif self.fairness_type == "gini":
                    fairness_penalty = gini_penalty(expected_claims_t)
                elif self.fairness_type == "envy_scaled":
                    if self.reward_scaling is None:
                        raise ValueError("Reward scaling must be provided for envy_scaled penalty.")
                    fairness_penalty = envy_scaled_penalty(expected_claims_t, self.reward_scaling)
                else:
                    raise ValueError(f"Unknown fairness type: {self.fairness_type}")

                total_expected_reward -= self.langrangian_weight * fairness_penalty

        for a, agent in enumerate(self.agents):
            for c, column in enumerate(agent.get_columns()):
                reward = np.sum(column["reward"])
                cost = np.sum(agent.fixed_cost_vector * column["claims"])  # new line
                net_value = reward - cost  # optionally add a cost_weight factor
                total_expected_reward += self.decision_vars[a][c] * net_value

        objective = cv.Maximize(total_expected_reward)
        self.lp = cv.Problem(objective, constraints)
        self.lp.solve(verbose=False)

        print("Master LP Status:", self.lp.status)
        print("Master LP Objective Value:", self.lp.value)

        return self.lp.value, self.get_decision_distribution()

    def get_dual_prices(self):
        return np.array([
            c.dual_value if c.dual_value is not None else 0.0
            for c in self.resource_constraints
        ])

    def get_fairness_duals(self):
        if not self.fairness_constraint:
            return None
        return np.array([
            c.dual_value if c.dual_value is not None else 1.0
            for c in self.fairness_constraints
        ])


    def get_decision_distribution(self):
        """
        Returns the full distribution over columns for each agent.
        """
        distributions = []
        for a_vars in self.decision_vars:
            weights = np.array([var.value if var.value is not None else 0.0 for var in a_vars])
            if np.sum(weights) == 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights /= np.sum(weights)
            distributions.append(weights)
        return distributions