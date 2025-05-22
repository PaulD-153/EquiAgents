import cvxpy as cv
import numpy as np

class MasterProblem:
    def __init__(self, agents, resource_capacity=1, fairness=False, fairness_constraint=False, langrangian=False, langrangian_weight=1.0):
        self.agents = agents
        self.horizon = agents[0].horizon
        self.num_agents = len(agents)
        self.resource_capacity = resource_capacity
        self.fairness = fairness
        self.fairness_constraint = fairness_constraint
        self.langrangian = langrangian
        self.langrangian_weight = langrangian_weight
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

        # Fairness constraints: expected rewards for each agent should be similar
        # This is a simple implementation; adjust the tolerance and method as needed to fit fairness definition
        # Note: This assumes that the expected rewards are calculated as the sum of the selected columns
        if self.fairness_constraint:
            print("Adding linear fairness constraints (per timestep)...")
            epsilon = 0.5  # tolerance

            for t in range(self.horizon):
                expected_claims_t = []
                for a in range(self.num_agents):
                    expr = 0
                    for c, column in enumerate(self.agents[a].get_columns()):
                        expr += self.decision_vars[a][c] * column["claims"][t]
                    expected_claims_t.append(expr)

                mean_claim_t = cv.sum(expected_claims_t) / self.num_agents
                for a in range(self.num_agents):
                    constraints.append(expected_claims_t[a] >= mean_claim_t - epsilon)
                    constraints.append(expected_claims_t[a] <= mean_claim_t + epsilon)
        # Objective: maximize expected total reward
        total_expected_reward = 0

        if self.langrangian:
            # Compute expected reward per agent
            print("Computing expected reward per agent using Langrangian...")
            for t in range(self.horizon):
                expected_claims_t = []
                for a in range(self.num_agents):
                    expr = 0
                    for c, column in enumerate(self.agents[a].get_columns()):
                        expr += self.decision_vars[a][c] * column["claims"][t]
                    expected_claims_t.append(expr)

                mean_claim_t = cv.sum(expected_claims_t) / self.num_agents
                diffs = cv.vstack([claim - mean_claim_t for claim in expected_claims_t])
                variance_t = cv.sum_squares(diffs)

                # Penalize deviation from equal expected claim per timestep
                total_expected_reward -= self.langrangian_weight * variance_t

        for a, agent in enumerate(self.agents):
            for c, column in enumerate(agent.get_columns()):
                reward = np.sum(column["reward"])
                cost = np.sum(agent.fixed_cost_vector * column["claims"])  # new line
                net_value = reward - cost  # optionally add a cost_weight factor
                total_expected_reward += self.decision_vars[a][c] * net_value


        if self.fairness_constraint:
            self.fairness_constraints = constraints[-1 * self.num_agents:]  # last added fairness constraints


        objective = cv.Maximize(total_expected_reward)

        self.lp = cv.Problem(objective, constraints)

        self.lp.solve(verbose=False)

        if self.fairness_constraint:
            for i, constraint in enumerate(self.fairness_constraints):
                print(f"Fairness constraint {i}: dual = {constraint.dual_value}")

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