import cvxpy as cv
import numpy as np
from util.fairness_penalties import (
    variance_penalty, variance_penalty_gradient, variance_penalty_numpy
)

class MasterProblem:
    """
    Central LP planner coordinating agent policies under resource and fairness constraints.
    Supports Lagrangian fairness (variance-based) and computes gradients if needed.
    """
    def __init__(self, agents, resource_capacity=1, langrangian_weight=1.0,
                 fairness_type="variance", fairness_scope="timestep",
                 capacity_schedule=None, use_gradient_fairness=False):
        self.agents = agents
        self.horizon = agents[0].horizon
        self.num_agents = len(agents)
        self.resource_capacity = resource_capacity
        self.langrangian_weight = langrangian_weight
        self.fairness_type = fairness_type
        self.fairness_scope = fairness_scope  # "timestep" or "cumulative"
        self.capacity_schedule = capacity_schedule  # list of length horizon
        self.use_gradient_fairness = use_gradient_fairness

        self.decision_vars = []
        self.lp = None
        self.resource_constraints = []

    def solve(self):
        """Builds and solves the master LP. Returns (objective_value, distribution, fairness_impact)."""
        constraints = []
        self.decision_vars = []

        # Create decision variables
        for agent in self.agents:
            agent_vars = [cv.Variable(nonneg=True) for _ in agent.get_columns()]
            self.decision_vars.append(agent_vars)

        # Constraint: each agent must choose a valid probability distribution
        for a in range(self.num_agents):
            constraints.append(cv.sum(self.decision_vars[a]) == 1)

        # Constraint: total expected claim ≤ capacity (per timestep)
        self.resource_constraints = []
        for t in range(self.horizon):
            expected_total_claims = sum(
                self.decision_vars[a][c] * column["claims"][t]
                for a, agent in enumerate(self.agents)
                for c, column in enumerate(agent.get_columns())
            )
            cap_t = self.capacity_schedule[t] if self.capacity_schedule else self.resource_capacity
            constraint = expected_total_claims <= cap_t
            constraints.append(constraint)
            self.resource_constraints.append(constraint)

        # Objective: maximize total expected reward (including fairness penalty)
        total_expected_reward = 0

        if self.langrangian_weight > 0:
            if self.fairness_scope == "timestep":
                for t in range(self.horizon):
                    expected_claims_t = [
                        sum(self.decision_vars[a][c] * column["claims"][t]
                            for c, column in enumerate(self.agents[a].get_columns()))
                        for a in range(self.num_agents)
                    ]
                    if self.fairness_type == "variance":
                        fairness_penalty = variance_penalty(expected_claims_t)
                        total_expected_reward -= self.langrangian_weight * fairness_penalty
            elif self.fairness_scope == "cumulative":
                expected_cumulative_claims = [
                    sum(self.decision_vars[a][c] * sum(column["claims"])
                        for c, column in enumerate(self.agents[a].get_columns()))
                    for a in range(self.num_agents)
                ]
                if self.fairness_type == "variance":
                    fairness_penalty = variance_penalty(expected_cumulative_claims)
                    total_expected_reward -= self.langrangian_weight * fairness_penalty
            else:
                raise ValueError("Unknown fairness_scope option")

        # Add reward - cost terms
        for a, agent in enumerate(self.agents):
            for c, column in enumerate(agent.get_columns()):
                R = np.sum(column["reward"])
                C = sum(agent.cost_fn(t, agent.SL_traj[t]) * column["claims"][t]
                        for t in range(self.horizon))
                alpha = getattr(agent, "cost_weight", 1.0)
                net = R - alpha * C
                total_expected_reward += self.decision_vars[a][c] * net

        self.lp = cv.Problem(cv.Maximize(total_expected_reward), constraints)
        self.lp.solve(verbose=False, solver="ECOS")

        fairness_penalty_realized = self.compute_realized_fairness_penalty()
        fairness_impact = self.langrangian_weight * fairness_penalty_realized

        return self.lp.value, self.get_decision_distribution(), fairness_impact

    def compute_fairness_gradients(self):
        """Computes per-agent gradient of the fairness penalty (used for best-response planning)."""
        gradients_per_agent = []

        if self.fairness_scope == "timestep":
            gradients = []
            for t in range(self.horizon):
                expected_claims_t = [
                    sum(self.decision_vars[a][c].value * column["claims"][t]
                        for c, column in enumerate(self.agents[a].get_columns()))
                    for a in range(self.num_agents)
                ]
                if self.fairness_type == "variance":
                    grad_t = variance_penalty_gradient(expected_claims_t)
                else:
                    raise NotImplementedError("Only variance fairness implemented in gradient.")
                gradients.append(grad_t)

            for a in range(self.num_agents):
                agent_grad = np.array([gradients[t][a] for t in range(self.horizon)])
                gradients_per_agent.append(agent_grad)

        elif self.fairness_scope == "cumulative":
            expected_cumulative_claims = [
                sum(self.decision_vars[a][c].value * sum(column["claims"])
                    for c, column in enumerate(self.agents[a].get_columns()))
                for a in range(self.num_agents)
            ]
            if self.fairness_type == "variance":
                grad_cumulative = variance_penalty_gradient(expected_cumulative_claims)
            else:
                raise NotImplementedError("Only variance fairness implemented in gradient.")

            for a in range(self.num_agents):
                gradients_per_agent.append(np.array([grad_cumulative[a]] * self.horizon))

        else:
            raise ValueError("Unknown fairness_scope option.")

        return gradients_per_agent

    def get_dual_prices(self):
        """Return the dual values of the resource constraints (used for pricing)."""
        return np.array([
            c.dual_value if c.dual_value is not None else 0.0
            for c in self.resource_constraints
        ])

    def get_decision_distribution(self):
        """Return per-agent distribution over columns (normalized decision variable values)."""
        distributions = []
        for a_vars in self.decision_vars:
            weights = np.array([var.value if var.value is not None else 0.0 for var in a_vars])
            if weights.sum() == 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights /= weights.sum()
            distributions.append(weights)
        return distributions

    def compute_realized_fairness_penalty(self):
        """Evaluate fairness penalty numerically using optimized variable values."""
        if self.fairness_scope == "timestep":
            penalty = 0.0
            for t in range(self.horizon):
                claims_t = [
                    sum(self.decision_vars[a][c].value * column["claims"][t]
                        for c, column in enumerate(self.agents[a].get_columns()))
                    for a in range(self.num_agents)
                ]
                penalty += variance_penalty_numpy(claims_t)
            return penalty

        elif self.fairness_scope == "cumulative":
            claims = [
                sum(self.decision_vars[a][c].value * np.sum(column["claims"])
                    for c, column in enumerate(self.agents[a].get_columns()))
                for a in range(self.num_agents)
            ]
            return variance_penalty_numpy(claims)

        else:
            raise ValueError("Unknown fairness_scope.")
