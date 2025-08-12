import numpy as np
import cvxpy as cv
from typing import Dict

class DecentralizedAgentWithColumns:
    def __init__(self,
                 agent_id: int,
                 horizon: int,
                 verbose: bool = True,
                 reward_profile: Dict[int, tuple] = None,
                 cost_profile: Dict[int, tuple] = None,
                 langrangian_weight: float = 1.0,
                 beta: float = 2.0,
                 cost_weight: float = 1.0):
        self.agent_id = agent_id
        self.horizon = horizon
        self.verbose = verbose
        self.langrangian_weight = langrangian_weight
        self.beta = beta
        self.cost_weight = cost_weight

        self.reward_profile = reward_profile or {agent_id: (0.3, 1.0)}
        self.cost_profile = cost_profile or {agent_id: (0.0, 1.0)}
        self.reward_fn = lambda t, sL: np.random.uniform(*self.reward_profile[self.agent_id])
        self.cost_fn = lambda t, sL: np.random.uniform(*self.cost_profile[self.agent_id]) * (1.0 + self.beta * (sL == 2))

        self.columns = []
        self.selected_plan = None
        self.fixed_reward_vector = None
        self.SL_traj = []
        self.capacity_schedule = []
        self.episode = 0
        self.last_column_reduced_cost = 0.0

    def generate_candidate_columns(self):
        self.columns = []

        if self.fixed_reward_vector is None:
            low, high = self.reward_profile[self.agent_id]
            self.fixed_reward_vector = np.random.uniform(low=low, high=high, size=self.horizon)

        self.columns.append({"claims": [0.0] * self.horizon, "reward": np.zeros(self.horizon)})

        greedy_claims = []
        for t in range(self.horizon):
            r = self.reward_fn(t, self.SL_traj[t]) if self.SL_traj else np.mean(self.reward_profile[self.agent_id])
            c = self.cost_fn(t, self.SL_traj[t]) if self.SL_traj else np.mean(self.cost_profile[self.agent_id])
            greedy_claims.append(1.0 if r - self.cost_weight * c > 0 else 0.0)

        self.columns.append({"claims": greedy_claims, "reward": self.fixed_reward_vector})
        return self.columns

    def generate_best_response_column(self, dual_prices, fairness_duals=None):
        reward_vec = np.array([
            self.reward_fn(t, self.SL_traj[t]) if len(self.SL_traj) == self.horizon
            else np.mean(self.reward_profile[self.agent_id])
            for t in range(self.horizon)
        ])
        cost_vec = np.array([self.cost_fn(t, self.SL_traj[t]) for t in range(self.horizon)])
        dual_vec = np.array(dual_prices, dtype=float)
        fairness_duals = np.zeros_like(dual_vec) if fairness_duals is None else np.array(fairness_duals, dtype=float)

        adjusted_reward = reward_vec - dual_vec * cost_vec - self.cost_weight * cost_vec
        if self.langrangian_weight:
            adjusted_reward -= fairness_duals

        claim_vars = cv.Variable(self.horizon)
        penalty = 0.05 * cv.sum_squares(claim_vars)
        objective = cv.Maximize(cv.sum(cv.multiply(adjusted_reward, claim_vars)) - penalty)
        constraints = [claim_vars >= 0.0, claim_vars <= 1.0]

        problem = cv.Problem(objective, constraints)
        problem.solve()

        plan = np.clip(claim_vars.value, 0.0, 1.0)
        rounded_plan = tuple(np.round(plan, 5))
        existing_plans = {tuple(np.round(c["claims"], 5)) for c in self.columns}

        if rounded_plan in existing_plans:
            self.last_column_reduced_cost = 0.0
            if self.verbose:
                print(f"[Agent {self.agent_id}] Duplicate column detected, skipping.")
            return {"claims": plan.tolist(), "reward": self.fixed_reward_vector}

        self.last_column_reduced_cost = -problem.value

        if self.verbose:
            print(f"[Agent {self.agent_id}] Generated new column with reduced cost = {-problem.value:.4f}")

        return {"claims": plan.tolist(), "reward": self.fixed_reward_vector}

    def get_columns(self):
        return self.columns

    def get_last_column_reduced_cost(self):
        return self.last_column_reduced_cost

    def end_episode(self):
        self.fixed_reward_vector = None

    def reset(self):
        self.columns = []
        self.selected_plan = None
