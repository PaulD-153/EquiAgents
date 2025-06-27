import numpy as np
from planners.lp import LinearProgrammingPlanner
from planners.lp_minimal import MinimalFairnessPlanner
import cvxpy as cv
from typing import Dict, List

np.seterr(invalid='ignore', divide='ignore')


class DecentralizedAgentWithColumns:
    def __init__(self, agent_id, horizon, resource_capacity, num_columns=5, verbose=True, reward_profile: Dict[int, tuple] = None, cost_profile: Dict[int, tuple] = None, langrangian_weight=1.0):
        self.agent_id = agent_id
        self.horizon = horizon
        self.resource_capacity = resource_capacity
        self.num_columns = num_columns
        self.verbose = verbose
        self.columns = []
        self.selected_plan = None

        self.reward_profile = reward_profile or {agent_id: (0.3, 1.0)}  # default fallback
        self.fixed_reward_vector = None
        self.cost_profile = cost_profile or {agent_id: (0.0, 1.0)}  # default fallback
        self.fixed_cost_vector = None
        self.langrangian_weight = langrangian_weight

        self.episode = 0

    def generate_candidate_columns(self):
        self.columns = []

        # Initialize reward vector ONCE per episode
        if self.fixed_reward_vector is None:
            low, high = self.reward_profile.get(self.agent_id)
            self.fixed_reward_vector = np.random.uniform(low=low, high=high, size=self.horizon)
        
        if self.fixed_cost_vector is None:
            low, high = self.cost_profile.get(self.agent_id)
            self.fixed_cost_vector = np.random.uniform(low=low, high=high, size=self.horizon)

        # Add a "do nothing" fallback column (ensures feasibility)
        zero_claim_plan = [0.0 for _ in range(self.horizon)]
        zero_reward_vector = np.zeros(self.horizon)

        self.columns.append({
            "claims": zero_claim_plan,
            "reward": zero_reward_vector
        })

        return self.columns


    def generate_new_column_based_on_feedback(self, dual_prices, fairness_duals=None):
        """
        Use dual prices to construct a new plan via LP:
        maximize sum((reward - dual) * claim)
        s.t. claim in [0, 1]
        """
        reward_vector = self.fixed_reward_vector
        cost_vector = self.fixed_cost_vector 

        # Combine fairness and congestion duals (optional)
        total_dual = np.array(dual_prices, dtype=float)
        if fairness_duals is not None:
            fairness_duals = np.array(fairness_duals, dtype=float)
            total_dual += fairness_duals  # simple additive for now

        # LP variables: claim probabilities per timestep
        claim_vars = cv.Variable(self.horizon)

        # Objective: weighted reward - dual penalty
        print(f"Agent {self.agent_id} dual prices: {total_dual}")
        if len(total_dual) != self.horizon:
            print(f"[Warning] total_dual length {len(total_dual)} != horizon {self.horizon}, defaulting to 1s.")
            total_dual = np.ones(self.horizon)
        adjusted_value = reward_vector - cost_vector * total_dual

        objective = cv.Maximize(cv.sum(cv.multiply(adjusted_value, claim_vars)))

        constraints = [
            claim_vars >= 0,
            claim_vars <= 1,
        ]

        problem = cv.Problem(objective, constraints)
        problem.solve()

        if self.verbose:
            print(f"[Agent {self.agent_id}] LP column gen status: {problem.status}")

        plan = claim_vars.value.tolist()
        plan = [max(0.0, min(1.0, p)) for p in plan]  # clip to [0,1] just in case
        print(f"Agent {self.agent_id} generated plan: {plan}")
        print(f"Agent {self.agent_id} generated reward: {reward_vector}")
        self.columns.append({
            "claims": plan,
            "reward": reward_vector  # Store reward separately
        })


        if self.verbose:
            print(f"[Agent {self.agent_id}] Generated plan with duals {dual_prices}: {plan}")



    def generate_lp_column(self, penalty_weight=0.1):
        """
        Generate a reward-maximizing initial plan using LP:
        maximize ∑ (reward_t - penalty_weight) * claim_t
        s.t. claim_t ∈ [0,1]
        """
        claim_vars = cv.Variable(self.horizon)
        
        # Get reward vector for this agent (external reward)
        low, high = self.reward_profile.get(self.agent_id)
        reward_vector = np.random.uniform(low=low, high=high, size=self.horizon)

        # Penalize claiming to avoid greedy overclaiming (encourages diversity)
        objective = cv.Maximize(cv.sum(cv.multiply(reward_vector, claim_vars)) - penalty_weight * cv.sum(claim_vars))
        
        constraints = [
            claim_vars >= 0,
            claim_vars <= 1
        ]

        problem = cv.Problem(objective, constraints)
        problem.solve()

        if self.verbose:
            print(f"[Agent {self.agent_id}] LP column gen status: {problem.status}")
            print(f"[Agent {self.agent_id}] Reward vector: {reward_vector}")
            print(f"[Agent {self.agent_id}] Generated plan: {claim_vars.value}")

        # Clip just in case of numerical error
        plan = np.clip(claim_vars.value, 0.0, 1.0).tolist()

        self.columns.append({
            "claims": plan,
            "reward": reward_vector  # Needed for expected reward computation
        })

    def generate_best_response_column(self, dual_prices, fairness_duals=None):
        reward_vector = self.fixed_reward_vector
        cost_vector = self.fixed_cost_vector

        total_dual = np.array(dual_prices, dtype=float)

        if fairness_duals is None:
            fairness_duals = np.zeros_like(total_dual)
        else:
            fairness_duals = np.array(fairness_duals, dtype=float)

        adjusted_reward = reward_vector - cost_vector * total_dual

        if self.langrangian_weight is not None and fairness_duals is not None:
            adjusted_reward -= self.langrangian_weight * fairness_duals

        claim_vars = cv.Variable(self.horizon)
        penalty = 0.05 * cv.sum_squares(claim_vars)
        objective = cv.Maximize(cv.sum(cv.multiply(adjusted_reward, claim_vars)) - penalty)
        constraints = [claim_vars >= 0, claim_vars <= 1]
        problem = cv.Problem(objective, constraints)

        # print(f"[Agent {self.agent_id}] Round RC check")
        # print(f"  cost_vector:    {np.round(cost_vector, 3)}")
        # print(f"  reward_vector:  {np.round(reward_vector, 3)}")
        # print(f"  dual_prices:    {np.round(total_dual, 3)}")
        # print(f"  fairness_grad:  {np.round(fairness_duals, 3)}")
        # print(f"  adjusted_reward:{np.round(adjusted_reward, 3)}")

        problem.solve()

        plan = np.clip(claim_vars.value, 0.0, 1.0)
        plan_tuple = tuple(np.round(plan, 5))
        existing_plan_tuples = {tuple(np.round(c['claims'], 5)) for c in self.columns}

        if plan_tuple in existing_plan_tuples:
            self.last_column_reduced_cost = 0
            if self.verbose:
                print(f"[Agent {self.agent_id}] Duplicate column detected, skipping.")
            return {"claims": plan.tolist(), "reward": self.fixed_reward_vector}

        reduced_cost = -problem.value
        column = {"claims": plan.tolist(), "reward": self.fixed_reward_vector}
        self.last_column_reduced_cost = reduced_cost
        return column

    def get_last_column_reduced_cost(self):
        return getattr(self, 'last_column_reduced_cost', 0)


    def get_columns(self):
        return self.columns
    
    def end_episode(self):
        """
        Reset or update anything needed between episodes.
        Currently nothing is needed for decentralized agents.
        """
        self.fixed_reward_vector = None  # Reset fixed reward vector if needed

    def reset(self):
        """
        Reset the agent's state.
        """
        self.columns = []
        self.selected_plan = None
