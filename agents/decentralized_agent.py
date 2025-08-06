import numpy as np
from planners.lp import LinearProgrammingPlanner
from planners.lp_minimal import MinimalFairnessPlanner
import cvxpy as cv
from typing import Dict, List

np.seterr(invalid='ignore', divide='ignore')


class DecentralizedAgentWithColumns:
    def __init__(self,
                 agent_id,
                 horizon,
                 verbose=True,
                 reward_profile: Dict[int, tuple] = None,
                 cost_profile: Dict[int, tuple] = None,
                 langrangian_weight=1.0,
                 beta: float = 2.0,
                 cost_weight: float = 1.0):
        self.agent_id = agent_id
        self.horizon = horizon
        self.verbose = verbose
        self.columns = []
        self.selected_plan = None

        # SL trajectory + capacity schedule will be injected by the trainer
        self.SL_traj = []           
        self.capacity_schedule = []

        self.reward_profile = reward_profile or {agent_id: (0.3, 1.0)}  # default fallback
        self.fixed_reward_vector = None
        self.cost_profile = cost_profile or {agent_id: (0.0, 1.0)}
        # β controls surge‐pricing: cost multiplier = 1 + β·1_{sL==2}
        self.beta = beta
        self.langrangian_weight = langrangian_weight
        self.cost_weight = cost_weight  # α in the paper, used to scale costs

        # Provide default state‐dependent reward/cost functions
        self.reward_fn = lambda t, sL: np.random.uniform(*self.reward_profile[self.agent_id])
        self.cost_fn   = lambda t, sL: np.random.uniform(*self.cost_profile[self.agent_id]) \
                                      * (1.0 + self.beta * (sL == 2))
 

        self.episode = 0

    def generate_candidate_columns(self):
        self.columns = []

        # Initialize reward vector ONCE per episode
        if self.fixed_reward_vector is None:
            low, high = self.reward_profile.get(self.agent_id)
            self.fixed_reward_vector = np.random.uniform(low=low, high=high, size=self.horizon)
        
        # 1) do‐nothing column (always safe, zero cost/reward)
        zero = [0.0]*self.horizon
        self.columns.append({"claims": zero, "reward": np.zeros(self.horizon)})

        # 2) greedy “claim whenever it’s profitable” column
        #    compare reward_fn(t,sL) vs. cost_weight * cost_fn(t,sL)
        greedy = []
        for t in range(self.horizon):
            r_t = self.reward_fn(t, self.SL_traj[t]) if self.SL_traj else np.mean(self.reward_profile[self.agent_id])
            c_t = self.cost_fn(t, self.SL_traj[t]) if self.SL_traj else np.mean(self.cost_profile[self.agent_id])
            # claim only if expected net reward positive
            greedy.append(1.0 if r_t - self.cost_weight*c_t > 0 else 0.0)
        self.columns.append({"claims": greedy, "reward": self.fixed_reward_vector})


        return self.columns

    def generate_best_response_column(self, dual_prices, fairness_duals=None):
        # Build per‐timestep reward & cost based on SL_traj if available,
        # otherwise fall back to a random draw from the profile.
        reward_vector = np.array([
            self.reward_fn(t, self.SL_traj[t]) if len(self.SL_traj) == self.horizon
            else np.random.uniform(*self.reward_profile[self.agent_id])
            for t in range(self.horizon)
        ])
        # Build the cost vector on the fly, using surge-pricing via cost_fn
        cost_vector = np.array([ self.cost_fn(t, self.SL_traj[t])
                                 for t in range(self.horizon) ])
        total_dual = np.array(dual_prices, dtype=float)

        if fairness_duals is None:
            fairness_duals = np.zeros_like(total_dual)
        else:
            fairness_duals = np.array(fairness_duals, dtype=float)

        # price out each timestep by its dual λ_t, *and* scale cost by agent’s α
        alpha = getattr(self, "cost_weight", 1.0)
        adjusted_reward = (
            reward_vector
            - cost_vector * total_dual      # congestion price
            - alpha       * cost_vector      # direct cost penalty
            )

            # scale fairness_duals by langrangian_weight
        if self.langrangian_weight and fairness_duals is not None:
            adjusted_reward -= fairness_duals

        claim_vars = cv.Variable(self.horizon)

        # Regular L2 penalty to discourage aggressive plans
        penalty = 0.05 * cv.sum_squares(claim_vars)
        objective = cv.Maximize(cv.sum(cv.multiply(adjusted_reward, claim_vars)) - penalty)
        constraints = [claim_vars >= 0, claim_vars <= 1]

        # Objective: reward - L2 penalty - KL regularization
        objective = cv.Maximize(cv.sum(cv.multiply(adjusted_reward, claim_vars)) - penalty)
        constraints = [claim_vars >= 0, claim_vars <= 1.0]
        problem = cv.Problem(objective, constraints)

        print(f"[Agent {self.agent_id}] Round RC check")
        print(f"  cost_vector:    {np.round(cost_vector, 3)}")
        print(f"  reward_vector:  {np.round(reward_vector, 3)}")
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
