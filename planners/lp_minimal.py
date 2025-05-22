# lp_minimal.py

import time
import numpy as np
import cvxpy as cv

class MinimalFairnessPlanner:
    def __init__(self,
                 initial_distribution: np.ndarray,
                 resource_capacity: int,
                 n_agents: int,
                 horizon: int = 3,
                 verbose: bool = False):
        """
        Simple planner: Agents can claim resource; capacity limits claims per timestep.
        """
        self.initial_distribution = initial_distribution  # shape: (n_agents,)
        self.resource_capacity = resource_capacity
        self.n_agents = n_agents
        self.horizon = horizon
        self.verbose = verbose

        self.claims = []  # will hold decision variables
        self.lp = None
        self.time_step = 0

    def solve(self):
        t0 = time.perf_counter()
        if self.verbose:
            print("Instantiating minimal LP...")

        self.instantiate_lp()

        if self.verbose:
            print(f"LP instantiated in {time.perf_counter() - t0:.2f} seconds")

        self.lp.solve(verbose=self.verbose)

        if self.verbose:
            print(f"LP solved in {time.perf_counter() - t0:.2f} seconds")
            print(f"Problem status: {self.lp.status}")

    def instantiate_lp(self):
        # Variables: claims[h][agent]
        claim_var = cv.Variable(self.n_agents, nonneg=True)

        self.claims = [claim_var for _ in range(self.horizon)]
        # Objective: maximize total claimed resources
        objective = cv.Maximize(
            sum(cv.sum(self.claims[h]) for h in range(self.horizon))
        )

        constraints = []

        for h in range(self.horizon):
            # Resource constraint: total claims <= resource_capacity
            constraints.append(cv.sum(self.claims[h]) <= self.resource_capacity)
            # Agents can claim at most 1 unit per step
            constraints += [self.claims[h][i] <= 1 for i in range(self.n_agents)]

        self.lp = cv.Problem(objective, constraints)

    def get_claim_plan(self):
        """
        Returns list of claim vectors over horizon [h][agent_claims]
        """
        return [c.value for c in self.claims]

    def expected_reward(self) -> float:
        return self.lp.value if self.lp is not None else None

    def seed(self, seed):
        np.random.seed(seed)
