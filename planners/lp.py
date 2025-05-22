import time
import numpy as np
import cvxpy as cv

class LinearProgrammingPlanner:
    def __init__(self,
                 transition: np.ndarray,
                 reward: np.ndarray,
                 initial_distribution: np.ndarray,
                 resource_capacity: int,
                 n_agents: int,
                 horizon: int = 3,
                 verbose: bool = False):
        """
        reward: shape (n_agents, n_resources, n_resources)
        transition: (optional) can stay (n_resources, n_resources) if deterministic
        """
        self.transition = transition  # shape: (n_resources, n_resources)
        self.reward = reward  # shape: (n_agents, n_resources, n_resources)
        self.initial_distribution = initial_distribution  # shape: (n_agents, n_resources)
        self.resource_capacity = resource_capacity
        self.n_agents = n_agents
        self.horizon = horizon
        self.verbose = verbose

        self.n_resources = transition.shape[0]
        self.flow = []
        self.lp = None
        self.time_step = 0

    def solve(self):
        t0 = time.perf_counter()
        if self.verbose:
            print("Instantiating LP...")

        self.instantiate_lp()

        if self.verbose:
            print(f"LP instantiated in {time.perf_counter() - t0:.2f} seconds")

        self.lp.solve(verbose=self.verbose)

        if self.verbose:
            print(f"LP solved in {time.perf_counter() - t0:.2f} seconds")
            print(f"Problem status: {self.lp.status}")

    def instantiate_lp(self):
        # Variables: flow[h][agent][i][j]
        self.flow = [
            [
                cv.Variable((self.n_resources, self.n_resources), nonneg=True)
                for _ in range(self.n_agents)
            ]
            for _ in range(self.horizon)
        ]

        # Objective: maximize total reward across horizon and agents
        objective = cv.Maximize(
            sum(
                cv.sum(cv.multiply(self.flow[h][agent], self.reward[agent]))
                for h in range(self.horizon)
                for agent in range(self.n_agents)
            )
        )

        constraints = []

        # Initial step: match each agent's initial distribution
        for agent in range(self.n_agents):
            for i in range(self.n_resources):
                constraints.append(cv.sum(self.flow[0][agent][i, :]) == self.initial_distribution[agent, i])

        # Flow conservation for each agent at each step
        for agent in range(self.n_agents):
            for h in range(1, self.horizon):
                for j in range(self.n_resources):
                    inflow = cv.sum(self.flow[h-1][agent][:, j])
                    outflow = cv.sum(self.flow[h][agent][j, :])
                    constraints.append(inflow == outflow)

        # Resource capacity constraint at each step
        for h in range(self.horizon):
            for j in range(self.n_resources):
                total_occupancy = cv.sum(
                    [cv.sum(self.flow[h][agent][:, j]) for agent in range(self.n_agents)]
                )
                constraints.append(total_occupancy <= self.resource_capacity)

        # Total number of agents constraint (optional, depends if needed)
        for h in range(self.horizon):
            total_agents = cv.sum(
                [cv.sum(self.flow[h][agent]) for agent in range(self.n_agents)]
            )
            constraints.append(total_agents == self.n_agents)

        self.lp = cv.Problem(objective, constraints)

    def get_flow_plan(self):
        """
        Returns a list of [horizon] elements, each a list of [n_agents] flow matrices (n_resources x n_resources)
        """
        return [
            [f.value for f in flow_h]
            for flow_h in self.flow
        ]

    def expected_reward(self) -> float:
        return self.lp.value if self.lp is not None else None

    def seed(self, seed):
        np.random.seed(seed)
