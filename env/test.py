import cvxpy as cv
import numpy as np

n_agents = 3
horizon = 5
resource_capacity = 2.0

columns = {
    agent_id: [np.full(horizon, 0.1)]
    for agent_id in range(n_agents)
}

decision_vars = [
    [cv.Variable(nonneg=True)]
    for _ in range(n_agents)
]

constraints = [cv.sum(decision_vars[a]) == 1 for a in range(n_agents)]

for t in range(horizon):
    total_claim = sum(decision_vars[a][0] * columns[a][0][t] for a in range(n_agents))
    constraints.append(total_claim <= resource_capacity)

objective = cv.Maximize(
    sum(decision_vars[a][0] * np.sum(columns[a][0]) for a in range(n_agents))
)

problem = cv.Problem(objective, constraints)
problem.solve(solver=cv.ECOS_BB)

print("Status:", problem.status)
print("Objective:", problem.value)
print("Decisions:", [v[0].value for v in decision_vars])
print("Duals (resource constraints):", [c.dual_value for c in constraints[n_agents:]])