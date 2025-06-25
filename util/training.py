import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from planners.master_problem import MasterProblem


def compute_fairness_score(values, fairness_type):
    if fairness_type == "jain":
        numerator = (np.sum(values)) ** 2
        denominator = len(values) * np.sum(values ** 2)
        return numerator / (denominator + 1e-8)
    elif fairness_type == "nsw":
        return np.exp(np.mean(np.log(values + 1e-6)))
    elif fairness_type == "minshare":
        return np.min(values)
    elif fairness_type == "gini":
        sorted_vals = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        gini_coeff = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals) + 1e-6) - (n + 1) / n
        return 1.0 - gini_coeff
    elif fairness_type == "variance":
        mean = np.mean(values)
        variance = np.mean((values - mean) ** 2)
        return -variance
    else:
        raise ValueError(f"Unknown fairness type: {fairness_type}")

def evaluate_all_fairness(values, fairness_metrics):
    scores = {}
    for fairness_type in fairness_metrics:
        score = compute_fairness_score(values, fairness_type)
        scores[fairness_type] = score
    return scores

def train_agents_with_dynamic_master(env, agents, number_of_episodes, max_column_generation_rounds=5, verbose=True,
                                      fairness=False, langrangian=False, langrangian_weight=None,
                                      fairness_type='variance', seed=None, fairness_metrics=None):

    metrics = {'master_value': [], 'expected_return': [], 'optimal usage': [], 'min_rc': []}
    min_rc_history = []

    for fairness_eval in fairness_metrics:
        metrics[f'expected_fairness_{fairness_eval}'] = []

    for agent in agents:
        agent.reset()

    for episode in range(number_of_episodes):
        rc_per_episode = []
        env.reset()
        for agent in agents:
            agent.columns = agent.generate_candidate_columns()

        round_idx = 0
        while True:
            master = MasterProblem(agents, resource_capacity=env.resource_capacity,
                                    fairness=fairness,
                                    langrangian=langrangian, langrangian_weight=langrangian_weight)
            master_value, _ = master.solve()

            if master.lp.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Master LP not feasible at round {round_idx}")
                break

            if langrangian:
                fairness_gradients_per_agent = master.compute_fairness_gradients()
            else:
                fairness_gradients_per_agent = [np.zeros(env.max_steps) for _ in range(len(agents))]

            dual_prices = master.get_dual_prices()

            new_columns = []
            for a_idx, agent in enumerate(agents):
                fairness_dual = fairness_gradients_per_agent[a_idx]
                column = agent.generate_best_response_column(dual_prices, fairness_duals=fairness_dual)
                reduced_cost = agent.get_last_column_reduced_cost()
                new_columns.append((column, reduced_cost))

            reduced_costs = [rc for _, rc in new_columns]
            min_rc = min(reduced_costs)
            print(f"Round {round_idx}: min reduced cost {min_rc:.6f}")

            rc_per_episode.append(min_rc)

            if min_rc > -1e-2 or round_idx >= max_column_generation_rounds:
                column_distributions = master.get_decision_distribution()
                break

            for agent, (column, rc) in zip(agents, new_columns):
                agent.columns.append(column)

            round_idx += 1

        # Compute expected reward (ignoring fairness penalty)
        total_expected_reward = 0
        for a, agent in enumerate(agents):
            for c, column in enumerate(agent.columns):
                reward = np.sum(column["reward"])
                cost = np.sum(agent.fixed_cost_vector * column["claims"])
                prob = column_distributions[a][c]
                net_value = reward - cost
                total_expected_reward += prob * net_value

        # Compute expected claims for fairness metrics
        expected_claims_per_timestep = np.zeros((env.max_steps, len(agents)))
        for t in range(env.max_steps):
            for a_idx, agent in enumerate(agents):
                for i, col in enumerate(agent.columns):
                    prob = column_distributions[a_idx][i]
                    expected_claims_per_timestep[t, a_idx] += prob * col["claims"][t]

        # Save histogram after last episode
        if episode == number_of_episodes - 1:
            agent_expected_claims = np.mean(expected_claims_per_timestep, axis=0)

        for fairness_eval in fairness_metrics:
            fairness_values = [
                compute_fairness_score(expected_claims_per_timestep[t], fairness_type=fairness_eval)
                for t in range(env.max_steps)]
            metrics[f'expected_fairness_{fairness_eval}'].append(np.mean(fairness_values))

        optimal_usage_score = np.sum(expected_claims_per_timestep) / (env.resource_capacity * env.max_steps)
        metrics['optimal usage'].append(optimal_usage_score)
        metrics['expected_return'].append(total_expected_reward)
        metrics['master_value'].append(master_value)
        metrics['min_rc'].append(min_rc)
        min_rc_history.append(rc_per_episode)

        print(f"Finished episode {episode+1}: expected_return={total_expected_reward:.2f}, master_value={master_value:.2f}, usage={optimal_usage_score:.2f}")
    return metrics, min_rc_history, agent_expected_claims