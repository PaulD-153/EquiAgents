import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from planners.master_problem import MasterProblem

def plot_all_metrics(metrics, seed, out_dir='plots', window=5, fairness=False, langrangian=False, fair_constraint=False):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(metrics)
    for key in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df[key], label='raw', alpha=0.5)
        if len(df[key]) >= window:
            smoothed = df[key].rolling(window=window).mean()
            plt.plot(smoothed, label=f'smoothed (window={window})', linestyle='--')
        plt.title(f"{key} over Episodes (Seed {seed})")
        plt.xlabel("Episode")
        plt.ylabel(key)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        filename = os.path.join(out_dir, f"{key}_seed{seed}(fairness={fairness},langrangian={langrangian},fair_constraint={fair_constraint}).png")
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved to {filename}")

def jain_index(x):
    numerator = (np.sum(x)) ** 2
    denominator = len(x) * np.sum(x ** 2)
    return numerator / (denominator + 1e-8)

def compute_fairness_score(values, fairness_type, reward_scaling=None):
    if fairness_type == "jain":
        return jain_index(values)
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
    elif fairness_type == "envy_scaled":
        n = len(values)
        scaled = values * reward_scaling
        envy_list = [max(scaled[j] - scaled[i], 0) for i in range(n) for j in range(n)]
        return np.sum(envy_list) / (n * (n-1) + 1e-6)
    else:
        raise ValueError(f"Unknown fairness type: {fairness_type}")

def evaluate_all_fairness(values, fairness_metrics, reward_scaling):
    scores = {}
    for fairness_type in fairness_metrics:
        if fairness_type == "envy_scaled":
            score = compute_fairness_score(values, fairness_type, reward_scaling)
        else:
            score = compute_fairness_score(values, fairness_type)
        scores[fairness_type] = score
    return scores

def train_agents_with_dynamic_master(env, agents, number_of_episodes, max_column_generation_rounds=5, verbose=True,
                                      fairness=False, fairness_constraint=False, langrangian=False, langrangian_weight=None,
                                      fairness_type=None, seed=None, reward_scaling=None, fairness_metrics=None):

    metrics = {'returns': [], 'optimal usage': [], 'actual_stepwise_fairness': []}
    for fairness_eval in fairness_metrics:
        metrics[f'expected_fairness_{fairness_eval}'] = []
        metrics[f'fairness_{fairness_eval}'] = []

    for agent in agents:
        agent.reset()

    for episode in range(number_of_episodes):
        env.reset()
        for agent in agents:
            agent.columns = agent.generate_candidate_columns()

        round_idx = 0
        while True:
            master = MasterProblem(agents, resource_capacity=env.resource_capacity,
                                    fairness=fairness, fairness_constraint=fairness_constraint,
                                    langrangian=langrangian, langrangian_weight=langrangian_weight,
                                    fairness_type=fairness_type, reward_scaling=reward_scaling)
            master_value, _ = master.solve()
            
            if master.lp.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Master LP not feasible at round {round_idx}")
                break

            dual_prices = master.get_dual_prices()
            dual_prices_fairness = master.get_fairness_duals() if fairness_constraint else None

            new_columns = []
            for agent in agents:
                column = agent.generate_best_response_column(dual_prices, dual_prices_fairness)
                reduced_cost = agent.get_last_column_reduced_cost()
                new_columns.append((column, reduced_cost))

            reduced_costs = [rc for _, rc in new_columns]
            min_rc = min(reduced_costs)
            print(f"Round {round_idx}: min reduced cost {min_rc:.6f}")
            
            if min_rc > -1e-2 or round_idx >= max_column_generation_rounds:
                column_distributions = master.get_decision_distribution()
                break

            for agent, (column, rc) in zip(agents, new_columns):
                agent.columns.append(column)

            round_idx += 1

        # Assign policies
        for agent_id, agent in enumerate(agents):
            agent.policy_distribution = column_distributions[agent_id]

        # Expected fairness from policy distributions
        expected_claims_per_timestep = np.zeros((env.max_steps, len(agents)))
        for t in range(env.max_steps):
            for a_idx, agent in enumerate(agents):
                for i, col in enumerate(agent.columns):
                    prob = column_distributions[a_idx][i]
                    expected_claims_per_timestep[t, a_idx] += prob * col["claims"][t]

        # Aggregate expected claims across timesteps for histogram
        if episode == number_of_episodes - 1:  # Only plot after last episode
            agent_expected_claims = np.mean(expected_claims_per_timestep, axis=0)

            plt.figure(figsize=(8, 5))
            plt.bar(range(len(agents)), agent_expected_claims)
            plt.xlabel("Agent ID")
            plt.ylabel("Average Expected Claim Probability")
            plt.title("Expected Claim Allocation per Agent")
            plt.grid(True)
            plt.tight_layout()
            histogram_path = os.path.join("plots", f"expected_claim_hist_seed{seed}.png")
            plt.savefig(histogram_path)
            plt.close()
            print(f"Saved histogram plot to {histogram_path}")

        for fairness_eval in fairness_metrics:
            fairness_values = [
                compute_fairness_score(expected_claims_per_timestep[t], fairness_type=fairness_eval,
                                        reward_scaling=reward_scaling if fairness_eval == "envy_scaled" else None)
                for t in range(env.max_steps)]
            metrics[f'expected_fairness_{fairness_eval}'].append(np.mean(fairness_values))

        # Simulate episode
        episode_return, agent_rewards, agent_claims, timestep = 0, np.zeros(len(agents)), np.zeros(len(agents)), 0
        done, fairness_per_timestep = False, []

        while not done:
            claim_vector = np.zeros(len(agents))
            for agent_id, agent in enumerate(agents):
                prob = sum(agent.policy_distribution[i] * column["claims"][min(timestep, len(column["claims"])-1)]
                           for i, column in enumerate(agent.columns))
                claim_vector[agent_id] = float(np.random.rand() < prob)

            fairness_per_timestep.append(
                compute_fairness_score(claim_vector, fairness_type=fairness_type, reward_scaling=reward_scaling)
            )

            state, _, done, _ = env.step(claim_vector)

            for agent_id, agent in enumerate(agents):
                reward_this_step = agent.fixed_reward_vector[min(timestep, len(agent.fixed_reward_vector)-1)] * claim_vector[agent_id]
                agent_rewards[agent_id] += reward_this_step
                episode_return += reward_this_step

            agent_claims += claim_vector
            timestep += 1

        for agent in agents:
            agent.end_episode()

        optimal_usage_score = np.sum(agent_claims) / (env.resource_capacity * env.max_steps)
        metrics['returns'].append(episode_return)
        metrics['optimal usage'].append(optimal_usage_score)
        metrics['actual_stepwise_fairness'].append(np.mean(fairness_per_timestep))

        # Post-hoc fairness evaluation on realized agent_rewards
        realized_scores = evaluate_all_fairness(agent_rewards, fairness_metrics, reward_scaling)
        for fairness_eval, score in realized_scores.items():
            metrics[f'fairness_{fairness_eval}'].append(score)

        print(f"Finished episode {episode+1}: return={episode_return:.2f}, usage={optimal_usage_score:.2f}, rewards={agent_rewards}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join('results', f"metrics_seed{seed}(fairness={fairness},langrangian={langrangian},fair_constraint={fairness_constraint}).csv"), index=False)
    plot_all_metrics(metrics=metrics, seed=seed, fairness=fairness, langrangian=langrangian, fair_constraint=fairness_constraint)
    return metrics
