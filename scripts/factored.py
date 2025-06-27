import os
import numpy as np
import pandas as pd
import json

from util.build_mdp import build_env_and_agents
from util.training import train_agents_with_dynamic_master, tune_log_lambda
from util.plotting import plot_min_rc_history_all_seeds, plot_average_expected_claims, plot_fairness_sweep, plot_lambda_vs_fairness

def main():
    FAIRNESS_SCOPE = "timestep"  # or "cumulative"
    FAIRNESS_METRICS = ["jain", "nsw", "minshare", "gini", "variance"]
    fairness_scores_dict = {metric: [] for metric in FAIRNESS_METRICS}
    fairness_scores_dict['return'] = []
    lambda_values = [0, 1, 5, 10, 25, 50, 100, 200, 400, 600, 1000]
    for lambda_fair in lambda_values:
        num_episodes = 5
        max_column_generation_rounds = 25000
        verbose = False

        horizon = 5
        num_agents = 5
        resource_capacity = 3
        seeds = range(5)


        reward_profile = {
            0: (5, 5),  # Agent 0 gets rewards
            1: (5, 5),  # Agent 1 gets rewards
            2: (5, 5),  # Agent 2 gets rewards
            3: (100, 100),  # Agent 3 gets higher rewards
            4: (5, 5)   # Agent 4 gets rewards
        }
        cost_profile = {
            0: (1, 1),  # Agent 0 has costs
            1: (1, 1),  # Agent 1 has costs
            2: (1, 1),  # Agent 2 has costs
            3: (1, 1),  # Agent 3 has higher costs
            4: (1, 1)   # Agent 4 has costs
        }

        if len(reward_profile) != num_agents:
            raise ValueError(f"Reward profile length {len(reward_profile)} does not match number of agents {num_agents}.")


        all_metrics = []
        all_min_rc_histories = []
        all_agent_expected_claims = []
        for seed in seeds:
            np.random.seed(seed)

            env, agents = build_env_and_agents(horizon, num_agents, resource_capacity, reward_profile, cost_profile, lambda_fair)

            # Train with master coordination
            print(f"Running column generation experiment for seed {seed}")
            metrics, min_rc_history, agent_expected_claims  = train_agents_with_dynamic_master(env, agents, num_episodes, verbose=verbose, max_column_generation_rounds=max_column_generation_rounds, langrangian_weight=lambda_fair, fairness_metrics=FAIRNESS_METRICS, fairness_scope=FAIRNESS_SCOPE)
            print(f"Metrics for seed {seed}: {metrics}")
            all_metrics.append(metrics)
            all_min_rc_histories.append(min_rc_history)
            all_agent_expected_claims.append(agent_expected_claims)
        with open(os.path.join('results', f"min_rc_history_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).json"), "w") as f:
            json.dump(all_min_rc_histories, f)
        # Combine metrics into one DataFrame
        dfs = [pd.DataFrame(m) for m in all_metrics]
        combined_df = pd.concat(dfs, ignore_index=True)

        # Write full combined metrics
        combined_df.to_csv(os.path.join('results', f"metrics_combined_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).csv"), index=False)
        plot_min_rc_history_all_seeds(result_dir='results', lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE)
        plot_average_expected_claims(all_agent_expected_claims, out_dir="results\plots", lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE)
        # Compute average fairness scores across seeds for this lambda
        averages_per_metric = {}

        for metric in FAIRNESS_METRICS:
            values = []
            for m in all_metrics:
                values.append(m[f'expected_fairness_{metric}'][0])  # Since num_episodes = 1
            avg_value = np.mean(values)
            averages_per_metric[metric] = avg_value

        for m in all_metrics:
            values.append(m[f'expected_return'][0])  # Since num_episodes = 1
        avg_value = np.mean(values)
        averages_per_metric['return'] = avg_value

        fairness_scores_dict['return'].append(averages_per_metric['return'])

        # Append to the global dictionary
        for metric in FAIRNESS_METRICS:
            fairness_scores_dict[metric].append(averages_per_metric[metric])
        
    
    # Now normalize returns:
    returns = fairness_scores_dict['return']
    min_return = 0
    max_return = max(returns)

    if max_return != min_return:
        normalized_returns = [(r - min_return) / (max_return - min_return) for r in returns]
    else:
        normalized_returns = [0.5 for r in returns]

    fairness_scores_dict['return_normalized'] = normalized_returns

    # Delete the original 'return' key
    del fairness_scores_dict['return']

    # Finally plot full sweep
    plot_fairness_sweep(lambda_values, fairness_scores_dict, out_dir="results", fairness_scope=FAIRNESS_SCOPE)


    build_fn = lambda lambda_val: build_env_and_agents(horizon, num_agents, resource_capacity, reward_profile, cost_profile, lambda_val)

    target = 0.98
    lam_star, f_star, hist = tune_log_lambda(
        build_fn,
        target_fairness=target,
        metric="jain",
        scope="timestep",
        log10_min=-4,     # λ from 1e-4
        log10_max=4,      #       to 1e4
        tol=0.001,        # within 0.5%
        max_iter=20,
        num_eps=5,        # average over 5 runs
        verbose=True
    )

    print(f"\n→ Converged: λ≈{lam_star:.4f}, fairness≈{f_star:.4f}")
    plot_lambda_vs_fairness(hist, target=target,
                            out_path="results/plots/lambda_search.png")





if __name__ == '__main__':
    main()
