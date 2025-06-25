import os
import numpy as np
import pandas as pd
import json

from agents.abs_opt_cmdp import DecentralizedAgentWithColumns
from util.training import train_agents_with_dynamic_master
from env.resource_mdp_env import ResourceMDPEnv
from util.plotting import plot_min_rc_history_all_seeds, plot_average_expected_claims, plot_fairness_sweep

def main():
    FAIRNESS_METRICS = ["jain", "nsw", "minshare", "gini", "variance"]
    fairness_scores_dict = {metric: [] for metric in FAIRNESS_METRICS}
    lambda_values = [1, 5, 10, 25, 50, 100, 200, 400, 600]
    for lambda_fair in lambda_values:
        horizon = 5
        num_episodes = 1
        max_column_generation_rounds = 25000
        num_agents = 5
        reward_profile = {
            0: (1, 5),  # Agent 0 gets rewards
            1: (1, 5),  # Agent 1 gets rewards
            2: (1, 5),  # Agent 2 gets rewards
            3: (50, 100),  # Agent 3 gets higher rewards
            4: (1, 5)   # Agent 4 gets rewards
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
        seeds = range(5)
        verbose = False
        FAIRNESS_ENABLED = True  # Toggle fairness constraint here
        FAIRNESS_METRICS = ["jain", "nsw", "minshare", "gini", "variance"]
        if FAIRNESS_ENABLED:
            LANGRANGIAN_ENABLED = True  # Toggle Langranian relaxation here
            LAMBDA_FAIR = lambda_fair  # Tune this
        else:
            LANGRANGIAN_ENABLED = False
            LAMBDA_FAIR = None

        out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])

        resource_capacity = 3

        env = ResourceMDPEnv(
            n_agents=num_agents,
            resource_capacity=resource_capacity,
            max_steps=horizon,
            reward_profile=reward_profile
        )
        all_metrics = []
        all_min_rc_histories = []
        all_agent_expected_claims = []
        for seed in seeds:
            np.random.seed(seed)

            # Instantiate decentralized agents
            agents = []
            for agent_id in range(num_agents):
                agent = DecentralizedAgentWithColumns(
                    agent_id=agent_id,
                    horizon=horizon,
                    resource_capacity=resource_capacity,
                    num_columns=3,
                    verbose=verbose,
                    reward_profile=reward_profile,
                    cost_profile=cost_profile,
                    langrangian_weight=LAMBDA_FAIR if LANGRANGIAN_ENABLED else None
                )
                agents.append(agent)

            # Train with master coordination
            print(f"Running column generation experiment for seed {seed}")
            metrics, min_rc_history, agent_expected_claims  = train_agents_with_dynamic_master(env, agents, num_episodes, verbose=verbose, max_column_generation_rounds=max_column_generation_rounds, fairness=FAIRNESS_ENABLED, langrangian=LANGRANGIAN_ENABLED, langrangian_weight=LAMBDA_FAIR, seed=seed, fairness_metrics=FAIRNESS_METRICS)
            print(f"Metrics for seed {seed}: {metrics}")
            all_metrics.append(metrics)
            all_min_rc_histories.append(min_rc_history)
            all_agent_expected_claims.append(agent_expected_claims)
        with open(os.path.join('results', f"min_rc_history_all_seeds_(fairness={FAIRNESS_ENABLED},lambda={LAMBDA_FAIR}).json"), "w") as f:
            json.dump(all_min_rc_histories, f)
        # Combine metrics into one DataFrame
        dfs = [pd.DataFrame(m) for m in all_metrics]
        combined_df = pd.concat(dfs, ignore_index=True)

        # Write full combined metrics
        combined_df.to_csv(os.path.join('results', f"metrics_combined_all_seeds_(fairness={FAIRNESS_ENABLED},lambda={LAMBDA_FAIR}).csv"), index=False)
        plot_min_rc_history_all_seeds(result_dir='results', fairness=FAIRNESS_ENABLED, lambda_fair=LAMBDA_FAIR)
        plot_average_expected_claims(all_agent_expected_claims, out_dir="results\plots", fairness=FAIRNESS_ENABLED, lambda_fair=LAMBDA_FAIR)
        # Compute average fairness scores across seeds for this lambda
        averages_per_metric = {}

        for metric in FAIRNESS_METRICS:
            values = []
            for m in all_metrics:
                values.append(m[f'expected_fairness_{metric}'][0])  # Since num_episodes = 1
            avg_value = np.mean(values)
            averages_per_metric[metric] = avg_value

        # Append to the global dictionary
        for metric in FAIRNESS_METRICS:
            fairness_scores_dict[metric].append(averages_per_metric[metric])
    # Finally plot full sweep
    plot_fairness_sweep(lambda_values, fairness_scores_dict, out_dir="results")


if __name__ == '__main__':
    main()
