import os
import numpy as np
import pandas as pd
import json
from skopt import gp_minimize
from skopt.space import Real
from util.build_mdp import build_env_and_agents
from util.training import train_agents_with_dynamic_master
from util.plotting import plot_min_rc_history_all_seeds, plot_average_expected_claims, plot_fairness_sweep, plot_cost_history_all_seeds, plot_fairness_sweep, plot_primal_dual_history
from util.evaluate import mc_evaluate_policy

def main():
    FAIRNESS_SCOPE = "timestep"  # or "cumulative"
    FAIRNESS_METRICS = ["jain", "nsw", "minshare", "gini", "variance"]
    fairness_scores_dict = {metric: [] for metric in FAIRNESS_METRICS}
    fairness_scores_dict['return'] = []
    cost_scores = []  # to collect average costs across λ

    num_episodes = 5
    max_column_generation_rounds = 25000
    verbose = False

    horizon = 15
    num_agents = 5
    resource_capacity = 3
    seeds = range(5)

    # --- New: exogenous capacity‐limit process SL ---
    SL_states = [0, 1, 2]                   # e.g. three discrete limit‐states
    # Transition matrix TL: P[s' | s]
    TL = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ])
    # Limit function: maps (timestep, sL) → capacity
    def limit_fn(t, sL):
        # for example, state 0 → high cap, 1 → medium, 2 → low
        caps = {0: resource_capacity,
                1: int(0.75 * resource_capacity),
                2: int(0.5 * resource_capacity)}
        return caps[sL]


    lambda_values = [0, 1, 5, 10, 25, 50, 100, 200, 400, 600, 1000]
    for lambda_fair in lambda_values:

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
            3: (1, 1),  # Agent 3 has costs
            4: (1, 1)   # Agent 4 has costs
        }

        if len(reward_profile) != num_agents:
            raise ValueError(f"Reward profile length {len(reward_profile)} does not match number of agents {num_agents}.")


        all_metrics = []
        all_min_rc_histories = []
        all_agent_expected_claims = []
        all_SL_trajectories = []
        for seed in seeds:
            np.random.seed(seed)

            env, agents = build_env_and_agents(
                horizon, num_agents,
                resource_capacity,
                reward_profile, cost_profile,
                lambda_fair,
                SL_states=SL_states,
                TL=TL,
                limit_fn=limit_fn
            )

            # Train with master coordination
            print(f"Running column generation experiment for seed {seed}")
            metrics, min_rc_history, cost_history, agent_expected_claims, saved_columns, saved_distributions = train_agents_with_dynamic_master(env, agents, num_episodes, verbose=verbose, max_column_generation_rounds=max_column_generation_rounds, langrangian_weight=lambda_fair, fairness_metrics=FAIRNESS_METRICS, fairness_scope=FAIRNESS_SCOPE, seed=seed)
        

            # Factories to rebuild env & agents from scratch (same config)
            env_factory   = lambda: build_env_and_agents(
                                horizon, num_agents, resource_capacity,
                                reward_profile, cost_profile, lambda_fair,
                                SL_states=SL_states, TL=TL, limit_fn=limit_fn
                             )[0]
            agent_factory = lambda: build_env_and_agents(
                                horizon, num_agents, resource_capacity,
                                reward_profile, cost_profile, lambda_fair,
                                SL_states=SL_states, TL=TL, limit_fn=limit_fn
                             )[1]
        
            eval_stats = mc_evaluate_policy(
                env_factory,
                agent_factory,
                saved_columns,
                saved_distributions,
                num_rollouts=500
            )
            print(f"[MC EVAL] avg_return={eval_stats['avg_return']:.2f}, "
                  f"violation_rate={eval_stats['capacity_violation_rate']:.3%}")
        
            # Store SL trajectories for analysis
            all_SL_trajectories.append(env.sL_history)
            print(f"Metrics for seed {seed}: {metrics}")
            all_metrics.append(metrics)
            all_min_rc_histories.append(min_rc_history)
            all_agent_expected_claims.append(agent_expected_claims)
        with open(os.path.join('results', f"min_rc_history_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).json"), "w") as f:
            json.dump(all_min_rc_histories, f)
        # then dump to JSON alongside your other results
        with open(os.path.join('results', f"SL_history_seed={seed}.json"), 'w') as f:
            json.dump(all_SL_trajectories, f)
        # Combine metrics into one DataFrame
        dfs = [pd.DataFrame(m) for m in all_metrics]
        combined_df = pd.concat(dfs, ignore_index=True)

        # Write full combined metrics
        combined_df.to_csv(os.path.join('results', f"metrics_combined_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).csv"), index=False)
        plot_min_rc_history_all_seeds(result_dir='results', lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE)
        plot_cost_history_all_seeds(out_dir="results/plots", lambda_fair=lambda_fair, cost_history=cost_history, fairness_scope=FAIRNESS_SCOPE)
        plot_average_expected_claims(all_agent_expected_claims, out_dir="results/plots", lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE)
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



    # --- New: online primal–dual training ---
    from util.training import train_agents_primal_dual

    # rebuild a fresh env & agents for primal–dual
    env, agents = build_env_and_agents(
        horizon, num_agents, resource_capacity,
        reward_profile, cost_profile,
        lambda_fair=1.0,        # initial λ
        SL_states=SL_states,
        TL=TL,
        limit_fn=limit_fn
    )

    pd_history = train_agents_primal_dual(
        env, agents,
        num_episodes=50,
        fairness_metric="jain",
        fairness_scope=FAIRNESS_SCOPE,
        init_lambda=1.0,
        beta_f=0.95,
        alpha=1,
        max_column_generation_rounds=500,
        verbose=True,
        seed=0
    )

    # Save history to CSV for later plotting
    pd_df = pd.DataFrame(pd_history)
    pd_df.to_csv("results/primal_dual_history.csv", index=False)
    plot_primal_dual_history("results/primal_dual_history.csv")

if __name__ == '__main__':
    main()
