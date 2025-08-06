import os
import numpy as np
import pandas as pd
import json
from util.build_mdp import build_env_and_agents
from util.training import train_agents_with_dynamic_master
from util.plotting import plot_min_rc_history_all_seeds, plot_average_expected_claims, plot_fairness_sweep, plot_cost_history_all_seeds, plot_fairness_sweep, plot_primal_dual_history
from util.evaluate import mc_evaluate_policy

def main():

    USE_GRADIENT_FAIRNESS = True
    FAIRNESS_SCOPE = "timestep"  # or "cumulative"
    FAIRNESS_METRICS = ["jain", "nsw", "minshare", "gini", "variance"]

    num_episodes = 1
    max_column_generation_rounds = 75
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
    
    env_settings = {
        "low": {
            "num_agents": 5,
            "horizon": 10,
            "base_capacity": 3,
            "TL": np.array([
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
            ])
        },
        "medium": {
            "num_agents": 8,
            "horizon": 15,
            "base_capacity": 3,
            "TL": np.array([
                [0.8, 0.1, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.2, 0.7],
            ])
        },
        "high": {
            "num_agents": 10,
            "horizon": 20,
            "base_capacity": 3,
            "TL": np.array([
                [0.6, 0.2, 0.2],
                [0.3, 0.4, 0.3],
                [0.2, 0.2, 0.6],
            ])
        }
    }
    for FAIRNESS_SCOPE in ["timestep", "cumulative"]:
        for complexity_level, params in env_settings.items():
            fairness_scores_dict = {}
            for metric in FAIRNESS_METRICS + ['return']:
                for scope in ["timestep","cumulative"]:
                    fairness_scores_dict[f"{metric}_{scope}"] = []
            print(f"\n--- Running {complexity_level.upper()} complexity setting ---")

            num_agents = params["num_agents"]
            horizon = params["horizon"]
            resource_capacity = params["base_capacity"]
            TL = params["TL"]

            lambda_values = [0, 1, 5, 10, 25, 50, 100, 200, 400, 600, 1000]

            for lambda_fair in lambda_values:
                reward_profile = {}
                cost_profile = {}

                all_metrics = []
                all_min_rc_histories = []
                all_agent_expected_claims = []
                all_SL_trajectories = []
                for seed in seeds:
                    np.random.seed(seed)

                    for a in range(num_agents):
                        if a == 3 and num_agents >= 4:
                            v = np.random.uniform(low=50, high=100)
                            reward_profile[a] = (v, v)  # high-reward agent
                        else:
                            v = np.random.uniform(low=1, high=5)
                            reward_profile[a] = (v, v)
                        cost_profile[a] = (1, 1)

                    if len(reward_profile) != num_agents:
                        raise ValueError(f"Reward profile length {len(reward_profile)} does not match number of agents {num_agents}.")


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
                    metrics, min_rc_history, cost_history, net_value_history, expected_return_history, agent_expected_claims, saved_columns, saved_distributions = train_agents_with_dynamic_master(env, agents, num_episodes, verbose=verbose, max_column_generation_rounds=max_column_generation_rounds, langrangian_weight=lambda_fair, fairness_metrics=FAIRNESS_METRICS, fairness_scope=FAIRNESS_SCOPE, use_gradient_fairness=USE_GRADIENT_FAIRNESS, seed=seed)

                    # if it's an ndarray, convert it
                    if isinstance(agent_expected_claims, np.ndarray):
                        agent_expected_claims = agent_expected_claims.tolist()

                    # if it's a dict of arrays, convert each entry
                    elif isinstance(agent_expected_claims, dict):
                        for k, v in agent_expected_claims.items():
                            if isinstance(v, np.ndarray):
                                agent_expected_claims[k] = v.tolist()

                    all_agent_expected_claims.append(agent_expected_claims)

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
                    print(f"[MC EVAL] avg_return={eval_stats['avg_return'][-2]:.2f}, "
                        f"violation_rate={eval_stats['capacity_violation_rate']:.3%}")
                    
                
                    # Store SL trajectories for analysis
                    all_SL_trajectories.append(env.sL_history)
                    print(f"Metrics for seed {seed}: {metrics}")
                    all_metrics.append(metrics)
                    all_min_rc_histories.append(min_rc_history)
                    all_agent_expected_claims.append(agent_expected_claims)
                    os.makedirs(f"results/{complexity_level}/{FAIRNESS_SCOPE}", exist_ok=True)
                    os.makedirs(f"results/{complexity_level}/{FAIRNESS_SCOPE}/plots", exist_ok=True)
                    # then dump to JSON alongside your other results
                    # Dump evaluation stats to JSON
                    with open(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"eval_stats_seed={seed}, lambda={lambda_fair}.json"), 'w') as f:
                        json.dump(eval_stats, f)
                    with open(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"SL_history_seed={seed}, lambda={lambda_fair}.json"), 'w') as f:
                        json.dump(all_SL_trajectories, f)
                    with open(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"net_value_history_seed={seed}, lambda={lambda_fair}.json"), 'w') as f:
                        json.dump(net_value_history, f)
                    with open(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"expected_return_history_seed={seed}, lambda={lambda_fair}.json"), 'w') as f:
                        json.dump(expected_return_history, f)
                    with open(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"cost_history_seed={seed}, lambda={lambda_fair}.json"), 'w') as f:
                        json.dump(cost_history, f)
                    with open(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"all_agent_expected_claims_seed={seed}, lambda={lambda_fair}.json"), 'w') as f:
                        json.dump(all_agent_expected_claims, f)
                with open(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"min_rc_history_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).json"), "w") as f:
                    json.dump(all_min_rc_histories, f)
                # Combine metrics into one DataFrame
                dfs = [pd.DataFrame(m) for m in all_metrics]
                combined_df = pd.concat(dfs, ignore_index=True)

                # Write full combined metrics
                combined_df.to_csv(os.path.join(f'results/{complexity_level}/{FAIRNESS_SCOPE}', f"metrics_combined_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).csv"), index=False)
                plot_min_rc_history_all_seeds(result_dir=f'results/{complexity_level}/{FAIRNESS_SCOPE}', lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE)
                plot_cost_history_all_seeds(out_dir=f"results/{complexity_level}/{FAIRNESS_SCOPE}/plots", lambda_fair=lambda_fair, cost_history=cost_history, fairness_scope=FAIRNESS_SCOPE)
                plot_average_expected_claims(all_agent_expected_claims, out_dir=f"results/{complexity_level}/{FAIRNESS_SCOPE}/plots", lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE)
                # NEW: compute mean return once
                return_values = [m['expected_return'][0] for m in all_metrics]
                mean_return   = np.mean(return_values)

                # For each fairness metric AND each scope, append one value per lambda
                for metric in FAIRNESS_METRICS:
                    for scope in ["timestep", "cumulative"]:
                        key_in_metrics = f"expected_fairness_{metric}_{scope}"
                        vals = [m[key_in_metrics][0] for m in all_metrics]
                        fairness_scores_dict[f"{metric}_{scope}"].append(np.mean(vals))

                # Append the mean_return under both scopes (so it lines up with fairness curves)
                for scope in ["timestep", "cumulative"]:
                    fairness_scores_dict[f"return_{scope}"].append(mean_return)

            # Finally plot full sweep
            plot_fairness_sweep(lambda_values, fairness_scores_dict, out_dir=f"results/{complexity_level}/{FAIRNESS_SCOPE}", fairness_scope=FAIRNESS_SCOPE)

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
                num_episodes=200,
                fairness_metric="jain",
                fairness_scope=FAIRNESS_SCOPE,
                init_lambda=1.0,
                beta_f=0.95,
                alpha=0.5,
                max_column_generation_rounds=500,
                verbose=True,
                seed=0
            )

            # Save history to CSV for later plotting
            pd_df = pd.DataFrame(pd_history)
            pd_df.to_csv(f"results/{complexity_level}/{FAIRNESS_SCOPE}/primal_dual_history.csv", index=False)
            fairness_path, lambda_path = plot_primal_dual_history(f"results/{complexity_level}/{FAIRNESS_SCOPE}/primal_dual_history.csv")
            print("Saved fairness plot to:", fairness_path)
            print("Saved λ plot to:", lambda_path)

if __name__ == '__main__':
    main()
