import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


from util.build_mdp import build_env_and_agents
from util.training import (
    train_agents_with_dynamic_master,
    train_agents_primal_dual
)
from util.plotting import (
    plot_min_rc_history_all_seeds,
    plot_average_expected_claims,
    plot_fairness_sweep,
    plot_cost_history_all_seeds,
    plot_primal_dual_history
)
from util.evaluate import mc_evaluate_policy


def main():
    """Runs fairness-aware training and evaluation experiments under varying environment complexities and fairness scopes."""

    USE_GRADIENT_FAIRNESS = True
    FAIRNESS_METRICS = ["jain", "nsw", "minshare", "gini", "variance"]
    num_episodes = 1
    max_column_generation_rounds = 75
    verbose = False  # Set to True to enable detailed output
    seeds = range(5)

    def log(msg):
        if verbose:
            print(msg)

    SL_states = [0, 1, 2]  # Exogenous capacity states

    def limit_fn(t, sL):
        """Map stochastic state to capacity."""
        caps = {
            0: 3,               # High capacity
            1: int(0.75 * 3),   # Medium
            2: int(0.5 * 3),    # Low
        }
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
            fairness_scores_dict = {f"{m}_{s}": [] for m in FAIRNESS_METRICS + ["return"] for s in ["timestep", "cumulative"]}

            num_agents = params["num_agents"]
            horizon = params["horizon"]
            resource_capacity = params["base_capacity"]
            TL = params["TL"]

            lambda_values = [0, 1, 5, 10, 25, 50, 100, 200, 400, 600, 1000]
            print(f"\n=== Experiment: complexity={complexity_level.upper()}, fairness_scope={FAIRNESS_SCOPE} ===")

            for lambda_fair in tqdm(lambda_values, desc=f"{complexity_level}_{FAIRNESS_SCOPE}", position=0):
                reward_profile = {}
                cost_profile = {}
                all_metrics = []
                all_min_rc_histories = []
                all_agent_expected_claims = []
                all_SL_trajectories = []

                for seed in seeds:
                    np.random.seed(seed)

                    for a in range(num_agents):
                        v = np.random.uniform(50, 100) if a == 3 and num_agents >= 4 else np.random.uniform(1, 5)
                        reward_profile[a] = (v, v)
                        cost_profile[a] = (1, 1)

                    env, agents = build_env_and_agents(
                        horizon, num_agents, resource_capacity,
                        reward_profile, cost_profile, lambda_fair,
                        SL_states=SL_states, TL=TL, limit_fn=limit_fn, verbose=verbose
                    )

                    log(f"Running column generation experiment for seed {seed}")
                    metrics, min_rc_history, cost_history, net_value_history, expected_return_history, agent_expected_claims, saved_columns, saved_distributions = train_agents_with_dynamic_master(
                        env, agents, num_episodes, verbose=verbose,
                        max_column_generation_rounds=max_column_generation_rounds,
                        langrangian_weight=lambda_fair,
                        fairness_metrics=FAIRNESS_METRICS,
                        fairness_scope=FAIRNESS_SCOPE,
                        use_gradient_fairness=USE_GRADIENT_FAIRNESS,
                        seed=seed
                    )

                    if isinstance(agent_expected_claims, np.ndarray):
                        agent_expected_claims = agent_expected_claims.tolist()
                    elif isinstance(agent_expected_claims, dict):
                        agent_expected_claims = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in agent_expected_claims.items()}

                    all_agent_expected_claims.append(agent_expected_claims)
                    all_SL_trajectories.append(env.sL_history)
                    all_metrics.append(metrics)
                    all_min_rc_histories.append(min_rc_history)

                    env_factory = lambda: build_env_and_agents(
                        horizon, num_agents, resource_capacity,
                        reward_profile, cost_profile, lambda_fair,
                        SL_states=SL_states, TL=TL, limit_fn=limit_fn, verbose=verbose
                    )[0]
                    agent_factory = lambda: build_env_and_agents(
                        horizon, num_agents, resource_capacity,
                        reward_profile, cost_profile, lambda_fair,
                        SL_states=SL_states, TL=TL, limit_fn=limit_fn, verbose=verbose
                    )[1]

                    eval_stats = mc_evaluate_policy(env_factory, agent_factory, saved_columns, saved_distributions, num_rollouts=500)

                    log(f"[MC EVAL] avg_return={eval_stats['avg_return'][-2]:.2f}, violation_rate={eval_stats['capacity_violation_rate']:.3%}")
                    log(f"Metrics for seed {seed}: {metrics}")

                    out_dir = f"results/{complexity_level}/{FAIRNESS_SCOPE}"
                    os.makedirs(out_dir + "/plots", exist_ok=True)

                    def save_json(obj, name):
                        with open(os.path.join(out_dir, name), 'w') as f:
                            json.dump(obj, f)

                    save_json(eval_stats, f"eval_stats_seed={seed}, lambda={lambda_fair}.json")
                    save_json(all_SL_trajectories, f"SL_history_seed={seed}, lambda={lambda_fair}.json")
                    save_json(net_value_history, f"net_value_history_seed={seed}, lambda={lambda_fair}.json")
                    save_json(expected_return_history, f"expected_return_history_seed={seed}, lambda={lambda_fair}.json")
                    save_json(cost_history, f"cost_history_seed={seed}, lambda={lambda_fair}.json")
                    save_json(all_agent_expected_claims, f"all_agent_expected_claims_seed={seed}, lambda={lambda_fair}.json")

                with open(os.path.join(out_dir, f"min_rc_history_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).json"), "w") as f:
                    json.dump(all_min_rc_histories, f)

                combined_df = pd.concat([pd.DataFrame(m) for m in all_metrics], ignore_index=True)
                combined_df.to_csv(os.path.join(out_dir, f"metrics_combined_all_seeds_({FAIRNESS_SCOPE},lambda={lambda_fair}).csv"), index=False)

                # Plotting
                plot_min_rc_history_all_seeds(result_dir=out_dir, lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE, verbose=verbose)
                plot_cost_history_all_seeds(out_dir=f"{out_dir}/plots", lambda_fair=lambda_fair, cost_history=cost_history, fairness_scope=FAIRNESS_SCOPE)
                plot_average_expected_claims(all_agent_expected_claims, out_dir=f"{out_dir}/plots", lambda_fair=lambda_fair, fairness_scope=FAIRNESS_SCOPE, verbose=verbose)

                return_values = [m['expected_return'][0] for m in all_metrics]
                mean_return = np.mean(return_values)

                for metric in FAIRNESS_METRICS:
                    for scope in ["timestep", "cumulative"]:
                        vals = [m[f"expected_fairness_{metric}_{scope}"][0] for m in all_metrics]
                        fairness_scores_dict[f"{metric}_{scope}"].append(np.mean(vals))

                for scope in ["timestep", "cumulative"]:
                    fairness_scores_dict[f"return_{scope}"].append(mean_return)

            plot_fairness_sweep(lambda_values, fairness_scores_dict, out_dir=out_dir, fairness_scope=FAIRNESS_SCOPE, verbose=verbose)

            # --- Primal–dual training ---
            env, agents = build_env_and_agents(
                horizon, num_agents, resource_capacity,
                reward_profile, cost_profile,
                lambda_fair=1.0,
                SL_states=SL_states, TL=TL, limit_fn=limit_fn, verbose=verbose
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
                verbose=verbose,
                seed=0
            )

            pd_df = pd.DataFrame(pd_history)
            pd_df.to_csv(f"{out_dir}/primal_dual_history.csv", index=False)

            fairness_path, lambda_path = plot_primal_dual_history(f"{out_dir}/primal_dual_history.csv")
            log("Saved fairness plot to: " + fairness_path)
            log("Saved λ plot to: " + lambda_path)


if __name__ == '__main__':
    main()
