import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cost_history_all_seeds(out_dir, lambda_fair, cost_history, fairness_scope):
    """Plot cost over column-generation rounds for all seeds."""
    plt.figure()
    for seed_idx, ch in enumerate(cost_history):
        plt.plot(ch, label=f'Seed {seed_idx}', alpha=0.6)
    plt.xlabel('Column Generation Round')
    plt.ylabel('Expected Cost')
    plt.title(f'Cost History (λ={lambda_fair}, {fairness_scope})')
    plt.legend()
    path = os.path.join(out_dir, f"cost_history_lambda={lambda_fair}.png")
    plt.savefig(path)
    plt.close()


def plot_min_rc_history_all_seeds(result_dir, lambda_fair=None, fairness_scope='timestep', verbose=True):
    """Plot convergence of minimal reduced cost across seeds."""
    filename = f"min_rc_history_all_seeds_({fairness_scope},lambda={lambda_fair}).json"
    filepath = os.path.join(result_dir, filename)

    with open(filepath, 'r') as f:
        all_min_rc_history = json.load(f)

    plt.figure(figsize=(10, 6))
    for seed_idx, min_rc_history in enumerate(all_min_rc_history):
        max_rounds = max(len(rc_list) for rc_list in min_rc_history)
        padded_rc = []
        for rc_list in min_rc_history:
            padded = rc_list + [rc_list[-1]] * (max_rounds - len(rc_list))
            padded_rc.append(padded)
        avg_rc = np.mean(np.array(padded_rc), axis=0)
        plt.plot(range(max_rounds), avg_rc, label=f"Seed {seed_idx}")

    plt.title("Reduced Cost Convergence Across Seeds")
    plt.xlabel("Column Generation Round")
    plt.ylabel("Minimal Reduced Cost (min_rc)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(result_dir, f"plots/min_rc_convergence_all_seeds_({fairness_scope},lambda={lambda_fair}).png")
    plt.savefig(output_path)
    plt.close()
    if verbose:
        print(f"Saved combined min_rc convergence plot to {output_path}")


def plot_average_expected_claims(all_agent_expected_claims, out_dir="results", lambda_fair=None, fairness_scope='timestep', verbose=True):
    """Plot average expected claim per agent across seeds with error bars."""
    os.makedirs(out_dir, exist_ok=True)

    expected_claims_array = np.array(all_agent_expected_claims)
    average_claims = np.mean(expected_claims_array, axis=0)
    std_claims = np.std(expected_claims_array, axis=0)

    plt.figure(figsize=(8, 5))
    agent_ids = np.arange(len(average_claims))
    plt.bar(agent_ids, average_claims, yerr=std_claims, capsize=5)
    plt.xlabel("Agent ID")
    plt.ylabel("Average Expected Claim")
    plt.title("Average Expected Claim Allocation Across Seeds")
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"expected_claims_average_all_seeds_({fairness_scope},lambda={lambda_fair}).png")
    plt.savefig(filename)
    plt.close()
    if verbose:
        print(f"Saved average expected claims plot to {filename}")


def plot_fairness_sweep(lambda_values, fairness_scores_dict, out_dir="results", fairness_scope='timestep', verbose=True):
    """Plot each fairness metric across different lambda values."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for fairness_metric, scores in fairness_scores_dict.items():
        plt.plot(lambda_values, scores, label=fairness_metric, marker='o')

    plt.xlabel("Lagrangian Fairness Weight (λ)")
    plt.ylabel("Fairness Metric Value")
    plt.title("Fairness Metrics vs. Lagrangian Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"plots/fairness_vs_lambda-{fairness_scope}.png")
    plt.savefig(filename)
    plt.close()
    if verbose:
        print(f"Saved fairness sweep plot to {filename}")


def plot_lambda_vs_fairness_history(history, target: float, metric: str, out_path: str):
    """Plot fairness value as a function of lambda (log scale)."""
    lambdas, fairness_vals = zip(*history)

    plt.figure()
    plt.semilogx(lambdas, fairness_vals, marker='o', linestyle='-', label=f"{metric.capitalize()} vs λ")
    plt.axhline(y=target, color='gray', linestyle='--', label=f"Target = {target:.2f}")

    # Highlight best λ ≥ target
    best_idx = max(
        (i for i in range(len(history)) if fairness_vals[i] >= target),
        default=None,
        key=lambda i: fairness_vals[i]
    )
    if best_idx is not None:
        best_lambda, best_fair = history[best_idx]
        plt.scatter([best_lambda], [best_fair], color='red', label=f"λ ≈ {best_lambda:.4g}")

    plt.xlabel("λ (log scale)")
    plt.ylabel(f"{metric.capitalize()} Fairness")
    plt.title(f"{metric.capitalize()} Fairness vs. λ")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def aggregate_histories_max(histories):
    """
    Pad histories to equal length, return mean, 25th, and 75th percentiles.
    Returns three lists of length L: mean, p25, p75.
    """
    L = max(len(h) for h in histories)
    padded = [h + [np.nan] * (L - len(h)) for h in histories]
    arr = np.array(padded, dtype=float)
    mean = np.nanmean(arr, axis=0).tolist()
    p25 = np.nanpercentile(arr, 25, axis=0).tolist()
    p75 = np.nanpercentile(arr, 75, axis=0).tolist()
    return mean, p25, p75


def plot_primal_dual_history(path_csv, out_dir="results/plots", prefix="primal_dual"):
    """Plot fairness and lambda values over primal-dual episodes."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(path_csv)

    # Plot fairness
    plt.figure()
    plt.plot(df['episode'], df['fairness'], 'b-o', label='Fairness')
    plt.xlabel('Episode')
    plt.ylabel('Fairness')
    plt.title('Fairness over Episodes')
    plt.legend(loc='best')
    plt.tight_layout()
    fairness_path = os.path.join(out_dir, f"{prefix}_fairness.png")
    plt.savefig(fairness_path)
    plt.close()

    # Plot λ
    plt.figure()
    plt.plot(df['episode'], df['lambda'], 'r-s', label='λ')
    plt.xlabel('Episode')
    plt.ylabel('λ (Lagrange Multiplier)')
    plt.title('Lagrange Multiplier over Episodes')
    plt.legend(loc='best')
    plt.tight_layout()
    lambda_path = os.path.join(out_dir, f"{prefix}_lambda.png")
    plt.savefig(lambda_path)
    plt.close()

    return fairness_path, lambda_path
