import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_min_rc_history_all_seeds(result_dir, lambda_fair=None, fairness_scope='timestep'):
    filename = f"min_rc_history_all_seeds_({fairness_scope},lambda={lambda_fair}).json"
    filepath = os.path.join(result_dir, filename)
    
    with open(filepath, 'r') as f:
        all_min_rc_history = json.load(f)
    
    plt.figure(figsize=(10, 6))
    for seed_idx, min_rc_history in enumerate(all_min_rc_history):
        # Each min_rc_history is list of episodes for this seed
        # We average over episodes inside seed
        max_rounds = max(len(rc_list) for rc_list in min_rc_history)
        padded_rc = []
        for rc_list in min_rc_history:
            if len(rc_list) < max_rounds:
                rc_list = rc_list + [rc_list[-1]] * (max_rounds - len(rc_list))
            padded_rc.append(rc_list)
        padded_rc = np.array(padded_rc)
        avg_rc = np.mean(padded_rc, axis=0)
        
        plt.plot(range(max_rounds), avg_rc, label=f"Seed {seed_idx}")
    
    plt.title("Reduced Cost Convergence across Seeds")
    plt.xlabel("Column Generation Round")
    plt.ylabel("Minimal Reduced Cost (min_rc)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(result_dir, f"plots\min_rc_convergence_all_seeds_({fairness_scope},lambda={lambda_fair}).png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined min_rc convergence plot to {output_path}")


def plot_average_expected_claims(all_agent_expected_claims, out_dir="results", lambda_fair=None, fairness_scope='timestep'):
    """
    Plots average expected claims per agent across seeds.
    
    Args:
        all_agent_expected_claims: list of numpy arrays (one per seed), each shape (num_agents,)
        out_dir: output directory for saving plot
    """
    os.makedirs(out_dir, exist_ok=True)

    expected_claims_array = np.array(all_agent_expected_claims)  # shape: (num_seeds, num_agents)
    average_claims = np.mean(expected_claims_array, axis=0)
    std_claims = np.std(expected_claims_array, axis=0)

    plt.figure(figsize=(8, 5))
    agent_ids = np.arange(len(average_claims))
    plt.bar(agent_ids, average_claims, yerr=std_claims, capsize=5)
    plt.xlabel("Agent ID")
    plt.ylabel("Average Expected Claim Probability")
    plt.title("Average Expected Claim Allocation across Seeds")
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"expected_claims_average_all_seeds_({fairness_scope},lambda={lambda_fair}).png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved average expected claims plot to {filename}")

def plot_fairness_sweep(lambda_values, fairness_scores_dict, out_dir="results", fairness_scope='timestep'):
    """
    Plot fairness metrics over different Lagrangian weights (lambda_fair).
    
    Args:
        lambda_values: list of lambda_fair values
        fairness_scores_dict: dict of fairness metric name -> list of average values (same order as lambda_values)
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for fairness_metric, scores in fairness_scores_dict.items():
        plt.plot(lambda_values, scores, label=fairness_metric, marker='o')

    plt.xlabel("Lagrangian Fairness Weight (λ)")
    plt.ylabel("Fairness Metric Value")
    plt.title("Fairness Metrics vs Lagrangian Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"plots/fairness_vs_lambda-{fairness_scope}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved fairness sweep plot to {filename}")