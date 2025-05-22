import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange
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
    """
    Computes Jain's Fairness Index for a vector of allocations.
    """
    numerator = (np.sum(x)) ** 2
    denominator = len(x) * np.sum(x ** 2)
    return numerator / (denominator + 1e-8)


def train_agents_with_dynamic_master(env, agents, number_of_episodes, max_column_generation_rounds=5, verbose=True, fairness=False, fairness_constraint=False, langrangian=False, langrangian_weight=None, seed=None):
    metrics = {
        'returns': [], 'fairness': [], 'optimal usage': [],
        'expected_stepwise_fairness': [], 'actual_stepwise_fairness': []
    }
    for agent in agents:
        agent.reset()   

    for episode in range(number_of_episodes):
        env.reset()

        # Step 1: Agents generate initial columns
        for agent in agents:
            agent.columns = agent.generate_candidate_columns()

        feasible = False

        round_idx = 0
        while True:
            master = MasterProblem(
                agents,
                resource_capacity=env.resource_capacity,
                fairness=fairness,
                fairness_constraint=fairness_constraint,
                langrangian=langrangian,
                langrangian_weight=langrangian_weight
            )
            master_value, _ = master.solve()

            if master.lp.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Master LP not feasible at round {round_idx} (status: {master.lp.status})")
                break  # Exit if truly infeasible

            # Step 1: Get duals
            dual_prices = master.get_dual_prices()
            dual_prices_fairness = master.get_fairness_duals() if fairness_constraint else None

            # Step 2: Generate one best-response column for each agent
            new_columns = []
            for agent in agents:
                column = agent.generate_best_response_column(dual_prices, dual_prices_fairness)
                reduced_cost = agent.get_last_column_reduced_cost()
                new_columns.append((column, reduced_cost))

            # Step 3: Check for improving columns with early stopping
            reduced_costs = [rc for _, rc in new_columns]
            min_rc = min(reduced_costs)
            early_stop_threshold = -1e-2 # You can tune this
            print(f"Round {round_idx}: min reduced cost {min_rc:.6f}")
            print(f"Round {round_idx}: reduced costs {reduced_costs}")
            print(f"[Round {round_idx}] Duals: {np.round(dual_prices, 4)}")

            if min_rc > early_stop_threshold:
                print(f"Early stopping: min reduced cost {min_rc:.6f} is above threshold {early_stop_threshold}.")
                column_distributions = master.get_decision_distribution()
                break

            if round_idx >= max_column_generation_rounds:
                print(f"Max column generation rounds reached: {max_column_generation_rounds}.")
                column_distributions = master.get_decision_distribution()
                break


            # Step 4: Add improving columns
            for agent, (column, rc) in zip(agents, new_columns):
                agent.columns.append(column)

            if verbose:
                for agent_id, agent in enumerate(agents):
                    print(f"Agent {agent_id} columns now: {len(agent.columns)}")
                    print(f"Agent {agent_id} reduced cost: {new_columns[agent_id][1]:.4f}")
                    print(f"Agent {agent_id} dual prices: {dual_prices}")
                    print(f"Agent {agent_id} columns: {agent.columns}")
            
            round_idx += 1


        if master.lp.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Failed to find feasible solution after {max_column_generation_rounds} rounds.")

        print(f"Episode {episode+1}: Master LP Value = {master_value:.2f}")

        # Step 2: Assign selected plans
        for agent_id, agent in enumerate(agents):
            agent.policy_distribution = column_distributions[agent_id]

        # Step 2.5: Compute expected fairness from current policy distributions
        expected_claims_per_timestep = np.zeros((env.max_steps, len(agents)))
        for t in range(env.max_steps):
            for a_idx, agent in enumerate(agents):
                for i, col in enumerate(agent.columns):
                    prob = column_distributions[a_idx][i]
                    expected_claims_per_timestep[t, a_idx] += prob * col["claims"][t]

        expected_fairness = np.mean([
            jain_index(expected_claims_per_timestep[t]) for t in range(env.max_steps)
        ])
        metrics['expected_stepwise_fairness'].append(expected_fairness)

        # Step 3: Simulate episode
        episode_return = 0
        agent_rewards = np.zeros(len(agents))
        agent_claims = np.zeros(len(agents))
        timestep = 0
        done = False
        fairness_per_timestep = []
        agents_rewards_vectors = []

        while not done:
            claim_vector = np.zeros(len(agents))
            for agent_id, agent in enumerate(agents):
                prob = 0.0
                for i, column in enumerate(agent.columns):
                    plan_prob = agent.policy_distribution[i]
                    plan_value = column["claims"][min(timestep, len(column["claims"]) - 1)]
                    prob += plan_prob * plan_value
                claim_vector[agent_id] = float(np.random.rand() < prob)

            fairness_per_timestep.append(jain_index(claim_vector))
            state, _, done, _ = env.step(claim_vector)  # Ignore env.reward if using internal rewards

            for agent_id, agent in enumerate(agents):
                timestep_clipped = min(timestep, len(agent.fixed_reward_vector) - 1)
                reward_this_step = agent.fixed_reward_vector[timestep_clipped] * claim_vector[agent_id]
                agent_rewards[agent_id] += reward_this_step
                episode_return += reward_this_step
                agents_rewards_vectors.append(agent.fixed_reward_vector)


            agent_claims += claim_vector
            timestep += 1

        for agent in agents:
            agent.end_episode()
            agent.episode+= 1

        # Step 4: Record metrics
        fairness_score = jain_index(agent_rewards)
        optimal_usage_score = np.sum(agent_claims) / (env.resource_capacity * env.max_steps)  # safe division

        metrics['returns'].append(episode_return)
        metrics['fairness'].append(fairness_score)
        metrics['optimal usage'].append(optimal_usage_score)
        metrics['actual_stepwise_fairness'].append(np.mean(fairness_per_timestep))

        print(f"Finished episode {episode+1} with actual return {episode_return:.2f}, fairness {fairness_score:.4f}, optimal usage {optimal_usage_score:.4f}, agent claims: {agent_claims}")

    plot_all_metrics(metrics, seed, fairness=fairness, langrangian=langrangian, fair_constraint=fairness_constraint)
    return metrics
