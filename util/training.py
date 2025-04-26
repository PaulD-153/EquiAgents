"""
This script contains the training and evaluation functions for multi-agent environments.
It includes functions to save and plot metrics, run experiments, and train agents using Monte Carlo methods.
"""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

from util.mdp import monte_carlo_evaluation


def compute_fairness_metrics(data):
    # Calculate the Gini coefficient and other fairness metrics.
    pass

def save_and_plot(data, moves, out_dir, seed, plot=True, window=25, testing=False):
    # Ensure the output directory exists.
    os.makedirs(out_dir, exist_ok=True)

    # Save detailed moves to CSV.
    moves_df = pd.DataFrame(moves)
    moves_csv = os.path.join(out_dir, f'moves_{seed}.csv')
    moves_df.to_csv(moves_csv, index=False)
    print(f"Info data saved to {moves_csv}")

    # Save the raw data to CSV.
    # Here we flatten the nested data: outer key = agent id, inner keys = metric names.
    flat_data = []
    for agent_id, metrics in data.items():
        # Add a column for agent id.
        df_agent = pd.DataFrame(metrics)
        df_agent.insert(0, 'agent_id', agent_id)
        flat_data.append(df_agent)
    data_df = pd.concat(flat_data, ignore_index=True)
    data_csv = os.path.join(out_dir, f'data_{seed}.csv')
    data_df.to_csv(data_csv, index=False)
    print(f"Raw data saved to {data_csv}")

    # Compute summary statistics per agent for each metric.
    # We will create a nested dictionary where the outer key is the agent id,
    # and the inner key is the metric name.
    summary_stats = {}
    for agent_id, metrics in data.items():
        summary_stats[agent_id] = {}
        for key, values in metrics.items():
            try:
                arr = np.array(values, dtype=float)
                summary_stats[agent_id][key] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'median': np.median(arr)
                }
            except Exception as e:
                print(f"Error processing agent {agent_id}, metric {key}: {e}")
    
    # Convert nested dictionary to a DataFrame.
    # We'll create one row per agent-metric combination.
    summary_list = []
    for agent_id, metrics in summary_stats.items():
        for metric, stats in metrics.items():
            stats['agent_id'] = agent_id
            stats['metric'] = metric
            summary_list.append(stats)
    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(out_dir, f'summary_{seed}.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to {summary_csv}")

    # Create plots for each agent and each metric if plotting is enabled.
    if plot:
        for agent_id, metrics in data.items():
            for key, values in metrics.items():
                try:
                    arr = np.array(values, dtype=float)
                    plt.figure(figsize=(8, 4))
                    plt.plot(arr, label=f'Agent {agent_id} {key}', alpha=0.7)
                    # Compute and plot the rolling mean if enough data exists.
                    if len(arr) >= window:
                        rolling_mean = pd.Series(arr).rolling(window=window).mean()
                        plt.plot(rolling_mean, label=f'Agent {agent_id} {key} (rolling mean, w={window})', linestyle='--')
                    plt.xlabel('Episode')
                    plt.ylabel(key)
                    plt.title(f'Agent {agent_id} {key} over Episodes (Seed {seed})')
                    plt.legend()
                    plot_file = os.path.join(out_dir, f'{key}_agent{agent_id}_seed{seed}.png')
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Plot for Agent {agent_id} {key} saved to {plot_file}")
                except Exception as e:
                    print(f"Error plotting Agent {agent_id} metric {key}: {e}")

    print(f"All outputs saved to {out_dir}")


def run_experiment(env, agents, seed,
                   number_of_episodes, out_dir, eval_episodes, fair_metrics=False):
    # Print agent names
    print("Running experiment with agents:", [agent.agent_id for agent in agents])
    metrics, moves = train_agents(agents, env, number_of_episodes, agents[0].horizon, seed, out_dir, fair_metrics=fair_metrics)
    save_and_plot(metrics, moves, out_dir, seed)

    # Now run evaluation on the learned policy
    eval_returns, eval_costs, eval_length = monte_carlo_evaluation(
        env, agents, agents[0].horizon, discount_factor=1, number_of_episodes=eval_episodes, verbose=True
    )
    print(f"Evaluation results: Return={eval_returns}, Cost={eval_costs}, Length={eval_length}")

def train_agents(agents, env, number_of_episodes, horizon, seed, out_dir, fair_metrics=False):
    # Initialize metrics per agent.
    metrics = {agent.agent_id: {'returns': []} for agent in agents}
    moves = {agent.agent_id: {'moves': []} for agent in agents}
    env.agent_types = [agent.agent_type for agent in agents]

    
    for episode in range(number_of_episodes):
        state = env.reset()  # shared observation
        # Reset each agent's planner time step if needed.
        for agent in agents:
            if hasattr(agent.planner, 'reset_time_step'):
                agent.planner.reset_time_step()
            else:
                agent.planner.time_step = 0
        
        episode_returns = {agent.agent_id: 0 for agent in agents}
        
        for t in range(horizon):
            # Each agent gets the same shared state.
            # Extract the base state from the shared observation
            base_state = state['state']
            
            actions = [agent.act(base_state) for agent in agents]

            next_state, rewards, done, infos = env.step(actions)

            # Update each agent with its own experience.
            for agent in agents:
                idx = agent.agent_id
                agent.add_transition(state['state'], rewards[idx], actions[idx], next_state['state'], done, infos[idx])
                episode_returns[idx] += rewards[idx]  # You can incorporate discounting as needed.

            state = next_state
            if done:
                break
        
        for agent in agents:
            print(f"Agent {agent.agent_id} finished episode {episode + 1}/{number_of_episodes} with return {episode_returns[agent.agent_id]}")
            moves[agent.agent_id]['moves'].append(agent.visited_resources)
            agent.end_episode()
            metrics[agent.agent_id]['returns'].append(episode_returns[agent.agent_id])

            
    
    # Optionally, export metrics to file.
    return metrics, moves

def run_experiments_batch(env, agents_specs, eval_episodes, number_of_episodes, out_dir, seeds, parallel=False, fair_metrics=False):
    """
    agents_specs: a list of tuples (agent_name, agentClass, agent_kwargs)
    Instead of looping over each agent spec individually, we will build all agents
    for a given seed and run one experiment.
    """
    experiments = []
    for seed in seeds:
        # For each seed, build all agents from the provided specifications.
        experiment_agents = []
        for (agent_name, agentClass, agent_kwargs) in agents_specs:
            # Here we assume your agent class provides a from_discrete_env() or similar constructor.
            agent = agentClass.from_discrete_env(env, **agent_kwargs)
            experiment_agents.append(agent)
        # Append a single experiment tuple with the list of agents.
        position = len(experiments)
        x = (env, experiment_agents, seed, number_of_episodes, out_dir, eval_episodes)
        experiments.append(x)
        print(x)
        
    if parallel:
        with Pool() as pool:
            pool.starmap(run_experiment, experiments)
    else:
        for e in experiments:
            run_experiment(*e)