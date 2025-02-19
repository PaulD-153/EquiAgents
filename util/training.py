import os
from multiprocessing import Pool

import gym

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

from util.mdp import monte_carlo_evaluation


def train_agent(agent, env, number_of_episodes, horizon, seed, out_dir=None, eval_episodes=10, discount_factor=1,
                label='', position=0, verbose=True):
    env.seed(seed)
    env.reset()
    agent.seed(seed)
    np.random.seed(seed)
    results = run_training_episodes(agent, env, number_of_episodes, horizon, eval_episodes, discount_factor,
                                    label, position, verbose=verbose)
    if out_dir is not None:
        save_and_plot(results, out_dir, seed)
    return results

def compute_fairness_metrics(data):
    allocations = np.array(data["training_returns"])  # Assume training return correlates with allocation
    if len(allocations) < 3:
        return None
    
    gini_coefficient = np.abs(np.subtract.outer(allocations, allocations)).sum() / (2 * len(allocations) * allocations.sum())
    max_diff = max(allocations) - min(allocations)
    return {"gini": gini_coefficient, "max_diff": max_diff}

def save_and_plot(data, out_dir, seed, plot=True, window=100, testing=False):
    fairness_metrics = compute_fairness_metrics(data)
    if fairness_metrics:
        print(f"Fairness at seed {seed}: Gini {fairness_metrics['gini']:.3f}, Max Difference {fairness_metrics['max_diff']:.3f}")
    
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(data)
    df["Gini"] = fairness_metrics["gini"]
    df["Max Difference"] = fairness_metrics["max_diff"]
    df.to_csv(os.path.join(out_dir, f'results_{seed}.csv'))
    
    if plot:
        plt.figure()
        plt.plot(df["training_returns"], label="Training Returns")
        plt.plot(df["Gini"], label="Gini Coefficient")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f'fairness_{seed}.png'))

def save_and_plot_2(metrics, out_dir, seed):
    # Convert metrics dictionary to a DataFrame.
    data = []
    for agent_id, d in metrics.items():
        data.append({
            'agent_id': agent_id,
            'action_count': d['action_count'],
            'avg_return': np.mean(d['returns'])
        })
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir, f'fairness_metrics_{seed}.csv'), index=False)
    # Plot a bar chart of action counts
    plt.figure()
    plt.bar(df['agent_id'], df['action_count'])
    plt.xlabel("Agent ID")
    plt.ylabel("Action Count")
    plt.title("Number of Actions Executed per Agent")
    plt.savefig(os.path.join(out_dir, f'agent_action_counts_{seed}.png'))

def run_training_episodes(agent, env, number_of_episodes, horizon, eval_episodes=1, discount_factor=1, label='', position=0,
                          log_freq=10, verbose=False, out_dir=None, seed=None):
    training_returns = []
    training_costs = []
    training_length = []
    training_fails = []
    evaluation_returns = []
    evaluation_costs = []
    evaluation_length = []
    evaluation_fail = []
    # New metric: count the number of times the agent got to act.
    action_counts = 0  
    desc = "training {}".format(label)

    with trange(number_of_episodes, desc=desc, unit="episode", position=position, disable=not verbose) as progress_bar:
        for i in progress_bar:
            state = env.reset()  # shared env returns a tuple of observations
            # Reset the planner's time step for this agent.
            if hasattr(agent.planner, 'reset_time_step'):
                agent.planner.reset_time_step()
            else:
                agent.planner.time_step = 0

            episode_return = 0
            episode_cost = 0
            fail = 0
            steps = 0
            
            for t in range(horizon):
                # Get a list of actions from all agents based on the shared state.
                # Here, for simplicity, we assume each agent receives the same observation.
                actions = [agent.act(state['state']) for _ in range(env.num_agents)]
                next_state, rewards, done, infos = env.step(actions)
                # Use the info for this agent (indexed by agent.agent_id) to update action count.
                # Assume agent.agent_id is the index for this agent.
                if infos[agent.agent_id].get('acted', False):
                    action_counts += 1

                # Update this agent's transition using its own info.
                agent.add_transition(state['state'], rewards[agent.agent_id], actions[agent.agent_id], next_state['state'], done, infos[agent.agent_id])
                episode_return += rewards[agent.agent_id] * discount_factor ** t
                episode_cost += infos[agent.agent_id].get('cost', 0) * discount_factor ** t
                state = next_state
                steps += 1
                if done:
                    if infos[agent.agent_id].get('fail', False):
                        fail = 1
                    break
            
            agent.end_episode()
            training_returns.append(episode_return)
            training_costs.append(episode_cost)
            training_length.append(steps)
            training_fails.append(fail)
            
            # Optionally, accumulate per-episode action count (or average over episodes)
            # Here we simply print the action count for this agent.
            progress_bar.set_postfix(action_count=action_counts)
            
            if eval_episodes > 0:
                evaluation = monte_carlo_evaluation(env, agent, agent.horizon, discount_factor, eval_episodes)
                evaluation_returns.append(evaluation[0])
                evaluation_costs.append(evaluation[1])
                evaluation_length.append(evaluation[2])
                evaluation_fail.append(evaluation[3])
            
            if not (i % log_freq):
                progress_bar.set_postfix(
                    t_ret=np.mean(training_returns[-log_freq:]),
                    t_cost=np.mean(training_costs[-log_freq:]),
                    e_ret=np.mean(evaluation_returns[-log_freq:]),
                    e_cost=np.mean(evaluation_costs[-log_freq:]),
                    act_count=action_counts
                )
                if out_dir is not None:
                    save_and_plot({
                        "training_returns": training_returns,
                        "training_costs": training_costs,
                        "training_length": training_length,
                        "training_fail": training_fails,
                        "evaluation_returns": evaluation_returns,
                        "evaluation_costs": evaluation_costs,
                        "evaluation_length": evaluation_length,
                        "evaluation_fail": evaluation_fail,
                        "action_counts": action_counts
                    }, out_dir, seed, plot=False)

    results = {
        "training_returns": training_returns,
        "training_costs": training_costs,
        "training_length": training_length,
        "training_fail": training_fails,
        "evaluation_returns": evaluation_returns,
        "evaluation_costs": evaluation_costs,
        "evaluation_length": evaluation_length,
        "evaluation_fail": evaluation_fail,
        "action_counts": action_counts
    }
    return results

def run_experiment(env, agent_name, agentClass, agent_kwargs, seed,
                   number_of_episodes, position, out_dir, eval_episodes):
    # For a shared resource environment, create one agent per agent index,
    # all sharing the same env.
    agents = []
    num_agents = env.num_agents  # from the shared wrapper
    for i in range(num_agents):
        kwargs_copy = agent_kwargs.copy()
        kwargs_copy.pop('env', None)  # remove any env key if present
        kwargs_copy['agent_id'] = i  # assign unique id to each agent
        agent_i = agentClass.from_discrete_env(env, **kwargs_copy)
        agents.append(agent_i)
    # Now, run training jointly for all agents.
    # Here, we assume a separate training function for the shared environment.
    metrics=train_agents(agents, env, number_of_episodes, agents[0].horizon, seed, out_dir, eval_episodes)
    save_and_plot_2(metrics, out_dir, seed)

def train_agents(agents, env, number_of_episodes, horizon, seed, out_dir, eval_episodes):
    # Initialize metrics per agent.
    metrics = {agent.agent_id: {'action_count': 0, 'returns': []} for agent in agents}
    
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
            resource = state['resource'][0]
            print(resource)
            print(env.num_agents)
            if resource >= env.num_agents:
                actions = [agent.act(base_state) for agent in agents]
            elif resource == env.num_agents - 1:
                # Get each agent's expected reward (their “bid”)
                value_estimates = [agent.get_expected_reward(base_state) for agent in agents]
                # Choose the agent with the highest expected reward
                best_agent_idx = int(np.argmax(value_estimates))
                # Choose the second best agent
                value_estimates[best_agent_idx] = -np.inf
                second_best_agent_idx = int(np.argmax(value_estimates))

                actions = []
                for i, agent in enumerate(agents):
                    if i == best_agent_idx:
                        # This agent gets to act—use its computed action.
                        actions.append(agent.act(base_state))
                    elif i == second_best_agent_idx:
                        actions.append(agent.act(base_state))
                    else:
                        # For agents not selected, send a dummy action (e.g., -1).
                        actions.append(-1)
            else:
                # Get each agent's expected reward (their “bid”)
                value_estimates = [agent.get_expected_reward(base_state) for agent in agents]
                # Choose the agent with the highest expected reward
                best_agent_idx = int(np.argmax(value_estimates))
                actions = []
                for i, agent in enumerate(agents):
                    if i == best_agent_idx:
                        # This agent gets to act—use its computed action.
                        actions.append(agent.act(base_state))
                    else:
                        # For agents not selected, send a dummy action (e.g., 0).
                        actions.append(-1)
            print(actions)
            next_state, rewards, done, infos = env.step(actions)
            print(next_state, rewards, done, infos)
            # Update each agent with its own experience.
            for agent in agents:
                idx = agent.agent_id
                agent.add_transition(state['state'], rewards[idx], actions[idx], next_state['state'], done, infos[idx])
                episode_returns[idx] += rewards[idx]  # You can incorporate discounting as needed.
                if infos[idx].get('acted', False):
                    metrics[idx]['action_count'] += 1
            state = next_state
            if done:
                break
        
        for agent in agents:
            agent.end_episode()
            metrics[agent.agent_id]['returns'].append(episode_returns[agent.agent_id])
    
    # Save or print fairness metrics.
    for agent_id, data in metrics.items():
        print(f"Agent {agent_id} acted {data['action_count']} times over {number_of_episodes} episodes. Average return: {np.mean(data['returns'])}")
    
    # Optionally, export metrics to file.
    return metrics

def run_experiments_batch(env, agents, eval_episodes, number_of_episodes, out_dir, seeds, parallel=False):
    experiments = []
    for seed in seeds:
        for (agent_name, agentClass, agent_kwargs) in agents:
            position = len(experiments)
            x = (env, agent_name, agentClass, agent_kwargs, seed, number_of_episodes,
                 position, out_dir, eval_episodes)
            experiments.append(x)
            print(x)
    if parallel:
        with Pool() as pool:
            pool.starmap(run_experiment, experiments)
    else:
        for e in experiments:
            run_experiment(*e)
