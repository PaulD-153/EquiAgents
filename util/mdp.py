from typing import Sequence
from tqdm import trange
import numpy as np
from gym_factored.envs.base import DiscreteEnv
from tqdm import trange

def get_mdp_functions(env: DiscreteEnv):
    """
    Extracts the full MDP functions (transition, reward, cost, terminal)
    from the given environment.
    """
    nS = env.nS
    # Use env.nA if it exists; otherwise, derive from the action space.
    nA = env.nA if hasattr(env, 'nA') else env.action_space.n
    transition = np.zeros(shape=(nS, nA, nS))
    reward = np.zeros(shape=(nS, nA))
    cost = np.zeros(shape=(nS, nA))
    terminal = np.zeros(shape=(nS,), dtype=bool)
    for s, state_transitions in env.P.items():
        for a, state_action_transitions in state_transitions.items():
            for tr in state_action_transitions:
                p, ns, r, done, info = get_transition_with_info(tr)
                reward[s, a] += p * r
                cost[s, a] += p * info.get('cost', 0)
                transition[s, a, ns] = p
                if done:
                    terminal[ns] = True
    return transition, reward, cost, terminal


def get_transition_with_info(tr: Sequence):
    if len(tr) == 5:
        return tr  # already has probability, next_state, reward, done, info
    elif len(tr) == 3:
        # For the simplified environment: assume done is False and info is an empty dict.
        return (tr[0], tr[1], tr[2], False, {})
    elif len(tr) == 4:
        # If only 4 elements are provided, assume the last is info and done is False.
        return (tr[0], tr[1], tr[2], False, tr[3])
    else:
        raise ValueError("Unexpected transition tuple length: {}".format(len(tr)))


def get_mdp_functions_partial(env: DiscreteEnv, features: Sequence):
    """
    Extracts an abstraction of the MDP that only considers the given features.
    """
    nS = env.nS
    nA = env.nA if hasattr(env, 'nA') else env.action_space.n

    # Determine feature domains from the decoded states.
    feature_values = [set() for _ in features]
    for s in range(nS):
        decoded_state = list(env.decode(s))
        for i, feature in enumerate(features):
            feature_values[i].add(decoded_state[feature])
    feature_domains = [sorted(fv) for fv in feature_values]

    number_of_abstract_states = 1
    for domain in feature_domains:
        number_of_abstract_states *= len(domain)
    w = number_of_abstract_states / nS

    transition = np.zeros(shape=(number_of_abstract_states, nA, number_of_abstract_states))
    reward = np.zeros(shape=(number_of_abstract_states, nA))
    cost = np.zeros(shape=(number_of_abstract_states, nA))
    terminal = np.ones(shape=number_of_abstract_states, dtype=bool)
    abs_map = np.zeros(shape=(number_of_abstract_states, nS), dtype=bool)
    
    for s, state_transitions in env.P.items():
        abstract_s = abstract(s, features, feature_domains, env)
        if env.isd[s] > 0:
            terminal[abstract_s] = False
        abs_map[abstract_s, s] = 1
        for a, state_action_transitions in state_transitions.items():
            for tr in state_action_transitions:
                p, ns, r, done, info = get_transition_with_info(tr)
                abstract_ns = abstract(ns, features, feature_domains, env)
                reward[abstract_s, a] += p * r * w
                cost[abstract_s, a] += p * info.get('cost', 0) * w
                transition[abstract_s, a, abstract_ns] += p * w
                if not done:
                    terminal[abstract_ns] = False
    return transition, reward, cost, terminal, abs_map


def abstract(state_id: int, features, feature_domains, env):
    """
    Maps an individual state to its abstract state index.
    """
    encoded_state = list(env.decode(state_id))
    abs_encoded_state = [encoded_state[v] for v in features]
    return encode(abs_encoded_state, feature_domains)


def encode(state_features: list, feature_domains: Sequence[Sequence[int]]):
    i = 0
    for v, value in enumerate(state_features):
        i *= len(feature_domains[v])
        i += value
    return i


def monte_carlo_evaluation(env, agents, horizon, discount_factor=1, number_of_episodes=1000, verbose=False):
    """
    Evaluates a list of agents over several episodes in evaluation mode.
    Returns per-agent average returns and costs, as well as overall average episode length and fail rate.
    Assumes env.step() returns (next_state, rewards, done, infos) where rewards and infos are lists (one per agent).
    """
    num_agents = len(agents)
    # Initialize per-agent metrics dictionaries.
    episodes_returns = {agent.agent_id: np.zeros(number_of_episodes) for agent in agents}
    episodes_costs = {agent.agent_id: np.zeros(number_of_episodes) for agent in agents}
    episodes_length = np.zeros(number_of_episodes)
    
    from tqdm import trange
    with trange(number_of_episodes, desc="monte carlo evaluation", unit='episodes', disable=not verbose) as progress:
        for i in progress:
            state = env.reset()
            # Initialize accumulators for this episode.
            episode_return = {agent.agent_id: 0 for agent in agents}
            episode_cost = {agent.agent_id: 0 for agent in agents}
            steps = 0
            fail = 0
            for t in range(horizon):
                # Use evaluation mode (so that exploration is disabled)
                actions = [agent.act(state['state'], evaluation=True) for agent in agents]
                next_state, rewards, done, infos = env.step(actions)
                for agent in agents:
                    idx = agent.agent_id
                    episode_return[idx] += rewards[idx] * (discount_factor ** t)
                    cost = infos[idx].get('cost', 0)
                    episode_cost[idx] += cost * (discount_factor ** t)
                state = next_state
                steps += 1
                if done:
                    # If any agent indicates failure, mark fail (could be refined per agent if desired)
                    fail = int(any(('fail' in info and info['fail']) for info in infos))
                    break
            # Record per-agent metrics for this episode.
            for agent in agents:
                episodes_returns[agent.agent_id][i] = episode_return[agent.agent_id]
                episodes_costs[agent.agent_id][i] = episode_cost[agent.agent_id]
                agent.end_episode(evaluation=True)
            episodes_length[i] = steps
    
    # Compute average returns and costs per agent.
    avg_returns = {agent_id: np.mean(returns) for agent_id, returns in episodes_returns.items()}
    avg_costs   = {agent_id: np.mean(costs) for agent_id, costs in episodes_costs.items()}
    avg_length  = episodes_length.mean()

    return avg_returns, avg_costs, avg_length
