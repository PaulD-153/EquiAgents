import os

from agents import AbsOptCMDPAgent
from util.training import run_experiments_batch
from env.multi_agent_env import MultiAgentEnv

def generate_agent_specs(agent_class, num_agents, base_params, default_types=None):
    """
    Generates a list of agent specifications.
    
    Args:
        agent_class: The agent class (e.g. AbsOptCMDPAgent).
        num_agents: Number of agents to create.
        base_params: Dictionary of parameters common to all agents (e.g., 'horizon', 'reward_scale', etc.)
        default_types: Optional list of types to assign; if not provided, cycle through a default list.
        
    Returns:
        A list of tuples of the form (agent_name, agent_class, agent_kwargs).
    """
    if default_types is None:
        default_types = ['A', 'B', 'C']
    specs = []
    for i in range(num_agents):
        params = base_params.copy()
        params['agent_id'] = i
        params['agent_type'] = default_types[i % len(default_types)]
        # You can generate a name automatically.
        agent_name = f"Agent_{i} (Type {params['agent_type']})"
        specs.append((agent_name, agent_class, params))
    return specs

def main():
    c = None
    horizon = 5
    num_episodes = 250
    seeds = range(5)
    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])
    eval_episodes = 100
    # Example: Define resource types and their available resources.
    resource_pool = {
        'A': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'B': ['B1', 'B2', 'B3'],
        'C': ['C1', 'C2', 'C3', 'C4']
}

    # Create the environment; here, nS is set to the horizon if you want one state per timestep.
    env = MultiAgentEnv(num_agents=50, max_steps=50, nS=horizon, resource_pool=resource_pool)
    
    # Set base parameters common to all agents.
    base_params = {
        'horizon': horizon,
        'reward_scale': 1,
        'cost_scale': 1,
        'cost_bound': c
    }
    
    # Generate specs for 50 agents.
    agents_specs = generate_agent_specs(AbsOptCMDPAgent, num_agents=50, base_params=base_params)
    
    # Run experiments using the generated agent specs.
    run_experiments_batch(env, agents_specs, eval_episodes, num_episodes, out_dir, seeds, fair_metrics=False)


if __name__ == '__main__':
    main()
