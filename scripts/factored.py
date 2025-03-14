import os

from agents import AbsOptCMDPAgent
from util.training import run_experiments_batch
from env.multi_agent_env import MultiAgentEnv

def main():
    c = None
    h = 5
    number_of_episodes = 250
    seeds = range(5)
    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])
    eval_episodes = 100
    # Example: Define resource types and their available resources.
    resource_pool = {
        'A': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'B': ['B1', 'B2', 'B3'],
        'C': ['C1', 'C2', 'C3', 'C4']
}

    # Pick replenishment rate between 1 and 3
    env = MultiAgentEnv(num_agents=3, max_steps=50, nS=h, resource_pool=resource_pool)


    agents = [
        ('LP Agent A', AbsOptCMDPAgent, {
            'agent_id': 0,
            'agent_type': 'A',
            'horizon': h,
            'reward_scale': 1,
            'cost_scale': 1,
            'cost_bound': c,
        }),
        ('LP Agent B', AbsOptCMDPAgent, {
            'agent_id': 1,
            'agent_type': 'B',
            'horizon': h,
            'reward_scale': 1,
            'cost_scale': 1,
            'cost_bound': c,
        }),
        ('LP Agent C', AbsOptCMDPAgent, {
            'agent_id': 2,
            'agent_type': 'C',
            'horizon': h,
            'reward_scale': 1,
            'cost_scale': 1,
            'cost_bound': c,
        }),
    ]

    run_experiments_batch(env, agents, eval_episodes, number_of_episodes, out_dir, seeds, fair_metrics=False)


if __name__ == '__main__':
    main()
