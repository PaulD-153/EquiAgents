import os

from agents import OptCMDPAgent
from agents import AbsOptCMDPAgent
from util.training import run_experiments_batch
from agents.multi_agent_env import SharedResourceCMDPWrapper
import random

def main():
    c = None
    h = 25
    number_of_episodes = 1000
    seeds = range(10)
    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])
    eval_episodes = 100
    # Pick replenishment rate between 1 and 3
    env = SharedResourceCMDPWrapper(num_agents=3, initial_resource=20, capacity_limit=3, max_steps=50, replenishment_rate=3)


    agents = [
        ('AbsOptCMDP A (Greedy)', AbsOptCMDPAgent, {
            'features': [0],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground',
            'agent_id': 0,
            'reward_scale': 1,
            'cost_scale': 1
        }),
        ('AbsOptCMDP B (Underserved)', AbsOptCMDPAgent, {
            'features': [0],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground',
            'agent_id': 1,
            'reward_scale': 0.5,
            'cost_scale': 2.0
        }),
        ('AbsOptCMDP C (Middle)', AbsOptCMDPAgent, {
            'features': [0],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground',
            'agent_id': 2,
            'reward_scale': 100,
            'cost_scale': 1.0
        }),
    ]

    run_experiments_batch(env, agents, eval_episodes, number_of_episodes, out_dir, seeds, fair_metrics=True)


if __name__ == '__main__':
    main()
