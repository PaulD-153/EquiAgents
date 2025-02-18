import os

from agents import OptCMDPAgent
from agents import AbsOptCMDPAgent
from util.training import run_experiments_batch
from agents.multi_agent_env import SharedResourceCMDPWrapper

def main():
    c = None
    h = 3
    number_of_episodes = 1000
    seeds = range(10)
    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])
    eval_episodes = 100
    env = SharedResourceCMDPWrapper(prob_y_zero=0.1, initial_resource=10, capacity_limit=2, num_agents=3, replenishment_rate=2)


    agents = [
        ('AbsOptCMDP A (Greedy)', AbsOptCMDPAgent, {
            'features': [0],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground',
            'agent_id': 0,
            'reward_scale': 100,
            'cost_scale': 0.5
        }),
        ('AbsOptCMDP B (Underserved)', AbsOptCMDPAgent, {
            'features': [0],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground',
            'agent_id': 1,
            'reward_scale': 0.5,
            'cost_scale': 1.0
        }),
        ('AbsOptCMDP C (Middle)', AbsOptCMDPAgent, {
            'features': [0],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground',
            'agent_id': 2,
            'reward_scale': 1.0,
            'cost_scale': 1.0
        }),
    ]

    run_experiments_batch(env, agents, eval_episodes, number_of_episodes, out_dir, seeds)


if __name__ == '__main__':
    main()
