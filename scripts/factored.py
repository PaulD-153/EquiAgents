import os
import numpy as np

from agents.abs_opt_cmdp import DecentralizedAgentWithColumns
from util.training import train_agents_with_dynamic_master
from env.resource_mdp_env import ResourceMDPEnv

def main():
    horizon = 5
    num_episodes = 25
    max_column_generation_rounds = 50
    num_agents = 5
    reward_profile = {
        0: (1, 5),  # Agent 0 gets rewards
        1: (1, 5),  # Agent 1 gets rewards
        2: (1, 5),  # Agent 2 gets rewards
        3: (50, 100),  # Agent 3 gets higher rewards
        4: (1, 5)   # Agent 4 gets rewards
    }
    cost_profile = {
        0: (1, 1),  # Agent 0 has costs
        1: (1, 1),  # Agent 1 has costs
        2: (1, 1),  # Agent 2 has costs
        3: (1, 1),  # Agent 3 has higher costs
        4: (1, 1)   # Agent 4 has costs
    }

    if len(reward_profile) != num_agents:
        raise ValueError(f"Reward profile length {len(reward_profile)} does not match number of agents {num_agents}.")
    seeds = range(3)
    verbose = False
    FAIRNESS_ENABLED = False  # Toggle fairness constraint here
    if FAIRNESS_ENABLED:
        FAIRNESS_CONSTRAINTS_ENABLED = False  # Toggle constraints here
        LANGRANGIAN_ENABLED = True  # Toggle Langranian relaxation here
        LAMBDA_FAIR = 50  # Tune this
        assert LANGRANGIAN_ENABLED != FAIRNESS_CONSTRAINTS_ENABLED, "Only one of FAIRNESS_CONSTRAINTS_ENABLED or LANGRANIAN_ENABLED should be True."
    else:
        FAIRNESS_CONSTRAINTS_ENABLED = False
        LANGRANGIAN_ENABLED = False
        LAMBDA_FAIR = None

    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])

    resource_capacity = 3

    env = ResourceMDPEnv(
        n_agents=num_agents,
        resource_capacity=resource_capacity,
        max_steps=horizon,
        reward_profile=reward_profile
    )

    for seed in seeds:
        np.random.seed(seed)

        # Instantiate decentralized agents
        agents = []
        for agent_id in range(num_agents):
            agent = DecentralizedAgentWithColumns(
                agent_id=agent_id,
                horizon=horizon,
                resource_capacity=resource_capacity,
                num_columns=3,
                verbose=verbose,
                reward_profile=reward_profile,
                cost_profile=cost_profile,
            )
            agents.append(agent)

        # Train with master coordination
        print(f"Running column generation experiment for seed {seed}")
        metrics = train_agents_with_dynamic_master(env, agents, num_episodes, verbose=verbose, max_column_generation_rounds=max_column_generation_rounds, fairness=FAIRNESS_ENABLED, fairness_constraint=FAIRNESS_CONSTRAINTS_ENABLED, langrangian=LANGRANGIAN_ENABLED, langrangian_weight=LAMBDA_FAIR, seed=seed)
        print(f"Metrics for seed {seed}: {metrics}")

if __name__ == '__main__':
    main()
