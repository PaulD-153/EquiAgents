from agents.decentralized_agent import DecentralizedAgentWithColumns
from env.resource_mdp_env import ResourceMDPEnv

def build_env_and_agents(horizon, num_agents, resource_capacity, reward_profile, cost_profile, lambda_fair):
    env = ResourceMDPEnv(
        n_agents=num_agents,
        resource_capacity=resource_capacity,
        max_steps=horizon,
        reward_profile=reward_profile
    )
    agents = [
        DecentralizedAgentWithColumns(
            agent_id=i,
            horizon=horizon,
            resource_capacity=resource_capacity,
            num_columns=3,
            verbose=False,
            reward_profile=reward_profile,
            cost_profile=cost_profile,
            langrangian_weight=lambda_fair
        )
        for i in range(num_agents)
    ]
    return env, agents

