from agents.decentralized_agent import DecentralizedAgentWithColumns
from env.resource_mdp_env import ResourceMDPEnv

def build_env_and_agents(horizon, num_agents, resource_capacity, reward_profile, cost_profile, lambda_fair, SL_states, TL, limit_fn):
    env = ResourceMDPEnv(
        n_agents=num_agents,
        resource_capacity=resource_capacity,
        max_steps=horizon,
        reward_profile=reward_profile,
        SL_states=SL_states,
        TL=TL,
        limit_fn=limit_fn
    )
    agents = [
        DecentralizedAgentWithColumns(
            agent_id=i,
            horizon=horizon,
            num_columns=3,
            verbose=False,
            reward_profile=reward_profile,
            cost_profile=cost_profile,
            langrangian_weight=lambda_fair
        )
        for i in range(num_agents)
    ]
    return env, agents

