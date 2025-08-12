from agents.decentralized_agent import DecentralizedAgentWithColumns
from env.resource_mdp_env import ResourceMDPEnv

def build_env_and_agents(horizon, num_agents, resource_capacity, reward_profile, cost_profile, lambda_fair, SL_states, TL, limit_fn, verbose=True):
    env = ResourceMDPEnv(
        n_agents=num_agents,
        resource_capacity=resource_capacity,
        max_steps=horizon,
        reward_profile=reward_profile,
        SL_states=SL_states,
        TL=TL,
        limit_fn=limit_fn,
        verbose=verbose
    )
    agents = [
        DecentralizedAgentWithColumns(
            agent_id=i,
            horizon=horizon,
            verbose=verbose,
            reward_profile=reward_profile,
            cost_profile=cost_profile,
            langrangian_weight=lambda_fair,
            # scale α so that high‐reward agents “feel” cost more:
            cost_weight=reward_profile[i][0]*0.05
        )
        for i in range(num_agents)
    ]
    return env, agents

