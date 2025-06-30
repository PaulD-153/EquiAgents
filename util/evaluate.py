import numpy as np

def mc_evaluate_policy(env_factory, agent_factory, saved_columns, column_distributions, num_rollouts=1000):
    """
    env_factory: fn() -> fresh ResourceMDPEnv (with SL etc. configured)
    agent_factory: fn() -> fresh agents (so we can attach the saved columns)
    column_distributions: list, one per agent, each an array of shape (num_columns,)
    saved_columns: list, one per agent, each a list of saved column plans
    num_rollouts: how many MC episodes to average
    Returns: dict of avg_return, avg_usage, avg_fairness, capacity_violation_rate, ...
    """
    returns = []
    usages = []            # total claims per episode
    fairness_scores = []   # e.g. final Jain index
    violations = 0

    for _ in range(num_rollouts):
        env = env_factory()
        agents = agent_factory()
        # overwrite each agent's columns & distributions
        for a, agent in enumerate(agents):
            agent.columns = saved_columns[a]
            agent.distribution = column_distributions[a]

        obs = env.reset()
        cum_reward = np.zeros(len(agents))
        cum_claims = np.zeros((env.max_steps, len(agents)))

        done = False
        t = 0
        while not done:
            # For each agent, sample a column c ~ P(c)
            claims = []
            for a, agent in enumerate(agents):
                c = np.random.choice(len(agent.columns), p=agent.distribution)
                claims.append(agent.columns[c]["claims"][t])
            action = np.array(claims)  # shape (num_agents,)

            obs, reward, done, info = env.step(action)
            cum_reward += reward
            cum_claims[t] = action
            # check capacity
            if action.sum() > env.resource_capacity:
                violations += 1
            t += 1

        # compute fairness at end of episode
        # e.g. Jain’s index on total claims:
        total_claims = cum_claims.sum(axis=0)
        jain = (total_claims.sum()**2) / (len(total_claims) * np.sum(total_claims**2) + 1e-8)

        returns.append(cum_reward.sum())
        usages.append(total_claims.sum() / (env.resource_capacity * env.max_steps))
        fairness_scores.append(jain)

    return {
        "avg_return": np.mean(returns),
        "avg_usage": np.mean(usages),
        "avg_fairness": np.mean(fairness_scores),
        "capacity_violation_rate": violations / (num_rollouts * env.max_steps)
    }