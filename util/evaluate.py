import numpy as np
import scipy.stats as st

def gini_index(x):
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    total = cumx[-1]
    if total == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cumx) / total) / n

def ci(data, confidence=0.95):
    """
    Compute 95% confidence interval for a list of values.
    Returns (mean, lower_bound, upper_bound)
    """
    mean = np.mean(data)
    sem = st.sem(data)
    margin = sem * st.t.ppf((1 + confidence) / 2., len(data)-1) if len(data) > 1 else 0
    return mean, mean - margin, mean + margin
import numpy as np

def mc_evaluate_policy(env_factory, agent_factory, saved_columns, column_distributions, num_rollouts=1000):
    returns, usages = [], []
    jains, minshares, nsws, ginis, variances = [], [], [], [], []
    jains_timestep_means, minshares_timestep_means = [], []
    nsws_timestep_means, ginis_timestep_means, variances_timestep_means = [], [], []
    violations = 0

    for _ in range(num_rollouts):
        env = env_factory()
        agents = agent_factory()

        for a, agent in enumerate(agents):
            agent.columns = saved_columns[a]
            agent.distribution = column_distributions[a]

        obs = env.reset()
        cum_reward = np.zeros(len(agents))
        cum_claims = np.zeros((env.max_steps, len(agents)))
        timestep_jain, timestep_minshare, timestep_nsw = [], [], []
        timestep_gini, timestep_variance = [], []

        done = False
        t = 0
        while not done:
            claims = []
            for a, agent in enumerate(agents):
                c = np.random.choice(len(agent.columns), p=agent.distribution)
                claims.append(agent.columns[c]["claims"][t])
            action = np.array(claims)

            obs, reward, done, info = env.step(action)
            cum_reward += reward
            cum_claims[t] = action

            if action.sum() > env.resource_capacity:
                violations += 1

            # Per-timestep fairness metrics
            jain_t = (action.sum() ** 2) / (len(action) * np.sum(action ** 2) + 1e-8)
            minshare_t = np.min(action) / env.resource_capacity
            nsw_t = np.exp(np.mean(np.log(action + 1e-8)))
            gini_t = gini_index(action)
            variance_t = np.var(action)

            timestep_jain.append(jain_t)
            timestep_minshare.append(minshare_t)
            timestep_nsw.append(nsw_t)
            timestep_gini.append(gini_t)
            timestep_variance.append(variance_t)

            t += 1

        total_claims = cum_claims.sum(axis=0)
        total_resource = env.resource_capacity * env.max_steps

        jain = (total_claims.sum() ** 2) / (len(total_claims) * np.sum(total_claims ** 2) + 1e-8)
        minshare = np.min(total_claims) / total_resource
        nsw = np.exp(np.mean(np.log(total_claims + 1e-8)))
        gini = gini_index(total_claims)
        variance = np.var(total_claims)

        returns.append(cum_reward.sum())
        usages.append(total_claims.sum() / total_resource)
        jains.append(jain)
        minshares.append(minshare)
        nsws.append(nsw)
        ginis.append(gini)
        variances.append(variance)

        # Average per-timestep fairness metrics
        jains_timestep_means.append(np.mean(timestep_jain))
        minshares_timestep_means.append(np.mean(timestep_minshare))
        nsws_timestep_means.append(np.mean(timestep_nsw))
        ginis_timestep_means.append(np.mean(timestep_gini))
        variances_timestep_means.append(np.mean(timestep_variance))

    return {
        "avg_return": ci(returns),
        "avg_usage": ci(usages),
        "avg_fairness_jain": ci(jains),
        "avg_fairness_minshare": ci(minshares),
        "avg_fairness_nsw": ci(nsws),
        "avg_fairness_gini": ci(ginis),
        "claim_variance": ci(variances),
        "avg_fairness_jain_timestep": ci(jains_timestep_means),
        "avg_fairness_minshare_timestep": ci(minshares_timestep_means),
        "avg_fairness_nsw_timestep": ci(nsws_timestep_means),
        "avg_fairness_gini_timestep": ci(ginis_timestep_means),
        "claim_variance_timestep": ci(variances_timestep_means),
        "capacity_violation_rate": np.round(violations / (num_rollouts * env.max_steps), 4)
    }

