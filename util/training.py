import numpy as np
from planners.master_problem import MasterProblem


def compute_fairness_score(values, fairness_type, fairness_scope='timestep', horizon=None):
    if fairness_scope == 'cumulative':
        values = np.array(values) / horizon 

    if fairness_type == "jain":
        numerator = (np.sum(values)) ** 2
        denominator = len(values) * np.sum(values ** 2)
        return numerator / (denominator + 1e-8)
    elif fairness_type == "nsw":
        return np.exp(np.mean(np.log(values + 1e-6)))
    elif fairness_type == "minshare":
        return np.min(values)
    elif fairness_type == "gini":
        sorted_vals = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        gini_coeff = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals) + 1e-6) - (n + 1) / n
        return 1.0 - gini_coeff
    elif fairness_type == "variance":
        mean = np.mean(values)
        variance = np.mean((values - mean) ** 2)
        return variance
    else:
        raise ValueError(f"Unknown fairness type: {fairness_type}")

def train_agents_with_dynamic_master(
    env,
    agents,
    number_of_episodes,
    max_column_generation_rounds=5,
    verbose=True,
    langrangian_weight=None,
    fairness_metrics=None,
    fairness_scope='timestep',
    use_gradient_fairness=True,
    seed=None
):
    def log(msg):
        if verbose:
            print(msg)

    metrics = {'fairness_impact': [], 'expected_return': [], 'optimal usage': [], 'min_rc': []}
    min_rc_history = []
    cost_history = []
    net_value_history = []
    expected_return_history = []

    for fair_scope in ['timestep', 'cumulative']:
        for fairness_eval in fairness_metrics:
            metrics[f'expected_fairness_{fairness_eval}_{fair_scope}'] = []

    for agent in agents:
        agent.reset()

    NUM_SCENARIOS = 5  # SL scenarios per episode

    for episode in range(number_of_episodes):
        cost_per_episode = []
        rc_per_episode = []
        expected_reward_per_episode = []
        net_value_per_episode = []
        env.reset()

        # --- 1) Sample SL scenarios and build capacity schedules ---
        scenario_caps = []
        scenario_SLs = []
        for k in range(NUM_SCENARIOS):
            sL_path = [env.SL_states[0]]
            for t in range(env.max_steps):
                p = env.TL[env.SL_states.index(sL_path[-1])]
                sL_path.append(np.random.choice(env.SL_states, p=p))
            sL_path = sL_path[:env.max_steps]
            caps_k = [env.limit_fn(t, sL_t) for t, sL_t in enumerate(sL_path)]
            scenario_SLs.append(sL_path)
            scenario_caps.append(caps_k)

        capacity_schedule = [
            sum(scenario_caps[k][t] for k in range(NUM_SCENARIOS)) / NUM_SCENARIOS
            for t in range(env.max_steps)
        ]

        # Assign scenario and schedule to agents
        rep_SL = scenario_SLs[0]
        for agent in agents:
            agent.SL_traj = rep_SL
            agent.capacity_schedule = capacity_schedule
            agent.columns = agent.generate_candidate_columns()

        round_idx = 0
        while True:
            log(f"[DEBUG] capacity_schedule: {capacity_schedule}")

            master = MasterProblem(
                agents,
                resource_capacity=env._base_capacity,
                capacity_schedule=capacity_schedule,
                langrangian_weight=langrangian_weight,
                fairness_scope=fairness_scope,
                use_gradient_fairness=use_gradient_fairness,
            )

            master_value, _, fairness_impact = master.solve()

            if master.lp.status not in ["optimal", "optimal_inaccurate"]:
                log(f"Master LP not feasible at round {round_idx}")
                break

            dist = master.get_decision_distribution()
            expected_cost_round = 0.0
            expected_net_value_round = 0.0

            for a, agent in enumerate(agents):
                for c, column in enumerate(agent.columns):
                    alpha = getattr(agent, "cost_weight", 1.0)
                    cost_c = sum(agent.cost_fn(t, agent.SL_traj[t]) * column["claims"][t]
                                 for t in range(agent.horizon))
                    reward = np.sum(column["reward"])
                    expected_cost_round += dist[a][c] * (alpha * cost_c)
                    expected_net_value_round += dist[a][c] * (reward - cost_c)

            log(f"[Seed {seed}] Episode {episode}, Round {round_idx}: "
                f"cost = {expected_cost_round:.4f}, net value = {expected_net_value_round:.4f}")

            cost_per_episode.append(expected_cost_round)
            net_value_per_episode.append(expected_net_value_round)
            expected_reward_per_episode.append(master_value)

            if langrangian_weight > 0 and use_gradient_fairness:
                fairness_gradients_per_agent = master.compute_fairness_gradients()
            else:
                fairness_gradients_per_agent = [np.zeros(env.max_steps) for _ in agents]

            dual_prices = master.get_dual_prices()

            new_columns = []
            for a_idx, agent in enumerate(agents):
                fairness_dual = langrangian_weight * fairness_gradients_per_agent[a_idx]
                column = agent.generate_best_response_column(dual_prices, fairness_duals=fairness_dual)
                reduced_cost = agent.get_last_column_reduced_cost()
                new_columns.append((column, reduced_cost))

            reduced_costs = [rc for _, rc in new_columns]
            min_rc = min(reduced_costs)

            log(f"Round {round_idx}: min reduced cost {min_rc:.6f}")
            rc_per_episode.append(min_rc)

            if min_rc > -1e-2 or round_idx >= max_column_generation_rounds:
                column_distributions = master.get_decision_distribution()
                break

            for agent, (column, rc) in zip(agents, new_columns):
                agent.columns.append(column)

            round_idx += 1

        # Compute expected reward (ignoring fairness penalty)
        total_expected_reward = 0
        total_expected_cost = 0
        for a, agent in enumerate(agents):
            for c, column in enumerate(agent.columns):
                reward = np.sum(column["reward"])
                cost_c = sum(
                    agent.cost_fn(t, agent.SL_traj[t]) * column["claims"][t]
                    for t in range(agent.horizon)
                )
                prob = column_distributions[a][c]
                net_value = reward - cost_c
                total_expected_reward += prob * net_value
                total_expected_cost += prob * cost_c

        # Log cost as a new metric
        metrics.setdefault('expected_cost', []).append(total_expected_cost)

        # Compute expected claims for fairness metrics
        expected_claims_per_timestep = np.zeros((env.max_steps, len(agents)))
        for t in range(env.max_steps):
            for a_idx, agent in enumerate(agents):
                for i, col in enumerate(agent.columns):
                    prob = column_distributions[a_idx][i]
                    expected_claims_per_timestep[t, a_idx] += prob * col["claims"][t]

        # Save histogram after last episode
        if episode == number_of_episodes - 1:
            agent_expected_claims = np.mean(expected_claims_per_timestep, axis=0)

        # Compute cumulative claims
        cumulative_claims = np.sum(expected_claims_per_timestep, axis=0)

        for fairness_eval in fairness_metrics:
            fairness_value = compute_fairness_score(cumulative_claims, fairness_type=fairness_eval, fairness_scope='cumulative', horizon=env.max_steps)
            metrics[f'expected_fairness_{fairness_eval}_cumulative'].append(fairness_value)
            fairness_values = [
                compute_fairness_score(expected_claims_per_timestep[t], fairness_type=fairness_eval, fairness_scope='timestep', horizon=env.max_steps)
                for t in range(env.max_steps)]
            metrics[f'expected_fairness_{fairness_eval}_timestep'].append(np.mean(fairness_values))

        optimal_usage_score = np.sum(expected_claims_per_timestep) / (env.resource_capacity * env.max_steps)
        metrics['optimal usage'].append(optimal_usage_score)
        metrics['expected_return'].append(total_expected_reward)
        metrics['fairness_impact'].append(fairness_impact)
        metrics['min_rc'].append(min_rc)
        min_rc_history.append(rc_per_episode)
        cost_history.append(cost_per_episode)
        net_value_history.append(net_value_per_episode)
        expected_return_history.append(expected_reward_per_episode)

    saved_columns = [list(agent.columns) for agent in agents]
    saved_distributions = [dist.copy() for dist in column_distributions]

    return (
        metrics,
        min_rc_history,
        cost_history,
        net_value_history,
        expected_return_history,
        agent_expected_claims,
        saved_columns,
        saved_distributions
    )

def train_agents_primal_dual(env,
                             agents,
                             num_episodes: int,
                             fairness_metric: str,
                             fairness_scope: str,
                             init_lambda: float = 1.0,
                             beta_f: float = 0.95,
                             alpha: float = 0.1,
                             max_column_generation_rounds: int = 2500,
                             verbose: bool = False,
                             seed: int = None):
    """
    Online primal–dual training:
      - env, agents: as usual
      - num_episodes: number of sequential episodes
      - fairness_metric: e.g. "jain"
      - fairness_scope: "timestep" or "cumulative"
      - init_lambda: starting fairness multiplier
      - beta_f: target fairness threshold
      - alpha: step size for dual update
    Returns:
      history: list of dicts with keys
               ['episode','lambda','fairness','return','cost']
    """
    λ = init_lambda
    history = []

    # reset agents once
    for agent in agents:
        agent.reset()

    f_ema = beta_f  # target fairness level

    for ep in range(num_episodes):
        if seed is not None:
            np.random.seed(seed + ep)

        # train exactly 1 episode under current λ
        metrics, _, cost_history, *_ = train_agents_with_dynamic_master(
            env, agents,
            number_of_episodes=1,
            max_column_generation_rounds=max_column_generation_rounds,
            langrangian_weight=λ,
            fairness_metrics=[fairness_metric],
            fairness_scope=fairness_scope,
            verbose=verbose,
            seed=seed
        )

        # extract the realized fairness and return
        f_val = metrics[f'expected_fairness_{fairness_metric}_{fairness_scope}'][-1]
        r_val = metrics['expected_return'][-1]
        c_val = metrics.get('expected_cost', [None])[-1]

        # hyperparams
        k = 30.0
        δ = 0.005

        rho   = 0.5   # smoothing factor (0<ρ<1)
        f_ema = rho*f_ema + (1-rho)*f_val

        # after computing f_val and beta_f:
        violation = beta_f - f_ema                             # positive if below target
        delta = np.sign(violation) * (np.exp(k * abs(violation)) - 1.0)
        λ = max(0.0, λ + alpha * delta)

        history.append({
            'episode':   ep,
            'lambda':    λ,
            'fairness':  f_ema,
            'return':    r_val,
            'cost':      c_val
        })

        if verbose:
            print(f"[Ep {ep}] fairness={f_ema:.3f}, return={r_val:.2f}, cost={c_val:.2f}, λ→{λ:.3f}, update={delta:.3f}, violation={violation:.3f}")

        # early stopping if we hit the target
        if abs(violation) <= δ:  # δ is a small tolerance
            if verbose:
                print(f"Reached target fairness {beta_f} at episode {ep}, stopping training.")
            break

    return history