## Pseudo-code:

factored.py:
```python
Define main():
    Set problem parameters:
        - planning horizon
        - number of episodes
        - maximum column generation rounds
        - number of agents
        - reward profile per agent
        - cost profile per agent

    Validate that reward profile matches number of agents

    Define random seeds for experiments

    Set flags for fairness handling:
        - enable fairness
        - choose between fairness constraints or Lagrangian relaxation
        - define Lagrangian penalty weight

    Create output directory path

    Initialize environment:
        - number of agents
        - resource capacity per timestep
        - planning horizon
        - per-agent reward profile

    For each seed:
        Set random seed

        Initialize agents:
            For each agent:
                - create a decentralized agent with:
                    - ID
                    - planning horizon
                    - resource constraints
                    - reward and cost profiles
                    - initial number of plan columns
                - add to agent list

        Print experiment info

        Train agents using column generation and fairness-aware master coordination:
            Call train_agents_with_dynamic_master() with:
                - environment
                - agent list
                - number of episodes
                - fairness settings
                - maximum column generation rounds
                - Lagrangian parameters
                - current seed

        Print the collected metrics

If file is executed directly:
    Call main()
```

training.py:
```python
define plot_all_metrics(metrics, seed, ...)

Create output directory

For each metric:
    Plot raw values
    If long enough, add a smoothed rolling average
    Save plot to file with descriptive filename

define jain_index(x)

Compute Jain’s fairness index:
    numerator = (sum of x)^2
    denominator = len(x) * sum of squares of x
Return ratio (numerator / denominator)

define train_agents_with_dynamic_master(...)

Initialize metric containers

For each agent:
    Reset internal state

For each episode:
    Reset environment

    Step 1: Generate initial columns for all agents

    Initialize column generation loop (round_idx = 0)

    WHILE True:
        - Build MasterProblem with agents, fairness settings
        - Solve LP

        IF LP not feasible:
            BREAK loop

        - Retrieve dual prices
        - (If fairness constraint) Retrieve fairness duals

        - Each agent generates a new best-response column using duals
        - Calculate reduced costs for all agents

        IF minimum reduced cost is above threshold:
            - Retrieve column distributions
            - BREAK (early stopping)

        IF max rounds exceeded:
            - Retrieve column distributions
            - BREAK

        - Add new columns to agents

        - If verbose, print diagnostic info

        Increment round index

    IF still infeasible:
        Raise error

    Print LP value for episode

    Step 2: Assign policy distribution to agents (based on LP solution)

    Step 2.5: Compute expected stepwise fairness:
        For each timestep:
            For each agent:
                Sum expected claims weighted by policy distribution
        Compute Jain index over expected claims per timestep
        Average them over horizon

    Step 3: Simulate episode with sampled actions:
        Initialize reward, claim, timestep counters
        WHILE not done:
            - For each agent, compute total action probability from policy
            - Sample binary action (claim or not)
            - Compute fairness index at this step
            - Step environment forward using claims
            - Compute actual reward from reward vector
            - Accumulate claims and rewards
            - Advance timestep

    After episode ends:
        - Finalize agents’ episode
        - Compute actual fairness and usage
        - Record all metrics

    Print summary of episode

After all episodes:
    Plot metrics

Return collected metrics
```

master_problem.py:
```python
Class MasterProblem:
    Constructor(agents, resource_capacity, fairness=False, fairness_constraint=False, langrangian=False, langrangian_weight=1.0):
        Set self.agents, self.resource_capacity, self.horizon, self.num_agents
        Set fairness and Lagrangian flags
        Initialize decision_vars, lp, resource_constraints, fairness_constraints

    Method solve():
        Initialize constraints = []
        Clear decision_vars

        For each agent:
            For each column in agent.get_columns():
                Create non-negative decision variable
            Append list of variables for agent to decision_vars

        For each agent:
            Add constraint: sum of decision_vars for that agent == 1

        For each timestep t in horizon:
            expected_total_claims = 0
            For each agent a:
                For each column c:
                    expected_total_claims += decision_var[a][c] * column["claims"][t]
            Add constraint: expected_total_claims <= resource_capacity
            Store this constraint in resource_constraints

        If fairness_constraint:
            For each timestep t:
                expected_claims_t = []
                For each agent a:
                    expr = sum over c of (decision_var[a][c] * column["claims"][t])
                    Append expr to expected_claims_t
                mean_claim_t = sum(expected_claims_t) / num_agents
                For each agent a:
                    Add constraints: expected_claims_t[a] within ±epsilon of mean_claim_t
            Store last fairness constraints in fairness_constraints

        total_expected_reward = 0

        If langrangian:
            For each timestep t:
                expected_claims_t = []
                For each agent a:
                    expr = sum over c of (decision_var[a][c] * column["claims"][t])
                    Append expr to expected_claims_t
                mean_claim_t = average of expected_claims_t
                diffs = [claim - mean for claim in expected_claims_t]
                variance_t = sum of squares of diffs
                Subtract langrangian_weight * variance_t from total_expected_reward

        For each agent a:
            For each column c:
                reward = sum(column["reward"])
                cost = sum(agent.fixed_cost_vector * column["claims"])
                net_value = reward - cost
                total_expected_reward += decision_var[a][c] * net_value

        Define objective = Maximize(total_expected_reward)
        Solve LP with objective and constraints
        Store LP problem in self.lp

        If fairness_constraint:
            Print duals of fairness_constraints

        Print LP status and objective value
        Return LP value and get_decision_distribution()

    Method get_dual_prices():
        Return array of dual values for resource_constraints

    Method get_fairness_duals():
        If fairness_constraint:
            Return array of dual values for fairness_constraints
        Else return None

    Method get_decision_distribution():
        For each agent’s decision_vars:
            Extract .value from each var
            Normalize into a probability distribution
        Return list of distributions (one per agent)
```

abs_opt_cmdp.py:
```python
Class DecentralizedAgentWithColumns:

    Constructor(agent_id, horizon, resource_capacity, num_columns, verbose, reward_profile, cost_profile):
        Initialize attributes:
            agent_id, horizon, resource_capacity, num_columns, verbose
            reward_profile (default fallback if None)
            cost_profile (default fallback if None)
        Initialize:
            columns = []
            selected_plan = None
            fixed_reward_vector = None
            fixed_cost_vector = None
            episode = 0

    Method generate_candidate_columns():
        Clear current columns
        If fixed_reward_vector is None:
            Sample reward vector from reward_profile
        If fixed_cost_vector is None:
            Sample cost vector from cost_profile
        Call generate_lp_column() to add a greedy column
        Add a fallback "do-nothing" column with zero claims and zero reward
        Return list of columns

    Method generate_lp_column(penalty_weight=0.1):
        Define LP variable: claim_vars for horizon timesteps
        Sample reward_vector from reward_profile
        Define objective:
            maximize sum(reward_vector * claim_vars) - penalty_weight * sum(claim_vars)
        Add constraints: claim_vars ∈ [0, 1]
        Solve the LP
        Clip solution to [0, 1]
        Append column with claims and reward_vector to self.columns

    Method generate_new_column_based_on_feedback(dual_prices, fairness_duals=None):
        Get fixed reward and cost vectors
        total_dual = dual_prices + fairness_duals (if present)
        adjusted_value = reward_vector - cost_vector * total_dual
        Define LP variable: claim_vars
        Define objective:
            maximize sum(adjusted_value * claim_vars) - regularization penalty
        Add constraints: claim_vars ∈ [0, 1]
        Solve LP
        Clip and store result
        Append new column with claims and reward to self.columns

    Method generate_best_response_column(dual_prices, fairness_duals=None):
        Get fixed reward and cost vectors
        total_dual = dual_prices + fairness_duals (if present)
        adjusted_reward = reward_vector - cost_vector * total_dual
        Define LP variable: claim_vars
        Define objective:
            maximize sum(adjusted_reward * claim_vars) - L2 penalty
        Add constraints: claim_vars ∈ [0, 1]
        Solve LP
        Clip result and check if it's already in self.columns
        If duplicate: return column with reduced_cost = 0
        Else: compute reduced_cost = -LP.objective_value
              append column and return it

    Method get_last_column_reduced_cost():
        Return most recent stored reduced cost or 0 if not present

    Method get_columns():
        Return self.columns

    Method end_episode():
        Reset fixed_reward_vector (optionally reset other episode-level state)

    Method reset():
        Clear columns and selected_plan
```

resource_mdp_env.py:
```python
Class ResourceMDPEnv (inherits from gym.Env):

    Constructor(n_agents, resource_capacity=1, max_steps=5, reward_profile=None):
        Set number of agents, resource capacity, and episode length
        Set reward profile (default uniform rewards if not specified)
        Initialize timestep to 0
        Initialize usage_vector and last_claims to zero arrays
        Define observation_space: vector of length 1 + n_agents, values in [0, 1]
        Define action_space: vector of length n_agents, values in [0, 1]

    Method reset():
        Reset timestep, usage_vector, and last_claims to zero
        Return current observation (via _get_obs)

    Method _get_obs():
        Return concatenated vector: [normalized_time] + usage_vector

    Method step(actions):
        Input: actions — array of claim probabilities ∈ [0,1] for each agent
        Assert actions have correct shape
        Sample binary claims from Bernoulli(actions)
        For each agent:
            If claimed:
                Sample reward from uniform(low, high) using agent’s reward_profile
        Update usage_vector with claims
        Update last_claims
        Increment timestep
        Determine if episode is done (timestep ≥ max_steps)
        Return: (observation, reward_vector, done_flag, {"claims": last_claims})

    Method render(mode='human'):
        Print current timestep, usage_vector, and last_claims
```
