## modules

* `agents`: model-based RL agents that interact with the environment
* `planners`: the planners used to compute each agent's policy
* `scripts`: contains entry point scripts to reproduce experiments
* `util`: common logic for building environments, training, evaluation, and plotting
* `environments`: custom multi-agent resource MDP environment

---

## lp solver

By default, the code uses [cvxpy](https://www.cvxpy.org/).
---

## usage

### 1. install dependencies

Create the environment from the included YAML file:

```bash
conda env create -f environment.yml
conda activate your-env-name  # replace with the name in the YAML
```

---

### 2. run the experiments

```bash
python scripts/factored.py
```

This will run all combinations of:

* **Complexity levels**: `low`, `medium`, `high`
* **Fairness scopes**: `timestep`, `cumulative`
* **Lagrangian weights**: various λ values
* **Seeds**: to ensure reproducibility

---

### 3. results location

Outputs will be saved under:

```
results/<complexity>/<scope>/
```

You’ll find:

* Evaluation `.csv` and `.json` logs
* Plots of fairness metrics, cost history, reduced costs, etc.

---