# TSP Objective Logic

This note summarizes the parameterized TSP objective used when parsing and training on TravellingSalesman XCSP3 instances.

## Config

The TSP objective is controlled from `configs/TSP/TSP_default.json`:

```json
"tsp_objective": {
    "enabled": true,
    "use_duplicate_penalty": true,
    "use_missing_edge_penalty": true,
    "use_tour_cost": true
}
```

If `enabled` is `false`, the model uses the normal ANYCSP reward based on unsatisfied constraints.

If `enabled` is `true`, the model uses a TSP-specific lexicographic objective. Each component can be turned on or off independently.

## Assignment Meaning

For TSP, each CSP variable represents a position in the tour:

```text
x_0 = first city visited
x_1 = second city visited
...
x_{N-1} = last city visited
```

Every variable has domain:

```text
{0, 1, ..., N-1}
```

The selected tour is:

```text
x_0 -> x_1 -> ... -> x_{N-1} -> x_0
```

## Penalties

### Duplicate Penalty

This measures violation of `allDifferent`.

For each city `c`, count how many times it appears in the assignment:

```text
count(c) = number of variables assigned to city c
```

Then:

```text
duplicate_penalty = sum_c max(count(c) - 1, 0)
```

Examples:

```text
[0, 1, 2, 3] -> duplicate_penalty = 0
[0, 1, 1, 3] -> duplicate_penalty = 1
[0, 0, 1, 1] -> duplicate_penalty = 2
```

### Missing Edge Penalty

This measures broken tour transitions.

For every adjacent pair in the tour, including the return edge:

```text
(x_0, x_1), (x_1, x_2), ..., (x_{N-1}, x_0)
```

the parser stores whether the edge exists:

```text
L_E(x_i, x_{i+1}) = 1 if edge exists
L_E(x_i, x_{i+1}) = 0 otherwise
```

Then:

```text
missing_edge_penalty = sum_i (1 - L_E(x_i, x_{i+1}))
```

### Tour Cost

For each selected adjacent edge, the parser stores a cost:

```text
cost(x_i, x_{i+1})
```

The total tour cost is:

```text
tour_cost = sum_i cost(x_i, x_{i+1})
```

If an edge does not exist, the missing edge penalty is what makes that assignment bad.

## BIG_M

The objective uses a large feasibility barrier:

```text
BIG_M = sum_i max_edge_cost_i + 1
```

where `max_edge_cost_i` is the largest edge cost available for transition `i`.

For the current TravellingSalesman files, each transition uses the same cost matrix, so this is effectively:

```text
BIG_M = N * max_edge_cost + 1
```

This makes one duplicate city or one missing edge worse than any expensive valid tour.

## Objective Formula

The full objective is:

```text
tsp_objective =
    duplicate_weight * BIG_M * duplicate_penalty
  + missing_edge_weight * BIG_M * missing_edge_penalty
  + cost_weight * tour_cost
```

The weights are controlled by config:

```text
duplicate_weight    = 1 if use_duplicate_penalty is true, else 0
missing_edge_weight = 1 if use_missing_edge_penalty is true, else 0
cost_weight         = 1 if use_tour_cost is true, else 0
```

With all components enabled:

```text
tsp_objective =
    BIG_M * duplicate_penalty
  + BIG_M * missing_edge_penalty
  + tour_cost
```

Lower is better.

## Reward

ANYCSP uses REINFORCE, so the model needs a reward to maximize. For TSP, lower objective is converted into positive improvement reward.

Let:

```text
O_0 = objective at the initial assignment
O_t = objective at step t
```

The raw TSP reward is:

```text
reward_t = (O_0 - O_t) / BIG_M
```

So:

```text
O_t < O_0 -> positive reward
O_t = O_0 -> zero reward
O_t > O_0 -> negative reward
```

When `reward` is set to `"improve"`, only new improvements are kept:

```text
reward_t = max(0, reward_t - best_previous_reward)
```

This matches the original ANYCSP idea: reward the model when it finds a better assignment than before.

## Ablation Examples

Learn only `allDifferent`:

```json
"tsp_objective": {
    "enabled": true,
    "use_duplicate_penalty": true,
    "use_missing_edge_penalty": false,
    "use_tour_cost": false
}
```

Learn only valid adjacent edges:

```json
"tsp_objective": {
    "enabled": true,
    "use_duplicate_penalty": false,
    "use_missing_edge_penalty": true,
    "use_tour_cost": false
}
```

Learn only tour cost:

```json
"tsp_objective": {
    "enabled": true,
    "use_duplicate_penalty": false,
    "use_missing_edge_penalty": false,
    "use_tour_cost": true
}
```

Full TSP objective:

```json
"tsp_objective": {
    "enabled": true,
    "use_duplicate_penalty": true,
    "use_missing_edge_penalty": true,
    "use_tour_cost": true
}
```

## Reading The Printed Metrics

Training and evaluation print:

```text
Mean Unsat Count
Solved
Mean TSP Objective
Mean Tour Cost
Duplicates
Missing Edges
```

Interpretation:

```text
Duplicates > 0      -> allDifferent is violated
Missing Edges > 0   -> at least one transition breaks the tour
Tour Cost           -> cost of selected adjacent edges
TSP Objective       -> lexicographic training objective
Solved = 100%       -> no unsatisfied constraints
```

For a valid TSP tour:

```text
Duplicates = 0
Missing Edges = 0
TSP Objective = Tour Cost
```

