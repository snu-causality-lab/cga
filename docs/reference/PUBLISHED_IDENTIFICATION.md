# Identification, fixing, and selection recovery

The Python functions below construct symbolic probability expressions from
graphs and declared data sources. A `Probability` leaf describes an available
observational, intervention, or selected source. `Product`, `Summation`, and
`Ratio` combine those leaves. Use `probability_expression_to_tex(result.formula)`
when an operation returns a formula. The caller supplies the underlying
distributions, support and model assumptions needed to interpret it.

## Known DAGs and acyclic directed mixed graphs

| Operation | Query and source | Basis |
| --- | --- | --- |
| `identify_causal_effect` | `P(Y \| do(X))` from the observed joint | [Shpitser and Pearl, AAAI 2006](https://aaai.org/Papers/AAAI/2006/AAAI06-191.pdf), [Shpitser, 2023](https://arxiv.org/abs/2307.03750) |
| `identify_conditional_causal_effect` | `P(Y \| do(X), Z)` from the observed joint | [Shpitser and Pearl, UAI 2006](https://escholarship.org/uc/item/4s66k31k) |
| `identify_generalized_causal_effect` | `P(Y \| do(X))` from complete hard-experiment kernels | [Kivva et al., UAI 2022](https://proceedings.mlr.press/v180/kivva22a.html) |
| `identify_conditional_generalized_causal_effect` | `P(Y \| do(X), Z)` from complete hard-experiment kernels | [Kivva, Etesami, and Kiyavash, UAI 2023](https://proceedings.mlr.press/v216/kivva23a.html) |

ID applies the intrinsic-district test to the full requested effect before
constructing a formula. A nonidentified strict subquery alone cannot establish
the result for the full query. The tests include the [exact-query hedge
example](../../tests/fixtures/papers/shpitser_2023_figure1c_hedge_counterexample.json).
IDC applies do-calculus Rule 2 and records any hedge for the reduced joint query.

For gID and c-gID, each entry in `experiment_interventions` supplies the full
kernel at every assignment of that target. For example, `((), ("X",))` supplies
the observational joint and the kernel for intervening on `X`. Strict positivity
of the full observational distribution and finite observed domains are premises
even when observation is absent from the input. c-gID checks each condition
candidate `z` by cutting outgoing arrows at `z` and separating the singleton
`{z}` from the outcomes, as required by Rule 2. The printed MaxBI line 5 uses
different sets for this check. The code then runs gID on the reduced joint query.

```python
from causal_graph_algorithms import DAG, IdentifiedEffect, identify_causal_effect

graph = DAG.from_text("Z --> X; Z --> Y; X --> Y")
result = identify_causal_effect(graph, outcomes="Y", interventions="X")
assert isinstance(result, IdentifiedEffect)
```

## Equivalence classes

- `identify_pag_causal_effect` covers marginal intervention effects in a
  no-selection `CausalPAG`. It uses the possible-descendant criterion in the
  extended IDP algorithm. [Jaber, Ribeiro, Zhang, and Bareinboim, R-86](https://causalai.net/r86.pdf)
- `identify_mpdag_causal_effect` applies the MPDAG identification condition
  and constructs the bucket factorization when it succeeds. `enumerate_mpdag_effects`
  implements the IDGraphs split into subclasses that each identify the effect.
  Both operations use the fully observed underlying DAG model.
  [Perković, UAI 2020](https://proceedings.mlr.press/v124/perkovic20a.html),
  [Guo and Perković, AISTATS 2021](https://proceedings.mlr.press/v130/guo21c.html)

## Conditional ADMGs and Tian adjustment

`CADMG` stores random and fixed vertices. `fixable_vertices`,
`fixing_sequence_is_valid`, `fix_vertices`, and `find_fixing_sequence` implement
graph fixing and reachability. `intrinsic_set_is_valid` tests one set and
`intrinsic_sets` enumerates the intrinsic sets. `CADMGKernel` and `fix_kernel`
implement symbolic kernel fixing. The positive nested-Markov premise supplies
the normalization and fixing-order guarantees. A bound on the number of
reachable states is available for enumeration. [Richardson, Evans, Robins,
and Shpitser, 2023](https://doi.org/10.1214/22-AOS2253)

`tian_adjustment_to_uca` provides the singleton Tian sufficient construction and
groups chain factors into consecutive blocks. The supplied topological order
contains every vertex and ends with the outcome. The result describes the
identification expression and its factors. [Jung, Tian, and Bareinboim, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/0c4bc137edaf0eb7f66a87275a8be706-Paper-Conference.pdf)

## Selection recovery

- `recover_conditional_distribution` implements the conditional recovery
  criterion for data observed at one fixed selection event.
  [Bareinboim, Tian, and Pearl, 2014](https://ftp.cs.ucla.edu/pub/stat_ser/r425.pdf)
- `selection_adjustment_pair_is_valid`, `find_selection_adjustment_pair`, and
  `iter_selection_adjustment_pairs` use constructive adjustment pairs with
  selected data and an optional same-population margin.
  [Correa, Tian, and Bareinboim, 2018](https://causalai.net/r29.pdf)
- `recover_causal_effect` performs selected-only RC. When the declared input is
  unconditional it uses ordinary ID. The tests include a six-vertex
  [selected-only example](../../tests/fixtures/papers/correa_tian_bareinboim_2019_rc.json)
  with a recovery expression beyond selected-only RC.
  [Bareinboim and Tian, 2015](https://ftp.cs.ucla.edu/pub/stat_ser/r445.pdf),
  [Correa, Tian, and Bareinboim, 2019](https://causalai.net/r38.pdf)
- `recover_causal_effect_with_external_data` uses IDSB/RCE on a complete
  selected joint and an optional same-population joint margin. The selected
  source is a `Probability` conditioned on one fixed directed sink. A returned
  `RecoveredExternalEffect` supplies a formula. `ExternalRecoveryNotEstablished`
  records that this route produced none. The premises for a returned formula
  include compatible sources and support for its ratios.
  [Correa, Tian, and Bareinboim, 2019](https://causalai.net/r38.pdf)

A recovery budget exception reports resource exhaustion. A failed adjustment
pair search reports exhaustion of the criterion's admissible pair set.

## Counterfactuals

The counterfactual value API represents explicit hard-intervention assignments
and events. `identify_counterfactual` and `identify_conditional_counterfactual`
apply the unit and contradiction rules and reduce supported single-world
queries to ID, IDC or gID. Out-of-scope queries can return
`UnresolvedCounterfactual`. The
[counterfactual calculus](COUNTERFACTUAL_CALCULUS.md) describes the independent-root
model class, AMWN separation, and the three rule checks from Correa and
Bareinboim's ICML 2025 paper.
