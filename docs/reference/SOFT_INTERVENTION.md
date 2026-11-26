# Soft-intervention equivalence

The routines take causal DAGs with any latent variables represented
explicitly. Pass the common observed labels as `observed`. Each routine
adds auxiliary F-roots, then uses the ancestral marginalization
routine to obtain a maximal ancestral graph (MAG) on the observed
vertices and the F-nodes. The equivalence routines compare these MAGs.

## Known controlled targets

`controlled_soft_intervention_mag(graph, observed=..., targets=...)`
implements the 2019 augmentation, with a root for each distinct nonempty
symmetric difference of the supplied target sets. An empty set denotes
an observational regime. Empty differences correspond to isolated roots
and are suppressed, as in Figure 2. The two-graph comparison uses the
same known targets for both DAGs.
[Kocaoglu et al. (2019), Definition 3 and Theorem 2](https://proceedings.neurips.cc/paper/2019/file/c3d96fbd5b1b45096ff04c04038fff5d-Paper.pdf#page=6)

```python
from causal_graph_algorithms import DAG, controlled_soft_intervention_markov_equivalent

left = DAG.from_text("X --> Y")
right = DAG.from_text("Y --> X")
assert not controlled_soft_intervention_markov_equivalent(
    left, right, observed=("X", "Y"), targets=((), ("X",)),
)
```

## Candidate targets with mechanism labels

`psi_soft_intervention_mag(graph, observed=..., regimes=...)` takes an
ordered iterable of mappings. Each mapping states the proposed target
variables and their mechanism labels for that regime. An absent variable
keeps its original mechanism. Labels may be strings or integers, and
are compared only within each variable. The F-nodes correspond to all
pairs of distinct positions, so equal difference sets still have their
separate comparison nodes. The comparison matches positions on both
sides and returns `False` when the regime tuple lengths differ.
[Jaber et al. (2020), Definitions 1 and 4 and Theorem 1](https://proceedings.neurips.cc/paper/2020/file/6cd9313ed34ef58bad3fdd504355e72c-Paper.pdf#page=4)

```python
from causal_graph_algorithms import DAG, psi_markov_equivalent

graph = DAG.from_text("X --> Y")
assert not psi_markov_equivalent(
    graph, ({"X": 1}, {"X": 2}), graph, ({"X": 1}, {"X": 1}),
    observed=("X", "Y"),
)
```

Target names must be observed variables. F-node names are generated to
avoid collisions with graph labels. The equivalence routines use one
common name allocation for the candidate pair. The package evaluates
graphical equivalence for the supplied candidates and regime labels.

[Lean companion for known targets](https://github.com/snu-causality-lab/open-causal-science/blob/main/docs/papers/kocaoglu-et-al-2019-controlled-soft-interventions.md) ·
[Lean companion for unknown targets](https://github.com/snu-causality-lab/open-causal-science/blob/main/docs/papers/jaber-et-al-2020-unknown-soft-targets.md)
