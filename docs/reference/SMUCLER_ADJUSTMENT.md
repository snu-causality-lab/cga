# Smucler–Sapienza–Rotnitzky efficient adjustment

## Overview

`smucler_adjustment_set` implements Definition 6, Theorem 2 and Algorithms 1–3
of *Efficient adjustment sets in causal graphical models with hidden variables*,
[arXiv:2004.10521v3](https://arxiv.org/pdf/2004.10521v3), dated 26 May 2020.
The manuscript has a related [Biometrika publication](https://doi.org/10.1093/biomet/asab018);
the implementation's checked source
is this specific preprint version, with SHA-256
`06a0559f953be52ee00a30a2ad23c0e1966d2a3ba0fd20db00ff64adc98796d1`.

## Contract and example

The input is a validated `DAG` containing both observed and hidden vertices.
`observed` specifies the measurable subset N; omission means all vertices.
Treatment A and outcome Y are single vertices, with Y a strict descendant of A.
The optional `policy_covariates` L are observed non-descendants of A and must
be included in the selected set. These are the inclusion assumptions on pages
4–5 of the source. Raw directed/bidirected topology, MAGs and PAGs are not inputs
to this algorithm. A hidden-variable DAG is not inferred from an endpoint graph.

```python
from causal_graph_algorithms import (
    DAG,
    DirectedBidirectedGraph,
    SmuclerAdjustmentSet,
    smucler_adjustment_set,
)

# Figure 5 / Example 2 at K=3, checked against the drawn edges.
graph = DAG.from_graph(DirectedBidirectedGraph(
    nodes=("A", "T", "W1", "W2", "W3", "W4", "Y"),
    directed=[("T", "A"), ("A", "Y"), ("W4", "Y")]
             + [(w, v) for w in ("W1", "W2", "W3") for v in ("T", "Y")],
))
for objective in ("optimal", "optimal_minimal", "optimal_minimum"):
    result = smucler_adjustment_set(graph, "A", "Y", objective=objective)
    assert isinstance(result, SmuclerAdjustmentSet)
    print(objective, result.vertices)
```

Expected output:

```text
optimal ('W1', 'W2', 'W3', 'W4')
optimal_minimal ('W1', 'W2', 'W3')
optimal_minimum ('T',)
```

| Objective | Guaranteed comparison class |
| --- | --- |
| `optimal` | All observed adjustment sets, if N=V or N is a subset of ancestors of {A,Y}∪L |
| `optimal_minimal` (default) | Inclusion-minimal observed adjustment sets containing L |
| `optimal_minimum` | Observed adjustment sets containing L of smallest cardinality |

The efficiency comparison is the paper's asymptotic variance of nonparametric
adjustment estimators of a point-policy value, under its estimation regularity
assumptions. It is not a guarantee about arbitrary estimators, finite-sample
error, measurement costs, multivariate exposures, or longitudinal policies.
“Minimal” and “minimum” are different, as the example demonstrates.

Malformed inputs raise `SmuclerAdjustmentDomainError`. Algorithm outcomes are
immutable, sealed values with graph declaration order preserved:

- `SmuclerAdjustmentSet.vertices` is the selected tuple, including mandatory L.
- `SmuclerAdjustmentNotApplicable.reason == "no_observed_adjustment_set"`
  means no valid observed adjustment set containing L exists. Its
  `valid_candidate` is `None`.
- `reason == "global_optimality_not_established"` means adjustment exists but
  the sufficient condition for `optimal` was not met. Its `valid_candidate`
  is a valid observed adjustment set without a global guarantee. This outcome
  does not establish that a global optimum is absent.

Every result exposes `reduction`: ordered H0 vertices and edges, H1 vertices
and edges, forbidden vertices, and ignored vertices. The reduction is an
undirected auxiliary graph, not a causal DAG or latent projection.
Selectors accept iterables of distinct declared string labels; a string means
one vertex, including labels with multiple characters. Duplicate labels,
undeclared labels, bytes and non-string labels are rejected.

## Algorithm and implementation choices

H0 is the moral graph after restricting to ancestors of {A,Y}∪L and removing
the first edge of each causal A-to-Y path. H1 removes hidden and forbidden
internal vertices, connects every retained boundary of an ignored connected
component into a clique, and adds A–L and Y–L edges. These additions require
every A–Y vertex cut to include L.

Under the inclusion assumptions, adjacent A and Y in H1 mean no observed
adjustment exists. Conversely, if they are nonadjacent, the neighbors of Y
form a cut and hence a valid adjustment set by Proposition 4. Checking this
condition after constructing H1 replaces the paper's preliminary FINDADJ call.
It preserves the existence decision but does not claim FINDADJ's complexity.

The `optimal` result is the neighbors of Y. Algorithm 3 computes
`optimal_minimal` by exploring from A and stopping at the first neighbors of
Y. Algorithms 1–2 compute `optimal_minimum`: find a maximum collection of
internally vertex-disjoint A–Y paths, then on each path choose the vertex
nearest Y that belongs to some minimum cut. To test membership, add A–v and
v–Y and check whether the minimum-cut cardinality is unchanged.

Maximum flows use unit-capacity split internal vertices and deterministic
Edmonds–Karp breadth-first augmentations, followed by integral flow
decomposition. The unchanged base minimum-cut cardinality is computed once.
There is no enumeration of adjustment sets or vertex cuts in production.
For n vertices, the direct moralization and ignored-component implementation
uses at most O(n³) work; a flow uses O(n(n+m)) work on H1 with m edges, and
Algorithms 1–2 make at most n+1 flows. A conservative overall bound is O(n⁴)
time and O(n²) space. This implementation is polynomial but does not claim the
paper's O(n^(7/2)) bound, which assumes a faster maximum-flow subroutine.

## Source discrepancies and interpretation

These observations apply to the checked arXiv v3, not automatically to later
published versions.

- The [arXiv version record](https://arxiv.org/abs/2004.10521) explicitly reports
  an author correction to Example 2. The fixtures use the corrected v3; they
  do not reconstruct the superseded example or claim that correction as a new
  finding.
- The displayed definition of O_min on page 11 passes G to
  `cc(nb_H1(Y), A, G)`. The following prose, Algorithm 3, and its proof on
  page 23 specify traversal in H1. The implementation follows those mutually
  consistent definitions. This is a graph-symbol correction, not an asserted
  counterexample to the theorem.
- Examples 3 and 4 on page 13 write the reverse noninclusion when discussing
  failure of Theorem 2's sufficient condition. The code checks the theorem's
  direction, N ⊆ ancestors({A,Y}∪L), with equality allowed. A regression
  includes an unobserved ancestor to distinguish the two directions.
- Proposition 4 gives cut/adjustment equivalence for sets within the ancestral
  restriction, and exact equivalence for minimal and minimum sets. It does not
  equate arbitrary observed adjustment sets outside that restriction with H1
  cuts. Proposition 5 gives a sufficient graphical relation for variance
  dominance, not an equivalence with every possible variance ordering.

## Verification

The initial focused run was RED (`ModuleNotFoundError`) before the module was
created. A subsequent RED run established missing figure fixtures and rejected
singleton string selectors before adding the fixtures and aligning selectors
with the package contract. The regression suite independently checks:

- all 64 DAGs consistent with one four-vertex order, every reachable ordered
  A,Y pair, every observed subset containing A,Y, and every observed
  non-descendant policy subset;
- 48 seeded six-vertex DAGs, including hidden vertices and static/dynamic
  policies;
- validity by enumerating original-DAG simple paths, excluding forbidden
  descendants and blocking every noncausal path, without calling production
  ancestry, moralization, separation or flow routines;
- the two sufficient d-separations for variance dominance against every valid
  competitor in the relevant class on those small graphs;
- independent exact summation of positive binary-CPT distributions and the
  nonparametric adjustment influence-function variance for the 48 seeded
  graphs that admit adjustment;
- all 512 five-vertex undirected graphs with no A–Y edge, comparing the flow
  output to an exhaustively computed minimum-cut lattice infimum;
- a residual-flow rerouting example and a 66-vertex graph with 2³² minimum
  cuts, which returns via flows without enumerating that collection;
- all five figures and their G/H0/H1 panels, transcribed from rendered PDF
  pages, with each adjustment output adjudicated by the independent path
  criterion; input rejection and immutable result contracts.

The versioned fixture is
[`smucler_2020_figures.json`](../../tests/fixtures/papers/smucler_2020_figures.json).
It records source provenance, reported results and adjudication separately.

| Figure / example | Reproduced behavior |
| --- | --- |
| 1, Example 1 | All DAG/H0/H1 edges; O = O_min = O_m = {L,F} |
| 2, Example 2 at K=3 | All DAG/H0=H1 edges; O={W1,W2,W3,W4}, O_min={W1,W2,W3}, O_m={T} |
| 3, Example 3 | All three panels; exactly three valid observed sets; empty O_min/O_m; global sufficient condition fails |
| 4, Example 4 | All three panels; O_min/O_m={L}; global sufficient condition fails although Proposition 2 establishes {L,F} as globally optimal |
| 5, Example 2 at K=4 | Both panels with the repeated arrow pattern expanded; O={W1,…,W5}, O_min={W1,…,W4}, O_m={T} |

For Figure 3, independently constructed positive binary models (seeds 0 and 6)
give opposite variance orders for the empty set and {Z1,Z2}. These are new
witnesses to the reported behavior, not a reproduction of numerical parameters
from the paper. Figure 4's {L,F} result is checked by the independent
Proposition 2 d-separation and exact binary variances for eight deterministic
dynamic-policy models; it is not mislabeled as an output of Theorem 2's
algorithm. Figure 5 is parameterized for arbitrarily large K; finite fixtures
instantiate K=3 and K=4 and do not claim coverage of every K. The source has
no additional figure panels or numerical plots to reproduce.

These checks support this implementation on the stated finite universes;
they are not a proof over all graphs or arbitrary distributions. No independent
oracle imports production graph traversal or reduction helpers.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/python/test_smucler_adjustment.py
python -m mypy --strict --cache-dir /tmp/cga-smucler-mypy-cache python/src/causal_graph_algorithms/smucler_adjustment.py tests/python/smucler_oracles.py tests/python/test_smucler_adjustment.py
```
