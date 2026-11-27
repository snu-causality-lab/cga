# Graph operations

## Latent projection, ancestral marginalization and selection

`latent_projection(graph, retained)` accepts a `DAG` or an
`AcyclicDirectedBidirectedGraph` and returns an acyclic directed–bidirected graph
on the retained vertices. The source definition allows a directed and a
bidirected edge between the same vertices. See [Richardson, Evans, Robins and
Shpitser (2023), Supplementary Material § A](https://doi.org/10.1214/22-AOS2253)
and [Evans (2021), Section 2](https://proceedings.mlr.press/v161/evans21a.html).

```python
from causal_graph_algorithms import DAG, latent_projection

graph = DAG.from_text("L --> X; L --> Y; X --> Y")
projected = latent_projection(graph, ("X", "Y"))
assert projected.topology.directed == {("X", "Y")}
assert projected.topology.bidirected == {("X", "Y")}
```

`ancestral_marginal_condition_transform(graph, conditioned=(), marginalized=())`
implements Richardson–Spirtes's `G[S,L]` on an `AncestralGraph` or
`MaximalAncestralGraph`. The result is a maximal ancestral graph on the vertices
outside the disjoint selection and latent sets. See [Richardson and Spirtes
(2002), Section 4](https://doi.org/10.1214/aos/1031689015). [Projection](../../tests/python/test_projection_contract.py)
and [ancestral-transform](../../tests/python/test_ancestral_transform_contract.py)
tests include exact figure regressions and definition-level reference checks.

`induced_subgraph(graph, retained)` restricts vertices and keeps edges between
retained vertices. It accepts raw `DirectedBidirectedGraph` and `EndpointMarkedGraph`
topologies as well as `DAG`, acyclic directed–bidirected, ancestral and maximal
ancestral views. The result preserves the input graph view. Use
`find_nonmaximality_witness(graph)` for a nonadjacent vertex pair with a primitive
inducing path, or `maximal_ancestral_extension(graph)` to preserve the graph's
m-separation model while adding the required edges. The [maximality tests](../../tests/python/test_maximality_contract.py)
reproduce Richardson–Spirtes Figures 9 and 13.

## Topology conversion, serialization and rendering

| Operation | Input and output |
| --- | --- |
| `directed_bidirected_to_endpoint_marked(graph)` | Converts raw directed–bidirected topology with at most one edge per pair to endpoint-marked topology. A bow raises `LossyConversionError`. |
| `endpoint_marked_to_directed_bidirected(graph)` | Converts raw topology containing directed and bidirected edges. Circle marks and undirected edges raise `LossyConversionError`. |
| `graph_to_canonical_json(graph)` and `graph_from_json(data)` | Round-trip raw topology with schema `cga.graph/1` in declared vertex order. Invalid data raises `SchemaError`. |
| `graph_to_svg(graph)` | Deterministic SVG rendering for small supported graph views and raw topologies. |

## PDAGs, Meek rules, and background knowledge

```text
PDAG.from_graph(endpoint_topology)
CPDAG.from_dag(dag)
CPDAG.from_graph(endpoint_topology)
MPDAG.from_graph(endpoint_topology)
BackgroundKnowledge(*, required=(), forbidden=())
find_consistent_dag_extension(pdag_or_cpdag_or_mpdag)
meek_closure(pdag_or_cpdag_or_mpdag, *, knowledge=None)
dag_markov_equivalent(dag1, dag2)
```

The three sealed views have distinct meanings. PDAG proves directed/undirected
marks and no directed cycle, but permits an empty consistent extension set.
CPDAG proves the full essential graph of a DAG MEC. MPDAG proves all invariant
orientations of its consistent extension set, potentially restricted by knowledge.
Semidirected cycles can be valid: `A → B`, `B — C`, `C — A` is extendable.
A chordless undirected four-cycle is acyclic but has no consistent DAG extension.
Tail-tail edges in these views express unresolved DAG orientation, unlike their
selection interpretation in an ancestral graph. A causal PAG is not a CPDAG.

Extensions preserve the original skeleton, directed edges, and **exact unshielded
colliders** using [Dor–Tarsi's potential-sink construction](https://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf).
Closure implements [Meek R1–R4](https://arxiv.org/abs/1302.4972);
the applicability to arbitrary extendable PDAGs is also described in
[Luttermann et al. 2023](https://proceedings.mlr.press/v213/luttermann23a.html).
CPDAG construction orients the DAG pattern and closes it. `CPDAG.from_graph`
validates by an exact essential-graph roundtrip; a two-node DAG `A → B` is not
accepted as a CPDAG since its full MEC is `A — B`.

Knowledge describes direct required/forbidden edges. Requiring a nonedge is
inconsistent; forbidding a nonedge is vacuous. Forbidding one orientation of an
adjacency forces its reverse; forbidding both removes all extensions. Unknown
labels raise `PDAGDomainError`. Knowledge must preserve the original collider
set: requiring `A → B ← C` in the CPDAG `A — B — C` is inconsistent even though
the proposed DAG is acyclic. All knowledge is imposed before closure; a new
collider or cycle cannot silently redefine the original class.

The extension finder returns a `DAG` or `NoConsistentExtension`; closure returns
an `MPDAG` or `NoConsistentExtension`. Failure distinguishes `input_unextendable`
and `background_inconsistent`. Malformed graph views and labels raise
`PDAGDomainError`. Failures retain the supplied `knowledge` object, and labels are
validated even when the input's extension set is empty. This is a reference
implementation with repeated scans, not a claim of optimal runtime. DAG equivalence
compares exact labels and ignores their declaration order. No DAG-class listing
or discovery-from-data algorithm is added.
Use `.topology` with existing text, JSON, and SVG adapters.

## GES operator validity

`ges_insert_is_valid` and `ges_delete_is_valid` test candidate Insert and Delete
moves on a `CPDAG`. For vertices `X` and `Y`, let `N(Y,X)` be the undirected
neighbors of `Y` adjacent to `X`. Insert requires nonadjacent endpoints and a
set `tails` of undirected neighbors of `Y` nonadjacent to `X`. The union of
`N(Y,X)` and `tails` must be a clique and meet every semidirected path from
`Y` to `X`. Delete requires an undirected `X — Y` or directed `X → Y` edge
and a subset `heads` of `N(Y,X)`. The nodes in `N(Y,X)` outside `heads`
must form a clique.
[Chickering (2002), Definitions 12–14 and Theorems 15 and 17](https://jmlr.org/papers/volume3/chickering02b/chickering02b.pdf#page=21)

Each set argument accepts one label or an iterable of labels. A candidate with
known vertices but incompatible edges or sets returns `False`. Unknown labels
and inputs outside the `CPDAG` view raise `PDAGDomainError`. These predicates
check whether editing one edge of a DAG representative realizes the operator's
result. [Lean proof on essential graphs](https://github.com/snu-causality-lab/open-causal-science/blob/51af06a9f7e0691cbfcc7e9fee8f81c0be8d6f29/docs/papers/chickering-2002-greedy-equivalence-search.md#the-validity-tests-together)

```python
from causal_graph_algorithms import CPDAG, DAG, ges_delete_is_valid, ges_insert_is_valid

chain = CPDAG.from_dag(DAG.from_text("X --> P --> Q --> Y"))
assert not ges_insert_is_valid(chain, "X", "Y")
assert ges_insert_is_valid(chain, "X", "Y", tails="Q")

diamond = CPDAG.from_dag(
    DAG.from_text("X --> Y; X --> P; X --> Q; Y --> P; Y --> Q")
)
assert not ges_delete_is_valid(diamond, "X", "Y")
assert ges_delete_is_valid(diamond, "X", "Y", heads="P")
```

## CI values, oracles, and graphoid deductions

`CIStatement(left, right, given=())` uses sets, canonicalizes symmetry, and requires
pairwise disjoint roles. Strings mean one vertex, not a character sequence. Empty
left/right denotes trivial CI. `CIScope(nodes)` declares the universe; unknown
labels are rejected when a statement is used in a scope.

`CIOracle` is a callable protocol taking one `CIStatement` and returning **bool**.
`SeparationCIOracle(graph)` accepts fully specified separation views and reports
exact graph separation. Under Markov, `True` entails distributional CI; interpreting
`False` as distributional dependence additionally requires faithfulness.
`CachedCIOracle(oracle, max_entries=10000)` caches symmetric queries and records
underlying `calls` and `cache_hits`. Eviction preserves answers. It does not certify
the oracle or convert a p-value/unknown into a Boolean. Statistical CI decisions
need a separate adapter and uncertainty contract.

```python
from causal_graph_algorithms import CIStatement, CIScope, derive_ci_closure

scope = CIScope(("X", "Y", "W", "Z"))
closure = derive_ci_closure(
    scope,
    (CIStatement("X", "Y", "Z"), CIStatement("X", "W", ("Y", "Z"))),
    max_statements=100,
)
assert closure.contains(CIStatement("X", ("Y", "W"), "Z"))
```

`derive_ci_closure(scope, premises, *, max_statements, axioms=GraphoidAxioms(),
max_rule_checks=1000000)` returns a completed selected-rule closure with ordered
`CIDerivation` records. Semi-graphoid symmetry, decomposition, weak union, and
contraction are always enabled. Intersection and composition require explicit
flags in `GraphoidAxioms`. The [primary axiom record, Sadeghi–Soo 2022, §2.2](https://jmlr.org/papers/v23/21-0425.html)
states their scopes: positivity supplies intersection, while composition is an
additional property and is not supplied by positivity alone. This function
validates syntax; it does not check that premises or optional assumptions are true.

`closure.contains(s) == False` means **underivable**, never dependent. The completed
closure covers these selected inference rules, not general probabilistic CI
implication. Trivial statements are implicit and consume no statement budget.
Either budget raises `CIResourceLimitError` atomically, without a partial closure
being presented as a completed answer. Bad labels, flags, or budgets raise
`CIDomainError`. Enumeration and pair scans can be exponential; budgets are part
of the public call, not a scalability claim.
