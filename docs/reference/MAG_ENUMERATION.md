# Exhaustive MAG enumeration

`enumerate_mags` generates maximal ancestral graphs (MAGs), not a Markov equivalence class. Use it to build small exhaustive graph collections; use `list_mag_equivalence_class` when a `ClosedPAG` already specifies the class of interest.

The class-listing regression reproduces [Wang, Du and Zhou (2025), Figure 1](https://proceedings.mlr.press/v267/wang25y.html) and records the independently checked class and background-knowledge counts in its [fixture](../../tests/fixtures/papers/wang_du_zhou_2025_figure1.json).

## Use

```python
from causal_graph_algorithms import enumerate_mags, graph_to_text

graphs = list(enumerate_mags(
    2, pinned=("X", "Y"), include_undirected=False, max_output=4,
))
print(len(graphs))                         # 4
print(graph_to_text(graphs[1].topology))
# X
# Y
# X --> Y
```

The four outputs are the empty graph, `X --> Y`, `X <-- Y`, and `X <-> Y`. The pinned labels prevent swapping X with Y. Without pins there are three isomorphism representatives; the two directed graphs then belong to the same orbit. This is not Markov equivalence: the different nonempty two-node graphs have the same independence model but are not all isomorphic.

```python
enumerate_mags(
    n: int, *, pinned: Sequence[str] = (),
    up_to_isomorphism: bool = True, max_output: int | None = None,
    include_undirected: bool = True,
) -> Iterator[MaximalAncestralGraph]
```

The default is the full MAG domain, including directed, bidirected and undirected edges. `include_undirected=False` selects the no-selection, directed–bidirected domain. All graphs have at most one edge per unordered pair, no self-loops and no circles. The existing `MaximalAncestralGraph` constructor validates every emitted graph.

Nodes are the pinned labels in the supplied order, followed by `V0`, `V1`, …, skipping collisions. For example, `n=4, pinned=("V0", "X")` gives `("V0", "X", "V1", "V2")`. Each pin is fixed pointwise; only unpinned nodes may be permuted. `up_to_isomorphism=False` emits every labelled graph. The empty and singleton universes each contain one graph.

Validation occurs when calling the function, before iteration, in this order: `n`, `pinned`, `up_to_isomorphism`, `include_undirected`, `max_output`. Invalid arguments raise `MAGListingDomainError`. `n` must be a nonnegative integer, not a Boolean. `pinned` must be a sequence of at most `n` distinct nonempty strings, not a single string, set or iterator; its contents are snapshotted. Both flags must be actual Booleans. `max_output` is `None` or a nonnegative integer, not a Boolean.

## Output order and limits

For node positions `0,1,…`, pairs are ordered `(0,1), (0,2), (1,2), (0,3), …`. States are ordered absent, forward directed, reverse directed, bidirected, undirected. Outputs follow lexicographic order in that encoding. With isomorphism reduction, the representative is the smallest code over all permutations that fix each pinned vertex. This order is not the same as the raw topology's edge-storage order.

`max_output` is an output budget, not silent truncation. After yielding that many graphs, discovering an additional graph that would otherwise be emitted raises `MAGListingResourceLimitError`; isomorphism-rejected candidates do not consume the budget. Previously yielded graphs remain valid, but their collection is incomplete. If the universe has exactly the budgeted size, iteration ends normally. Even `n=0` has one output, so a zero budget raises on the first `next()`. Both `MAGListingDomainError` and `MAGListingResourceLimitError` are imported from `causal_graph_algorithms`.

To intentionally inspect only a prefix, use `itertools.islice(enumerate_mags(...), count)`. Stopping early cannot establish completeness; neither can merely receiving exactly `max_output` graphs. Only normal exhaustion establishes that the selected universe was fully enumerated. The existing `list_mag_equivalence_class(..., max_members=...)` instead returns a complete tuple atomically, or raises without returning a partial tuple.

Neither output budget bounds search work, working memory or elapsed time. Search uses explicit stacks, incremental directed-reachability masks, monotone ancestral-violation pruning, and exact canonical checks at completed vertex groups. Permutations are explored lazily with prefix comparisons and interchangeable-vertex pruning; there is no factorial permutation table or output-sized deduplication set. Memory is independent of how many outputs have already been consumed, unless the caller retains them. Enumeration remains exponential in the number of vertex pairs, and difficult isomorphism checks can require factorial work. This is a small-graph API, without a polynomial-delay or large-graph performance guarantee.

## Mathematical basis and independent checks

The graph domain and maximality meaning come from Richardson–Spirtes (2002), [Sections 3.1, 3.4 and 3.7; Corollary 4.3 and Lemma 4.5](https://www.cmu.edu/dietrich/philosophy/docs/spirtes/annals2002.pdf). Maximality requires an m-separating set for each nonadjacent pair; it does not mean a maximum number of edges. Production uses an incremental ancestral prefix check and a primitive-inducing-path prefilter, followed by the existing public MAG validator. The enumeration and canonical-search code are original repository implementations, not a transcription of an external prototype.

The pruning rules have separate justifications:

- A directed cycle, an almost-directed cycle or an arrowhead at an undirected-edge endpoint cannot disappear when more edges are assigned. The search rejects these ancestral violations immediately. Maximality is checked only at completed leaves, because adding an edge can repair nonmaximality.
- At a completed initial vertex group, every allowed permutation within that group extends by the identity on later vertices. If it makes the prefix smaller, it makes every completion smaller too. Thus no globally least representative is removed. The final group checks the entire graph.
- A partial relabeling with a larger completed code cannot later become smaller; one with an equal code is extended. Interchangeable vertices are skipped only when their transposition is an automorphism of the completed prefix. No score, degree-only grouping or Markov-equivalence test replaces that exact condition.

Together with exhaustive pair-state branching and leaf maximality, these rules preserve every labelled MAG when reduction is off and precisely one least representative per allowed orbit when it is on. The finite tests below check the implementation of these arguments; they do not replace the arguments.

The [independent oracle](../../tests/python/mag_enumeration_oracles.py) enumerates the complete Cartesian edge-assignment universe, without ancestral prefix pruning or canonical skipping. It tests ancestrality at the leaf, then checks every conditioning subset against vertex-simple m-connecting paths for each nonedge. A separate full-permutation calculation groups graphs while fixing pins. It imports no production graph traversal, maximality or canonicalization helper. The test-only leaf evaluator is itself checked against the repository's earlier literal topological-order and m-separation oracles through four vertices.

The [oracle tests](../../tests/python/test_mag_enumeration_oracle.py) compare exact labelled sets and canonical orbit sets, not only counts. Routine tests exhaust both domains through four vertices and every possible number of pins. The optional five-node replay compares the entire labelled set and the zero-, one- and two-pin orbit sets. These repository-derived counts are not numbers asserted by the paper:

| Domain | n | Labelled MAGs | No pins | One pin | Two pins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Directed–bidirected | 3 | 56 | 14 | 32 | 56 |
| Directed–bidirected | 4 | 2,492 | 142 | 474 | 1,302 |
| Directed–bidirected | 5 | 328,924 | 3,268 | 15,078 | 57,350 |
| Full, including undirected | 3 | 72 | 19 | 42 | 72 |
| Full, including undirected | 4 | 3,045 | 183 | 592 | 1,603 |
| Full, including undirected | 5 | 379,052 | 3,867 | 17,617 | 66,512 |

Run from the repository root after [installing the package and development requirements](../../README.md#development):

```console
python -B -m pytest -p no:cacheprovider tests/python/test_mag_enumeration_contract.py tests/python/test_mag_enumeration_oracle.py -q
CGA_EXHAUSTIVE_MAG_N5=1 python -B -m pytest -p no:cacheprovider tests/python/test_mag_enumeration_oracle.py -k n5 -q
```

Six-node prefix tests check independently computed validity and distinct pinned orbits, not whole-universe completeness or a six-node census. The tests cannot establish correctness at arbitrary sizes by finite agreement alone. This enumerator can supply test inputs, but a consumer must not use production MAG validation as its sole expected answer when testing that same validator.
