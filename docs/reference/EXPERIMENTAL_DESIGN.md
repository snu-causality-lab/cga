# Experimental design

## A minimum-cost intervention for one district

`hedge_hull(graph, target)` computes the fixed point in Algorithm 1 of Akbari,
Etesami and Kiyavash (2025) by alternating directed ancestor restriction and
bidirected component restriction. It returns the target together with the
union of its proper hedges. When there is no proper hedge, the algorithm
returns the target, the base case used in Appendix E. The result lists
vertices in their declared graph order.

`find_minimum_cost_intervention(graph, target, costs=..., allowed=...)`
computes a single intervention set $A$ for identifying
$Q[S] = P(S \mid do(V \setminus S))$ from the interventional kernel
$Q[V \setminus A]$ in the paper’s positive semi-Markovian model class. The input is a
validated `AcyclicDirectedBidirectedGraph`. Both operations take a vertex name or an
iterable of names for the nonempty target `S`, which must be
bidirected-connected in the induced graph. Lemma 3 turns the search into a weighted
hitting set problem for the proper hedges of `S` with `S` removed from
each hedge. The implementation branches on the current hedge hull whenever
it strictly contains `S`, then recomputes the hull after deleting each
candidate. The search is exact and can take exponential time.

Supply a mapping from vertex names to nonnegative integer or `Fraction`
costs. An omitted cost defaults to one. By default, all vertices outside the
target are eligible. Use `allowed` to restrict the search to an eligible
vertex or an iterable of eligible vertices.
A successful empty intervention means that the observational distribution
suffices. A return value of `None` means the allowed vertices cannot hit
every hedge. Ties are resolved by the number of intervened vertices and
then by the declared vertex order. All costs are added exactly.

The paper's Figure 2(a) and §3.3 example give the two hedge differences
`{v1, v2}` and `{v2}` for `S = {s1, s2}`. Choosing `v2`
suffices. [JMLR 2025](https://jmlr.org/papers/v26/22-1516.html)

## Comparing interventional equivalence of DAGs

`dag_intervention_markov_equivalent(left, right, targets)` implements the
cut-graph criterion of Hauser and Bühlmann (2012), Theorem 10. The paper's
statistical interpretation uses positive densities and perfect interventions. For each
regime, remove incoming arrows at its perfect-intervention targets and compare
the resulting skeletons and unshielded colliders. Regimes can have multiple
targets. The collection must leave each vertex untouched in at least one
regime. For example, `targets=[(), ("4",)]` supplies an observational
regime and a regime that intervenes on vertex `4`. The comparison returns
`False` immediately if the DAGs have different vertex labels. For DAGs on the
same labels, empty or nonconservative collections raise
`MarkovEquivalenceDomainError`, and unknown target labels raise
`GraphValidationError`.

In Figure 2, all three DAGs agree observationally. With the observational
regime and intervention on vertex 4, only the first two are still equivalent.
[Hauser and Bühlmann, JMLR 2012](https://www.jmlr.org/papers/v13/hauser12a.html)

## Designing a proxy experiment for adjustment

`find_minimum_cost_proxy_adjustment(graph, interventions, outcomes, costs=..., allowed=...)`
returns one experiment and adjustment set meeting the graphical criterion of
Elahi et al. (2024), §4. It drops irrelevant query treatments using Rule 3,
restricts to the ancestors of the remaining query, and solves the resulting
split-network cut with exact rational capacities. The result also gives the
remaining query treatments and the total intervention cost. `None` denotes
infeasibility for this adjustment criterion and these allowed intervention
vertices. Other identification strategies may still be available.

An omitted vertex cost defaults to one. Supply nonnegative integer or
`Fraction` costs. `allowed` restricts intervention vertices. The cut
protects both endpoint sets from adjustment and protects every member of
the target ancestor set from intervention. The latter restriction preserves
the target mechanisms used in the adjustment formula. The implementation
also keeps descendants of query treatments out of the adjustment set,
using their descendants before the proxy intervention. This sufficient
restriction still permits intervening on such a covariate when it is
outside the target ancestor set. Optimality is with respect to this
protected cut problem.

For example, on
`X → Y` with a bidirected edge and a lower cost at `Y`, the printed
cut can select `do(Y)`. The protected cut selects `do(X)` when allowed.
Figure 2(a) gives a working proxy experiment on `v1` for the joint effect
of `v2` and `v3` on `y`.
[Elahi et al., Figure 2 and §4](https://papers.nips.cc/paper/2024/file/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Paper-Conference.pdf)
