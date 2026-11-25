# C library

Build the C11 static library with `make library`. Link `build/libcga.a` and
add `c/include` to your header search path. The headers declare ABI version 1.
`cga_backend_query_operation` reports the supported operations and maximum
vertex count for each backend. The two backends accept indices from 0 to 15
within the graph's declared size. Selectors use `uint16_t` bit masks. See
[backend.h](../c/include/cga/backend.h) and [status.h](../c/include/cga/status.h).

| Operation | Graph and result |
| --- | --- |
| m-separation | An ancestral endpoint-marked graph or an acyclic directed-bidirected graph. The two query sets are nonempty and both are disjoint from each other and the conditioning set. |
| Ancestrality and maximality | An endpoint-marked graph. Each classification returns a Boolean. |
| MAG Markov equivalence | Two maximal ancestral graphs with the same vertex count and correspondence by index. |
| Induced subgraph | Either graph backend. Output indices compact in ascending retained order. |
| Latent projection | An acyclic directed-bidirected graph, including bows. Output indices compact in ascending retained order. |
| Covariate adjustment | A known DAG without bidirected edges, one treatment, and one outcome. A reusable query checks whether a supplied mask is valid. |
| Loyal equivalent graphs | A maximally oriented no-selection PAG supplied as an endpoint-marked graph. The representative and iterator describe the class members whose bidirected edges are invariant. |

The ancestral-graph and equivalence operations follow the graph semantics of
Richardson and Spirtes and of Ali, Richardson, and Spirtes. Latent projection
preserves the induced directed and bidirected relations among retained vertices.
[Richardson and Spirtes, 2002](https://doi.org/10.1214/aos/1031689015),
[Ali, Richardson, and Spirtes, 2009](https://doi.org/10.1214/08-AOS626),
[Richardson, Evans, Robins, and Shpitser, 2023](https://doi.org/10.1214/22-AOS2253)

The DAG adjustment query caches the forbidden set and proper back-door graph.
Its validity check makes no allocations. The caller chooses the available
adjustment vertices. [adjustment.h](../c/include/cga/adjustment.h),
[van der Zander, Liśkiewicz, and Textor, 2014](https://proceedings.mlr.press/r12/zander14a.html)

LEG construction completes the circle edges of the supplied PAG. The iterator
streams the representative and then each remaining class LEG once, using
legitimate reversals and reverse search. The caller establishes PAG maximal
orientation and the no-selection premise. The C checks cover endpoint
structure, circle-subgraph chordality and the completed representative's MAG
status. [leg.h](../c/include/cga/leg.h),
[Zhang, 2008](https://doi.org/10.1016/j.artint.2008.08.001),
[Zhang and Spirtes, 2005](https://arxiv.org/abs/1207.1419),
[Avis and Fukuda, 1996](https://doi.org/10.1016/0166-218X%2895%2900026-N)

Call the corresponding destroy function for each created handle and each graph
returned by a graph-producing call. An adjustment query or LEG iterator owns
its snapshot, so the original graph can be destroyed after creation. A non-OK
status preserves output arguments, and a failed iterator call preserves
traversal state. Normal iterator exhaustion reports `CGA_STATUS_OK` with a
null graph and a false `out_has_graph`.

```sh
make library
cc -std=c11 -Ic/include c/examples/leg_enumeration.c build/libcga.a -o build/leg_enumeration
./build/leg_enumeration
```

The example uses a three-vertex no-selection PAG.
[Source](../c/examples/leg_enumeration.c)
