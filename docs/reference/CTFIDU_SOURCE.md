# Source selection for a counterfactual factor

`find_ctfidu_source` implements the source-selection step of CTFIDU for an
already formed, consistent counterfactual factor. Supply its primitive
vertices through `district`. They must induce a nonempty bidirected-connected
subgraph in the supplied `DAG` or `AcyclicDirectedBidirectedGraph`.
[Correa, Lee and Bareinboim (2021), Algorithm 1, lines 5–11](https://proceedings.neurips.cc/paper/2021/file/36bedb6eb7152f39b16328448942822b-Paper.pdf#page=8)

Supply the available regimes in order through `experiment_interventions`.
Each regime names the vertices set by a hard intervention and denotes a
joint kernel over the other vertices at the intervention assignments.
Each accepts a single label or an iterable of labels. Use `()` to
include observation. A regime is admissible when its free
vertices contain the district and a valid fixing sequence reaches exactly
that district. The Lean companion proves this equivalence for the
paper's source search on an already supplied district target.
[OCS proof record](https://github.com/snu-causality-lab/open-causal-science/blob/51af06a9f7e0691cbfcc7e9fee8f81c0be8d6f29/docs/papers/correa-lee-bareinboim-2021-counterfactual-identification.md#ctfidu-source-selection)

A `CTFIDUSource` gives the first admissible regime's zero-based
`source_index`, its hard `interventions` in graph vertex order, and a
`fixing_sequence`. The sequence starts from the CADMG with the
intervention vertices fixed. It removes arrows into those vertices and
bidirected edges incident to them, and retains their arrows to free
vertices. `None` means none of the supplied
regimes reaches the district. The caller supplies the factor, its
consistent value assignments, and any kernel assumptions needed to
evaluate it. The result is a graphical plan.

Figure 5(a) has the bow `X → Z`, `X ↔ Z`. The factor on `Z` in
equation (17) can use the source that intervenes on `X`. The factor on
`Y` can use observation.
[Figure 5(a) and equation (17)](https://proceedings.neurips.cc/paper/2021/file/36bedb6eb7152f39b16328448942822b-Paper.pdf#page=7)

```python
from causal_graph_algorithms import AcyclicDirectedBidirectedGraph, find_ctfidu_source

graph = AcyclicDirectedBidirectedGraph.from_text(
    "X --> Z; X <-> Z; X --> Y; Z --> Y"
)
assert find_ctfidu_source(graph, district="Z", experiment_interventions=[()]) is None

regimes = ((), ("X",))
z_source = find_ctfidu_source(graph, district="Z", experiment_interventions=regimes)
assert z_source is not None
assert z_source.source_index == 1
assert z_source.interventions == ("X",)
assert z_source.fixing_sequence == ("Y",)

y_source = find_ctfidu_source(graph, district="Y", experiment_interventions=regimes)
assert y_source is not None
assert y_source.source_index == 0
assert y_source.interventions == ()
```
