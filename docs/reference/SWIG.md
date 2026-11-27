# Single-world intervention templates

`swig_template` takes a `DAG` and intervention targets given by one
vertex label or an iterable of labels. It retains one random vertex for
each original variable and adds one fixed vertex for each target. Each
arrow still ends at the random half of its destination. It starts at
the fixed half of its source when the source is a target and at the
random vertex otherwise. [Richardson and Robins (2013), §3.3.1](https://csss.uw.edu/Papers/2013/wp128.pdf#page=27)

`SWIGTemplate` exposes `random_vertices`, `fixed_vertices`, and `directed`.
Each `SWIGVertex` records its original `variable` and an `is_fixed` flag.
The vertex lists follow input vertex order, and edges are sorted by their
source and then their destination in that order. For a random vertex,
`fixed_ancestors(vertex)` returns the original names of its fixed ancestors
in input vertex order. These names label the vertex in the template before
assigning concrete intervention values.
[Open Causal Science proof record](https://github.com/snu-causality-lab/open-causal-science/blob/51af06a9f7e0691cbfcc7e9fee8f81c0be8d6f29/docs/papers/richardson-robins-2013-swig.md)

The two panels of Figure 2 share the same template. Its one edge starts
at the fixed half of `X`. Figure 10(iv) splits two consecutive targets.
Fixed `Z` labels random `M`, and fixed `M` labels random `Y`.
[Figure 2](https://csss.uw.edu/Papers/2013/wp128.pdf#page=10) ·
[Figure 10(iv)](https://csss.uw.edu/Papers/2013/wp128.pdf#page=30)

```python
from causal_graph_algorithms import DAG, SWIGVertex, swig_template

pair = swig_template(DAG.from_text("X --> Y"), interventions="X")
assert pair.directed == ((SWIGVertex("X", is_fixed=True), SWIGVertex("Y")),)
assert pair.fixed_ancestors("X") == ()
assert pair.fixed_ancestors("Y") == ("X",)

chain = swig_template(DAG.from_text("Z --> M --> Y"), interventions=("Z", "M"))
assert chain.fixed_ancestors("M") == ("Z",)
assert chain.fixed_ancestors("Y") == ("M",)
```
