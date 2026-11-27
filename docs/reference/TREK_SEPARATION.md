# Trek separation in DAGs

`trek_separated` checks whether given left and right vertex sets block
every trek between two endpoint sets in a `DAG`. A trek has two directed
paths from the same top to the two endpoints. The blockers act on the
paths on their respective sides, including each path's top and endpoint.
Paths may have zero edges, and all four sets may overlap.
[Sullivant, Talaska, and Draisma (2010), Definition 2.7](https://pure.tue.nl/ws/portalfiles/portal/3002904/Metis230934.pdf#page=6)

The function removes each side's blockers and finds the reflexive ancestors
of the surviving endpoints on that side. An unblocked trek exists exactly
when these two ancestor sets intersect. This is the same reachability test
as the paper's two-copy construction. The Lean reference proof relates
two-copy routes to unblocked treks and checks the shared-endpoint boundary.
[Proposition 3.5](https://pure.tue.nl/ws/portalfiles/portal/3002904/Metis230934.pdf#page=14) ·
[OCS proof record](https://github.com/snu-causality-lab/open-causal-science/blob/51af06a9f7e0691cbfcc7e9fee8f81c0be8d6f29/docs/papers/sullivant-et-al-2010-trek-separation.md)

In Figure 1, vertex 4 is a choke point on the right side of treks
from {1, 3} to {4, 5}. Blocking it on the left leaves the trek
from vertex 1 to vertex 4 intact.
[Figure 1](https://pure.tue.nl/ws/portalfiles/portal/3002904/Metis230934.pdf#page=5) ·
[Example 2.9](https://pure.tue.nl/ws/portalfiles/portal/3002904/Metis230934.pdf#page=7)

```python
from causal_graph_algorithms import DAG, trek_separated

graph = DAG.from_text("V1 --> V4; V2 --> V4; V2 --> V5; V3 --> V4 --> V5")
assert trek_separated(graph, ("V1", "V3"), ("V4", "V5"), right_blocked="V4")
assert not trek_separated(graph, ("V1", "V3"), ("V4", "V5"), left_blocked="V4")
```
