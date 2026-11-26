# IDA parent sets

The IDA graph routines take a validated `CPDAG` for a causally sufficient DAG class. They enumerate parent choices for population IDA and joint-IDA. Computing causal effects from those choices in the published linear SEMs also needs the population covariance matrix. [Local IDA, Algorithm 3](https://doi.org/10.1214/09-AOS685) · [Joint-IDA, Algorithm 5.1](https://doi.org/10.1214/16-AOS1462)

| Operation | Result |
| -- | -- |
| `ida_parent_sets(graph, treatment)` | Distinct parent sets at one treatment. Each combines the compelled parents with a clique of undirected neighbors. |
| `joint_ida_parent_sets(graph, treatments)` | Distinct tuples realizable by a shared representative DAG. Each tuple uses the supplied treatment order. |
| `joint_ida_parent_counts(graph, treatments)` | A dictionary of the same tuples with their semi-local multiplicities. The full class multiplies every count by the same number of orientations of the untouched components. |

The joint routines enumerate collider-free DAG orientations only in chain components containing a treatment. Enumeration and output can be exponential. The treatments must be distinct known vertices. A single string represents one treatment.

```python
from causal_graph_algorithms import (
    CPDAG, DAG, ida_parent_sets, joint_ida_parent_sets,
)

graph = CPDAG.from_dag(DAG.from_text("B -> A; B -> C"))
empty, b = frozenset(), frozenset({"B"})

assert ida_parent_sets(graph, "A") == {empty, b}
assert joint_ida_parent_sets(graph, ("A", "C")) == {(empty, b), (b, empty), (b, b)}
```

Each endpoint of `A — B — C` can have no parents, but selecting both empty sets would require the excluded collider `A → B ← C`. This is Figure 2 and Example 3 of the joint-IDA paper with renamed vertices. [Joint-IDA paper, §5](https://arxiv.org/pdf/1407.2451#page=16)
