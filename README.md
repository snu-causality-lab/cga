# Causal Graph Algorithms

Causal Graph Algorithms implements published methods for causal graphs in Python and provides a C11 library for selected bounded graph operations. It includes paper examples and regression tests for specific changes to the published procedures.

```python
from causal_graph_algorithms import DAG, d_separated, find_adjustment_set

graph = DAG.from_text("Z --> X --> Y; Z --> Y")
assert not d_separated(graph, "X", "Y", given="Z")
assert find_adjustment_set(graph, interventions="X", outcomes="Y").vertices == ("Z",)
```

## Install

Use Python 3.11 or later. Download the wheel from the [GitHub release](https://github.com/snu-causality-lab/cga/releases) or install from a checkout.

```sh
python -m pip install .
```

## Algorithms and references

| Topic | Operations and contracts |
| --- | --- |
| Graphs, equivalence and CI | Graph views, d- and m-separation, latent projection, ancestral marginalization and selection, maximal extension, CPDAG/MPDAG construction, DAG and MAG equivalence, and conditional independence. [Graph operations](docs/reference/GRAPH_OPERATIONS.md) · [MAG and PAG construction and listing](docs/reference/MAG_ENUMERATION.md) |
| Constrained separators | Construct, minimize or enumerate separators with required, allowed or weighted vertices. [Separator search and the authors’ 2023 corrigendum](docs/reference/SEPARATOR_SEARCH.md) |
| Adjustment | General adjustment, efficient DAG and optimal mixed-graph sets, conditional and sequential adjustment, and local sufficient selection rules. [DAG and O-set](docs/reference/DAG_ADJUSTMENT.md) · [Runge](docs/reference/RUNGE_ADJUSTMENT.md) · [Smucler et al.](docs/reference/SMUCLER_ADJUSTMENT.md) · [Conditional](docs/reference/CONDITIONAL_ADJUSTMENT.md) · [Sequential](docs/reference/SEQUENTIAL_ADJUSTMENT.md) · [Local selection](docs/reference/LSAS_SELECTION.md) |
| Identification and recovery | ID, IDC, gID, c-gID, marginal PAG and MPDAG identification, Tian adjustment, CADMG and kernel fixing, and selection recovery with optional external data. [Identification and recovery](docs/reference/PUBLISHED_IDENTIFICATION.md) |
| Path-specific effects | Find a recanting district and its path witnesses in a DAG or ADMG. [Recanting districts](docs/reference/PUBLISHED_IDENTIFICATION.md#recanting-districts) |
| IDA | Local and joint parent-set enumeration for CPDAGs. [IDA parent sets](docs/reference/IDA.md) |
| Experimental design | Perfect-intervention DAG equivalence minimum-cost intervention for a district kernel, and proxy experiment design for adjustment. [Experimental design](docs/reference/EXPERIMENTAL_DESIGN.md) |
| Soft interventions | Augmented MAGs and equivalence for controlled and unknown-target soft regimes. [Soft intervention graphs](docs/reference/SOFT_INTERVENTION.md) |
| Missing data | Graphical target and full-law recovery criteria and direct testability. [Missing-data graphs](docs/reference/MISSINGNESS.md) |
| Counterfactuals | Single-world reductions and applicability checks for three rules of counterfactual calculus. [Identification](docs/reference/PUBLISHED_IDENTIFICATION.md#counterfactuals) · [Calculus](docs/reference/COUNTERFACTUAL_CALCULUS.md) |
| C11 library | Separation, projection, ancestral graph checks, MAG equivalence, loyal equivalent graph enumeration, and DAG adjustment validity. [Native API and bounds](docs/native.md) |

Results retain the assumptions and failure meanings of their operation. An empty adjustment set can be a successful result. Identification returns symbolic expressions with their declared probability sources. The caller supplies the distributions and causal premises needed to interpret a result.

## Verification

The [paper fixtures](tests/fixtures/papers/) reproduce published examples and record regression cases. The [Open Causal Science proof map](https://github.com/snu-causality-lab/open-causal-science/blob/main/docs/cga.md) identifies the exact Lean statements related to CGA. Both projects build independently.

```sh
python -m pip install -e '.[test]'
python -m pytest
```

Pytest includes the paper fixtures. With Make and a C11 compiler that supports ASan and UBSan, `make test` also runs the C contracts and differential checks.

### Lean replays

In a checkout of the [matching Lean revision](https://github.com/snu-causality-lab/open-causal-science/tree/fb92293741a2aac754c07193eb369d995d28745b), use the pinned toolchain to export the verdicts.

```sh
lake build
lake env lean replay/MSeparation.lean > m-separation.json
lake env lean replay/DSeparation.lean > d-separation.json
```

In this CGA checkout with the Python package installed, compare the 432 queries from the nine published mixed graphs and the 7,040 queries on four-vertex DAGs.

```sh
python tests/compare_lean_mseparation.py /path/to/m-separation.json
python tests/compare_lean_dseparation.py /path/to/d-separation.json
```

The [mixed-graph worklist proof and replay](https://github.com/snu-causality-lab/open-causal-science/blob/main/docs/cga.md#compare-the-mixed-graph-arrival-search)
also cover bow-capable ADMGs and ancestral graphs with undirected edges. Generate
`mixed-separation.json` with the linked Lean exporter and replay it against CGA
and the independent simple-edge-path oracle.

```sh
python tests/compare_lean_mixed_separation.py /path/to/mixed-separation.json
```

## License

Causal Graph Algorithms is available under the [MIT License](LICENSE).
