# Causal Graph Algorithms

Causal Graph Algorithms provides graph representations and reference implementations for separation, covariate adjustment, and Markov equivalence. It includes exact tests of examples from the original papers.

## Install and run

Use Python 3.11 or later and run these commands from a checkout.

```sh
python -m pip install -e '.[test]'
python -m pytest
```

```python
from causal_graph_algorithms import DAG, d_separated, find_adjustment_set

graph = DAG.from_text("Z --> X --> Y; Z --> Y")
assert not d_separated(graph, "X", "Y", given="Z")
assert find_adjustment_set(graph, interventions="X", outcomes="Y").vertices == ("Z",)
```

## Supported results

| Operation | Scope and references |
| --- | --- |
| d- and m-separation | Validated DAGs, acyclic directed-bidirected graphs, and ancestral graphs. The tests include [Richardson and Spirtes (2002), Figure 9](https://doi.org/10.1214/aos/1031689015). |
| Constrained separators | A separator or an inclusion-minimal separator containing the required vertices and using only the allowed vertices. [Van der Zander, Liśkiewicz and Textor (2019)](https://doi.org/10.1016/j.artint.2018.12.006) · [Authors’ 2023 corrigendum](https://doi.org/10.1016/j.artint.2023.103938) · [FindMinSep, Proposition 5.2](https://proceedings.mlr.press/v115/van-der-zander20a.html) |
| General adjustment | Check, construct, or minimize an adjustment set for a total effect in a DAG, a known acyclic directed-bidirected graph, a CPDAG, or a MAG/PAG in the no-selection class. Each operation uses the graphical and causal premises of its input domain. [Perković et al. (2018)](https://jmlr.org/papers/v18/16-319.html) |
| Markov equivalence and partial orientations | DAG and MAG equivalence, CPDAG construction and consistent extension, and PAG construction from a known MAG. The PAG constructors use either the full MAG universe or the no-selection universe. [Verma and Pearl](https://ftp.cs.ucla.edu/tech-report/1991-reports/910020.pdf), [Ali, Richardson and Spirtes (2009)](https://doi.org/10.1214/08-AOS626), [Zhang (2008)](https://doi.org/10.1016/j.artint.2008.08.001) |
| MAG equivalence class listing | `list_mag_equivalence_class(pag, max_members=...)` returns all MAGs represented by a `ClosedPAG`, including members with undirected edges. The required `max_members` is a nonnegative integer. If the class exceeds that limit, the call raises `MAGListingResourceLimitError` without returning a partial list. [Wang, Du and Zhou (2025)](https://proceedings.mlr.press/v267/wang25y.html) |

An adjustment search returns a set or an `AdjustmentNotFound` result for the specified graph and query. An empty `AdjustmentSet` is a valid result when no covariates need conditioning. Graph and query validation errors are raised separately. The library returns graph-theoretic results, with the causal interpretation and support conditions supplied by the caller.

The [paper fixtures](tests/fixtures/papers/) record source examples and regressions derived from published definitions. The suite also checks selected small graphs against independent path or equivalence enumeration.

The ordinary-adjustment tests replay [Witte et al. (2020), Figures 2B and 2F](https://jmlr.org/papers/v21/20-175.html).
The [Lean checks for Figure 2B](https://github.com/sanghack81/open-causal-science/blob/5a08ef6d08b757ba0f0689421721889ea5f20303/docs/paper-checks.md#witte-et-al-2020-figure-2b) and [Figure 2F](https://github.com/sanghack81/open-causal-science/blob/5a08ef6d08b757ba0f0689421721889ea5f20303/docs/paper-checks.md#witte-et-al-2020-figure-2f) formalize the path and projection calculations used in those examples.

The separator reference follows the authors’ 2023 runtime correction. The Lean proof counts parent-pair checks in the specified augmentation loop. [Proof scope](https://github.com/sanghack81/open-causal-science/blob/5a08ef6d08b757ba0f0689421721889ea5f20303/OpenCausalScience/Papers/VanDerZanderEtAl2023/AugmentationParentPairs.lean)

## Lean and paper replays

[Open Causal Science](https://github.com/sanghack81/open-causal-science/blob/5a08ef6d08b757ba0f0689421721889ea5f20303/docs/cga.md) records the exact Lean statements corresponding to CGA graph operations. The tests include the original Verma–Pearl Figure 2 and two DAG cases for the forbidden-set construction. Each project builds independently.

To compare the nine published mixed graphs, generate `m-separation.json` with the Lean command in that proof map. From this checkout with CGA installed, compare it against CGA's own paper fixtures.

```sh
python tests/compare_lean_mseparation.py /path/to/m-separation.json
```
