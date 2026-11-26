"""Compare Lean's mixed-graph replay with CGA and a simple edge-path oracle."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    AncestralGraph,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    Mark,
    m_separated_sets,
)

from python.oracles import (
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    m_separated_by_simple_edge_paths,
)


NODES3 = ("0", "1", "2")
NODES4 = ("0", "1", "2", "3")
PAIRS3 = tuple(combinations(NODES3, 2))
ENDPOINTS = (None, (False, False), (False, True), (True, False), (True, True))
BOW4: tuple[MixedEdgeAtom, ...] = (
    ("0", False, "1", True),
    ("0", True, "1", True),
    ("1", True, "2", True),
    ("1", False, "3", True),
)
ANCESTRAL4: tuple[MixedEdgeAtom, ...] = (
    ("0", False, "1", False),
    ("1", False, "2", True),
    ("2", True, "3", True),
)


def unique_object(entries: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(entries)
    if len(result) != len(entries):
        raise ValueError("export contains duplicate object keys")
    return result


def indexed_entries(entries: object, domain: str) -> dict[int, object]:
    if not isinstance(entries, list):
        raise ValueError(f"{domain}: expected a list of indexed entries")
    result: dict[int, object] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2 or type(entry[0]) is not int:
            raise ValueError(f"{domain}: expected an integer identifier and a value")
        key, value = entry
        if key in result:
            raise ValueError(f"{domain}: duplicate identifier {key}")
        result[key] = value
    return result


def admg3_edges(mask: int) -> tuple[MixedEdgeAtom, ...]:
    directed = tuple(
        (u, False, v, True) for i, (u, v) in enumerate(PAIRS3) if mask & (1 << i)
    )
    bidirected = tuple(
        (u, True, v, True) for i, (u, v) in enumerate(PAIRS3) if mask & (1 << (i + 3))
    )
    return directed + bidirected


def ancestral3_edges(code: int) -> tuple[MixedEdgeAtom, ...]:
    result = []
    for i, (u, v) in enumerate(PAIRS3):
        marks = ENDPOINTS[(code // 5**i) % 5]
        if marks is not None:
            result.append((u, marks[0], v, marks[1]))
    return tuple(result)


def admg(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> AcyclicDirectedBidirectedGraph:
    directed = [(u, v) for u, a, v, b in edges if not a and b]
    directed.extend((v, u) for u, a, v, b in edges if a and not b)
    bidirected = [(u, v) for u, a, v, b in edges if a and b]
    return AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected)
    )


def ancestral_graph(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> AncestralGraph:
    marks = {False: Mark.TAIL, True: Mark.ARROW}
    return AncestralGraph.from_graph(
        EndpointMarkedGraph(
            nodes=nodes, edges=[(u, marks[a], v, marks[b]) for u, a, v, b in edges]
        )
    )


def queries(nodes: tuple[str, ...]):
    for labels in product(range(4), repeat=len(nodes)):
        groups = tuple(
            tuple(node for node, label in zip(nodes, labels, strict=True) if label == group)
            for group in range(1, 4)
        )
        if groups[0] and groups[1]:
            mask = sum(label * 4**i for i, label in enumerate(labels))
            yield mask, groups


def compare_graph(
    domain: str,
    family: str,
    nodes: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    exported: object,
) -> int:
    expected_queries = dict(queries(nodes))
    lean = indexed_entries(exported, domain)
    if lean.keys() != expected_queries.keys():
        raise ValueError(f"{domain}: missing or unexpected query masks")
    graph = admg(nodes, edges) if family == "ADMG" else ancestral_graph(nodes, edges)
    for mask, (left, right, given) in expected_queries.items():
        lean_value = lean[mask]
        cga_value = m_separated_sets(graph, left, right, given)
        oracle_value = m_separated_by_simple_edge_paths(nodes, edges, left, right, given)
        if type(lean_value) is not bool or not (lean_value == cga_value == oracle_value):
            raise ValueError(
                f"{domain}, query {mask}: Lean={lean_value!r}, "
                f"CGA={cga_value}, simple path={oracle_value}"
            )
    return len(expected_queries)


def compare_export(exported: object) -> tuple[int, int]:
    if not isinstance(exported, dict) or exported.keys() != {
        "admg3", "ancestral3", "bow4", "ancestral4"
    }:
        raise ValueError("export has missing or unexpected graph families")

    admg_entries = indexed_entries(exported["admg3"], "ADMG")
    if admg_entries.keys() != set(range(64)):
        raise ValueError("ADMG exports are missing or include unexpected masks")
    ancestral_entries = indexed_entries(exported["ancestral3"], "ancestral")
    ancestral_cases = {}
    for code in range(125):
        edges = ancestral3_edges(code)
        if endpoint_graph_is_ancestral_by_definition(NODES3, edges):
            ancestral_cases[code] = edges
    if ancestral_entries.keys() != ancestral_cases.keys():
        raise ValueError("ancestral exports differ from the definition-level graph census")

    count = 0
    for mask, verdicts in admg_entries.items():
        count += compare_graph(f"ADMG {mask}", "ADMG", NODES3, admg3_edges(mask), verdicts)
    for code, edges in ancestral_cases.items():
        count += compare_graph(
            f"ancestral {code}", "ancestral", NODES3, edges, ancestral_entries[code]
        )
    count += compare_graph("bow4", "ADMG", NODES4, BOW4, exported["bow4"])
    count += compare_graph(
        "ancestral4", "ancestral", NODES4, ANCESTRAL4, exported["ancestral4"]
    )
    return len(admg_entries) + len(ancestral_entries) + 2, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="JSON from the OCS mixed-separation replay")
    args = parser.parse_args()
    try:
        exported = json.loads(args.export.read_text(encoding="utf-8"),
                              object_pairs_hook=unique_object)
        graphs, comparisons = compare_export(exported)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"{graphs} mixed graphs, {comparisons} Lean/CGA/simple-path comparisons agree")


if __name__ == "__main__":
    main()
