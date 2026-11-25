"""Compare CGA's set d-separation with the OCS Lean export on four vertices."""

import argparse
import json
from itertools import combinations, product
from pathlib import Path

from causal_graph_algorithms import DAG, DirectedBidirectedGraph, d_separated_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="JSON from the OCS Lean replay")
    args = parser.parse_args()
    entries = json.loads(args.export.read_text())
    lean = dict(entries)
    nodes = tuple(str(index) for index in range(4))
    edges = tuple(combinations(nodes, 2))
    cases = {}
    for labels in product(range(4), repeat=len(nodes)):
        query = tuple(tuple(node for node, label in zip(nodes, labels) if label == group)
                      for group in range(1, 4))
        if query[0] and query[1]:
            cases[sum(label * 4**index for index, label in enumerate(labels))] = query
    if len(entries) != len(lean) or set(lean) != set(range(1 << len(edges))):
        raise SystemExit("Lean export has unexpected DAG masks")
    count = 0
    for mask, verdicts in lean.items():
        directed = [edge for index, edge in enumerate(edges) if mask & (1 << index)]
        graph = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=directed))
        expected = dict(verdicts)
        if len(verdicts) != len(expected) or set(expected) != set(cases):
            raise SystemExit(f"DAG mask {mask}: Lean export has unexpected query masks")
        for query_mask, (left, right, given) in cases.items():
            lean_value = expected[query_mask]
            cga_value = d_separated_sets(graph, left, right, given)
            if type(lean_value) is not bool or cga_value != lean_value:
                raise SystemExit(f"DAG {mask}, query {query_mask}: "
                                 f"CGA {cga_value}, Lean {lean_value}")
            count += 1
    print(f"{len(lean)} DAGs, {count} CGA/Lean set d-separation comparisons agree")


if __name__ == "__main__":
    main()
