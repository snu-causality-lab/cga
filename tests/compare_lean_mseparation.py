"""Compare the OCS m-separation export with CGA's paper fixtures."""

import argparse
import json
from pathlib import Path

from causal_graph_algorithms import AncestralGraph, EndpointMarkedGraph, Mark, m_separated


def figure_graphs():
    fixtures = Path(__file__).parent / "fixtures" / "papers"
    figure_9 = json.loads((fixtures / "richardson_spirtes_2002_figure9.json").read_text())
    figure_4 = json.loads((fixtures / "ali_richardson_spirtes_2009_figure4.json").read_text())
    figure_6 = json.loads((fixtures / "ali_richardson_spirtes_2009_figure6.json").read_text())
    yield "9a", figure_9["nodes"], figure_9["edges"]
    yield "9b", figure_9["nodes"], figure_9["edges"] + figure_9["maximal_extension"]["added_edges"]
    for name, edges in figure_4["graphs"].items():
        yield f"4.{name}", figure_4["nodes"], edges
    for panel, graphs in figure_6["panels"].items():
        for name, edges in graphs.items():
            yield f"6.{panel}.{name}", figure_6["nodes"], edges


def cga_signature(nodes, edges):
    graph = AncestralGraph.from_graph(EndpointMarkedGraph(
        nodes=nodes,
        edges=[(left, Mark(at_left), right, Mark(at_right))
               for left, at_left, right, at_right in edges],
    ))
    verdicts = []
    for i, left in enumerate(nodes):
        for j, right in enumerate(nodes):
            if i == j:
                continue
            for mask in range(1 << len(nodes)):
                if mask & ((1 << i) | (1 << j)):
                    continue
                given = [node for k, node in enumerate(nodes) if mask & (1 << k)]
                verdicts.append(not m_separated(graph, left, right, given))
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="JSON from the OCS Lean replay")
    args = parser.parse_args()
    entries = json.loads(args.export.read_text())
    lean = dict(entries)
    cases = list(figure_graphs())
    if len(entries) != len(lean) or set(lean) != {name for name, _, _ in cases}:
        raise SystemExit("Lean export has unexpected figure identifiers")
    count = 0
    for name, nodes, edges in cases:
        cga = cga_signature(nodes, edges)
        expected = lean[name]
        if len(cga) != len(expected):
            raise SystemExit(f"{name}: expected {len(cga)} queries, got {len(expected)}")
        for i, (ours, theirs) in enumerate(zip(cga, expected), start=1):
            if type(theirs) is not bool or ours != theirs:
                raise SystemExit(f"{name}: query {i} differs (CGA {ours}, Lean {theirs})")
        count += len(cga)
    print(f"{len(cases)} graphs, {count} CGA/Lean m-connection comparisons agree")


if __name__ == "__main__":
    main()
