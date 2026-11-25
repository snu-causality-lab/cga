"""Replay C adjustment answers against literal original-DAG path conditions.

The independent oracle enumerates proper causal and noncausal simple paths; it
does not construct the production proper back-door graph or call separation.
The existing Python public API is checked as a third implementation, not used
as ground truth. Run exhaustive/randomized verification on a server or CI.
"""
from __future__ import annotations

import itertools
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
sys.path.insert(0, str(ROOT / "python" / "src"))

from dag_adjustment_oracles import all_dags, valid  # noqa: E402
from causal_graph_algorithms import (  # noqa: E402
    DAG, DirectedBidirectedGraph, dag_adjustment_set_is_valid,
)


def run_dump(executable: Path) -> bytes:
    result = subprocess.run([str(executable)], check=True, capture_output=True)
    if result.stderr:
        raise AssertionError(f"unexpected stderr: {result.stderr}")
    return result.stdout


def decode(n: int, code: int) -> frozenset[tuple[str, str]]:
    if code < 0 or code >= 1 << (n * n):
        raise AssertionError(f"out-of-range graph code: {n}, {code}")
    edges = frozenset(
        (str(a), str(b)) for a in range(n) for b in range(n)
        if code & (1 << (a * n + b))
    )
    if any(a == b or (b, a) in edges for a, b in edges):
        raise AssertionError(f"self-edge or opposing edges in dump: {edges}")
    return edges


def has_order(nodes: tuple[str, ...], edges: frozenset[tuple[str, str]]) -> bool:
    """Definition-level acyclicity via a topological permutation witness."""
    return any(
        all(order.index(a) < order.index(b) for a, b in edges)
        for order in itertools.permutations(nodes)
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_c_adjustment_dump.py PATH_TO_CGA_ADJUSTMENT_DUMP")
    executable = Path(sys.argv[1]).resolve()
    raw_output = run_dump(executable)
    if run_dump(executable) != raw_output:
        raise AssertionError("C adjustment dump is not byte-for-byte deterministic")
    rows = raw_output.decode("ascii").splitlines()
    if len(rows) != 729 + 64:
        raise AssertionError(f"expected 793 graph rows, got {len(rows)}")
    nodes4 = tuple(map(str, range(4)))
    expected_dags = set(all_dags(nodes4))
    expected_topologies = {
        frozenset(
            pair if state == 1 else pair[::-1]
            for pair, state in zip(itertools.combinations(nodes4, 2), states, strict=True)
            if state
        )
        for states in itertools.product(range(3), repeat=6)
    }
    seen_topologies: set[frozenset[tuple[str, str]]] = set()
    seen_dags: set[frozenset[tuple[str, str]]] = set()
    graph_count = 0
    query_count = 0
    for index, row in enumerate(rows):
        fields = row.split("\t")
        if len(fields) != 3:
            raise AssertionError(f"malformed row {index}: {row!r}")
        n, code = map(int, fields[:2])
        expected_n = 4 if index < 729 else 5 if index < 761 else 6
        if n != expected_n:
            raise AssertionError(f"row {index}: expected n={expected_n}, got {n}")
        edges = decode(n, code)
        nodes = tuple(map(str, range(n)))
        acyclic = edges in expected_dags if n == 4 else has_order(nodes, edges)
        if n == 4:
            if edges in seen_topologies:
                raise AssertionError(f"duplicate exhaustive graph at row {index}")
            seen_topologies.add(edges)
        elif not acyclic:
            raise AssertionError(f"random DAG row {index} contains a cycle")
        if not acyclic:
            if fields[2] != "I":
                raise AssertionError(f"cyclic graph not rejected at row {index}")
            continue
        if n == 4:
            seen_dags.add(edges)
        expected: list[str] = []
        graph = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=edges))
        for a in nodes:
            for y in nodes:
                if a == y:
                    continue
                for mask in range(1 << n):
                    z = {v for i, v in enumerate(nodes) if mask & (1 << i)}
                    if z & {a, y}:
                        continue
                    literal = valid(nodes, edges, {a}, {y}, z)
                    python = dag_adjustment_set_is_valid(
                        graph, interventions=a, outcomes=y, adjustment=z,
                    )
                    if literal != python:
                        raise AssertionError(
                            f"Python/path disagreement: edges={edges}, A={a}, Y={y}, Z={z}"
                        )
                    expected.append("1" if literal else "0")
                    query_count += 1
        if fields[2] != "".join(expected):
            raise AssertionError(f"C/path disagreement: row={index}, n={n}, edges={edges}")
        graph_count += 1
    if seen_topologies != expected_topologies or seen_dags != expected_dags:
        raise AssertionError("exhaustive labelled n=4 universe mismatch")
    print(
        "C DAG adjustment differential: "
        f"{len(seen_dags)} four-node DAGs + 64 seeded five/six-node DAGs, "
        f"{query_count} C/Python/literal-path checks passed; "
        f"{len(seen_topologies) - len(seen_dags)} cyclic inputs rejected"
    )


if __name__ == "__main__":
    main()
