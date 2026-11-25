"""Check bounded C MAG classification by two definition-level oracles.

The maximality oracle literally enumerates every conditioning subset for each
nonedge and calls the independent vertex-simple m-separation path oracle.  A
second check independently enumerates primitive inducing paths.  Neither check
imports the production Python package or mirrors the bounded C bitset search.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from python.oracles import (  # noqa: E402
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    m_separated_by_simple_edge_paths,
)


NODES = ("0", "1", "2", "3")
PAIRS = (
    ("0", "1"),
    ("0", "2"),
    ("0", "3"),
    ("1", "2"),
    ("1", "3"),
    ("2", "3"),
)
PMG_ROWS = 5**len(PAIRS)
EXPECTED_ANCESTRAL = 3057
EXPECTED_MAXIMAL_ANCESTRAL = 3045
EXPECTED_NONMAXIMAL_ANCESTRAL = 12


def fail(message: str) -> None:
    raise SystemExit(message)


def run_dump(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        fail(
            f"{executable} exited {completed.returncode}: "
            f"{completed.stderr.rstrip()}"
        )
    if completed.stderr:
        fail(f"{executable} wrote unexpected stderr: {completed.stderr.rstrip()}")
    return completed.stdout


def decode_choices(row: int) -> tuple[int, ...]:
    return tuple(
        (row // (5**pair_index)) % 5
        for pair_index in range(len(PAIRS))
    )


def endpoint_atoms(choices: tuple[int, ...]) -> tuple[MixedEdgeAtom, ...]:
    states: tuple[None | tuple[bool, bool], ...] = (
        None,
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    return tuple(
        (left, state[0], right, state[1])
        for (left, right), choice in zip(PAIRS, choices, strict=True)
        if (state := states[choice]) is not None
    )


def adjacent(
    edges: tuple[MixedEdgeAtom, ...], first: str, second: str
) -> bool:
    return any(
        (edge_first == first and edge_second == second)
        or (edge_first == second and edge_second == first)
        for edge_first, _, edge_second, _ in edges
    )


def maximal_by_all_conditioning_subsets(
    edges: tuple[MixedEdgeAtom, ...],
) -> bool:
    """Richardson--Spirtes maximality definition, evaluated literally."""

    for left, right in PAIRS:
        if adjacent(edges, left, right):
            continue
        remaining = tuple(node for node in NODES if node not in (left, right))
        has_separator = False
        for mask in range(1 << len(remaining)):
            given = tuple(
                node
                for index, node in enumerate(remaining)
                if mask & (1 << index)
            )
            if m_separated_by_simple_edge_paths(
                NODES, edges, (left,), (right,), given
            ):
                has_separator = True
                break
        if not has_separator:
            return False
    return True


def directed_edges(
    edges: tuple[MixedEdgeAtom, ...],
) -> frozenset[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for first, arrow_first, second, arrow_second in edges:
        if not arrow_first and arrow_second:
            result.add((first, second))
        elif arrow_first and not arrow_second:
            result.add((second, first))
    return frozenset(result)


def has_positive_directed_path(
    directed: frozenset[tuple[str, str]], source: str, target: str
) -> bool:
    """Literal simple directed paths, separate from the m-separation walk."""

    pending: list[tuple[str, frozenset[str]]] = [
        (source, frozenset((source,)))
    ]
    while pending:
        current, visited = pending.pop()
        for edge_source, neighbor in directed:
            if edge_source != current or neighbor in visited:
                continue
            if neighbor == target:
                return True
            pending.append((neighbor, visited | frozenset((neighbor,))))
    return False


def incident_marks(
    edges: tuple[MixedEdgeAtom, ...], first: str, second: str
) -> None | tuple[bool, bool]:
    for edge_first, arrow_first, edge_second, arrow_second in edges:
        if edge_first == first and edge_second == second:
            return arrow_first, arrow_second
        if edge_first == second and edge_second == first:
            return arrow_second, arrow_first
    return None


def is_primitive_inducing_path(
    path: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    directed: frozenset[tuple[str, str]],
) -> bool:
    left = path[0]
    right = path[-1]

    for path_index in range(1, len(path) - 1):
        previous_marks = incident_marks(
            edges, path[path_index - 1], path[path_index]
        )
        next_marks = incident_marks(
            edges, path[path_index], path[path_index + 1]
        )
        if previous_marks is None or next_marks is None:
            return False
        incoming_arrow = previous_marks[1]
        outgoing_arrow = next_marks[0]
        if not incoming_arrow or not outgoing_arrow:
            return False
        internal = path[path_index]
        if not (
            has_positive_directed_path(directed, internal, left)
            or has_positive_directed_path(directed, internal, right)
        ):
            return False
    return True


def has_primitive_inducing_path(
    edges: tuple[MixedEdgeAtom, ...], left: str, right: str
) -> bool:
    directed = directed_edges(edges)
    possible_internal = tuple(
        node for node in NODES if node not in (left, right)
    )
    for internal_count in range(1, len(possible_internal) + 1):
        for internal in itertools.permutations(
            possible_internal, internal_count
        ):
            path = (left, *internal, right)
            if is_primitive_inducing_path(path, edges, directed):
                return True
    return False


def maximal_by_primitive_inducing_paths(
    edges: tuple[MixedEdgeAtom, ...],
) -> bool:
    return all(
        adjacent(edges, left, right)
        or not has_primitive_inducing_path(edges, left, right)
        for left, right in PAIRS
    )


def parse_row(line: str, expected_row: int) -> tuple[bool, bool]:
    fields = line.split("\t")
    if len(fields) != 5:
        fail(f"row {expected_row} has {len(fields)} fields, expected 5")
    kind, row_text, choices_text, ancestry_text, maximality_text = fields
    if kind != "PMG4":
        fail(f"row {expected_row} has kind {kind!r}, expected 'PMG4'")
    if row_text != str(expected_row):
        fail(f"row order mismatch at {expected_row}: {row_text!r}")
    if len(choices_text) != len(PAIRS) or any(
        character not in "01234" for character in choices_text
    ):
        fail(f"row {expected_row} has malformed choices {choices_text!r}")
    actual_choices = tuple(int(character) for character in choices_text)
    expected_choices = decode_choices(expected_row)
    if actual_choices != expected_choices:
        fail(
            f"row {expected_row} choices {actual_choices} != "
            f"odometer {expected_choices}"
        )
    if ancestry_text not in ("A", "N"):
        fail(f"row {expected_row} has invalid ancestry class {ancestry_text!r}")
    if maximality_text not in ("M", "N"):
        fail(
            f"row {expected_row} has invalid maximality class "
            f"{maximality_text!r}"
        )
    return ancestry_text == "A", maximality_text == "M"


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_c_maximality_dump.py PATH_TO_CGA_MAXIMALITY_DUMP")
    executable = Path(sys.argv[1]).resolve()
    first = run_dump(executable)
    second = run_dump(executable)
    if first != second:
        fail("C maximality dump is not byte-for-byte deterministic")
    if first and not first.endswith("\n"):
        fail("C maximality dump lacks its final newline")

    rows = first.splitlines()
    if len(rows) != PMG_ROWS:
        fail(f"dump has {len(rows)} rows, expected {PMG_ROWS}")

    ancestral_count = 0
    maximal_count = 0
    nonmaximal_ancestral_count = 0
    for row, line in enumerate(rows):
        c_ancestral, c_maximal = parse_row(line, row)
        choices = decode_choices(row)
        edges = endpoint_atoms(choices)
        oracle_ancestral = endpoint_graph_is_ancestral_by_definition(
            NODES, edges
        )
        if c_ancestral != oracle_ancestral:
            fail(
                f"row {row} {choices}: C ancestry={c_ancestral}, "
                f"oracle={oracle_ancestral}"
            )

        if not oracle_ancestral:
            oracle_maximal = False
        else:
            ancestral_count += 1
            all_z_maximal = maximal_by_all_conditioning_subsets(edges)
            inducing_maximal = maximal_by_primitive_inducing_paths(edges)
            if all_z_maximal != inducing_maximal:
                fail(
                    f"row {row} {choices}: all-Z maximality="
                    f"{all_z_maximal}, primitive-inducing-path maximality="
                    f"{inducing_maximal}"
                )
            oracle_maximal = all_z_maximal
            if oracle_maximal:
                maximal_count += 1
            else:
                nonmaximal_ancestral_count += 1

        if c_maximal != oracle_maximal:
            fail(
                f"row {row} {choices}: C maximal-ancestral={c_maximal}, "
                f"oracle={oracle_maximal}"
            )

    actual_census = (
        ancestral_count,
        maximal_count,
        nonmaximal_ancestral_count,
    )
    expected_census = (
        EXPECTED_ANCESTRAL,
        EXPECTED_MAXIMAL_ANCESTRAL,
        EXPECTED_NONMAXIMAL_ANCESTRAL,
    )
    if actual_census != expected_census:
        fail(
            f"independent PMG4 census is {actual_census}, "
            f"expected {expected_census}"
        )

    print(
        "C maximal-ancestral differential: "
        f"{PMG_ROWS} PMG4 rows; {ancestral_count} ancestral, "
        f"{maximal_count} maximal ancestral, "
        f"{nonmaximal_ancestral_count} nonmaximal ancestral passed"
    )


if __name__ == "__main__":
    main()
