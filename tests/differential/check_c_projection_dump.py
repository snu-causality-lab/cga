"""Check the bounded C latent-projection dump by the published path definition.

This oracle deliberately enumerates directed paths and mixed edge-atom paths.  It
does not use the production package or the sequential latent-elimination rule.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


NODES = (0, 1, 2)
PAIRS = ((0, 1), (0, 2), (1, 2))
RAW_DB_ROWS = 216
ACYCLIC_DB_ROWS = 200
MASK_COUNT = 8
EXPECTED_ROWS = ACYCLIC_DB_ROWS * MASK_COUNT
OUTPUT_STRIDE = 3

# (first, arrowhead at first, second, arrowhead at second)
EdgeAtom = tuple[int, bool, int, bool]


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


def decode_choices(raw_row: int) -> tuple[int, ...]:
    return tuple((raw_row // (6**pair_index)) % 6 for pair_index in range(3))


def directed_edges(choices: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for (left, right), choice in zip(PAIRS, choices, strict=True):
        if choice in (1, 4):
            edges.add((left, right))
        elif choice in (2, 5):
            edges.add((right, left))
    return frozenset(edges)


def edge_atoms(choices: tuple[int, ...]) -> tuple[EdgeAtom, ...]:
    atoms: list[EdgeAtom] = []
    for (left, right), choice in zip(PAIRS, choices, strict=True):
        if choice in (1, 4):
            atoms.append((left, False, right, True))
        elif choice in (2, 5):
            atoms.append((right, False, left, True))
        if choice in (3, 4, 5):
            atoms.append((left, True, right, True))
    return tuple(atoms)


def has_topological_order(edges: frozenset[tuple[int, int]]) -> bool:
    remaining = set(NODES)
    while remaining:
        source = next(
            (
                node
                for node in sorted(remaining)
                if not any(
                    target == node and parent in remaining
                    for parent, target in edges
                )
            ),
            None,
        )
        if source is None:
            return False
        remaining.remove(source)
    return True


def valid_source_graphs() -> tuple[tuple[int, tuple[int, ...]], ...]:
    result = tuple(
        (raw_row, choices)
        for raw_row in range(RAW_DB_ROWS)
        if has_topological_order(
            directed_edges(choices := decode_choices(raw_row))
        )
    )
    if len(result) != ACYCLIC_DB_ROWS:
        fail(
            f"independent source census is {len(result)}, "
            f"expected {ACYCLIC_DB_ROWS}"
        )
    return result


def incident_marks(
    atoms: Iterable[EdgeAtom], first: int, second: int
) -> tuple[tuple[bool, bool], ...]:
    marks: list[tuple[bool, bool]] = []
    for atom_first, arrow_first, atom_second, arrow_second in atoms:
        if atom_first == first and atom_second == second:
            marks.append((arrow_first, arrow_second))
        elif atom_first == second and atom_second == first:
            marks.append((arrow_second, arrow_first))
    return tuple(marks)


def has_directed_projection_path(
    source: int,
    target: int,
    retained: frozenset[int],
    directed: frozenset[tuple[int, int]],
) -> bool:
    stack: list[tuple[int, frozenset[int]]] = [
        (source, frozenset((source,)))
    ]
    while stack:
        current, visited = stack.pop()
        for edge_source, neighbor in directed:
            if edge_source != current or neighbor in visited:
                continue
            if neighbor == target:
                return True
            if neighbor in retained:
                continue
            stack.append((neighbor, visited | frozenset((neighbor,))))
    return False


def has_bidirected_projection_path(
    source: int,
    target: int,
    retained: frozenset[int],
    atoms: tuple[EdgeAtom, ...],
) -> bool:
    # State records the arrowhead presented at the current vertex by the
    # preceding edge atom.  Every expanded nonendpoint is therefore latent.
    stack: list[tuple[int, bool | None, frozenset[int]]] = [
        (source, None, frozenset((source,)))
    ]
    while stack:
        current, incoming_arrow, visited = stack.pop()
        for neighbor in NODES:
            if neighbor in visited:
                continue
            for arrow_current, arrow_neighbor in incident_marks(
                atoms, current, neighbor
            ):
                if current == source:
                    if not arrow_current:
                        continue
                elif incoming_arrow and arrow_current:
                    # Both incident atoms have an arrowhead at this internal
                    # vertex, so this path has a forbidden latent collider.
                    continue

                if neighbor == target:
                    if arrow_neighbor:
                        return True
                    continue
                if neighbor in retained:
                    continue
                stack.append(
                    (
                        neighbor,
                        arrow_neighbor,
                        visited | frozenset((neighbor,)),
                    )
                )
    return False


def projection_by_literal_paths(
    choices: tuple[int, ...], retained_mask: int
) -> tuple[int, int, int]:
    retained_nodes = tuple(
        node for node in NODES if retained_mask & (1 << node)
    )
    retained = frozenset(retained_nodes)
    old_to_new = {
        old_vertex: new_vertex
        for new_vertex, old_vertex in enumerate(retained_nodes)
    }
    directed = directed_edges(choices)
    atoms = edge_atoms(choices)
    directed_bits = 0
    bidirected_bits = 0

    for old_source in retained_nodes:
        for old_target in retained_nodes:
            if old_source == old_target:
                continue
            if has_directed_projection_path(
                old_source, old_target, retained, directed
            ):
                source = old_to_new[old_source]
                target = old_to_new[old_target]
                directed_bits |= 1 << (OUTPUT_STRIDE * source + target)

    for first_index, old_first in enumerate(retained_nodes):
        for old_second in retained_nodes[first_index + 1 :]:
            if has_bidirected_projection_path(
                old_first, old_second, retained, atoms
            ):
                first = old_to_new[old_first]
                second = old_to_new[old_second]
                bidirected_bits |= 1 << (OUTPUT_STRIDE * first + second)

    return len(retained_nodes), directed_bits, bidirected_bits


def parse_decimal(text: str, field: str, line_number: int) -> int:
    try:
        value = int(text, 10)
    except ValueError:
        fail(f"line {line_number} has malformed {field}: {text!r}")
    if str(value) != text:
        fail(f"line {line_number} has noncanonical {field}: {text!r}")
    return value


def parse_hex(text: str, field: str, line_number: int) -> int:
    try:
        value = int(text, 16)
    except ValueError:
        fail(f"line {line_number} has malformed {field}: {text!r}")
    if f"{value:03x}" != text:
        fail(f"line {line_number} has noncanonical {field}: {text!r}")
    return value


def check_row(
    line: str,
    line_number: int,
    expected_graph: int,
    expected_raw_row: int,
    expected_choices: tuple[int, ...],
    expected_mask: int,
) -> None:
    fields = line.split("\t")
    if len(fields) != 8:
        fail(f"line {line_number} has {len(fields)} fields, expected 8")
    (
        kind,
        graph_text,
        raw_row_text,
        choices_text,
        mask_text,
        vertices_text,
        directed_text,
        bidirected_text,
    ) = fields
    if kind != "LP":
        fail(f"line {line_number} has kind {kind!r}, expected 'LP'")
    if parse_decimal(graph_text, "graph id", line_number) != expected_graph:
        fail(f"line {line_number} graph order mismatch")
    if parse_decimal(raw_row_text, "raw row", line_number) != expected_raw_row:
        fail(f"line {line_number} raw-row order mismatch")
    expected_choices_text = "".join(str(choice) for choice in expected_choices)
    if choices_text != expected_choices_text:
        fail(
            f"line {line_number} choices {choices_text!r} != "
            f"{expected_choices_text!r}"
        )
    if parse_decimal(mask_text, "retained mask", line_number) != expected_mask:
        fail(f"line {line_number} retained-mask order mismatch")

    expected_vertices, expected_directed, expected_bidirected = (
        projection_by_literal_paths(expected_choices, expected_mask)
    )
    actual_vertices = parse_decimal(vertices_text, "vertex count", line_number)
    actual_directed = parse_hex(directed_text, "directed atoms", line_number)
    actual_bidirected = parse_hex(
        bidirected_text, "bidirected atoms", line_number
    )
    if actual_vertices != expected_vertices:
        fail(
            f"line {line_number}, raw={expected_raw_row}, "
            f"mask={expected_mask}: C vertex count={actual_vertices}, "
            f"oracle={expected_vertices}"
        )
    if actual_directed != expected_directed:
        fail(
            f"line {line_number}, raw={expected_raw_row}, "
            f"mask={expected_mask}: C directed={actual_directed:03x}, "
            f"oracle={expected_directed:03x}"
        )
    if actual_bidirected != expected_bidirected:
        fail(
            f"line {line_number}, raw={expected_raw_row}, "
            f"mask={expected_mask}: C bidirected={actual_bidirected:03x}, "
            f"oracle={expected_bidirected:03x}"
        )


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_c_projection_dump.py PATH_TO_CGA_PROJECTION_DUMP")
    executable = Path(sys.argv[1]).resolve()
    first = run_dump(executable)
    second = run_dump(executable)
    if first != second:
        fail("C latent-projection dump is not byte-for-byte deterministic")
    if first and not first.endswith("\n"):
        fail("C latent-projection dump lacks its final newline")

    rows = first.splitlines()
    if len(rows) != EXPECTED_ROWS:
        fail(f"dump has {len(rows)} rows, expected {EXPECTED_ROWS}")

    graphs = valid_source_graphs()
    for line_index, line in enumerate(rows):
        graph_index = line_index // MASK_COUNT
        retained_mask = line_index % MASK_COUNT
        raw_row, choices = graphs[graph_index]
        check_row(
            line,
            line_index + 1,
            graph_index,
            raw_row,
            choices,
            retained_mask,
        )

    print(
        "C latent-projection differential: "
        f"{ACYCLIC_DB_ROWS} bow-capable ADMGs x {MASK_COUNT} masks "
        f"= {EXPECTED_ROWS} literal-path checks passed"
    )


if __name__ == "__main__":
    main()
