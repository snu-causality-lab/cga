"""Check every bounded C m-separation dump row with independent Python oracles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from python.oracles import (  # noqa: E402
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    has_topological_order,
    m_separated_by_canonical_dag,
    m_separated_by_simple_edge_paths,
)


NODES = ("0", "1", "2")
PAIRS = (("0", "1"), ("0", "2"), ("1", "2"))
DB_ROWS = 216
PMG_ROWS = 125
QUERY_COUNT = 18


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


def directed_bidirected_atoms(choices: tuple[int, ...]) -> tuple[MixedEdgeAtom, ...]:
    atoms: list[MixedEdgeAtom] = []
    for (left, right), choice in zip(PAIRS, choices, strict=True):
        if choice in (1, 4):
            atoms.append((left, False, right, True))
        if choice in (2, 5):
            atoms.append((right, False, left, True))
        if choice in (3, 4, 5):
            atoms.append((left, True, right, True))
    return tuple(atoms)


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


def mask_nodes(mask: int) -> tuple[str, ...]:
    return tuple(node for index, node in enumerate(NODES) if mask & (1 << index))


def all_valid_queries():
    mask_limit = 1 << len(NODES)
    for left_mask in range(1, mask_limit):
        for right_mask in range(1, mask_limit):
            for given_mask in range(mask_limit):
                if left_mask & right_mask or left_mask & given_mask:
                    continue
                if right_mask & given_mask:
                    continue
                yield (
                    mask_nodes(left_mask),
                    mask_nodes(right_mask),
                    mask_nodes(given_mask),
                )


QUERIES = tuple(all_valid_queries())
assert len(QUERIES) == QUERY_COUNT


def oracle_signature(edges: tuple[MixedEdgeAtom, ...]) -> str:
    signature: list[str] = []
    for left, right, given in QUERIES:
        path_result = m_separated_by_simple_edge_paths(
            NODES, edges, left, right, given
        )
        dag_result = m_separated_by_canonical_dag(
            NODES, edges, left, right, given
        )
        if path_result != dag_result:
            fail(
                "independent Python oracles disagree for "
                f"edges={edges}, query=({left}, {right} | {given})"
            )
        signature.append("S" if path_result else "C")
    return "".join(signature)


def parse_choices(text: str, base: int, row_kind: str, row_id: int) -> tuple[int, ...]:
    if len(text) != len(PAIRS) or any(
        character < "0" or character >= str(base) for character in text
    ):
        fail(f"malformed {row_kind} choices at row {row_id}: {text!r}")
    return tuple(int(character) for character in text)


def check_db_row(fields: list[str], expected_id: int) -> None:
    if len(fields) != 4:
        fail(f"DB row {expected_id} has {len(fields)} fields, expected 4")
    kind, row_text, choice_text, actual_signature = fields
    if kind != "DB" or row_text != str(expected_id):
        fail(f"DB row order mismatch at {expected_id}: {fields!r}")
    choices = parse_choices(choice_text, 6, kind, expected_id)
    expected_choices = tuple(
        (expected_id // (6**pair_index)) % 6
        for pair_index in range(len(PAIRS))
    )
    if choices != expected_choices:
        fail(
            f"DB row {expected_id} choices {choices} != "
            f"odometer {expected_choices}"
        )

    edges = directed_bidirected_atoms(choices)
    directed = frozenset(
        (first, second)
        for first, arrow_first, second, arrow_second in edges
        if not arrow_first and arrow_second
    )
    expected_signature = (
        oracle_signature(edges)
        if has_topological_order(NODES, directed)
        else "I" * QUERY_COUNT
    )
    if actual_signature != expected_signature:
        fail(
            f"DB row {expected_id} {choices}: "
            f"C={actual_signature}, oracle={expected_signature}"
        )


def check_pmg_row(fields: list[str], expected_id: int) -> None:
    if len(fields) != 5:
        fail(f"PMG row {expected_id} has {len(fields)} fields, expected 5")
    kind, row_text, choice_text, actual_class, actual_signature = fields
    if kind != "PMG" or row_text != str(expected_id):
        fail(f"PMG row order mismatch at {expected_id}: {fields!r}")
    choices = parse_choices(choice_text, 5, kind, expected_id)
    expected_choices = tuple(
        (expected_id // (5**pair_index)) % 5
        for pair_index in range(len(PAIRS))
    )
    if choices != expected_choices:
        fail(
            f"PMG row {expected_id} choices {choices} != "
            f"odometer {expected_choices}"
        )

    edges = endpoint_atoms(choices)
    ancestral = endpoint_graph_is_ancestral_by_definition(NODES, edges)
    expected_class = "A" if ancestral else "N"
    expected_signature = (
        oracle_signature(edges) if ancestral else "I" * QUERY_COUNT
    )
    if actual_class != expected_class:
        fail(
            f"PMG row {expected_id} {choices}: "
            f"C class={actual_class}, oracle={expected_class}"
        )
    if actual_signature != expected_signature:
        fail(
            f"PMG row {expected_id} {choices}: "
            f"C={actual_signature}, oracle={expected_signature}"
        )


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_c_msep_dump.py PATH_TO_CGA_MSEP_DUMP")
    executable = Path(sys.argv[1]).resolve()
    first = run_dump(executable)
    second = run_dump(executable)
    if first != second:
        fail("C m-separation dump is not byte-for-byte deterministic")

    rows = first.splitlines()
    expected_total = DB_ROWS + PMG_ROWS
    if len(rows) != expected_total:
        fail(f"dump has {len(rows)} rows, expected {expected_total}")
    for row_id, line in enumerate(rows[:DB_ROWS]):
        check_db_row(line.split("\t"), row_id)
    for row_id, line in enumerate(rows[DB_ROWS:]):
        check_pmg_row(line.split("\t"), row_id)

    print(
        "C m-separation differential: "
        f"{DB_ROWS} directed-bidirected + {PMG_ROWS} endpoint-marked rows, "
        f"{QUERY_COUNT} mask queries each passed"
    )


if __name__ == "__main__":
    main()
