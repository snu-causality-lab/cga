"""Check bounded C MAG equivalence against literal independence models.

The checker does not import the production Python package.  It classifies a
PMG as a MAG from the definition: ancestral validity plus, for every nonedge,
some conditioning subset that m-separates its endpoints.  Its equivalence
signature then literally records every pairwise all-Z m-separation result.
"""

from __future__ import annotations

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
PMG_ROWS = 5 ** len(PAIRS)
EXPECTED_MAGS = 3045
EXPECTED_CLASSES = 251
EXPECTED_LARGEST_CLASS = 242


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


def all_z_signature_if_mag(
    edges: tuple[MixedEdgeAtom, ...],
) -> None | tuple[bool, ...]:
    """Return the literal pair/all-subset CI model, or None outside MAGs."""

    if not endpoint_graph_is_ancestral_by_definition(NODES, edges):
        return None

    signature: list[bool] = []
    for left, right in PAIRS:
        remaining = tuple(node for node in NODES if node not in (left, right))
        pair_separations: list[bool] = []
        for mask in range(1 << len(remaining)):
            given = tuple(
                node
                for index, node in enumerate(remaining)
                if mask & (1 << index)
            )
            separated = m_separated_by_simple_edge_paths(
                NODES, edges, (left,), (right,), given
            )
            pair_separations.append(separated)
            signature.append(separated)
        if not adjacent(edges, left, right) and not any(pair_separations):
            return None
    return tuple(signature)


def parse_row(line: str, expected_row: int) -> tuple[bool, None | int]:
    fields = line.split("\t")
    if len(fields) != 5:
        fail(f"row {expected_row} has {len(fields)} fields, expected 5")
    kind, row_text, choices_text, validity_text, representative_text = fields
    if kind != "PMG4-EQ":
        fail(f"row {expected_row} has kind {kind!r}, expected 'PMG4-EQ'")
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

    if validity_text == "I":
        if representative_text != "-":
            fail(
                f"invalid row {expected_row} has representative "
                f"{representative_text!r}"
            )
        return False, None
    if validity_text != "M":
        fail(f"row {expected_row} has invalid class {validity_text!r}")
    try:
        representative = int(representative_text)
    except ValueError:
        fail(
            f"MAG row {expected_row} has malformed representative "
            f"{representative_text!r}"
        )
    if representative < 0 or representative > expected_row:
        fail(
            f"MAG row {expected_row} has out-of-order representative "
            f"{representative}"
        )
    return True, representative


def main() -> None:
    if len(sys.argv) != 2:
        fail(
            "usage: check_c_mag_equivalence_dump.py "
            "PATH_TO_CGA_MAG_EQUIVALENCE_DUMP"
        )
    executable = Path(sys.argv[1]).resolve()
    first = run_dump(executable)
    second = run_dump(executable)
    if first != second:
        fail("C MAG-equivalence dump is not byte-for-byte deterministic")
    if first and not first.endswith("\n"):
        fail("C MAG-equivalence dump lacks its final newline")

    rows = first.splitlines()
    if len(rows) != PMG_ROWS:
        fail(f"dump has {len(rows)} rows, expected {PMG_ROWS}")

    first_by_signature: dict[tuple[bool, ...], int] = {}
    size_by_signature: dict[tuple[bool, ...], int] = {}
    mag_count = 0
    for row, line in enumerate(rows):
        c_is_mag, c_representative = parse_row(line, row)
        choices = decode_choices(row)
        edges = endpoint_atoms(choices)
        signature = all_z_signature_if_mag(edges)
        oracle_is_mag = signature is not None
        if c_is_mag != oracle_is_mag:
            fail(
                f"row {row} {choices}: C MAG={c_is_mag}, "
                f"all-Z oracle MAG={oracle_is_mag}"
            )
        if signature is None:
            continue

        mag_count += 1
        expected_representative = first_by_signature.setdefault(signature, row)
        size_by_signature[signature] = size_by_signature.get(signature, 0) + 1
        if c_representative != expected_representative:
            fail(
                f"row {row} {choices}: C first representative="
                f"{c_representative}, all-Z oracle first representative="
                f"{expected_representative}"
            )

    class_count = len(first_by_signature)
    largest_class = max(size_by_signature.values(), default=0)
    actual_census = (mag_count, class_count, largest_class)
    expected_census = (
        EXPECTED_MAGS,
        EXPECTED_CLASSES,
        EXPECTED_LARGEST_CLASS,
    )
    if actual_census != expected_census:
        fail(
            f"independent PMG4 MAG/MEC census is {actual_census}, "
            f"expected {expected_census}"
        )

    print(
        "C MAG-equivalence differential: "
        f"{PMG_ROWS} PMG4 rows; {mag_count} MAGs, "
        f"{class_count} classes, largest class {largest_class} passed"
    )


if __name__ == "__main__":
    main()
