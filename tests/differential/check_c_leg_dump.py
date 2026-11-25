"""Compare complete C LEG streams to independent, literal CI equivalence classes.

Every expected member is enumerated independently as a directed/bidirected
endpoint graph, checked for ancestral validity and maximality, and grouped by
all pair/all-conditioning-set m-separations. A class LEG has exactly the
bidirected edges invariant in that class. This is the class-wide collection;
it does not preserve the directed edges of one chosen original DMAG.

No production Python package or C validity/transformation routine supplies an
expected answer. Run on a server allocation, not a development Mac.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from python.oracles import (  # noqa: E402
    ConditionalIndependenceSignature,
    EndpointMarkAtom,
    FullPAGOracleClass,
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    full_pag_classes_by_all_conditioning_subsets,
    m_separation_signature_by_all_conditioning_subsets,
    no_selection_census,
    nonmaximal_pair_by_all_conditioning_sets,
)


PAPER_FIXTURE = REPO_ROOT / "tests/fixtures/papers/zhang_spirtes_2005_legs.json"


@dataclass(frozen=True)
class Case:
    label: str
    n: int
    pag: str
    expected: frozenset[str]


def encode(nodes: tuple[str, ...], edges: tuple[EndpointMarkAtom, ...]) -> str:
    marks = {"tail": "1", "arrow": "2", "circle": "3"}
    by_pair = {(left, right): marks[lm] + marks[rm] for left, lm, right, rm in edges}
    return "".join(
        by_pair.get((left, right), "00")
        for left, right in itertools.combinations(nodes, 2)
    ) or "-"


def encode_mag(nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]) -> str:
    return encode(nodes, tuple(
        (left, "arrow" if lm else "tail", right, "arrow" if rm else "tail")
        for left, lm, right, rm in edges
    ))


def class_case(
    label: str,
    nodes: tuple[str, ...],
    members: tuple[tuple[MixedEdgeAtom, ...], ...],
    pag: tuple[EndpointMarkAtom, ...],
) -> Case:
    invariant_bidirected = {
        (left, right)
        for left, lm, right, rm in pag
        if lm == rm == "arrow"
    }
    expected = frozenset(
        encode_mag(nodes, edges)
        for edges in members
        if {(left, right) for left, lm, right, rm in edges if lm and rm}
        == invariant_bidirected
    )
    if not expected:
        raise AssertionError(f"independent class has no LEG: {label}")
    return Case(label, len(nodes), encode(nodes, pag), expected)


def skeleton_classes(
    nodes: tuple[str, ...], pairs: tuple[tuple[str, str], ...]
) -> dict[ConditionalIndependenceSignature, FullPAGOracleClass]:
    """All three edge orientations on a fixed skeleton, without C pruning."""

    mags = []
    for choices in itertools.product(((False, True), (True, False), (True, True)),
                                     repeat=len(pairs)):
        edges = tuple(
            (left, lm, right, rm)
            for (left, right), (lm, rm) in zip(pairs, choices, strict=True)
        )
        if (endpoint_graph_is_ancestral_by_definition(nodes, edges)
                and nonmaximal_pair_by_all_conditioning_sets(nodes, edges) is None):
            mags.append(edges)
    return full_pag_classes_by_all_conditioning_subsets(nodes, mags)


def paper_cases() -> tuple[list[Case], dict[str, object]]:
    fixture = json.loads(PAPER_FIXTURE.read_text(encoding="utf-8"))
    nodes = tuple(fixture["nodes"])

    def atoms(rows: list[list[str]]) -> tuple[MixedEdgeAtom, ...]:
        if any(lm not in ("tail", "arrow") or rm not in ("tail", "arrow")
               for _, lm, _, rm in rows):
            raise AssertionError("paper graph has a non-MAG endpoint")
        return tuple((left, lm == "arrow", right, rm == "arrow")
                     for left, lm, right, rm in rows)

    anchor = atoms(fixture["graphs"]["H1"]["edges"])
    pairs = tuple((left, right) for left, _, right, _ in anchor)
    classes = skeleton_classes(nodes, pairs)
    signature = m_separation_signature_by_all_conditioning_subsets(nodes, anchor)
    members, pag = classes[signature]
    case = class_case("Zhang-Spirtes-2005-Figure4", nodes, members, pag)
    claimed_pag = tuple(tuple(edge) for edge in fixture["expected_causal_pag"])
    claimed_legs = frozenset(encode_mag(nodes, atoms(graph["edges"]))
                            for graph in fixture["expected_class_legs"])
    if case.pag != encode(nodes, claimed_pag) or case.expected != claimed_legs:
        raise AssertionError("paper fixture PAG/LEG claims disagree with independent census")
    if len(case.expected) != fixture["expected_class_leg_count"]:
        raise AssertionError("paper fixture LEG count disagrees with its graph set")
    actual_separations = {(left, right, given) for left, right, given, separated
                          in signature if separated}
    claimed_separations = {(item["pair"][0], item["pair"][1], tuple(item["conditioning"]))
                           for item in fixture["expected_separations"]}
    if actual_separations != claimed_separations:
        raise AssertionError("paper fixture CI claim disagrees with literal paths")
    graph_atoms = {name: atoms(graph["edges"]) for name, graph in fixture["graphs"].items()}
    for name, edges in graph_atoms.items():
        if edges not in members:
            raise AssertionError(f"paper graph {name} is not in the independent MAG class")
        graph = fixture["graphs"][name]
        if (encode_mag(nodes, edges) in case.expected) != graph["expected_is_class_leg"]:
            raise AssertionError(f"paper graph {name}: incorrect LEG claim")
        if "expected_legitimate_reversals" in graph:
            reversals = set()
            for edge_index, (left, lm, right, rm) in enumerate(edges):
                if lm == rm:
                    continue
                reversed_edges = (edges[:edge_index] + ((left, rm, right, lm),)
                                  + edges[edge_index + 1:])
                if encode_mag(nodes, reversed_edges) in case.expected:
                    reversals.add((right, left) if lm else (left, right))
            if reversals != {tuple(edge) for edge in graph["expected_legitimate_reversals"]}:
                raise AssertionError(f"paper graph {name}: incorrect reversal claim")
    for item in fixture["source_specific_loyalty"]:
        source = graph_atoms[item["source"]]
        candidate = graph_atoms[item["candidate"]]
        preserved = all(edge in candidate for edge in source if edge[1] != edge[3])
        loyal = encode_mag(nodes, candidate) in case.expected and preserved
        if loyal != item["expected"]:
            raise AssertionError("paper fixture source-specific loyalty claim disagrees")
    sequence = [graph_atoms[name] for name in fixture["figure5_order"]]
    for before, after in zip(sequence, sequence[1:]):
        changes = sum((old[1] != new[1]) + (old[3] != new[3])
                      for old, new in zip(before, after, strict=True))
        if changes != 1:
            raise AssertionError("Figure 5 transition does not change exactly one mark")

    # Figure 6 independently establishes the excluded undirected-edge domain.
    scope = fixture["figure6_scope_counterexample"]
    scope_nodes = tuple(scope["nodes"])
    undirected, alternative = atoms(scope["M1"]), atoms(scope["M2"])
    for edges in (undirected, alternative):
        if (not endpoint_graph_is_ancestral_by_definition(scope_nodes, edges)
                or nonmaximal_pair_by_all_conditioning_sets(scope_nodes, edges) is not None):
            raise AssertionError("Figure 6 example is not maximal ancestral")
    equivalent = (m_separation_signature_by_all_conditioning_subsets(scope_nodes, undirected)
                  == m_separation_signature_by_all_conditioning_subsets(scope_nodes, alternative))
    if equivalent != scope["expected_markov_equivalent"]:
        raise AssertionError("Figure 6 equivalence claim disagrees with literal CI")
    valid_changes = 0
    for edge_index, (left, lm, right, rm) in enumerate(undirected):
        for changed in ((left, not lm, right, rm), (left, lm, right, not rm)):
            candidate = undirected[:edge_index] + (changed,) + undirected[edge_index + 1:]
            valid_changes += endpoint_graph_is_ancestral_by_definition(scope_nodes, candidate)
    if valid_changes != scope["expected_single_mark_changes_from_M1_preserving_ancestrality"]:
        raise AssertionError("Figure 6 single-mark-change claim disagrees with definition")

    shorthand = fixture["bidirected_spouse_shorthand_control"]
    two_nodes = tuple(shorthand["nodes"])
    bidirected = atoms(shorthand["bidirected"])
    variants = [bidirected, *(atoms(edges) for edges in shorthand["orientations"])]
    signatures = {m_separation_signature_by_all_conditioning_subsets(two_nodes, edges)
                  for edges in variants}
    if (len(signatures) == 1) != shorthand["expected_all_markov_equivalent"]:
        raise AssertionError("two-node spouse shorthand CI claim disagrees")
    spouses = {node: {right if left == node else left
                     for left, lm, right, rm in bidirected
                     if lm and rm and node in (left, right)} for node in two_nodes}
    equal = spouses[two_nodes[0]] == spouses[two_nodes[1]]
    if equal != shorthand["expected_original_spouse_sets_equal"]:
        raise AssertionError("two-node spouse shorthand set claim disagrees")

    cases = [class_case(f"paper-skeleton-class{index}", nodes, members, pag)
             for index, (members, pag) in enumerate(classes.values())]
    return cases, {"fixture": str(PAPER_FIXTURE.relative_to(REPO_ROOT)),
                   "sha256": hashlib.sha256(PAPER_FIXTURE.read_bytes()).hexdigest(),
                   "classes": len(classes), "illustrated_graphs": len(graph_atoms),
                   "class_legs": len(case.expected),
                   "ci_pag_legs_reversals_loyalty_scope_checked": True}


def oracle_cases() -> tuple[list[Case], dict[str, object]]:
    cases: list[Case] = []
    censuses = []
    for n in range(5):
        nodes = tuple(str(index) for index in range(n))
        census = no_selection_census(nodes)
        for index, (members, pag) in enumerate(census.classes.values()):
            cases.append(class_case(f"exhaustive-n{n}-class{index}", nodes, members, pag))
        censuses.append({"n": n, "raw": census.raw_count,
                         "ancestral": census.ancestral_count,
                         "mags": len(census.mags), "classes": len(census.classes)})

    # The class-wide API must include the reversed one-edge DAG. A LEG of the
    # particular original 0 -> 1 preserving that graph's arrows cannot do so.
    edge_class = next(case for case in cases if case.n == 2 and case.pag == "33")
    if edge_class.expected != frozenset({"12", "21"}):
        raise AssertionError("independent two-node class does not separate LEG definitions")

    nodes = tuple(str(index) for index in range(5))
    all_pairs = tuple(itertools.combinations(nodes, 2))
    common_spouses = (("0", "1"), ("0", "2"), ("1", "2"),
                      ("1", "3"), ("2", "3"), ("3", "4"))
    rng = random.Random(20261005)
    skeletons = [
        ("path", (("0", "1"), ("1", "2"), ("2", "3"), ("3", "4"))),
        ("cycle", (("0", "1"), ("0", "4"), ("1", "2"), ("2", "3"), ("3", "4"))),
        ("common-spouses", common_spouses),
        ("random-six-edges", tuple(sorted(rng.sample(all_pairs, 6)))),
    ]
    skeleton_summaries = []
    for label, pairs in skeletons:
        for relabel_index in range(3):
            permutation = list(nodes)
            if relabel_index:
                rng.shuffle(permutation)
            rename = dict(zip(nodes, permutation, strict=True))
            renamed_pairs = tuple(sorted(
                tuple(sorted((rename[left], rename[right]))) for left, right in pairs
            ))
            classes = skeleton_classes(nodes, renamed_pairs)
            for index, (members, pag) in enumerate(classes.values()):
                cases.append(class_case(
                    f"n5-{label}-relabel{relabel_index}-class{index}", nodes, members, pag
                ))
            skeleton_summaries.append({"label": label, "relabel": permutation,
                                       "raw": 3 ** len(pairs), "classes": len(classes)})
            if label == "common-spouses" and relabel_index == 0:
                # Synthetic shared-spouse regression: derive the class and PAG
                # independently before checking the proposed two orientations.
                anchor = (("0", False, "1", True), ("0", False, "2", True),
                          ("1", False, "2", True), ("1", True, "3", True),
                          ("2", True, "3", True), ("3", True, "4", False))
                signature = m_separation_signature_by_all_conditioning_subsets(nodes, anchor)
                members, pag = classes[signature]
                checked = class_case("synthetic-common-spouses", nodes, members, pag)
                proposed_pag = (("0", "circle", "1", "arrow"),
                                ("0", "circle", "2", "arrow"),
                                ("1", "circle", "2", "circle"),
                                ("1", "arrow", "3", "arrow"),
                                ("2", "arrow", "3", "arrow"),
                                ("3", "arrow", "4", "circle"))
                reversed_middle = tuple(
                    (left, not lm, right, not rm) if (left, right) == ("1", "2")
                    else (left, lm, right, rm) for left, lm, right, rm in anchor
                )
                if checked.pag != encode(nodes, proposed_pag) or checked.expected != frozenset({
                    encode_mag(nodes, anchor), encode_mag(nodes, reversed_middle)
                }):
                    raise AssertionError("synthetic shared-spouse fixture needs adjudication")
    published_cases, paper_summary = paper_cases()
    cases.extend(published_cases)
    return cases, {"exhaustive": censuses, "n5_skeletons": skeleton_summaries,
                   "published_fixture": paper_summary,
                   "narrow_definition_witness_checked": True,
                   "synthetic_common_spouses_checked": True}


def check_dump(output: str, cases: list[Case]) -> int:
    rows = iter(output.splitlines())
    total = 0
    if not output.endswith("\n"):
        raise AssertionError("C dump lacks its final newline")
    for index, case in enumerate(cases):
        seed = next(rows, "").split("\t")
        if len(seed) != 4 or seed[:3] != ["SEED", str(index), str(case.n)]:
            raise AssertionError(f"{case.label}: malformed seed record {seed}")
        if seed[3] not in case.expected:
            raise AssertionError(f"{case.label}: seed {seed[3]} is outside independent LEG set")
        actual: list[str] = []
        for row in rows:
            fields = row.split("\t")
            if fields[0] == "END":
                if fields != ["END", str(index), str(len(actual))]:
                    raise AssertionError(f"{case.label}: inconsistent end record {fields}")
                break
            if len(fields) != 4 or fields[:3] != ["LEG", str(index), str(case.n)]:
                raise AssertionError(f"{case.label}: malformed LEG record {fields}")
            actual.append(fields[3])
        else:
            raise AssertionError(f"{case.label}: missing END record")
        actual_set = set(actual)
        if not actual or actual[0] != seed[3]:
            raise AssertionError(f"{case.label}: iterator does not start at the representative")
        if len(actual) != len(actual_set):
            raise AssertionError(f"{case.label}: duplicate LEG output")
        if actual_set != case.expected:
            raise AssertionError(
                f"{case.label}: missing={sorted(case.expected - actual_set)}, "
                f"unexpected={sorted(actual_set - case.expected)}"
            )
        total += len(actual)
    extra = next(rows, None)
    if extra is not None:
        raise AssertionError(f"extra C output after final case: {extra!r}")
    return total


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = [REPO_ROOT / "Makefile", Path(__file__).resolve(),
             REPO_ROOT / "tests/python/oracles.py",
             REPO_ROOT / "c/tests/cga_leg_dump.c", PAPER_FIXTURE]
    paths.extend(path for path in (REPO_ROOT / "c/src").rglob("*")
                 if path.suffix in {".c", ".h"})
    paths.extend((REPO_ROOT / "c/include/cga").glob("*.h"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO_ROOT)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_c_leg_dump.py PATH_TO_CGA_LEG_DUMP")
    executable = Path(sys.argv[1]).resolve(strict=True)
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    source_sha256 = source_digest()
    cases, summary = oracle_cases()
    payload = "".join(f"{case.n} {case.pag}\n" for case in cases)
    outputs = []
    for _ in range(2):
        result = subprocess.run([str(executable)], input=payload, capture_output=True,
                                text=True, encoding="utf-8", timeout=300, check=False)
        if result.returncode != 0 or result.stderr:
            raise AssertionError(f"C dump exit {result.returncode}: {result.stderr.rstrip()}")
        outputs.append(result.stdout)
    if outputs[0] != outputs[1]:
        raise AssertionError("C LEG dump is not byte-for-byte deterministic")
    total = check_dump(outputs[0], cases)
    if (hashlib.sha256(executable.read_bytes()).hexdigest() != executable_sha256
            or source_digest() != source_sha256):
        raise AssertionError("source or executable changed during verification")
    print(json.dumps({"check": "C LEG independent differential", "passed": True,
                      "cases": len(cases), "leg_rows": total,
                      "executable": str(executable), "executable_sha256": executable_sha256,
                      "source_sha256": source_sha256,
                      "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "oracle": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
