from __future__ import annotations

import copy
import inspect
import json
import pickle
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, get_args

import pytest

import causal_graph_algorithms as cga
from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    TianAdjustmentDomainError,
    TianAdjustmentNotApplicable,
    TianToUCAResult,
    TianUCAFactor,
    TianUCARepresentation,
    tian_adjustment_to_uca,
)

from .id_oracles import (
    TinyADMG,
    all_bow_admgs_n3,
    exact_binary_interventional_probability,
    exact_binary_observational_distribution,
)
from .tian_uca_oracles import (
    OracleTianFactor,
    corrected_tian_factors_by_maximal_runs,
    criterion_children_by_definition,
    equation4_binary_outcome_mean,
    grouped_uca_binary_outcome_mean,
    topological_orders_by_permutation,
)


DOMAIN_MESSAGE = "Tian-to-UCA requires an acyclic directed-bidirected or DAG view"
DISTINCT_MESSAGE = "treatment and outcome must be distinct"
PERMUTATION_MESSAGE = "variable_order must contain every graph vertex exactly once"
TOPOLOGICAL_MESSAGE = "variable_order must be topological for the graph"
OUTCOME_LAST_MESSAGE = "outcome must be last in variable_order"
FIXTURE_NAME = "jung_tian_bareinboim_2024_tian_uca.json"


def _adbg(
    *,
    nodes: tuple[str, ...] | list[str],
    directed: tuple[tuple[str, str], ...] | list[list[str]] = (),
    bidirected: tuple[tuple[str, str], ...] | list[list[str]] = (),
) -> AcyclicDirectedBidirectedGraph:
    return AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(
            nodes=nodes,
            directed=directed,
            bidirected=bidirected,
        )
    )


def _dag(
    *,
    nodes: tuple[str, ...] | list[str],
    directed: tuple[tuple[str, str], ...] | list[list[str]] = (),
) -> DAG:
    return DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=directed))


def _load_fixture() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / FIXTURE_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _factor_fields(
    factors: tuple[TianUCAFactor, ...],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (factor.variables, factor.conditioned_on, factor.fixed_conditioning)
        for factor in factors
    )


def _fixture_factor_fields(
    raw_factors: list[dict[str, list[str]]],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (
            tuple(factor["variables"]),
            tuple(factor["conditioned_on"]),
            tuple(factor["fixed_conditioning"]),
        )
        for factor in raw_factors
    )


def _labeled_graph(
    graph: TinyADMG,
) -> tuple[AcyclicDirectedBidirectedGraph, tuple[str, ...]]:
    labels = tuple(f"V{node}" for node in graph.nodes)
    label = dict(zip(graph.nodes, labels, strict=True))
    return (
        _adbg(
            nodes=labels,
            directed=tuple(
                (label[parent], label[child]) for parent, child in graph.directed
            ),
            bidirected=tuple(
                (label[left], label[right]) for left, right in graph.bidirected
            ),
        ),
        labels,
    )


def _integer_factors(
    factors: tuple[TianUCAFactor, ...],
) -> tuple[OracleTianFactor, ...]:
    def integer(label: str) -> int:
        assert label.startswith("V")
        return int(label[1:])

    return tuple(
        OracleTianFactor(
            tuple(integer(node) for node in factor.variables),
            tuple(integer(node) for node in factor.conditioned_on),
            tuple(integer(node) for node in factor.fixed_conditioning),
        )
        for factor in factors
    )


def test_tian_to_uca_public_surface_and_signature_are_narrow() -> None:
    assert {
        "TianAdjustmentDomainError",
        "TianUCAFactor",
        "TianUCARepresentation",
        "TianAdjustmentNotApplicable",
        "TianToUCAResult",
        "tian_adjustment_to_uca",
    } <= set(cga.__all__)
    assert issubclass(TianAdjustmentDomainError, ValueError)
    assert set(get_args(TianToUCAResult)) == {
        TianUCARepresentation,
        TianAdjustmentNotApplicable,
    }
    assert TianUCAFactor.__slots__ == (
        "variables",
        "conditioned_on",
        "fixed_conditioning",
    )
    assert TianUCARepresentation.__slots__ == (
        "graph",
        "treatment",
        "outcome",
        "variable_order",
        "factors",
    )
    assert TianAdjustmentNotApplicable.__slots__ == (
        "graph",
        "treatment",
        "outcome",
        "variable_order",
        "children_in_treatment_district",
    )

    signature = inspect.signature(tian_adjustment_to_uca)
    assert tuple(signature.parameters) == (
        "graph",
        "treatment",
        "outcome",
        "variable_order",
    )
    assert signature.parameters["graph"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["graph"].default is inspect.Parameter.empty
    for name in ("treatment", "outcome", "variable_order"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    with pytest.raises(TypeError):
        tian_adjustment_to_uca(  # type: ignore[misc]
            graph,
            "X",
            "Y",
            ("X", "Y"),
        )
    with pytest.raises(TypeError):
        tian_adjustment_to_uca(  # type: ignore[call-arg]
            graph,
            treatment="X",
            outcome="Y",
        )
    for unsupported in (
        {"treatment_values": (0, 1)},
        {"return_formula": True},
        {"estimate": True},
        {"max_vertices": 2},
    ):
        with pytest.raises(TypeError):
            tian_adjustment_to_uca(  # type: ignore[call-arg]
                graph,
                treatment="X",
                outcome="Y",
                variable_order=("X", "Y"),
                **unsupported,
            )


def test_front_door_and_verma_fixtures_use_corrected_blocks_and_final_y() -> None:
    fixture = _load_fixture()
    assert fixture["source"]["venue"] == "NeurIPS 2024"
    assert fixture["source"]["doi"] == "10.52202/079017-0208"
    assert fixture["source"]["official_pdf_sha256"] == (
        "1c9bc6f603047b80443d4eff421810a19390cf152d68fd97ad507f3eec5e81ab"
    )
    assert fixture["independent_adjudication"]["status"] == (
        "paper pseudocode requires correction"
    )

    cases = {case["name"]: case for case in fixture["cases"]}
    assert set(cases) == {"front_door_figure_1a", "verma_figure_1b"}
    for case in cases.values():
        graph = _adbg(
            nodes=case["nodes"],
            directed=case["directed"],
            bidirected=case["bidirected"],
        )
        result = tian_adjustment_to_uca(
            graph,
            treatment=case["treatment"],
            outcome=case["outcome"],
            variable_order=case["variable_order"],
        )

        assert isinstance(result, TianUCARepresentation)
        assert result.graph is graph
        assert result.treatment == case["treatment"]
        assert result.outcome == case["outcome"]
        assert result.variable_order == tuple(case["variable_order"])
        assert _factor_fields(result.factors) == _fixture_factor_fields(
            case["expected_factors"]
        )
        assert all(isinstance(factor, TianUCAFactor) for factor in result.factors)
        assert result.factors[-1].variables == (case["outcome"],)


def test_printed_algorithm_one_off_by_one_and_branch_mutants_are_not_contracts() -> None:
    fixture = _load_fixture()
    reported = fixture["paper_reported"]
    adjudicated = fixture["independent_adjudication"]
    assert reported["algorithm_1_line_8"] == "m <- i"
    assert adjudicated["line_8_correction"] == "m <- i - 1"
    assert adjudicated["literal_printed_status"] == (
        "nontermination_on_front_door"
    )

    results: dict[str, TianUCARepresentation] = {}
    for case in fixture["cases"]:
        graph = _adbg(
            nodes=case["nodes"],
            directed=case["directed"],
            bidirected=case["bidirected"],
        )
        result = tian_adjustment_to_uca(
            graph,
            treatment=case["treatment"],
            outcome=case["outcome"],
            variable_order=case["variable_order"],
        )
        assert isinstance(result, TianUCARepresentation)
        results[case["name"]] = result

    front_door = results["front_door_figure_1a"]
    verma = results["verma_figure_1b"]
    assert len(front_door.factors) - 1 == adjudicated["front_door_corrected_m"]
    assert len(front_door.factors) - 1 != adjudicated[
        "front_door_line8_only_m_after_branch_correction"
    ]
    assert len(verma.factors) - 1 == adjudicated["verma_corrected_m"]
    assert len(verma.factors) - 1 != adjudicated["verma_line8_only_m"]

    branch_mutant = fixture["cases"][0]["paper_branch_mutant"]
    assert branch_mutant == {
        "C1_subset_S_X": False,
        "printed_next_membership": "inside_S_X",
        "literal_effect": (
            "no unassigned inside-S_X vertex remains before Y, so the printed "
            "loop cannot advance"
        ),
        "correct_next_vertex": "Z",
        "correct_next_membership": "outside_S_X",
    }
    assert front_door.factors[0].variables == ("C", "X")
    assert front_door.factors[1].variables == ("Z",)
    assert front_door.factors[1].fixed_conditioning == ("X",)


def test_corrected_maximal_runs_preserve_supplied_order_and_membership() -> None:
    graph = _adbg(
        nodes=("Y", "E", "D", "C", "B", "A", "X"),
        directed=(
            ("A", "X"),
            ("X", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "Y"),
        ),
        bidirected=(("X", "B"), ("B", "D")),
    )
    order = ["A", "X", "B", "C", "D", "E", "Y"]
    result = tian_adjustment_to_uca(
        graph,
        treatment="X",
        outcome="Y",
        variable_order=order,
    )
    order.reverse()

    assert isinstance(result, TianUCARepresentation)
    assert result.variable_order == ("A", "X", "B", "C", "D", "E", "Y")
    assert _factor_fields(result.factors) == (
        (("A", "X", "B"), (), ()),
        (("C",), ("A", "B"), ("X",)),
        (("D",), ("A", "X", "B", "C"), ()),
        (("E",), ("A", "B", "C", "D"), ("X",)),
        (("Y",), ("A", "B", "C", "D", "E"), ("X",)),
    )


def test_failed_tian_criterion_is_an_ordered_typed_outcome() -> None:
    graph = _adbg(
        nodes=("Y", "C1", "Z", "C2", "X"),
        directed=(("X", "C1"), ("X", "C2"), ("C1", "Y"), ("C2", "Y")),
        bidirected=(("X", "C1"), ("X", "Z"), ("Z", "C2")),
    )
    order = ["X", "C2", "Z", "C1", "Y"]
    result = tian_adjustment_to_uca(
        graph,
        treatment="X",
        outcome="Y",
        variable_order=order,
    )
    order.clear()

    assert isinstance(result, TianAdjustmentNotApplicable)
    assert result.graph is graph
    assert result.treatment == "X"
    assert result.outcome == "Y"
    assert result.variable_order == ("X", "C2", "Z", "C1", "Y")
    assert result.children_in_treatment_district == ("C2", "C1")

    dag = _dag(
        nodes=("X", "C2", "Z", "C1", "Y"),
        directed=(("X", "C1"), ("X", "C2"), ("C1", "Y"), ("C2", "Y")),
    )
    dag_result = tian_adjustment_to_uca(
        dag,
        treatment="X",
        outcome="Y",
        variable_order=("X", "C2", "Z", "C1", "Y"),
    )
    assert isinstance(dag_result, TianUCARepresentation)


@pytest.mark.parametrize(
    "invalid_graph",
    [
        DirectedBidirectedGraph(nodes=("X", "Y"), directed=(("X", "Y"),)),
        EndpointMarkedGraph(nodes=("X", "Y")),
        object(),
        None,
    ],
)
def test_tian_to_uca_rejects_raw_and_wrong_graph_views(invalid_graph: object) -> None:
    with pytest.raises(
        TianAdjustmentDomainError,
        match=f"^{re.escape(DOMAIN_MESSAGE)}$",
    ):
        tian_adjustment_to_uca(  # type: ignore[arg-type]
            invalid_graph,
            treatment="X",
            outcome="Y",
            variable_order=("X", "Y"),
        )


def test_validation_order_and_exact_messages_fail_closed() -> None:
    class ExplodingSequence:
        def __len__(self) -> int:
            raise AssertionError("variable order must not be inspected")

        def __getitem__(self, index: int) -> str:
            del index
            raise AssertionError("variable order must not be inspected")

    exploding = ExplodingSequence()
    raw = DirectedBidirectedGraph(nodes=("X", "Z", "Y"))
    with pytest.raises(TianAdjustmentDomainError, match=f"^{DOMAIN_MESSAGE}$"):
        tian_adjustment_to_uca(  # type: ignore[arg-type]
            raw,
            treatment="bad-treatment",
            outcome="bad-outcome",
            variable_order=exploding,  # type: ignore[arg-type]
        )

    graph = _dag(nodes=("X", "Z", "Y"), directed=(("X", "Z"),))
    with pytest.raises(GraphValidationError, match="^unknown vertex: 'bad-treatment'$"):
        tian_adjustment_to_uca(
            graph,
            treatment="bad-treatment",
            outcome="bad-outcome",
            variable_order=exploding,  # type: ignore[arg-type]
        )
    with pytest.raises(GraphValidationError, match="^unknown vertex: 'bad-outcome'$"):
        tian_adjustment_to_uca(
            graph,
            treatment="X",
            outcome="bad-outcome",
            variable_order=exploding,  # type: ignore[arg-type]
        )
    with pytest.raises(TianAdjustmentDomainError, match=f"^{DISTINCT_MESSAGE}$"):
        tian_adjustment_to_uca(
            graph,
            treatment="X",
            outcome="X",
            variable_order=exploding,  # type: ignore[arg-type]
        )

    malformed_orders: tuple[object, ...] = (
        "XZY",
        ("X", "Y"),
        ("X", "Z", "Z"),
        ("X", "Z", "Y", "Y"),
        ("X", "bad-order-label", "Y"),
        None,
        1,
    )
    for malformed in malformed_orders:
        with pytest.raises(
            TianAdjustmentDomainError,
            match=f"^{PERMUTATION_MESSAGE}$",
        ):
            tian_adjustment_to_uca(
                graph,
                treatment="X",
                outcome="Y",
                variable_order=malformed,  # type: ignore[arg-type]
            )

    with pytest.raises(
        TianAdjustmentDomainError,
        match=f"^{TOPOLOGICAL_MESSAGE}$",
    ):
        tian_adjustment_to_uca(
            graph,
            treatment="X",
            outcome="Y",
            variable_order=("Z", "X", "Y"),
        )
    with pytest.raises(
        TianAdjustmentDomainError,
        match=f"^{OUTCOME_LAST_MESSAGE}$",
    ):
        tian_adjustment_to_uca(
            graph,
            treatment="X",
            outcome="Y",
            variable_order=("X", "Y", "Z"),
        )


def test_results_and_factors_are_factory_only_final_immutable_values() -> None:
    graph = _adbg(
        nodes=("X", "A", "B", "Y"),
        directed=(("X", "A"), ("A", "B"), ("B", "Y")),
        bidirected=(("X", "B"), ("A", "Y")),
    )
    result = tian_adjustment_to_uca(
        graph,
        treatment="X",
        outcome="Y",
        variable_order=("X", "A", "B", "Y"),
    )
    assert isinstance(result, TianUCARepresentation)
    same = tian_adjustment_to_uca(
        graph,
        treatment="X",
        outcome="Y",
        variable_order=["X", "A", "B", "Y"],
    )
    assert result == same
    assert hash(result) == hash(same)

    for value in (result, *result.factors):
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError):
            pickle.dumps(value)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.treatment = "changed"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.factors[0].variables = ("changed",)  # type: ignore[misc]

    failure_graph = _adbg(
        nodes=("X", "M", "Y"),
        directed=(("X", "M"), ("M", "Y")),
        bidirected=(("X", "M"),),
    )
    failure = tian_adjustment_to_uca(
        failure_graph,
        treatment="X",
        outcome="Y",
        variable_order=("X", "M", "Y"),
    )
    assert isinstance(failure, TianAdjustmentNotApplicable)
    assert copy.copy(failure) is failure
    assert copy.deepcopy(failure) is failure
    assert not hasattr(failure, "__dict__")
    with pytest.raises(TypeError):
        pickle.dumps(failure)

    with pytest.raises(TypeError, match="factory-created"):
        TianUCAFactor(  # type: ignore[call-arg]
            object(),
            variables=("X",),
            conditioned_on=(),
            fixed_conditioning=(),
        )
    with pytest.raises(TypeError, match="factory-created"):
        TianUCARepresentation(  # type: ignore[call-arg]
            object(),
            graph=graph,
            treatment="X",
            outcome="Y",
            variable_order=("X", "A", "B", "Y"),
            factors=result.factors,
        )
    with pytest.raises(TypeError, match="factory-created"):
        TianAdjustmentNotApplicable(  # type: ignore[call-arg]
            object(),
            graph=failure_graph,
            treatment="X",
            outcome="Y",
            variable_order=("X", "M", "Y"),
            children_in_treatment_district=("M",),
        )

    with pytest.raises(TypeError):

        class BadFactor(TianUCAFactor):
            pass

    with pytest.raises(TypeError):

        class BadRepresentation(TianUCARepresentation):
            pass

    with pytest.raises(TypeError):

        class BadFailure(TianAdjustmentNotApplicable):
            pass


def test_large_bidirected_district_and_block_construction_are_nonrecursive() -> None:
    size = 1_205
    nodes = tuple(f"V{index}" for index in range(size))
    graph = _adbg(
        nodes=nodes,
        bidirected=tuple(
            (nodes[index], nodes[index + 1]) for index in range(size - 1)
        ),
    )
    result = tian_adjustment_to_uca(
        graph,
        treatment=nodes[0],
        outcome=nodes[-1],
        variable_order=nodes,
    )

    assert isinstance(result, TianUCARepresentation)
    assert len(result.factors) == 2
    assert result.factors[0].variables == nodes[:-1]
    assert result.factors[0].conditioned_on == ()
    assert result.factors[0].fixed_conditioning == ()
    assert result.factors[1].variables == (nodes[-1],)
    assert result.factors[1].conditioned_on == nodes[:-1]
    assert result.factors[1].fixed_conditioning == ()


def test_exhaustive_n3_equation4_and_finite_scm_oracle() -> None:
    graphs = all_bow_admgs_n3()
    assert len(graphs) == 200

    structural_rows = 0
    applicable_rows = 0
    nonapplicable_rows = 0
    applicable_numeric_rows = 0
    applicable_mismatches = 0
    nonapplicable_numeric_rows = 0
    nonapplicable_mismatches = 0

    for tiny_graph in graphs:
        public_graph, labels = _labeled_graph(tiny_graph)
        for integer_order in topological_orders_by_permutation(tiny_graph):
            outcome = integer_order[-1]
            for treatment in integer_order[:-1]:
                structural_rows += 1
                variable_order = tuple(labels[node] for node in integer_order)
                result = tian_adjustment_to_uca(
                    public_graph,
                    treatment=labels[treatment],
                    outcome=labels[outcome],
                    variable_order=variable_order,
                )
                criterion_children = criterion_children_by_definition(
                    tiny_graph,
                    treatment=treatment,
                    variable_order=integer_order,
                )

                if criterion_children:
                    nonapplicable_rows += 1
                    assert isinstance(result, TianAdjustmentNotApplicable)
                    assert result.graph is public_graph
                    assert result.treatment == labels[treatment]
                    assert result.outcome == labels[outcome]
                    assert result.variable_order == variable_order
                    assert result.children_in_treatment_district == tuple(
                        labels[node] for node in criterion_children
                    )
                else:
                    applicable_rows += 1
                    assert isinstance(result, TianUCARepresentation)
                    actual_factors = _integer_factors(result.factors)
                    expected_factors = corrected_tian_factors_by_maximal_runs(
                        tiny_graph,
                        treatment=treatment,
                        outcome=outcome,
                        variable_order=integer_order,
                    )
                    assert actual_factors == expected_factors

                for variant in (0, 1):
                    table = exact_binary_observational_distribution(
                        tiny_graph,
                        variant,
                    )
                    for treatment_value in (0, 1):
                        equation4 = equation4_binary_outcome_mean(
                            tiny_graph,
                            table=table,
                            treatment=treatment,
                            outcome=outcome,
                            variable_order=integer_order,
                            treatment_value=treatment_value,
                        )
                        direct = exact_binary_interventional_probability(
                            tiny_graph,
                            outcomes=(outcome,),
                            interventions=(treatment,),
                            values={treatment: treatment_value, outcome: 1},
                            variant=variant,
                        )
                        if criterion_children:
                            nonapplicable_numeric_rows += 1
                            nonapplicable_mismatches += equation4 != direct
                        else:
                            assert isinstance(result, TianUCARepresentation)
                            grouped = grouped_uca_binary_outcome_mean(
                                tiny_graph,
                                table=table,
                                outcome=outcome,
                                variable_order=integer_order,
                                treatment_value=treatment_value,
                                factors=_integer_factors(result.factors),
                            )
                            applicable_numeric_rows += 1
                            applicable_mismatches += equation4 != direct
                            assert grouped == equation4 == direct

    assert structural_rows == 768
    assert applicable_rows == 456
    assert nonapplicable_rows == 312
    assert applicable_numeric_rows == 1_824
    assert applicable_mismatches == 0
    assert nonapplicable_numeric_rows == 1_248
    assert nonapplicable_mismatches == 1_104
    assert nonapplicable_numeric_rows - nonapplicable_mismatches == 144
