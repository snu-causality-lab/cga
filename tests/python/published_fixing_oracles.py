"""Literal integer-graph oracles; no production traversal or fixing imports."""
from __future__ import annotations

import itertools

from .id_oracles import TinyADMG
from .gid_oracles import generalized_identification_oracle
from .oracles import m_separated_by_canonical_dag


def conditional_generalized_oracle(graph, *, outcomes, interventions, conditioned_on, sources):
    feasible = []
    conditions = sorted(conditioned_on)
    for mask in range(1 << len(conditions)):
        moved = frozenset(v for i,v in enumerate(conditions) if mask & (1 << i))
        edges = tuple((a,False,b,True) for a,b in graph.directed if b not in interventions and a not in moved)
        edges += tuple((a,True,b,True) for a,b in graph.bidirected if a not in interventions and b not in interventions)
        if m_separated_by_canonical_dag(graph.nodes, edges, outcomes, moved,
            interventions | (conditioned_on-moved)):
            feasible.append(moved)
    maximal = frozenset().union(*feasible)
    assert maximal in feasible
    result = generalized_identification_oracle(graph, outcomes=outcomes | (conditioned_on-maximal),
        interventions=interventions | maximal, experiment_interventions=sources)
    return result.identified, maximal


def literal_reachable_states(graph: TinyADMG):
    states = {frozenset(graph.nodes)}
    districts = set()
    for length in range(len(graph.nodes)+1):
        for sequence in itertools.permutations(graph.nodes, length):
            active = set(graph.nodes)
            valid = True
            for vertex in sequence:
                descendants = {vertex}
                for _ in graph.nodes:
                    descendants |= {b for a,b in graph.directed if a in descendants and a in active and b in active}
                district = {vertex}
                for _ in graph.nodes:
                    district |= {b for a,b in graph.bidirected if a in district and b in active}
                    district |= {a for a,b in graph.bidirected if b in district and a in active}
                if descendants & district != {vertex}:
                    valid = False
                    break
                active.remove(vertex)
            if not valid:
                continue
            states.add(frozenset(active))
            unseen = set(active)
            while unseen:
                district = {min(unseen)}
                for _ in graph.nodes:
                    district |= {b for a,b in graph.bidirected if a in district and b in active}
                    district |= {a for a,b in graph.bidirected if b in district and a in active}
                districts.add(frozenset(district))
                unseen -= district
    return states, districts
