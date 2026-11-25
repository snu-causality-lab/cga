#include "cga/backend.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define CGA_BACKEND_MAX_TOPOLOGY_VERTICES ((size_t)16u)
#define CGA_MAX_EDGE_ATOMS ((size_t)360u)

#ifndef CGA_CALLOC
#define CGA_CALLOC calloc
#endif

/* These representations are deliberately distinct and private. */
struct cga_pmg16 {
    size_t vertex_count;
    uint32_t endpoint_marks[CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
};

struct cga_directed_bidirected16 {
    size_t vertex_count;
    uint16_t directed_out[CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
    uint16_t bidirected[CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
};

_Static_assert(
    sizeof(uint16_t) * CHAR_BIT >= CGA_BACKEND_MAX_TOPOLOGY_VERTICES,
    "directed-bidirected bitsets must cover every backend vertex");
_Static_assert(
    sizeof(uint32_t) * CHAR_BIT >= 2u * CGA_BACKEND_MAX_TOPOLOGY_VERTICES,
    "endpoint-mark rows must hold two bits for every backend vertex");

typedef struct {
    size_t first;
    size_t second;
    bool arrow_first;
    bool arrow_second;
} cga_edge_atom;

typedef struct {
    size_t vertex;
    bool incoming_arrow;
} cga_walk_state;

typedef bool cga_pair_set
    [CGA_BACKEND_MAX_TOPOLOGY_VERTICES]
    [CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
typedef bool cga_triple_set
    [CGA_BACKEND_MAX_TOPOLOGY_VERTICES]
    [CGA_BACKEND_MAX_TOPOLOGY_VERTICES]
    [CGA_BACKEND_MAX_TOPOLOGY_VERTICES];

static uint16_t cga_vertex_bit(size_t vertex)
{
    return (uint16_t)(UINT32_C(1) << vertex);
}

static uint16_t cga_valid_vertex_mask(size_t vertex_count)
{
    if (vertex_count == CGA_BACKEND_MAX_TOPOLOGY_VERTICES) {
        return UINT16_MAX;
    }
    return (uint16_t)((UINT32_C(1) << vertex_count) - UINT32_C(1));
}

static bool cga_valid_distinct_pair(
    size_t vertex_count, size_t first, size_t second)
{
    return first < vertex_count && second < vertex_count && first != second;
}

static bool cga_retained_mask_is_valid(
    size_t vertex_count, uint16_t retained)
{
    const uint16_t valid_mask = cga_valid_vertex_mask(vertex_count);

    return (retained & (uint16_t)~valid_mask) == UINT16_C(0);
}

static size_t cga_compaction_map(
    size_t vertex_count,
    uint16_t retained,
    size_t old_to_new[CGA_BACKEND_MAX_TOPOLOGY_VERTICES])
{
    size_t old_vertex;
    size_t retained_count = 0u;

    for (old_vertex = 0u; old_vertex < vertex_count; ++old_vertex) {
        if ((retained & cga_vertex_bit(old_vertex)) != UINT16_C(0)) {
            old_to_new[old_vertex] = retained_count;
            ++retained_count;
        } else {
            old_to_new[old_vertex] = SIZE_MAX;
        }
    }
    return retained_count;
}

static bool cga_endpoint_is_edge_mark(cga_endpoint_mark mark)
{
    return mark == CGA_ENDPOINT_TAIL || mark == CGA_ENDPOINT_ARROW ||
        mark == CGA_ENDPOINT_CIRCLE;
}

static cga_endpoint_mark cga_pmg_mark(
    const cga_pmg16 *graph, size_t endpoint, size_t neighbor)
{
    const size_t shift = (size_t)2u * neighbor;
    const uint32_t value =
        (graph->endpoint_marks[endpoint] >> shift) & UINT32_C(3);

    return (cga_endpoint_mark)value;
}

static void cga_put_pmg_mark(
    uint32_t *row, size_t neighbor, cga_endpoint_mark mark)
{
    const size_t shift = (size_t)2u * neighbor;
    const uint32_t mask = UINT32_C(3) << shift;
    const uint32_t value = (uint32_t)mark << shift;

    *row = (*row & ~mask) | value;
}

static void cga_put_bidirected(
    uint16_t rows[CGA_BACKEND_MAX_TOPOLOGY_VERTICES],
    size_t first,
    size_t second)
{
    rows[first] |= cga_vertex_bit(second);
    rows[second] |= cga_vertex_bit(first);
}

static bool cga_pmg_storage_is_well_formed(const cga_pmg16 *graph)
{
    size_t first;
    size_t second;

    for (first = 0u; first < graph->vertex_count; ++first) {
        if (cga_pmg_mark(graph, first, first) != CGA_ENDPOINT_NONE) {
            return false;
        }
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            const cga_endpoint_mark first_mark =
                cga_pmg_mark(graph, first, second);
            const cga_endpoint_mark second_mark =
                cga_pmg_mark(graph, second, first);
            const bool first_absent = first_mark == CGA_ENDPOINT_NONE;
            const bool second_absent = second_mark == CGA_ENDPOINT_NONE;

            if (first_absent != second_absent) {
                return false;
            }
            if (!first_absent &&
                (!cga_endpoint_is_edge_mark(first_mark) ||
                 !cga_endpoint_is_edge_mark(second_mark))) {
                return false;
            }
        }
    }
    return true;
}

static void cga_directed_transitive_closure(
    uint16_t rows[CGA_BACKEND_MAX_TOPOLOGY_VERTICES],
    size_t vertex_count)
{
    size_t intermediate;
    size_t source;

    for (intermediate = 0u; intermediate < vertex_count; ++intermediate) {
        const uint16_t intermediate_bit = cga_vertex_bit(intermediate);

        for (source = 0u; source < vertex_count; ++source) {
            if ((rows[source] & intermediate_bit) != UINT16_C(0)) {
                rows[source] |= rows[intermediate];
            }
        }
    }
}

static bool cga_directed_rows_are_acyclic(
    const uint16_t directed[CGA_BACKEND_MAX_TOPOLOGY_VERTICES],
    size_t vertex_count)
{
    uint16_t reach[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {0};
    const uint16_t valid_mask = cga_valid_vertex_mask(vertex_count);
    size_t vertex;

    for (vertex = 0u; vertex < vertex_count; ++vertex) {
        if ((directed[vertex] & (uint16_t)~valid_mask) != UINT16_C(0) ||
            (directed[vertex] & cga_vertex_bit(vertex)) != UINT16_C(0)) {
            return false;
        }
        reach[vertex] = directed[vertex];
    }
    cga_directed_transitive_closure(reach, vertex_count);
    for (vertex = 0u; vertex < vertex_count; ++vertex) {
        if ((reach[vertex] & cga_vertex_bit(vertex)) != UINT16_C(0)) {
            return false;
        }
    }
    return true;
}

static void cga_pmg_directed_rows(
    const cga_pmg16 *graph,
    uint16_t directed[CGA_BACKEND_MAX_TOPOLOGY_VERTICES])
{
    size_t first;
    size_t second;

    for (first = 0u; first < graph->vertex_count; ++first) {
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            const cga_endpoint_mark first_mark =
                cga_pmg_mark(graph, first, second);
            const cga_endpoint_mark second_mark =
                cga_pmg_mark(graph, second, first);

            if (first_mark == CGA_ENDPOINT_TAIL &&
                second_mark == CGA_ENDPOINT_ARROW) {
                directed[first] |= cga_vertex_bit(second);
            } else if (first_mark == CGA_ENDPOINT_ARROW &&
                       second_mark == CGA_ENDPOINT_TAIL) {
                directed[second] |= cga_vertex_bit(first);
            }
        }
    }
}

static bool cga_pmg_ancestral_value(const cga_pmg16 *graph)
{
    uint16_t directed[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {0};
    uint16_t reach[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {0};
    uint16_t undirected_endpoints = UINT16_C(0);
    uint16_t arrowhead_endpoints = UINT16_C(0);
    size_t first;
    size_t second;

    for (first = 0u; first < graph->vertex_count; ++first) {
        if (cga_pmg_mark(graph, first, first) != CGA_ENDPOINT_NONE) {
            return false;
        }
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            const cga_endpoint_mark first_mark =
                cga_pmg_mark(graph, first, second);
            const cga_endpoint_mark second_mark =
                cga_pmg_mark(graph, second, first);
            const bool first_absent = first_mark == CGA_ENDPOINT_NONE;
            const bool second_absent = second_mark == CGA_ENDPOINT_NONE;

            if (first_absent != second_absent) {
                return false;
            }
            if (first_absent) {
                continue;
            }
            if (!cga_endpoint_is_edge_mark(first_mark) ||
                !cga_endpoint_is_edge_mark(second_mark) ||
                first_mark == CGA_ENDPOINT_CIRCLE ||
                second_mark == CGA_ENDPOINT_CIRCLE) {
                return false;
            }
            if (first_mark == CGA_ENDPOINT_TAIL &&
                second_mark == CGA_ENDPOINT_TAIL) {
                undirected_endpoints |=
                    (uint16_t)(cga_vertex_bit(first) | cga_vertex_bit(second));
            }
            if (first_mark == CGA_ENDPOINT_ARROW) {
                arrowhead_endpoints |= cga_vertex_bit(first);
            }
            if (second_mark == CGA_ENDPOINT_ARROW) {
                arrowhead_endpoints |= cga_vertex_bit(second);
            }
        }
    }
    if ((undirected_endpoints & arrowhead_endpoints) != UINT16_C(0)) {
        return false;
    }

    cga_pmg_directed_rows(graph, directed);
    if (!cga_directed_rows_are_acyclic(directed, graph->vertex_count)) {
        return false;
    }
    for (first = 0u; first < graph->vertex_count; ++first) {
        reach[first] = directed[first];
    }
    cga_directed_transitive_closure(reach, graph->vertex_count);
    for (first = 0u; first < graph->vertex_count; ++first) {
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            if (cga_pmg_mark(graph, first, second) == CGA_ENDPOINT_ARROW &&
                cga_pmg_mark(graph, second, first) == CGA_ENDPOINT_ARROW &&
                (((reach[first] & cga_vertex_bit(second)) != UINT16_C(0)) ||
                 ((reach[second] & cga_vertex_bit(first)) != UINT16_C(0)))) {
                return false;
            }
        }
    }
    return true;
}

static bool cga_pmg_has_bidirected_path(
    const cga_pmg16 *graph,
    size_t source,
    size_t target,
    uint16_t allowed)
{
    size_t pending[CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
    bool seen[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {false};
    size_t next = 0u;
    size_t pending_count = 1u;

    pending[0] = source;
    seen[source] = true;
    while (next < pending_count) {
        const size_t current = pending[next++];
        size_t neighbor;

        for (neighbor = 0u; neighbor < graph->vertex_count; ++neighbor) {
            if ((allowed & cga_vertex_bit(neighbor)) == UINT16_C(0) ||
                seen[neighbor] || neighbor == current) {
                continue;
            }
            if (cga_pmg_mark(graph, current, neighbor) !=
                    CGA_ENDPOINT_ARROW ||
                cga_pmg_mark(graph, neighbor, current) !=
                    CGA_ENDPOINT_ARROW) {
                continue;
            }
            if (neighbor == target) {
                return true;
            }
            seen[neighbor] = true;
            pending[pending_count++] = neighbor;
        }
    }
    return false;
}

static bool cga_pmg_maximal_ancestral_value(const cga_pmg16 *graph)
{
    uint16_t directed[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {0};
    uint16_t reach[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {0};
    size_t first;
    size_t second;

    if (!cga_pmg_ancestral_value(graph)) {
        return false;
    }
    cga_pmg_directed_rows(graph, directed);
    for (first = 0u; first < graph->vertex_count; ++first) {
        reach[first] = directed[first];
    }
    cga_directed_transitive_closure(reach, graph->vertex_count);

    for (first = 0u; first < graph->vertex_count; ++first) {
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            uint16_t allowed;
            size_t vertex;

            if (cga_pmg_mark(graph, first, second) != CGA_ENDPOINT_NONE) {
                continue;
            }
            allowed = (uint16_t)(
                cga_vertex_bit(first) | cga_vertex_bit(second));
            for (vertex = 0u; vertex < graph->vertex_count; ++vertex) {
                if ((reach[vertex] &
                     (uint16_t)(cga_vertex_bit(first) |
                                cga_vertex_bit(second))) != UINT16_C(0)) {
                    allowed |= cga_vertex_bit(vertex);
                }
            }
            if (cga_pmg_has_bidirected_path(
                    graph, first, second, allowed)) {
                return false;
            }
        }
    }
    return true;
}

static bool cga_pmg_vertices_are_adjacent(
    const cga_pmg16 *graph, size_t first, size_t second)
{
    return cga_pmg_mark(graph, first, second) != CGA_ENDPOINT_NONE;
}

static bool cga_pmg_vertex_is_parent(
    const cga_pmg16 *graph, size_t parent, size_t child)
{
    return cga_pmg_mark(graph, parent, child) == CGA_ENDPOINT_TAIL &&
        cga_pmg_mark(graph, child, parent) == CGA_ENDPOINT_ARROW;
}

static bool cga_pmg_vertices_are_spouses(
    const cga_pmg16 *graph, size_t first, size_t second)
{
    return cga_pmg_mark(graph, first, second) == CGA_ENDPOINT_ARROW &&
        cga_pmg_mark(graph, second, first) == CGA_ENDPOINT_ARROW;
}

static bool cga_pmg_triple_is_collider(
    const cga_pmg16 *graph, size_t first, size_t middle, size_t third)
{
    return first != middle && middle != third && first != third &&
        cga_pmg_mark(graph, middle, first) == CGA_ENDPOINT_ARROW &&
        cga_pmg_mark(graph, middle, third) == CGA_ENDPOINT_ARROW;
}

/*
 * The multidimensional array inputs below are logically read-only.  Keep
 * their element types unqualified because qualification-changing conversions
 * between pointers to arrays are not portable before C23.
 */
static void cga_pmg_reachable_pair_vertices(
    size_t vertex_count,
    cga_pair_set pair_vertices,
    cga_triple_set triples,
    size_t source_first,
    size_t source_second,
    cga_pair_set reached)
{
    size_t pending_first
        [CGA_BACKEND_MAX_TOPOLOGY_VERTICES *
         CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
    size_t pending_second
        [CGA_BACKEND_MAX_TOPOLOGY_VERTICES *
         CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
    size_t next = 0u;
    size_t pending_count = 1u;

    memset(reached, 0, sizeof(cga_pair_set));
    reached[source_first][source_second] = true;
    pending_first[0] = source_first;
    pending_second[0] = source_second;

    while (next < pending_count) {
        const size_t first = pending_first[next];
        const size_t middle = pending_second[next];
        size_t third;

        ++next;
        for (third = 0u; third < vertex_count; ++third) {
            if (!pair_vertices[middle][third] ||
                reached[middle][third] ||
                !triples[first][middle][third]) {
                continue;
            }
            reached[middle][third] = true;
            pending_first[pending_count] = middle;
            pending_second[pending_count] = third;
            ++pending_count;
        }
    }
}

static bool cga_pmg_candidate_has_discriminating_endpoint(
    const cga_pmg16 *graph,
    cga_triple_set triples,
    size_t first,
    size_t middle,
    size_t third)
{
    cga_pair_set pair_vertices = {{false}};
    cga_pair_set reachable = {{false}};
    size_t left_parent;
    size_t right_parent;
    size_t before;
    size_t at;
    size_t endpoint;

    pair_vertices[middle][first] = true;
    for (left_parent = 0u;
         left_parent < graph->vertex_count;
         ++left_parent) {
        if (!cga_pmg_vertex_is_parent(graph, left_parent, third)) {
            continue;
        }
        for (right_parent = 0u;
             right_parent < graph->vertex_count;
             ++right_parent) {
            if (cga_pmg_vertex_is_parent(graph, right_parent, third) &&
                cga_pmg_vertices_are_spouses(
                    graph, left_parent, right_parent)) {
                pair_vertices[left_parent][right_parent] = true;
            }
        }
    }

    cga_pmg_reachable_pair_vertices(
        graph->vertex_count, pair_vertices, triples,
        middle, first, reachable);
    for (before = 0u; before < graph->vertex_count; ++before) {
        for (at = 0u; at < graph->vertex_count; ++at) {
            if (!reachable[before][at]) {
                continue;
            }
            for (endpoint = 0u;
                 endpoint < graph->vertex_count;
                 ++endpoint) {
                if (triples[before][at][endpoint] &&
                    !cga_pmg_vertices_are_adjacent(
                        graph, endpoint, third)) {
                    return true;
                }
            }
        }
    }
    return false;
}

static void cga_pmg_equivalence_triples(
    const cga_pmg16 *graph, cga_triple_set triples)
{
    bool changed;
    size_t first;
    size_t middle;
    size_t third;

    memset(triples, 0, sizeof(cga_triple_set));
    for (first = 0u; first < graph->vertex_count; ++first) {
        for (middle = 0u; middle < graph->vertex_count; ++middle) {
            for (third = 0u; third < graph->vertex_count; ++third) {
                if (cga_pmg_triple_is_collider(
                        graph, first, middle, third) &&
                    !cga_pmg_vertices_are_adjacent(
                        graph, first, third)) {
                    triples[first][middle][third] = true;
                }
            }
        }
    }

    do {
        cga_triple_set additions = {{{false}}};

        changed = false;
        for (first = 0u; first < graph->vertex_count; ++first) {
            for (middle = 0u; middle < graph->vertex_count; ++middle) {
                for (third = 0u; third < graph->vertex_count; ++third) {
                    if (triples[first][middle][third] ||
                        !cga_pmg_triple_is_collider(
                            graph, first, middle, third) ||
                        !cga_pmg_vertices_are_spouses(
                            graph, first, middle) ||
                        !cga_pmg_vertex_is_parent(
                            graph, first, third)) {
                        continue;
                    }
                    if (cga_pmg_candidate_has_discriminating_endpoint(
                            graph, triples, first, middle, third)) {
                        additions[first][middle][third] = true;
                        additions[third][middle][first] = true;
                    }
                }
            }
        }
        for (first = 0u; first < graph->vertex_count; ++first) {
            for (middle = 0u; middle < graph->vertex_count; ++middle) {
                for (third = 0u; third < graph->vertex_count; ++third) {
                    if (additions[first][middle][third] &&
                        !triples[first][middle][third]) {
                        triples[first][middle][third] = true;
                        changed = true;
                    }
                }
            }
        }
    } while (changed);
}

static bool cga_pmg_markov_equivalence_value(
    const cga_pmg16 *left, const cga_pmg16 *right)
{
    cga_triple_set left_triples;
    cga_triple_set right_triples;
    size_t first;
    size_t middle;
    size_t third;

    for (first = 0u; first < left->vertex_count; ++first) {
        for (third = first + 1u;
             third < left->vertex_count;
             ++third) {
            if (cga_pmg_vertices_are_adjacent(left, first, third) !=
                cga_pmg_vertices_are_adjacent(right, first, third)) {
                return false;
            }
        }
    }

    cga_pmg_equivalence_triples(left, left_triples);
    cga_pmg_equivalence_triples(right, right_triples);
    for (first = 0u; first < left->vertex_count; ++first) {
        for (middle = 0u; middle < left->vertex_count; ++middle) {
            for (third = 0u; third < left->vertex_count; ++third) {
                if ((left_triples[first][middle][third] &&
                     !cga_pmg_triple_is_collider(
                         right, first, middle, third)) ||
                    (right_triples[first][middle][third] &&
                     !cga_pmg_triple_is_collider(
                         left, first, middle, third))) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool cga_db_storage_is_well_formed(
    const cga_directed_bidirected16 *graph)
{
    const uint16_t valid_mask = cga_valid_vertex_mask(graph->vertex_count);
    size_t first;
    size_t second;

    for (first = 0u; first < graph->vertex_count; ++first) {
        if ((graph->directed_out[first] & (uint16_t)~valid_mask) !=
                UINT16_C(0) ||
            (graph->directed_out[first] & cga_vertex_bit(first)) !=
                UINT16_C(0) ||
            (graph->bidirected[first] & (uint16_t)~valid_mask) !=
                UINT16_C(0) ||
            (graph->bidirected[first] & cga_vertex_bit(first)) !=
                UINT16_C(0)) {
            return false;
        }
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            const bool forward =
                (graph->bidirected[first] & cga_vertex_bit(second)) !=
                UINT16_C(0);
            const bool reverse =
                (graph->bidirected[second] & cga_vertex_bit(first)) !=
                UINT16_C(0);

            if (forward != reverse) {
                return false;
            }
        }
    }
    return true;
}

static bool cga_db_is_acyclic(const cga_directed_bidirected16 *graph)
{
    if (!cga_db_storage_is_well_formed(graph)) {
        return false;
    }
    return cga_directed_rows_are_acyclic(
        graph->directed_out, graph->vertex_count);
}

static size_t cga_db_edge_atoms(
    const cga_directed_bidirected16 *graph,
    cga_edge_atom atoms[CGA_MAX_EDGE_ATOMS])
{
    size_t count = 0u;
    size_t first;
    size_t second;

    for (first = 0u; first < graph->vertex_count; ++first) {
        for (second = 0u; second < graph->vertex_count; ++second) {
            if ((graph->directed_out[first] & cga_vertex_bit(second)) !=
                UINT16_C(0)) {
                atoms[count].first = first;
                atoms[count].second = second;
                atoms[count].arrow_first = false;
                atoms[count].arrow_second = true;
                ++count;
            }
        }
    }
    for (first = 0u; first < graph->vertex_count; ++first) {
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            if ((graph->bidirected[first] & cga_vertex_bit(second)) !=
                UINT16_C(0)) {
                atoms[count].first = first;
                atoms[count].second = second;
                atoms[count].arrow_first = true;
                atoms[count].arrow_second = true;
                ++count;
            }
        }
    }
    return count;
}

static size_t cga_pmg_edge_atoms(
    const cga_pmg16 *graph,
    cga_edge_atom atoms[CGA_MAX_EDGE_ATOMS])
{
    size_t count = 0u;
    size_t first;
    size_t second;

    for (first = 0u; first < graph->vertex_count; ++first) {
        for (second = first + 1u; second < graph->vertex_count; ++second) {
            const cga_endpoint_mark first_mark =
                cga_pmg_mark(graph, first, second);
            const cga_endpoint_mark second_mark =
                cga_pmg_mark(graph, second, first);

            if (first_mark != CGA_ENDPOINT_NONE) {
                atoms[count].first = first;
                atoms[count].second = second;
                atoms[count].arrow_first = first_mark == CGA_ENDPOINT_ARROW;
                atoms[count].arrow_second = second_mark == CGA_ENDPOINT_ARROW;
                ++count;
            }
        }
    }
    return count;
}

static bool cga_query_masks_are_valid(
    size_t vertex_count, uint16_t left, uint16_t right, uint16_t given)
{
    const uint16_t valid_mask = cga_valid_vertex_mask(vertex_count);
    const uint16_t combined = (uint16_t)(left | right | given);

    return left != UINT16_C(0) && right != UINT16_C(0) &&
        (combined & (uint16_t)~valid_mask) == UINT16_C(0) &&
        (left & right) == UINT16_C(0) &&
        (left & given) == UINT16_C(0) &&
        (right & given) == UINT16_C(0);
}

static bool cga_m_separated_atoms(
    const cga_edge_atom atoms[CGA_MAX_EDGE_ATOMS],
    size_t atom_count,
    uint16_t left,
    uint16_t right,
    uint16_t given)
{
    cga_walk_state pending[(size_t)2u * CGA_BACKEND_MAX_TOPOLOGY_VERTICES];
    bool seen[CGA_BACKEND_MAX_TOPOLOGY_VERTICES][2] = {{false}};
    uint16_t ancestors = given;
    size_t pending_count = 0u;
    size_t atom_index;
    bool changed;

    do {
        changed = false;
        for (atom_index = 0u; atom_index < atom_count; ++atom_index) {
            const cga_edge_atom atom = atoms[atom_index];
            const uint16_t first_bit = cga_vertex_bit(atom.first);
            const uint16_t second_bit = cga_vertex_bit(atom.second);

            if (!atom.arrow_first && atom.arrow_second &&
                (ancestors & second_bit) != UINT16_C(0) &&
                (ancestors & first_bit) == UINT16_C(0)) {
                ancestors |= first_bit;
                changed = true;
            } else if (atom.arrow_first && !atom.arrow_second &&
                       (ancestors & first_bit) != UINT16_C(0) &&
                       (ancestors & second_bit) == UINT16_C(0)) {
                ancestors |= second_bit;
                changed = true;
            }
        }
    } while (changed);

    for (atom_index = 0u; atom_index < atom_count; ++atom_index) {
        const cga_edge_atom atom = atoms[atom_index];
        size_t neighbor;
        bool incoming_arrow;
        size_t mark_index;

        if ((left & cga_vertex_bit(atom.first)) != UINT16_C(0)) {
            neighbor = atom.second;
            incoming_arrow = atom.arrow_second;
            if ((right & cga_vertex_bit(neighbor)) != UINT16_C(0)) {
                return false;
            }
            mark_index = incoming_arrow ? 1u : 0u;
            if (!seen[neighbor][mark_index]) {
                seen[neighbor][mark_index] = true;
                pending[pending_count].vertex = neighbor;
                pending[pending_count].incoming_arrow = incoming_arrow;
                ++pending_count;
            }
        }
        if ((left & cga_vertex_bit(atom.second)) != UINT16_C(0)) {
            neighbor = atom.first;
            incoming_arrow = atom.arrow_first;
            if ((right & cga_vertex_bit(neighbor)) != UINT16_C(0)) {
                return false;
            }
            mark_index = incoming_arrow ? 1u : 0u;
            if (!seen[neighbor][mark_index]) {
                seen[neighbor][mark_index] = true;
                pending[pending_count].vertex = neighbor;
                pending[pending_count].incoming_arrow = incoming_arrow;
                ++pending_count;
            }
        }
    }

    while (pending_count > 0u) {
        const cga_walk_state state = pending[--pending_count];

        for (atom_index = 0u; atom_index < atom_count; ++atom_index) {
            const cga_edge_atom atom = atoms[atom_index];
            size_t neighbor;
            bool arrow_current;
            bool arrow_neighbor;
            bool collider;
            size_t mark_index;

            if (atom.first == state.vertex) {
                neighbor = atom.second;
                arrow_current = atom.arrow_first;
                arrow_neighbor = atom.arrow_second;
            } else if (atom.second == state.vertex) {
                neighbor = atom.first;
                arrow_current = atom.arrow_second;
                arrow_neighbor = atom.arrow_first;
            } else {
                continue;
            }

            collider = state.incoming_arrow && arrow_current;
            if ((collider &&
                 (ancestors & cga_vertex_bit(state.vertex)) == UINT16_C(0)) ||
                (!collider &&
                 (given & cga_vertex_bit(state.vertex)) != UINT16_C(0))) {
                continue;
            }
            if ((right & cga_vertex_bit(neighbor)) != UINT16_C(0)) {
                return false;
            }
            mark_index = arrow_neighbor ? 1u : 0u;
            if (!seen[neighbor][mark_index]) {
                seen[neighbor][mark_index] = true;
                pending[pending_count].vertex = neighbor;
                pending[pending_count].incoming_arrow = arrow_neighbor;
                ++pending_count;
            }
        }
    }
    return true;
}

cga_status cga_backend_get_capabilities(
    cga_backend_family family,
    cga_backend_capabilities *out_capabilities)
{
    cga_backend_capabilities capabilities = {0};

    if (out_capabilities == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    capabilities.abi_version = CGA_ABI_VERSION;
    capabilities.max_topology_vertices = CGA_BACKEND_MAX_TOPOLOGY_VERTICES;
    switch (family) {
    case CGA_BACKEND_PMG16:
        capabilities.endpoint_marked = true;
        capabilities.supports_bows = false;
        break;
    case CGA_BACKEND_DIRECTED_BIDIRECTED16:
        capabilities.endpoint_marked = false;
        capabilities.supports_bows = true;
        break;
    default:
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    *out_capabilities = capabilities;
    return CGA_STATUS_OK;
}

cga_status cga_backend_query_operation(
    cga_backend_family family,
    cga_backend_operation operation,
    bool *out_supported,
    size_t *out_max_vertices)
{
    bool supported;
    size_t max_vertices;

    if (out_supported == NULL || out_max_vertices == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (family != CGA_BACKEND_PMG16 &&
        family != CGA_BACKEND_DIRECTED_BIDIRECTED16) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    switch (operation) {
    case CGA_OPERATION_M_SEPARATION:
        supported = true;
        max_vertices = CGA_BACKEND_MAX_TOPOLOGY_VERTICES;
        break;
    case CGA_OPERATION_ANCESTRAL_CLASSIFICATION:
        supported = family == CGA_BACKEND_PMG16;
        max_vertices = supported ? CGA_BACKEND_MAX_TOPOLOGY_VERTICES : 0u;
        break;
    case CGA_OPERATION_INDUCED_SUBGRAPH:
        supported = true;
        max_vertices = CGA_BACKEND_MAX_TOPOLOGY_VERTICES;
        break;
    case CGA_OPERATION_LATENT_PROJECTION:
    case CGA_OPERATION_DAG_ADJUSTMENT:
        supported = family == CGA_BACKEND_DIRECTED_BIDIRECTED16;
        max_vertices = supported ? CGA_BACKEND_MAX_TOPOLOGY_VERTICES : 0u;
        break;
    case CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION:
        supported = family == CGA_BACKEND_PMG16;
        max_vertices = supported ? CGA_BACKEND_MAX_TOPOLOGY_VERTICES : 0u;
        break;
    case CGA_OPERATION_MAG_MARKOV_EQUIVALENCE:
    case CGA_OPERATION_REPRESENTATIVE_LEG:
    case CGA_OPERATION_LEG_ENUMERATION:
        supported = family == CGA_BACKEND_PMG16;
        max_vertices = supported ? CGA_BACKEND_MAX_TOPOLOGY_VERTICES : 0u;
        break;
    default:
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    *out_supported = supported;
    *out_max_vertices = max_vertices;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_create(size_t vertex_count, cga_pmg16 **out_graph)
{
    cga_pmg16 *graph;

    if (out_graph == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (vertex_count > CGA_BACKEND_MAX_TOPOLOGY_VERTICES) {
        return CGA_STATUS_UNSUPPORTED_SCOPE;
    }

    graph = CGA_CALLOC((size_t)1u, sizeof(*graph));
    if (graph == NULL) {
        return CGA_STATUS_ALLOCATION_FAILED;
    }
    graph->vertex_count = vertex_count;
    *out_graph = graph;
    return CGA_STATUS_OK;
}

void cga_pmg16_destroy(cga_pmg16 *graph)
{
    free(graph);
}

cga_status cga_pmg16_vertex_count(
    const cga_pmg16 *graph,
    size_t *out_vertex_count)
{
    if (graph == NULL || out_vertex_count == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    *out_vertex_count = graph->vertex_count;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_set_edge(
    cga_pmg16 *graph,
    size_t first,
    cga_endpoint_mark first_mark,
    size_t second,
    cga_endpoint_mark second_mark)
{
    uint32_t first_row;
    uint32_t second_row;

    if (graph == NULL ||
        !cga_valid_distinct_pair(graph->vertex_count, first, second) ||
        !cga_endpoint_is_edge_mark(first_mark) ||
        !cga_endpoint_is_edge_mark(second_mark)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    first_row = graph->endpoint_marks[first];
    second_row = graph->endpoint_marks[second];
    cga_put_pmg_mark(&first_row, second, first_mark);
    cga_put_pmg_mark(&second_row, first, second_mark);
    graph->endpoint_marks[first] = first_row;
    graph->endpoint_marks[second] = second_row;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_remove_edge(
    cga_pmg16 *graph,
    size_t first,
    size_t second)
{
    uint32_t first_row;
    uint32_t second_row;

    if (graph == NULL ||
        !cga_valid_distinct_pair(graph->vertex_count, first, second)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    first_row = graph->endpoint_marks[first];
    second_row = graph->endpoint_marks[second];
    cga_put_pmg_mark(&first_row, second, CGA_ENDPOINT_NONE);
    cga_put_pmg_mark(&second_row, first, CGA_ENDPOINT_NONE);
    graph->endpoint_marks[first] = first_row;
    graph->endpoint_marks[second] = second_row;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_get_edge(
    const cga_pmg16 *graph,
    size_t first,
    size_t second,
    cga_endpoint_mark *out_first_mark,
    cga_endpoint_mark *out_second_mark)
{
    cga_endpoint_mark first_mark;
    cga_endpoint_mark second_mark;

    if (graph == NULL || out_first_mark == NULL || out_second_mark == NULL ||
        !cga_valid_distinct_pair(graph->vertex_count, first, second)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    first_mark = cga_pmg_mark(graph, first, second);
    second_mark = cga_pmg_mark(graph, second, first);
    if ((first_mark == CGA_ENDPOINT_NONE) !=
        (second_mark == CGA_ENDPOINT_NONE)) {
        return CGA_STATUS_INVALID_GRAPH;
    }
    *out_first_mark = first_mark;
    *out_second_mark = second_mark;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_is_ancestral(
    const cga_pmg16 *graph,
    bool *out_ancestral)
{
    bool ancestral;

    if (graph == NULL || out_ancestral == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    ancestral = cga_pmg_ancestral_value(graph);
    *out_ancestral = ancestral;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_is_maximal_ancestral(
    const cga_pmg16 *graph,
    bool *out_maximal_ancestral)
{
    bool maximal_ancestral;

    if (graph == NULL || out_maximal_ancestral == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    maximal_ancestral = cga_pmg_maximal_ancestral_value(graph);
    *out_maximal_ancestral = maximal_ancestral;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_are_markov_equivalent(
    const cga_pmg16 *left,
    const cga_pmg16 *right,
    bool *out_equivalent)
{
    bool equivalent;

    if (left == NULL || right == NULL || out_equivalent == NULL ||
        left->vertex_count != right->vertex_count) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_pmg_maximal_ancestral_value(left) ||
        !cga_pmg_maximal_ancestral_value(right)) {
        return CGA_STATUS_INVALID_GRAPH;
    }

    equivalent = cga_pmg_markov_equivalence_value(left, right);
    *out_equivalent = equivalent;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_m_separated(
    const cga_pmg16 *graph,
    uint16_t left,
    uint16_t right,
    uint16_t given,
    bool *out_separated)
{
    cga_edge_atom atoms[CGA_MAX_EDGE_ATOMS];
    size_t atom_count;
    bool separated;

    if (graph == NULL || out_separated == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_query_masks_are_valid(
            graph->vertex_count, left, right, given)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_pmg_ancestral_value(graph)) {
        return CGA_STATUS_INVALID_GRAPH;
    }

    atom_count = cga_pmg_edge_atoms(graph, atoms);
    separated = cga_m_separated_atoms(
        atoms, atom_count, left, right, given);
    *out_separated = separated;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_induced_subgraph(
    const cga_pmg16 *source,
    uint16_t retained,
    cga_pmg16 **out_graph)
{
    cga_pmg16 *result = NULL;
    size_t old_to_new[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {0};
    size_t retained_count;
    size_t old_first;
    size_t old_second;
    cga_status status;

    if (source == NULL || out_graph == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_retained_mask_is_valid(source->vertex_count, retained)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_pmg_storage_is_well_formed(source)) {
        return CGA_STATUS_INVALID_GRAPH;
    }

    retained_count = cga_compaction_map(
        source->vertex_count, retained, old_to_new);
    status = cga_pmg16_create(retained_count, &result);
    if (status != CGA_STATUS_OK) {
        return status;
    }

    for (old_first = 0u; old_first < source->vertex_count; ++old_first) {
        if (old_to_new[old_first] == SIZE_MAX) {
            continue;
        }
        for (old_second = old_first + 1u;
             old_second < source->vertex_count;
             ++old_second) {
            cga_endpoint_mark first_mark;
            cga_endpoint_mark second_mark;
            size_t new_first;
            size_t new_second;

            if (old_to_new[old_second] == SIZE_MAX) {
                continue;
            }
            first_mark = cga_pmg_mark(source, old_first, old_second);
            if (first_mark == CGA_ENDPOINT_NONE) {
                continue;
            }
            second_mark = cga_pmg_mark(source, old_second, old_first);
            new_first = old_to_new[old_first];
            new_second = old_to_new[old_second];
            cga_put_pmg_mark(
                &result->endpoint_marks[new_first], new_second, first_mark);
            cga_put_pmg_mark(
                &result->endpoint_marks[new_second], new_first, second_mark);
        }
    }

    *out_graph = result;
    return CGA_STATUS_OK;
}

cga_status cga_directed_bidirected16_create(
    size_t vertex_count,
    cga_directed_bidirected16 **out_graph)
{
    cga_directed_bidirected16 *graph;

    if (out_graph == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (vertex_count > CGA_BACKEND_MAX_TOPOLOGY_VERTICES) {
        return CGA_STATUS_UNSUPPORTED_SCOPE;
    }

    graph = CGA_CALLOC((size_t)1u, sizeof(*graph));
    if (graph == NULL) {
        return CGA_STATUS_ALLOCATION_FAILED;
    }
    graph->vertex_count = vertex_count;
    *out_graph = graph;
    return CGA_STATUS_OK;
}

void cga_directed_bidirected16_destroy(cga_directed_bidirected16 *graph)
{
    free(graph);
}

cga_status cga_directed_bidirected16_vertex_count(
    const cga_directed_bidirected16 *graph,
    size_t *out_vertex_count)
{
    if (graph == NULL || out_vertex_count == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    *out_vertex_count = graph->vertex_count;
    return CGA_STATUS_OK;
}

cga_status cga_directed_bidirected16_set_directed(
    cga_directed_bidirected16 *graph,
    size_t source,
    size_t target,
    bool present)
{
    const uint16_t target_bit =
        target < CGA_BACKEND_MAX_TOPOLOGY_VERTICES
            ? cga_vertex_bit(target)
            : UINT16_C(0);

    if (graph == NULL ||
        !cga_valid_distinct_pair(graph->vertex_count, source, target)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    if (present) {
        graph->directed_out[source] |= target_bit;
    } else {
        graph->directed_out[source] &= (uint16_t)~target_bit;
    }
    return CGA_STATUS_OK;
}

cga_status cga_directed_bidirected16_set_bidirected(
    cga_directed_bidirected16 *graph,
    size_t first,
    size_t second,
    bool present)
{
    uint16_t first_row;
    uint16_t second_row;
    uint16_t first_bit;
    uint16_t second_bit;

    if (graph == NULL ||
        !cga_valid_distinct_pair(graph->vertex_count, first, second)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }

    first_bit = cga_vertex_bit(first);
    second_bit = cga_vertex_bit(second);
    first_row = graph->bidirected[first];
    second_row = graph->bidirected[second];
    if (present) {
        first_row |= second_bit;
        second_row |= first_bit;
    } else {
        first_row &= (uint16_t)~second_bit;
        second_row &= (uint16_t)~first_bit;
    }
    graph->bidirected[first] = first_row;
    graph->bidirected[second] = second_row;
    return CGA_STATUS_OK;
}

cga_status cga_directed_bidirected16_has_directed(
    const cga_directed_bidirected16 *graph,
    size_t source,
    size_t target,
    bool *out_present)
{
    bool present;

    if (graph == NULL || out_present == NULL ||
        !cga_valid_distinct_pair(graph->vertex_count, source, target)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    present =
        (graph->directed_out[source] & cga_vertex_bit(target)) != UINT16_C(0);
    *out_present = present;
    return CGA_STATUS_OK;
}

cga_status cga_directed_bidirected16_has_bidirected(
    const cga_directed_bidirected16 *graph,
    size_t first,
    size_t second,
    bool *out_present)
{
    bool forward;
    bool reverse;

    if (graph == NULL || out_present == NULL ||
        !cga_valid_distinct_pair(graph->vertex_count, first, second)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    forward =
        (graph->bidirected[first] & cga_vertex_bit(second)) != UINT16_C(0);
    reverse =
        (graph->bidirected[second] & cga_vertex_bit(first)) != UINT16_C(0);
    if (forward != reverse) {
        return CGA_STATUS_INVALID_GRAPH;
    }
    *out_present = forward;
    return CGA_STATUS_OK;
}

cga_status cga_directed_bidirected16_m_separated(
    const cga_directed_bidirected16 *graph,
    uint16_t left,
    uint16_t right,
    uint16_t given,
    bool *out_separated)
{
    cga_edge_atom atoms[CGA_MAX_EDGE_ATOMS];
    size_t atom_count;
    bool separated;

    if (graph == NULL || out_separated == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_query_masks_are_valid(
            graph->vertex_count, left, right, given)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_db_is_acyclic(graph)) {
        return CGA_STATUS_INVALID_GRAPH;
    }

    atom_count = cga_db_edge_atoms(graph, atoms);
    separated = cga_m_separated_atoms(
        atoms, atom_count, left, right, given);
    *out_separated = separated;
    return CGA_STATUS_OK;
}

static cga_status cga_db_compact_copy(
    const cga_directed_bidirected16 *source,
    uint16_t retained,
    cga_directed_bidirected16 **out_graph)
{
    cga_directed_bidirected16 *result = NULL;
    size_t old_to_new[CGA_BACKEND_MAX_TOPOLOGY_VERTICES] = {0};
    size_t retained_count;
    size_t old_first;
    size_t old_second;
    cga_status status;

    retained_count = cga_compaction_map(
        source->vertex_count, retained, old_to_new);
    status = cga_directed_bidirected16_create(retained_count, &result);
    if (status != CGA_STATUS_OK) {
        return status;
    }

    for (old_first = 0u; old_first < source->vertex_count; ++old_first) {
        const size_t new_first = old_to_new[old_first];

        if (new_first == SIZE_MAX) {
            continue;
        }
        for (old_second = 0u;
             old_second < source->vertex_count;
             ++old_second) {
            const size_t new_second = old_to_new[old_second];

            if (new_second != SIZE_MAX &&
                (source->directed_out[old_first] &
                 cga_vertex_bit(old_second)) != UINT16_C(0)) {
                result->directed_out[new_first] |=
                    cga_vertex_bit(new_second);
            }
        }
        for (old_second = old_first + 1u;
             old_second < source->vertex_count;
             ++old_second) {
            const size_t new_second = old_to_new[old_second];

            if (new_second != SIZE_MAX &&
                (source->bidirected[old_first] &
                 cga_vertex_bit(old_second)) != UINT16_C(0)) {
                cga_put_bidirected(
                    result->bidirected, new_first, new_second);
            }
        }
    }

    *out_graph = result;
    return CGA_STATUS_OK;
}

cga_status cga_directed_bidirected16_induced_subgraph(
    const cga_directed_bidirected16 *source,
    uint16_t retained,
    cga_directed_bidirected16 **out_graph)
{
    if (source == NULL || out_graph == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_retained_mask_is_valid(source->vertex_count, retained)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_db_storage_is_well_formed(source)) {
        return CGA_STATUS_INVALID_GRAPH;
    }
    return cga_db_compact_copy(source, retained, out_graph);
}

cga_status cga_directed_bidirected16_latent_projection(
    const cga_directed_bidirected16 *source,
    uint16_t retained,
    cga_directed_bidirected16 **out_graph)
{
    cga_directed_bidirected16 staged;
    uint16_t active;
    size_t latent;

    if (source == NULL || out_graph == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_retained_mask_is_valid(source->vertex_count, retained)) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    if (!cga_db_is_acyclic(source)) {
        return CGA_STATUS_INVALID_GRAPH;
    }

    staged = *source;
    active = cga_valid_vertex_mask(source->vertex_count);
    for (latent = 0u; latent < source->vertex_count; ++latent) {
        const uint16_t latent_bit = cga_vertex_bit(latent);
        uint16_t parents = UINT16_C(0);
        uint16_t children;
        uint16_t spouses;
        size_t vertex;

        if ((retained & latent_bit) != UINT16_C(0)) {
            continue;
        }
        children = (uint16_t)(staged.directed_out[latent] & active);
        spouses = (uint16_t)(staged.bidirected[latent] & active);
        for (vertex = 0u; vertex < source->vertex_count; ++vertex) {
            if ((active & cga_vertex_bit(vertex)) != UINT16_C(0) &&
                (staged.directed_out[vertex] & latent_bit) != UINT16_C(0)) {
                parents |= cga_vertex_bit(vertex);
            }
        }

        for (vertex = 0u; vertex < source->vertex_count; ++vertex) {
            const uint16_t vertex_bit = cga_vertex_bit(vertex);

            if ((parents & vertex_bit) != UINT16_C(0)) {
                staged.directed_out[vertex] |=
                    (uint16_t)(children & (uint16_t)~vertex_bit);
            }
        }
        for (vertex = 0u; vertex < source->vertex_count; ++vertex) {
            const uint16_t vertex_bit = cga_vertex_bit(vertex);
            size_t other;

            if ((children & vertex_bit) == UINT16_C(0)) {
                continue;
            }
            for (other = vertex + 1u;
                 other < source->vertex_count;
                 ++other) {
                if ((children & cga_vertex_bit(other)) != UINT16_C(0)) {
                    cga_put_bidirected(staged.bidirected, vertex, other);
                }
            }
        }
        for (vertex = 0u; vertex < source->vertex_count; ++vertex) {
            const uint16_t vertex_bit = cga_vertex_bit(vertex);
            size_t child;

            if ((spouses & vertex_bit) == UINT16_C(0)) {
                continue;
            }
            for (child = 0u; child < source->vertex_count; ++child) {
                if (child != vertex &&
                    (children & cga_vertex_bit(child)) != UINT16_C(0)) {
                    cga_put_bidirected(
                        staged.bidirected, vertex, child);
                }
            }
        }

        staged.directed_out[latent] = UINT16_C(0);
        staged.bidirected[latent] = UINT16_C(0);
        for (vertex = 0u; vertex < source->vertex_count; ++vertex) {
            staged.directed_out[vertex] &= (uint16_t)~latent_bit;
            staged.bidirected[vertex] &= (uint16_t)~latent_bit;
        }
        active &= (uint16_t)~latent_bit;
    }

    if (!cga_db_is_acyclic(&staged)) {
        return CGA_STATUS_INTERNAL_ERROR;
    }
    return cga_db_compact_copy(&staged, retained, out_graph);
}
