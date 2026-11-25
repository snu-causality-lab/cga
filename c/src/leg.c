#include "cga/leg.h"

#include <stdlib.h>
#include <string.h>

#define CGA_LEG_VERTICES ((size_t)16u)
#define CGA_LEG_EDGES ((size_t)120u)

#ifndef CGA_CALLOC
#define CGA_CALLOC calloc
#endif

/* Directions: 1 = first -> second, 2 = first <- second, 3 = bidirected. */
typedef struct {
    size_t first;
    size_t second;
    unsigned char direction;
    unsigned char root_direction;
} cga_leg_edge;

struct cga_leg16_iterator {
    size_t vertex_count;
    size_t edge_count;
    cga_leg_edge edges[CGA_LEG_EDGES];
    uint16_t parents[CGA_LEG_VERTICES];
    uint16_t spouses[CGA_LEG_VERTICES];
    /* One current graph; each DFS frame stores only traversal edge indices. */
    size_t next_edge[CGA_LEG_EDGES + 1u];
    size_t incoming_edge[CGA_LEG_EDGES + 1u];
    size_t depth;
    bool started;
    bool exhausted;
};

static uint16_t leg_bit(size_t vertex)
{
    return (uint16_t)(UINT32_C(1) << vertex);
}

static cga_status leg_export(
    const cga_leg16_iterator *state, cga_pmg16 **out_graph)
{
    cga_pmg16 *graph = NULL;
    cga_status status = cga_pmg16_create(state->vertex_count, &graph);
    size_t i;

    if (status != CGA_STATUS_OK) {
        return status;
    }
    for (i = 0u; i < state->edge_count; ++i) {
        const cga_leg_edge *edge = &state->edges[i];
        const cga_endpoint_mark first_mark = edge->direction == 1u ?
            CGA_ENDPOINT_TAIL : CGA_ENDPOINT_ARROW;
        const cga_endpoint_mark second_mark = edge->direction == 2u ?
            CGA_ENDPOINT_TAIL : CGA_ENDPOINT_ARROW;

        status = cga_pmg16_set_edge(
            graph, edge->first, first_mark, edge->second, second_mark);
        if (status != CGA_STATUS_OK) {
            cga_pmg16_destroy(graph);
            return status;
        }
    }
    *out_graph = graph;
    return CGA_STATUS_OK;
}

/* MCS chooses the lowest vertex index among equal cardinalities. Earlier
 * selected neighbors must form a clique; orienting earlier -> later then
 * gives an acyclic orientation without a new circle-subgraph collider. */
static cga_status leg_circle_order(
    size_t vertex_count,
    const uint16_t circles[CGA_LEG_VERTICES],
    size_t rank[CGA_LEG_VERTICES])
{
    size_t weights[CGA_LEG_VERTICES] = {0u};
    uint16_t selected = UINT16_C(0);
    size_t step;

    for (step = 0u; step < vertex_count; ++step) {
        size_t best = SIZE_MAX;
        size_t vertex;
        uint16_t earlier;

        for (vertex = 0u; vertex < vertex_count; ++vertex) {
            if ((selected & leg_bit(vertex)) == UINT16_C(0) &&
                (best == SIZE_MAX || weights[vertex] > weights[best])) {
                best = vertex;
            }
        }
        if (best == SIZE_MAX) {
            return CGA_STATUS_INTERNAL_ERROR;
        }
        earlier = (uint16_t)(circles[best] & selected);
        for (vertex = 0u; vertex < vertex_count; ++vertex) {
            if ((earlier & leg_bit(vertex)) != UINT16_C(0) &&
                (earlier & (uint16_t)~(circles[vertex] | leg_bit(vertex))) !=
                    UINT16_C(0)) {
                return CGA_STATUS_INVALID_GRAPH;
            }
            if ((circles[best] & leg_bit(vertex)) != UINT16_C(0)) {
                ++weights[vertex];
            }
        }
        rank[best] = step;
        selected |= leg_bit(best);
    }
    return CGA_STATUS_OK;
}

static cga_status leg_seed(
    const cga_pmg16 *pag,
    cga_leg16_iterator *state,
    cga_pmg16 **out_graph)
{
    uint16_t circles[CGA_LEG_VERTICES] = {UINT16_C(0)};
    size_t rank[CGA_LEG_VERTICES] = {0u};
    cga_pmg16 *graph = NULL;
    cga_status status;
    bool valid = false;
    size_t first;
    size_t second;
    size_t i;

    memset(state, 0, sizeof(*state));
    status = cga_pmg16_vertex_count(pag, &state->vertex_count);
    if (status != CGA_STATUS_OK) {
        return status;
    }
    if (state->vertex_count > CGA_LEG_VERTICES) {
        return CGA_STATUS_UNSUPPORTED_SCOPE;
    }
    for (first = 0u; first < state->vertex_count; ++first) {
        for (second = first + 1u; second < state->vertex_count; ++second) {
            cga_endpoint_mark am;
            cga_endpoint_mark bm;
            cga_leg_edge *edge;

            status = cga_pmg16_get_edge(pag, first, second, &am, &bm);
            if (status != CGA_STATUS_OK) {
                return status;
            }
            if (am == CGA_ENDPOINT_NONE) {
                continue;
            }
            if ((am == CGA_ENDPOINT_TAIL && bm != CGA_ENDPOINT_ARROW) ||
                (bm == CGA_ENDPOINT_TAIL && am != CGA_ENDPOINT_ARROW)) {
                return CGA_STATUS_UNSUPPORTED_SCOPE;
            }
            edge = &state->edges[state->edge_count++];
            edge->first = first;
            edge->second = second;
            if (am == CGA_ENDPOINT_CIRCLE && bm == CGA_ENDPOINT_CIRCLE) {
                circles[first] |= leg_bit(second);
                circles[second] |= leg_bit(first);
                /* Zero is an internal pending circle-circle orientation. */
            } else if (am == CGA_ENDPOINT_ARROW) {
                edge->direction = bm == CGA_ENDPOINT_ARROW ? 3u : 2u;
            } else {
                edge->direction = 1u;
            }
        }
    }
    status = leg_circle_order(state->vertex_count, circles, rank);
    if (status != CGA_STATUS_OK) {
        return status;
    }
    for (i = 0u; i < state->edge_count; ++i) {
        cga_leg_edge *edge = &state->edges[i];

        if (edge->direction == 0u) {
            edge->direction = rank[edge->first] < rank[edge->second] ? 1u : 2u;
        }
        edge->root_direction = edge->direction;
        if (edge->direction == 3u) {
            state->spouses[edge->first] |= leg_bit(edge->second);
            state->spouses[edge->second] |= leg_bit(edge->first);
        } else if (edge->direction == 1u) {
            state->parents[edge->second] |= leg_bit(edge->first);
        } else {
            state->parents[edge->first] |= leg_bit(edge->second);
        }
    }
    status = leg_export(state, &graph);
    if (status != CGA_STATUS_OK) {
        return status;
    }
    /* This validates the resulting MAG, not the caller's PAG-closure claim. */
    status = cga_pmg16_is_maximal_ancestral(graph, &valid);
    if (status != CGA_STATUS_OK || !valid) {
        cga_pmg16_destroy(graph);
        return status == CGA_STATUS_OK ? CGA_STATUS_INVALID_GRAPH : status;
    }
    *out_graph = graph;
    return CGA_STATUS_OK;
}

static bool leg_reversible(const cga_leg16_iterator *state, size_t index)
{
    const cga_leg_edge *edge = &state->edges[index];
    const size_t source = edge->direction == 1u ? edge->first : edge->second;
    const size_t target = edge->direction == 1u ? edge->second : edge->first;

    return edge->direction != 3u &&
        state->parents[target] ==
            (uint16_t)(state->parents[source] | leg_bit(source)) &&
        state->spouses[source] == state->spouses[target];
}

static void leg_reverse(cga_leg16_iterator *state, size_t index)
{
    cga_leg_edge *edge = &state->edges[index];
    const size_t source = edge->direction == 1u ? edge->first : edge->second;
    const size_t target = edge->direction == 1u ? edge->second : edge->first;

    state->parents[target] &= (uint16_t)~leg_bit(source);
    state->parents[source] |= leg_bit(target);
    edge->direction = (unsigned char)(3u - edge->direction);
}

static size_t leg_parent_edge(const cga_leg16_iterator *state)
{
    size_t i;

    for (i = 0u; i < state->edge_count; ++i) {
        if (state->edges[i].direction != state->edges[i].root_direction &&
            leg_reversible(state, i)) {
            return i;
        }
    }
    return SIZE_MAX;
}

static cga_status leg_advance(cga_leg16_iterator *state, bool *out_has_graph)
{
    if (!state->started) {
        state->started = true;
        *out_has_graph = true;
        return CGA_STATUS_OK;
    }
    while (!state->exhausted) {
        size_t index;

        if (state->next_edge[state->depth] == state->edge_count) {
            if (state->depth == 0u) {
                state->exhausted = true;
                break;
            }
            leg_reverse(state, state->incoming_edge[state->depth]);
            --state->depth;
            continue;
        }
        index = state->next_edge[state->depth]++;
        if (!leg_reversible(state, index)) {
            continue;
        }
        leg_reverse(state, index);
        if (leg_parent_edge(state) != index) {
            leg_reverse(state, index);
            continue;
        }
        /* Parent steps decrease root disagreement by one, so depth <= m. */
        if (state->depth >= state->edge_count) {
            return CGA_STATUS_INTERNAL_ERROR;
        }
        ++state->depth;
        state->incoming_edge[state->depth] = index;
        state->next_edge[state->depth] = 0u;
        *out_has_graph = true;
        return CGA_STATUS_OK;
    }
    *out_has_graph = false;
    return CGA_STATUS_OK;
}

cga_status cga_pmg16_representative_leg(
    const cga_pmg16 *pag, cga_pmg16 **out_graph)
{
    cga_leg16_iterator state;

    if (pag == NULL || out_graph == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    return leg_seed(pag, &state, out_graph);
}

cga_status cga_leg16_iterator_create(
    const cga_pmg16 *pag, cga_leg16_iterator **out_iterator)
{
    cga_leg16_iterator state;
    cga_leg16_iterator *iterator;
    cga_pmg16 *seed = NULL;
    cga_status status;

    if (pag == NULL || out_iterator == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    status = leg_seed(pag, &state, &seed);
    if (status != CGA_STATUS_OK) {
        return status;
    }
    cga_pmg16_destroy(seed);
    iterator = CGA_CALLOC((size_t)1u, sizeof(*iterator));
    if (iterator == NULL) {
        return CGA_STATUS_ALLOCATION_FAILED;
    }
    *iterator = state;
    *out_iterator = iterator;
    return CGA_STATUS_OK;
}

cga_status cga_leg16_iterator_next(
    cga_leg16_iterator *iterator,
    cga_pmg16 **out_graph,
    bool *out_has_graph)
{
    cga_leg16_iterator next;
    cga_pmg16 *graph = NULL;
    cga_status status;
    bool has_graph = false;

    if (iterator == NULL || out_graph == NULL || out_has_graph == NULL) {
        return CGA_STATUS_INVALID_ARGUMENT;
    }
    /* A bounded local copy makes every failure, including export, retryable. */
    next = *iterator;
    status = leg_advance(&next, &has_graph);
    if (status != CGA_STATUS_OK) {
        return status;
    }
    if (has_graph) {
        status = leg_export(&next, &graph);
        if (status != CGA_STATUS_OK) {
            return status;
        }
    }
    *iterator = next;
    *out_graph = graph;
    *out_has_graph = has_graph;
    return CGA_STATUS_OK;
}

void cga_leg16_iterator_destroy(cga_leg16_iterator *iterator)
{
    free(iterator);
}
